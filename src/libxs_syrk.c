/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_syrk.h>
#include <libxs/libxs_malloc.h>
#include <libxs/libxs_sync.h>
#include "libxs_gemm_internal.h"
#include "libxs_mpi.h"

/* widest tile: narrower ones pack A more often, wider ones balance worse */
#if !defined(LIBXS_SYRK_WIDTH)
# define LIBXS_SYRK_WIDTH 256
#endif
/* shapes up to this size are served by a generated kernel, if there is one */
#if !defined(LIBXS_SYRK_KERNEL_N)
# define LIBXS_SYRK_KERNEL_N 32
#endif
#if !defined(LIBXS_SYRK_KERNEL_K)
# define LIBXS_SYRK_KERNEL_K 48
#endif
/* n*k beyond which a BLAS SYRK beats a kernel, which computes both triangles */
#if !defined(LIBXS_SYRK_KERNEL_NK)
# define LIBXS_SYRK_KERNEL_NK 768
#endif
/* extent of the squares on the diagonal that are computed in full and then scattered */
#define INTERNAL_SYRK_SQUARE 64
#if (INTERNAL_SYRK_SQUARE < LIBXS_SYRK_KERNEL_N)
# error "LIBXS_SYRK_KERNEL_N exceeds the square of the diagonal"
#endif
/* panels in flight of a distributed update: one is reduced while the next one is computed */
#define INTERNAL_SYRK_WINDOW 2
/* pauses of a task without work before it yields the core */
#define INTERNAL_SYRK_NPAUSE 64
/* integers the ranks compare at dispatch */
#define INTERNAL_SYRK_NAGREE 13
#define INTERNAL_SYRK_DIST_COUNT 0
#define INTERNAL_SYRK_DIST_PACK 1
#define INTERNAL_SYRK_DIST_UNPACK 2

/* C := alpha*T (or alpha*(T + T^T)) + beta*C in the triangle of a square, a loop per case to vectorize */
#define INTERNAL_SYRK_SCATTER(TYPE, CC, LDC, T, N, UPPER, TWOK, ALPHA, BETA) \
  do { \
    int ii_, jj_; \
    for (jj_ = 0; jj_ < (N); ++jj_) { \
      const int begin_ = ((UPPER) ? 0 : jj_), end_ = ((UPPER) ? (jj_ + 1) : (N)); \
      TYPE *const cc_ = (TYPE*)(CC) + (size_t)jj_ * (LDC); \
      const TYPE *const tt_ = (const TYPE*)(T) + (size_t)jj_ * (N); \
      const TYPE *const tr_ = (const TYPE*)(T) + jj_; \
      if (!(TWOK) && 0 == (BETA)) { \
        LIBXS_PRAGMA_SIMD \
        for (ii_ = begin_; ii_ < end_; ++ii_) cc_[ii_] = (ALPHA) * tt_[ii_]; \
      } \
      else if (!(TWOK)) { \
        LIBXS_PRAGMA_SIMD \
        for (ii_ = begin_; ii_ < end_; ++ii_) cc_[ii_] = (ALPHA) * tt_[ii_] + (BETA) * cc_[ii_]; \
      } \
      else if (0 == (BETA)) { \
        LIBXS_PRAGMA_SIMD \
        for (ii_ = begin_; ii_ < end_; ++ii_) cc_[ii_] = (ALPHA) * (tt_[ii_] + tr_[ii_ * (N)]); \
      } \
      else { \
        LIBXS_PRAGMA_SIMD \
        for (ii_ = begin_; ii_ < end_; ++ii_) { \
          cc_[ii_] = (ALPHA) * (tt_[ii_] + tr_[ii_ * (N)]) + (BETA) * cc_[ii_]; \
        } \
      } \
    } \
  } while(0)

/* C := beta*C in a rectangle, or in the triangle of a square */
#define INTERNAL_SYRK_SCALE(TYPE, CC, LDC, M, N, DIAG, UPPER, BETA) \
  do { \
    int ii_, jj_; \
    for (jj_ = 0; jj_ < (N); ++jj_) { \
      const int begin_ = ((DIAG) && !(UPPER)) ? jj_ : 0; \
      const int end_ = ((DIAG) && (UPPER)) ? (jj_ + 1) : (M); \
      TYPE *const cc_ = (TYPE*)(CC) + (size_t)jj_ * (LDC); \
      for (ii_ = begin_; ii_ < end_; ++ii_) cc_[ii_] = (0 == (BETA) ? (TYPE)0 : ((BETA) * cc_[ii_])); \
    } \
  } while(0)


/* trailing hidden lengths of uplo and trans as with libxs_gemm_dblas_t */
typedef void (*internal_libxs_dsyrk_t)(const char*, const char*, const int*, const int*,
  const double*, const double*, const int*, const double*, double*, const int*,
  size_t, size_t);
typedef void (*internal_libxs_ssyrk_t)(const char*, const char*, const int*, const int*,
  const float*, const float*, const int*, const float*, float*, const int*,
  size_t, size_t);
typedef void (*internal_libxs_dsyr2k_t)(const char*, const char*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*, size_t, size_t);
typedef void (*internal_libxs_ssyr2k_t)(const char*, const char*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*, size_t, size_t);

/* The arguments of one update, as every task passes them */
typedef struct internal_libxs_syrk_args_t {
  const libxs_syrk_t* plan;
  const void *a, *b;
  void* c;
  double alpha, beta;
} internal_libxs_syrk_args_t;

/* Tile of the triangle: m rows from r0 and w columns from c0 (the first tile of a panel is the diagonal) */
typedef struct internal_libxs_syrk_tile_t {
  int r0, c0, m, w, diag;
} internal_libxs_syrk_tile_t;

/* Panel of a distributed update: columns c0 to c0+pw of C, rows r0 to r0+nrows */
typedef struct internal_libxs_syrk_panel_t {
  const libxs_pdist_t* layout;
  int upper, c0, pw, r0, nrows, ldc;
} internal_libxs_syrk_panel_t;

/* Shared between the tasks and hence a cache line of its own */
typedef union internal_libxs_syrk_word_t {
  volatile int i;
  char pad[LIBXS_CACHELINE];
} internal_libxs_syrk_word_t;

/**
 * State of a distributed plan. The tasks claim tiles in the order of panels (next), count what is
 * complete per panel (done), and one at a time (busy) reduces the panels in order (reduced). A task
 * waits only for what another one holds, which is running by definition: hence there is neither a
 * team nor a barrier. The last task to leave (left) rewinds the state for the next update.
 */
typedef struct internal_libxs_syrk_state_t {
  internal_libxs_syrk_word_t next, reduced, busy, left, failed;
  internal_libxs_mpi_comm_t comm;
  libxs_pdist_t layout;
  volatile int* done;
  int* first;
  char *panel[INTERNAL_SYRK_WINDOW], *recvbuf, *sendbuf;
  int npanels, nwindow, serialized;
} internal_libxs_syrk_state_t;


LIBXS_APIVAR_DEFINE(internal_libxs_dsyrk_t internal_libxs_dsyrk_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_ssyrk_t internal_libxs_ssyrk_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_dsyr2k_t internal_libxs_dsyr2k_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_ssyr2k_t internal_libxs_ssyr2k_blas);


LIBXS_API_INTERN void internal_libxs_syrk_init(void)
{
#if defined(LIBXS_INTERCEPT_DYNAMIC)
  const char *const env = getenv("LIBXS_SYRK_BLAS");
  const int syrk_blas = (NULL == env ? 1/*default*/ : atoi(env));
  void* dl;
  if (0 != syrk_blas) {
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(dsyrk)));
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_dsyrk_t, internal_libxs_dsyrk_blas, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(ssyrk)));
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_ssyrk_t, internal_libxs_ssyrk_blas, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(dsyr2k)));
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_dsyr2k_t, internal_libxs_dsyr2k_blas, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(ssyr2k)));
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_ssyr2k_t, internal_libxs_ssyr2k_blas, dl);
    }
  }
#endif
}


/**
 * Tiles depend on n alone: neither threads nor tasks move a boundary. A small n has narrower tiles
 * to split at all, a medium one square tiles to balance, and a large one taller tiles to pack less.
 */
LIBXS_API_INLINE void internal_libxs_syrk_tiling(int n, int* width, int* height)
{
  const int full = LIBXS_SYRK_WIDTH;
  const int w = (n <= full ? (full / 2) : (n < 3 * full ? (3 * full / 4) : full));
  *width = LIBXS_MAX(w, 1);
  *height = (n < 6 * full ? *width : (2 * *width));
}


/* Tiles of panel p: the diagonal and the rest of its part of the triangle. */
LIBXS_API_INLINE int internal_libxs_syrk_panel_ntiles(const libxs_syrk_t* plan, int p)
{
  const int n = plan->shape.n, c0 = p * plan->width, pw = LIBXS_MIN(plan->width, n - c0);
  const int rest = ('U' == plan->shape.uplo ? c0 : (n - c0 - pw));
  return 1 + LIBXS_UPDIV(rest, plan->height);
}


LIBXS_API_INLINE void internal_libxs_syrk_tile(const libxs_syrk_t* plan, int p, int i,
  internal_libxs_syrk_tile_t* tile)
{
  const int n = plan->shape.n, upper = ('U' == plan->shape.uplo);
  tile->c0 = p * plan->width;
  tile->w = LIBXS_MIN(plan->width, n - tile->c0);
  tile->diag = (0 == i);
  if (0 != tile->diag) {
    tile->r0 = tile->c0;
    tile->m = tile->w;
  }
  else {
    const int begin = (0 != upper ? 0 : (tile->c0 + tile->w));
    const int end = (0 != upper ? tile->c0 : n);
    tile->r0 = begin + (i - 1) * plan->height;
    tile->m = LIBXS_MIN(plan->height, end - tile->r0);
  }
}


/* Operand at index i of n: a row of A, or a column of it if transposed. */
LIBXS_API_INLINE const void* internal_libxs_syrk_operand(const libxs_syrk_t* plan,
  const void* x, int ld, int i)
{
  const size_t offset = ('N' == plan->shape.trans ? (size_t)i : ((size_t)i * ld));
  return (const char*)x + offset * LIBXS_TYPESIZE(plan->shape.datatype);
}


LIBXS_API_INLINE void internal_libxs_syrk_scale(const libxs_syrk_t* plan,
  const internal_libxs_syrk_tile_t* tile, double beta, void* c, int ldc)
{
  const int upper = ('U' == plan->shape.uplo);
  if (1 != beta) {
    if (LIBXS_DATATYPE_F64 == plan->shape.datatype) {
      INTERNAL_SYRK_SCALE(double, c, ldc, tile->m, tile->w, tile->diag, upper, beta);
    }
    else {
      const float fb = (float)beta;
      INTERNAL_SYRK_SCALE(float, c, ldc, tile->m, tile->w, tile->diag, upper, fb);
    }
  }
}


/* Rectangle within the triangle, which BLAS writes without scratch. */
LIBXS_API_INLINE void internal_libxs_syrk_rect(const internal_libxs_syrk_args_t* args,
  const internal_libxs_syrk_tile_t* tile, double alpha, double beta, void* c, int ldc)
{
  const libxs_syrk_t *const plan = args->plan;
  const int k = plan->shape.k, lda = plan->shape.lda, twok = (0 != plan->shape.ldb);
  const int ldb = (0 != twok ? plan->shape.ldb : lda);
  const void *const b = (0 != twok ? args->b : args->a);
  internal_libxs_gemm_blas(&plan->gemm,
    internal_libxs_syrk_operand(plan, args->a, lda, tile->r0),
    internal_libxs_syrk_operand(plan, b, ldb, tile->c0),
    c, tile->m, tile->w, k, lda, ldb, ldc, alpha, beta);
  if (0 != twok) {
    internal_libxs_gemm_blas(&plan->gemm,
      internal_libxs_syrk_operand(plan, b, ldb, tile->r0),
      internal_libxs_syrk_operand(plan, args->a, lda, tile->c0),
      c, tile->m, tile->w, k, ldb, lda, ldc, alpha, 1.0);
  }
}


/* BLAS SYRK or SYR2K of a block on the diagonal, zero if there is no such entry point. */
LIBXS_API_INLINE int internal_libxs_syrk_blas(const internal_libxs_syrk_args_t* args,
  const internal_libxs_syrk_tile_t* tile, double alpha, double beta, void* c, int ldc)
{
  const libxs_syrk_t *const plan = args->plan;
  const char *const uplo = &plan->shape.uplo, *const trans = &plan->shape.trans;
  const int *const n = &tile->w, *const k = &plan->shape.k;
  const int *const lda = &plan->shape.lda, *const ldb = &plan->shape.ldb;
  const int f64 = (LIBXS_DATATYPE_F64 == plan->shape.datatype);
  const void *const a = internal_libxs_syrk_operand(plan, args->a, *lda, tile->c0);
  const float fa = (float)alpha, fb = (float)beta;
  int result = 1;
  if (0 == *ldb) {
    if (0 != f64 && NULL != internal_libxs_dsyrk_blas) {
      internal_libxs_dsyrk_blas(uplo, trans, n, k, &alpha, (const double*)a, lda,
        &beta, (double*)c, &ldc, 1, 1);
    }
    else if (0 == f64 && NULL != internal_libxs_ssyrk_blas) {
      internal_libxs_ssyrk_blas(uplo, trans, n, k, &fa, (const float*)a, lda,
        &fb, (float*)c, &ldc, 1, 1);
    }
    else result = 0;
  }
  else {
    const void *const b = internal_libxs_syrk_operand(plan, args->b, *ldb, tile->c0);
    if (0 != f64 && NULL != internal_libxs_dsyr2k_blas) {
      internal_libxs_dsyr2k_blas(uplo, trans, n, k, &alpha, (const double*)a, lda,
        (const double*)b, ldb, &beta, (double*)c, &ldc, 1, 1);
    }
    else if (0 == f64 && NULL != internal_libxs_ssyr2k_blas) {
      internal_libxs_ssyr2k_blas(uplo, trans, n, k, &fa, (const float*)a, lda,
        (const float*)b, ldb, &fb, (float*)c, &ldc, 1, 1);
    }
    else result = 0;
  }
  return result;
}


/**
 * Block on the diagonal: a generated kernel if the plan has one (a small shape), the BLAS SYRK,
 * or otherwise small squares, which are computed in full on the stack and scattered.
 */
LIBXS_API_INLINE void internal_libxs_syrk_diag(const internal_libxs_syrk_args_t* args,
  const internal_libxs_syrk_tile_t* tile, double alpha, double beta, void* c, int ldc)
{
  const libxs_syrk_t *const plan = args->plan;
  const libxs_gemm_config_t *const gemm = &plan->gemm;
  const int kernel = (NULL == plan->dist && 1 == plan->ntiles
    && ((NULL != gemm->jitter && (NULL != gemm->dgemm_jit || NULL != gemm->sgemm_jit))
      || NULL != gemm->xgemm));
  if (0 != kernel || 0 == internal_libxs_syrk_blas(args, tile, alpha, beta, c, ldc)) {
    const int upper = ('U' == plan->shape.uplo), twok = (0 != plan->shape.ldb);
    const int lda = plan->shape.lda, ldb = (0 != twok ? plan->shape.ldb : lda);
    const int f64 = (LIBXS_DATATYPE_F64 == plan->shape.datatype);
    const size_t elemsize = LIBXS_TYPESIZE(plan->shape.datatype);
    const int step = (0 != kernel ? tile->w : INTERNAL_SYRK_SQUARE);
    const void *const b = (0 != twok ? args->b : args->a);
    union { double d[INTERNAL_SYRK_SQUARE * INTERNAL_SYRK_SQUARE]; float f[1]; } square;
    int j;
    for (j = 0; j < tile->w; j += step) {
      const int size = LIBXS_MIN(step, tile->w - j), pos = tile->c0 + j;
      char *const cc = (char*)c + ((size_t)j * ldc + j) * elemsize;
      internal_libxs_syrk_tile_t part;
      part.c0 = pos; part.w = size; part.diag = 0;
      part.r0 = (0 != upper ? tile->c0 : (pos + size));
      part.m = (0 != upper ? j : (tile->w - j - size));
      if (0 < part.m) {
        internal_libxs_syrk_rect(args, &part, alpha, beta,
          (char*)c + ((size_t)j * ldc + (part.r0 - tile->c0)) * elemsize, ldc);
      }
      if (0 != kernel) libxs_gemm_call(gemm, args->a, b, &square);
      else {
        internal_libxs_gemm_blas(gemm,
          internal_libxs_syrk_operand(plan, args->a, lda, pos),
          internal_libxs_syrk_operand(plan, b, ldb, pos),
          &square, size, size, plan->shape.k, lda, ldb, size, 1.0, 0.0);
      }
      if (0 != f64) {
        INTERNAL_SYRK_SCATTER(double, cc, ldc, &square, size, upper, twok, alpha, beta);
      }
      else {
        const float fa = (float)alpha, fb = (float)beta;
        INTERNAL_SYRK_SCATTER(float, cc, ldc, &square, size, upper, twok, fa, fb);
      }
    }
  }
}


/* A tile of C, or of a partial sum: c points at the tile. */
LIBXS_API_INLINE void internal_libxs_syrk_compute(const internal_libxs_syrk_args_t* args,
  const internal_libxs_syrk_tile_t* tile, double alpha, double beta, void* c, int ldc)
{
  if (0 == args->plan->shape.k || 0 == alpha) internal_libxs_syrk_scale(args->plan, tile, beta, c, ldc);
  else if (0 != tile->diag) internal_libxs_syrk_diag(args, tile, alpha, beta, c, ldc);
  else internal_libxs_syrk_rect(args, tile, alpha, beta, c, ldc);
}


/* Tiles of a task if local: stateless, and split by work unless there is a task per tile. */
LIBXS_API_INLINE void internal_libxs_syrk_local(const internal_libxs_syrk_args_t* args, int tid, int ntasks)
{
  const libxs_syrk_t *const plan = args->plan;
  const int n = plan->shape.n, ldc = plan->shape.ldc;
  const int npanels = LIBXS_UPDIV(n, plan->width);
  const size_t elemsize = LIBXS_TYPESIZE(plan->shape.datatype);
  const double total = (double)n * (n + 1); /* twice the triangle */
  double work = 0;
  int p, i, t = 0;
  for (p = 0; p < npanels; ++p) {
    const int count = internal_libxs_syrk_panel_ntiles(plan, p);
    if (ntasks < plan->ntiles || (t <= tid && tid < t + count)) {
      for (i = 0; i < count; ++i) {
        internal_libxs_syrk_tile_t tile;
        int owner = t + i;
        internal_libxs_syrk_tile(plan, p, i, &tile);
        if (ntasks < plan->ntiles) {
          const double cost = (0 != tile.diag
            ? ((double)tile.w * (tile.w + 1)) : (2.0 * tile.m * tile.w));
          owner = LIBXS_MIN((int)((work + 0.5 * cost) * ntasks / total), ntasks - 1);
          work += cost;
        }
        if (tid == owner) {
          internal_libxs_syrk_compute(args, &tile, args->alpha, args->beta,
            (char*)args->c + ((size_t)tile.c0 * ldc + tile.r0) * elemsize, ldc);
        }
      }
    }
    t += count;
  }
}


/**
 * Walks the elements e0 to e1 of a panel (column-major, the triangle only) in runs of one block of C,
 * and treats those owned by rank who: counts them, packs them from the panel into buf, or adds buf
 * into the local C. Packing and unpacking walk in the same order, hence agree on the stream.
 */
LIBXS_API_INLINE size_t internal_libxs_syrk_dist_runs(const internal_libxs_syrk_panel_t* panel,
  size_t e0, size_t e1, int who, int mode, libxs_data_t datatype, const void* pbuf, void* buf,
  void* c, double alpha, double beta)
{
  const libxs_pdist_t *const layout = panel->layout;
  const size_t elemsize = LIBXS_TYPESIZE(datatype), nrows = (size_t)panel->nrows;
  size_t e = e0, result = 0;
  while (e < e1) {
    const size_t jj = e / nrows, cend = LIBXS_MIN(e1, (jj + 1) * nrows);
    const int j = panel->c0 + (int)jj, d = j - panel->r0;
    const int lo = (0 != panel->upper ? 0 : d), hi = (0 != panel->upper ? (d + 1) : panel->nrows);
    int a = LIBXS_MAX((int)(e - jj * nrows), lo);
    const int b = LIBXS_MIN((int)(cend - jj * nrows), hi);
    while (a < b) {
      const int i = panel->r0 + a, bi = i / layout->mb, bj = j / layout->nb;
      const int run = LIBXS_MIN(b, (bi + 1) * layout->mb - panel->r0) - a;
      const int owner = layout->ranks[(bi + layout->rsrc) % layout->nprow
        + ((bj + layout->csrc) % layout->npcol) * layout->nprow];
      if (who == owner) {
        if (INTERNAL_SYRK_DIST_PACK == mode) {
          memcpy((char*)buf + result * elemsize, (const char*)pbuf + (jj * nrows + a) * elemsize,
            (size_t)run * elemsize);
        }
        else if (INTERNAL_SYRK_DIST_UNPACK == mode) { /* as INDXG2L: the source process is irrelevant */
          const size_t li = (size_t)(bi / layout->nprow) * layout->mb + (size_t)(i % layout->mb);
          const size_t lj = (size_t)(bj / layout->npcol) * layout->nb + (size_t)(j % layout->nb);
          const size_t off = li + lj * panel->ldc;
          int x;
          if (LIBXS_DATATYPE_F64 == datatype) {
            double *const cc = (double*)c + off;
            const double *const bb = (const double*)buf + result;
            if (0 != beta) for (x = 0; x < run; ++x) cc[x] = beta * cc[x] + alpha * bb[x];
            else for (x = 0; x < run; ++x) cc[x] = alpha * bb[x];
          }
          else {
            float *const cc = (float*)c + off;
            const float *const bb = (const float*)buf + result;
            const float fa = (float)alpha, fb = (float)beta;
            if (0 != beta) for (x = 0; x < run; ++x) cc[x] = fb * cc[x] + fa * bb[x];
            else for (x = 0; x < run; ++x) cc[x] = fa * bb[x];
          }
        }
        result += (size_t)run;
      }
      a += run;
    }
    e = cend;
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_syrk_dist_panel(const libxs_syrk_t* plan, int p,
  internal_libxs_syrk_panel_t* panel)
{
  const internal_libxs_syrk_state_t *const state = (const internal_libxs_syrk_state_t*)plan->dist;
  const int n = plan->shape.n;
  panel->layout = &state->layout;
  panel->upper = ('U' == plan->shape.uplo);
  panel->ldc = plan->shape.ldc;
  panel->c0 = p * plan->width;
  panel->pw = LIBXS_MIN(plan->width, n - panel->c0);
  panel->r0 = (0 != panel->upper ? 0 : panel->c0);
  panel->nrows = (0 != panel->upper ? (panel->c0 + panel->pw) : (n - panel->c0));
}


/* Partial sum of this rank in a tile of panel p, alpha and beta are applied by the owners of C. */
LIBXS_API_INLINE void internal_libxs_syrk_dist_tile(const internal_libxs_syrk_args_t* args,
  internal_libxs_syrk_state_t* state, int p, int i)
{
  const libxs_syrk_t *const plan = args->plan;
  const size_t elemsize = LIBXS_TYPESIZE(plan->shape.datatype);
  internal_libxs_syrk_panel_t panel;
  internal_libxs_syrk_tile_t tile;
  char* out;
  int compute = (0 != plan->shape.k && 0 != args->alpha), j;
  internal_libxs_syrk_dist_panel(plan, p, &panel);
  internal_libxs_syrk_tile(plan, p, i, &tile);
  out = state->panel[p % state->nwindow]
    + ((size_t)(tile.c0 - panel.c0) * panel.nrows + (tile.r0 - panel.r0)) * elemsize;
  if (0 != compute && (NULL == args->a || (0 != plan->shape.ldb && NULL == args->b))) {
    LIBXS_ATOMIC_STORE(&state->failed.i, 1, LIBXS_ATOMIC_SEQ_CST); /* takes part nevertheless */
    compute = 0;
  }
  if (0 == compute || 0 != tile.diag) { /* a chunk travels with the other triangle of the diagonal */
    for (j = 0; j < tile.w; ++j) {
      memset(out + (size_t)j * panel.nrows * elemsize, 0, (size_t)tile.m * elemsize);
    }
  }
  if (0 != compute) internal_libxs_syrk_compute(args, &tile, 1.0, 0.0, out, panel.nrows);
}


/**
 * One panel: a ring reduce-scatter of equal chunks (every chunk summed in the same rank order, hence
 * reproducible), then every rank sends the elements of its chunk to their owners in C.
 */
LIBXS_API_INLINE int internal_libxs_syrk_dist_reduce(const internal_libxs_syrk_args_t* args,
  internal_libxs_syrk_state_t* state, int p)
{
  const libxs_data_t datatype = args->plan->shape.datatype;
  const size_t elemsize = LIBXS_TYPESIZE(datatype);
  const internal_libxs_mpi_comm_t *const comm = &state->comm;
  const int nranks = comm->size, me = comm->rank;
  const int next = (me + 1) % nranks, prev = (me + nranks - 1) % nranks;
  char *const pbuf = state->panel[p % state->nwindow];
  char *const tbuf = state->recvbuf, *const sbuf = state->sendbuf;
  internal_libxs_syrk_panel_t panel;
  size_t ntotal;
  int result = EXIT_SUCCESS, t;
  internal_libxs_syrk_dist_panel(args->plan, p, &panel);
  ntotal = (size_t)panel.nrows * panel.pw;
  for (t = 0; t < nranks - 1 && EXIT_SUCCESS == result; ++t) {
    const int cs = (me + 2 * nranks - t - 1) % nranks, cr = (me + 2 * nranks - t - 2) % nranks;
    const size_t s0 = ntotal * cs / nranks, s1 = ntotal * (cs + 1) / nranks;
    const size_t r0 = ntotal * cr / nranks, r1 = ntotal * (cr + 1) / nranks;
    result = internal_libxs_mpi_sendrecv(comm, pbuf + s0 * elemsize,
      (s1 - s0) * elemsize, next, tbuf, (r1 - r0) * elemsize, prev);
    if (EXIT_SUCCESS == result) {
      size_t x;
      if (LIBXS_DATATYPE_F64 == datatype) {
        double *const sum = (double*)(void*)pbuf + r0;
        for (x = 0; x < r1 - r0; ++x) sum[x] += ((const double*)(const void*)tbuf)[x];
      }
      else {
        float *const sum = (float*)(void*)pbuf + r0;
        for (x = 0; x < r1 - r0; ++x) sum[x] += ((const float*)(const void*)tbuf)[x];
      }
    }
  }
  for (t = 0; t < nranks && EXIT_SUCCESS == result; ++t) {
    const int dst = (me + t) % nranks, src = (me + nranks - t) % nranks;
    const size_t m0 = ntotal * me / nranks, m1 = ntotal * (me + 1) / nranks;
    const size_t q0 = ntotal * src / nranks, q1 = ntotal * (src + 1) / nranks;
    const size_t nsend = internal_libxs_syrk_dist_runs(&panel, m0, m1, dst,
      INTERNAL_SYRK_DIST_PACK, datatype, pbuf, sbuf, NULL, 0, 0);
    const size_t nrecv = (0 != t ? internal_libxs_syrk_dist_runs(&panel, q0, q1, me,
      INTERNAL_SYRK_DIST_COUNT, datatype, NULL, NULL, NULL, 0, 0) : nsend);
    if (0 != t) {
      result = internal_libxs_mpi_sendrecv(comm, sbuf, nsend * elemsize, dst,
        tbuf, nrecv * elemsize, src);
    }
    if (EXIT_SUCCESS == result && 0 != nrecv) {
      if (NULL != args->c) {
        internal_libxs_syrk_dist_runs(&panel, q0, q1, me, INTERNAL_SYRK_DIST_UNPACK, datatype,
          NULL, 0 != t ? tbuf : sbuf, args->c, args->alpha, args->beta);
      }
      else LIBXS_ATOMIC_STORE(&state->failed.i, 1, LIBXS_ATOMIC_SEQ_CST); /* the exchange goes on */
    }
  }
  return result;
}


/* Task of a distributed update (see internal_libxs_syrk_state_t). */
LIBXS_API_INLINE int internal_libxs_syrk_dist(const internal_libxs_syrk_args_t* args, int ntasks)
{
  const libxs_syrk_t *const plan = args->plan;
  internal_libxs_syrk_state_t *const state = (internal_libxs_syrk_state_t*)plan->dist;
  const int *const first = state->first;
  const int npanels = state->npanels;
  const int communicates = (0 != state->serialized || 0 != internal_libxs_mpi_main_thread());
  int result = EXIT_SUCCESS, npause = 0, current;
  if (0 == communicates && 1 == ntasks) result = EXIT_FAILURE; /* nobody else could communicate */
  else {
    while (npanels > (current = LIBXS_ATOMIC_LOAD(&state->reduced.i, LIBXS_ATOMIC_SEQ_CST))) {
      int progress = 0;
      if (0 != communicates && first[current + 1] - first[current]
        == (int)LIBXS_ATOMIC_LOAD(state->done + current, LIBXS_ATOMIC_SEQ_CST))
      {
        int idle = 0;
        if (LIBXS_ATOMIC_CMPSWP(&state->busy.i, idle, 1, LIBXS_ATOMIC_SEQ_CST)) {
          /* only the holder advances, and another one may have done so meanwhile */
          const int p = LIBXS_ATOMIC_LOAD(&state->reduced.i, LIBXS_ATOMIC_SEQ_CST);
          if (p < npanels && first[p + 1] - first[p]
            == (int)LIBXS_ATOMIC_LOAD(state->done + p, LIBXS_ATOMIC_SEQ_CST))
          {
            if (EXIT_SUCCESS != internal_libxs_syrk_dist_reduce(args, state, p)) {
              LIBXS_ATOMIC_STORE(&state->failed.i, 1, LIBXS_ATOMIC_SEQ_CST);
            }
            LIBXS_ATOMIC_STORE(&state->reduced.i, p + 1, LIBXS_ATOMIC_SEQ_CST);
            progress = 1;
          }
          LIBXS_ATOMIC_STORE(&state->busy.i, 0, LIBXS_ATOMIC_SEQ_CST);
        }
      }
      if (0 == progress) { /* claim the next tile within the window */
        int t = LIBXS_ATOMIC_LOAD(&state->next.i, LIBXS_ATOMIC_SEQ_CST);
        const int end = first[LIBXS_MIN(current + state->nwindow, npanels)];
        if (t < end) {
          const int claimed = t;
          if (LIBXS_ATOMIC_CMPSWP(&state->next.i, t, claimed + 1, LIBXS_ATOMIC_SEQ_CST)) {
            int p = current;
            while (first[p + 1] <= claimed) ++p;
            internal_libxs_syrk_dist_tile(args, state, p, claimed - first[p]);
            LIBXS_ATOMIC_ADD_FETCH(state->done + p, 1, LIBXS_ATOMIC_SEQ_CST);
          }
          progress = 1; /* also if another task was faster to claim */
        }
      }
      if (0 != progress) npause = 0;
      else if (INTERNAL_SYRK_NPAUSE > npause) {
        ++npause;
        LIBXS_SYNC_PAUSE;
      }
      else {
        LIBXS_SYNC_YIELD;
      }
    }
    if (0 != LIBXS_ATOMIC_LOAD(&state->failed.i, LIBXS_ATOMIC_SEQ_CST)) result = EXIT_FAILURE;
    if (ntasks == (int)LIBXS_ATOMIC_ADD_FETCH(&state->left.i, 1, LIBXS_ATOMIC_SEQ_CST)) {
      int p;
      for (p = 0; p < npanels; ++p) LIBXS_ATOMIC_STORE(state->done + p, 0, LIBXS_ATOMIC_SEQ_CST);
      LIBXS_ATOMIC_STORE(&state->next.i, 0, LIBXS_ATOMIC_SEQ_CST);
      LIBXS_ATOMIC_STORE(&state->reduced.i, 0, LIBXS_ATOMIC_SEQ_CST);
      LIBXS_ATOMIC_STORE(&state->failed.i, 0, LIBXS_ATOMIC_SEQ_CST);
      LIBXS_ATOMIC_STORE(&state->left.i, 0, LIBXS_ATOMIC_SEQ_CST);
    }
  }
  return result;
}


/* Rows of the block-cyclic distribution held by grid row (or column) iproc (as ScaLAPACK's NUMROC). */
LIBXS_API_INLINE int internal_libxs_syrk_numroc(int n, int nb, int iproc, int isrc, int nprocs)
{
  const int nblocks = n / nb, mydist = (nprocs + iproc - isrc) % nprocs, extra = nblocks % nprocs;
  int result = (nblocks / nprocs) * nb;
  if (mydist < extra) result += nb;
  else if (mydist == extra) result += n % nb;
  return result;
}


/* A rank's own view of a layout: the grid names valid ranks, this one once at most, and ldc holds its part. */
LIBXS_API_INLINE int internal_libxs_syrk_layout_valid(const libxs_syrk_t* plan,
  const libxs_pdist_t* layout, const internal_libxs_mpi_comm_t* comm)
{
  int result = (NULL != layout->ranks && 0 < layout->mb && 0 < layout->nb
    && 0 < layout->nprow && layout->nprow <= comm->size
    && 0 < layout->npcol && layout->npcol <= comm->size / layout->nprow
    && 0 <= layout->rsrc && layout->rsrc < layout->nprow
    && 0 <= layout->csrc && layout->csrc < layout->npcol);
  if (0 != result) {
    const int ngrid = layout->nprow * layout->npcol;
    int i, nlocal = 0, nfound = 0;
    for (i = 0; i < ngrid && 0 != result; ++i) {
      const int rank = layout->ranks[i];
      if (0 > rank || comm->size <= rank) result = 0;
      else if (comm->rank == rank) {
        nlocal = internal_libxs_syrk_numroc(plan->shape.n, layout->mb,
          i % layout->nprow, layout->rsrc, layout->nprow);
        ++nfound;
      }
    }
    if (1 < nfound || plan->shape.ldc < LIBXS_MAX(nlocal, 1)) result = 0;
  }
  return result;
}


LIBXS_API_INLINE internal_libxs_syrk_state_t* internal_libxs_syrk_state(const libxs_syrk_t* plan,
  const libxs_pdist_t* layout, const internal_libxs_mpi_comm_t* comm)
{
  const size_t elemsize = LIBXS_TYPESIZE(plan->shape.datatype);
  const int npanels = LIBXS_UPDIV(plan->shape.n, plan->width), ngrid = layout->nprow * layout->npcol;
  const int nwindow = LIBXS_CLMP(npanels, 1, INTERNAL_SYRK_WINDOW);
  const size_t nelems = (size_t)plan->shape.n * plan->width;
  const size_t psize = LIBXS_UP2(nelems * elemsize, (size_t)LIBXS_CACHELINE);
  const size_t csize = LIBXS_UP2(LIBXS_UPDIV(nelems, (size_t)comm->size) * elemsize, (size_t)LIBXS_CACHELINE);
  const size_t hsize = LIBXS_UP2(sizeof(internal_libxs_syrk_state_t)
    + sizeof(int) * (2 * ((size_t)npanels + 1) + ngrid), (size_t)LIBXS_CACHELINE);
  internal_libxs_syrk_state_t *const result = (internal_libxs_syrk_state_t*)libxs_malloc(
    libxs_default_pool(), hsize + nwindow * psize + 2 * csize, LIBXS_CACHELINE);
  if (NULL != result) {
    char *const buffers = (char*)result + hsize;
    int *const done = (int*)(result + 1);
    int *const first = done + npanels + 1, *const ranks = first + npanels + 1;
    int i;
    memset(result, 0, hsize);
    for (i = 0; i < npanels; ++i) first[i + 1] = first[i] + internal_libxs_syrk_panel_ntiles(plan, i);
    memcpy(ranks, layout->ranks, sizeof(int) * ngrid);
    for (i = 0; i < nwindow; ++i) result->panel[i] = buffers + i * psize;
    result->recvbuf = buffers + nwindow * psize;
    result->sendbuf = result->recvbuf + csize;
    result->comm = *comm;
    result->layout = *layout;
    result->layout.ranks = ranks;
    result->done = done;
    result->first = first;
    result->npanels = npanels;
    result->nwindow = nwindow;
    result->serialized = internal_libxs_mpi_serialized();
  }
  return result;
}


/**
 * Collective part of the dispatch. The ranks compare what must be equal with their predecessor in a
 * ring, and the verdict is the AND over all: one rank refusing alone would leave the others waiting.
 */
LIBXS_API_INLINE int internal_libxs_syrk_dispatch_dist(libxs_syrk_t* plan, int valid,
  const libxs_pdist_t* layout)
{
  internal_libxs_mpi_comm_t comm;
  /* without (known) MPI, every rank refuses alike */
  int result = (EXIT_SUCCESS == internal_libxs_mpi_comm(&comm, layout->fcomm)
    && EXIT_SUCCESS == internal_libxs_mpi_dup(&comm));
  if (0 != result) {
    const int next = (comm.rank + 1) % comm.size, prev = (comm.rank + comm.size - 1) % comm.size;
    internal_libxs_syrk_state_t* state = NULL;
    int mine[INTERNAL_SYRK_NAGREE], other[INTERNAL_SYRK_NAGREE];
    memset(mine, 0, sizeof(mine));
    if (0 != valid) valid = internal_libxs_syrk_layout_valid(plan, layout, &comm);
    if (0 != valid) {
      const int ngrid = layout->nprow * layout->npcol;
      unsigned int digest = 0;
      int i;
      for (i = 0; i < ngrid; ++i) digest = 31 * digest + (unsigned int)layout->ranks[i] + 1;
      mine[0] = (int)plan->shape.datatype;
      mine[1] = plan->shape.uplo;
      mine[2] = plan->shape.trans;
      mine[3] = (0 != plan->shape.ldb);
      mine[4] = plan->shape.n;
      mine[5] = plan->width;
      mine[6] = layout->mb;
      mine[7] = layout->nb;
      mine[8] = layout->nprow;
      mine[9] = layout->npcol;
      mine[10] = layout->rsrc;
      mine[11] = layout->csrc;
      mine[12] = (int)(digest & INT_MAX);
      state = internal_libxs_syrk_state(plan, layout, &comm);
      if (NULL == state) valid = 0;
    }
    memcpy(other, mine, sizeof(mine));
    if (1 < comm.size && EXIT_SUCCESS != internal_libxs_mpi_sendrecv(&comm,
      mine, sizeof(mine), next, other, sizeof(other), prev))
    {
      valid = 0;
    }
    if (0 != memcmp(mine, other, sizeof(mine))) valid = 0;
    if (EXIT_SUCCESS != internal_libxs_mpi_and(&comm, &valid)) valid = 0;
    if (0 != valid) plan->dist = state;
    else {
      libxs_free(state);
      internal_libxs_mpi_free(&comm);
    }
    result = valid;
  }
  return result;
}


LIBXS_API int libxs_syrk_dispatch_rt(libxs_syrk_t* plan,
  const libxs_syrk_shape_t* shape, const libxs_pdist_t* dist,
  const libxs_gemm_backend_t* backend, void* registry)
{
  libxs_syrk_t local;
  int result = (NULL != plan && NULL != shape);
  LIBXS_MEMZERO(&local);
  if (0 != result) {
    const char u = shape->uplo, t = shape->trans;
    const int notrans = ('N' == t || 'n' == t);
    const int nrows = LIBXS_MAX(0 != notrans ? shape->n : shape->k, 1);
    result = ((LIBXS_DATATYPE_F64 == shape->datatype || LIBXS_DATATYPE_F32 == shape->datatype)
      && ('U' == u || 'u' == u || 'L' == u || 'l' == u)
      && (0 != notrans || 'T' == t || 't' == t || 'C' == t || 'c' == t)
      && 0 <= shape->n && 0 <= shape->k && 0 <= shape->ldb
      && (0 == shape->k || (nrows <= shape->lda && (0 == shape->ldb || nrows <= shape->ldb)))
      && (NULL != dist || LIBXS_MAX(shape->n, 1) <= shape->ldc));
  }
  if (0 != result) {
    const int n = shape->n;
    libxs_gemm_shape_t gemm;
    int p, npanels;
    internal_libxs_gemm_init();
    local.shape = *shape;
    local.shape.uplo = (char)(('U' == shape->uplo || 'u' == shape->uplo) ? 'U' : 'L');
    local.shape.trans = (char)(('N' == shape->trans || 'n' == shape->trans) ? 'N' : 'T');
    internal_libxs_syrk_tiling(n, &local.width, &local.height);
    npanels = LIBXS_UPDIV(n, local.width);
    for (p = 0; p < npanels; ++p) local.ntiles += internal_libxs_syrk_panel_ntiles(&local, p);
    LIBXS_MEMZERO(&gemm);
    gemm.datatype = shape->datatype;
    gemm.transa = local.shape.trans;
    gemm.transb = (char)('N' == gemm.transa ? 'T' : 'N');
    gemm.m = n; gemm.n = n; gemm.k = shape->k;
    gemm.lda = shape->lda;
    gemm.ldb = (0 != shape->ldb ? shape->ldb : shape->lda);
    gemm.ldc = n;
    gemm.alpha = 1.0;
    /* a GEMM of exactly this shape, hence it shares the registry's entry of that GEMM rightfully */
    if (NULL != dist || 1 != local.ntiles || 0 == shape->k
      || LIBXS_SYRK_KERNEL_N < n || LIBXS_SYRK_KERNEL_K < shape->k || LIBXS_SYRK_KERNEL_NK < n * shape->k
      || 0 == libxs_gemm_dispatch_cpy_rt(&local.gemm, &gemm, NULL, backend, registry))
    {
      LIBXS_MEMZERO(&local.gemm);
      local.gemm.shape = gemm;
      internal_libxs_gemm_blas_init(&local.gemm, backend,
        INTERNAL_GEMM_BACKEND_BLAS >= internal_libxs_gemm_backend);
    }
  }
  if (NULL != dist) result = internal_libxs_syrk_dispatch_dist(&local, result, dist);
  if (NULL != plan) {
    if (0 != result) *plan = local;
    else LIBXS_MEMZERO(plan);
  }
  return result;
}


LIBXS_API int libxs_syrk_task(const libxs_syrk_t* plan,
  double alpha, const void* a, const void* b, double beta, void* c,
  int tid, int ntasks)
{
  int result = EXIT_FAILURE;
  if (NULL != plan && 0 <= tid && tid < ntasks
    && (LIBXS_DATATYPE_F64 == plan->shape.datatype || LIBXS_DATATYPE_F32 == plan->shape.datatype))
  {
    const int twok = (0 != plan->shape.ldb);
    const int reads = (0 != plan->shape.k && 0 != alpha);
    /* an operand B handed to a SYRK is a SYR2K that was planned as SYRK */
    const int misplaced = (0 == twok && NULL != b && a != b);
    internal_libxs_syrk_args_t args;
    args.plan = plan;
    args.a = a; args.b = b; args.c = c;
    args.alpha = alpha; args.beta = beta;
    if (NULL != plan->dist) { /* no early exit: the other ranks and tasks count on this one */
      result = internal_libxs_syrk_dist(&args, ntasks);
      if (0 != misplaced) result = EXIT_FAILURE;
    }
    else if (0 == plan->shape.n) result = EXIT_SUCCESS;
    else if (NULL != c && 0 == misplaced
      && (0 == reads || (NULL != a && (0 == twok || NULL != b))))
    {
      internal_libxs_syrk_local(&args, tid, ntasks);
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


LIBXS_API int libxs_syrk(const libxs_syrk_t* plan,
  double alpha, const void* a, const void* b, double beta, void* c)
{
  return libxs_syrk_task(plan, alpha, a, b, beta, c, 0, 1);
}


LIBXS_API int libxs_syrk_ntasks(const libxs_syrk_t* plan, int nthreads)
{
  int result = 1;
  if (NULL != plan && 1 < nthreads && 0 < plan->ntiles) {
    internal_libxs_gemm_init();
    /* a pthread-based OpenBLAS does not notice the team and would oversubscribe it */
    if (NULL == internal_libxs_openblas_parallel || NULL == internal_libxs_openblas_nthreads
      || 1 != internal_libxs_openblas_parallel() || 1 >= internal_libxs_openblas_nthreads())
    {
      result = (NULL != plan->dist ? nthreads : plan->ntiles);
    }
  }
  return result;
}


LIBXS_API void libxs_syrk_release(libxs_syrk_t* plan)
{
  if (NULL != plan) {
    internal_libxs_syrk_state_t *const state = (internal_libxs_syrk_state_t*)plan->dist;
    if (NULL != state) {
      internal_libxs_mpi_free(&state->comm);
      libxs_free(state);
    }
    LIBXS_MEMZERO(plan);
  }
}
