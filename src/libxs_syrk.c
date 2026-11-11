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
#include <libxs/libxs_math.h>
#include <libxs/libxs_malloc.h>
#include "libxs_gemm_internal.h"

/* A panel narrower than this is served better by the k-blocking than by one call
 * of full k. The two libraries disagree: MKL falls off a cliff at a width of 128
 * with full k, whereas OpenBLAS prefers full k at every width measured. */
#if !defined(LIBXS_SYRK_KFULL)
# define LIBXS_SYRK_KFULL 192
#endif
#if !defined(LIBXS_SYRK_KFULL_OPENBLAS)
# define LIBXS_SYRK_KFULL_OPENBLAS 0
#endif
#if !defined(LIBXS_SYRK_MINTASKS)
# define LIBXS_SYRK_MINTASKS 4
#endif

#define INTERNAL_SYRK_IRANGE(UPPER, JJ, IB, JB, CM, ISTART, IEND) \
  do { \
    const int dij_ = (JB) - (IB) + (JJ); \
    (ISTART) = (UPPER) ? 0 : (dij_ > 0 ? dij_ : 0); \
    (IEND)   = (UPPER) ? (dij_ + 1 < (CM) ? dij_ + 1 : (CM)) : (CM); \
  } while(0)

#define INTERNAL_SYRK_SCATTER(TYPE, CC, LDC, T, LDT, \
  IB, JB, CM, CN, UPPER, DIAG, ALPHA, BETA) \
  do { \
    int ii_, jj_; \
    if (DIAG) { \
      for (jj_ = 0; jj_ < (CN); ++jj_) { \
        int istart_, iend_; \
        INTERNAL_SYRK_IRANGE(UPPER, jj_, IB, JB, CM, istart_, iend_); \
        for (ii_ = istart_; ii_ < iend_; ++ii_) { \
          ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] = \
            (BETA) * ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] \
            + (ALPHA) * ((const TYPE*)(T))[ii_ + (size_t)jj_ * (LDT)]; \
        } \
      } \
    } \
    else { \
      for (jj_ = 0; jj_ < (CN); ++jj_) { \
        for (ii_ = 0; ii_ < (CM); ++ii_) { \
          ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] = \
            (BETA) * ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] \
            + (ALPHA) * ((const TYPE*)(T))[ii_ + (size_t)jj_ * (LDT)]; \
        } \
      } \
    } \
  } while(0)

#define INTERNAL_SYR2K_SCATTER(TYPE, CC, LDC, T1, T2, LDT, \
  IB, JB, CM, CN, UPPER, DIAG, SYM, ALPHA, BETA) \
  do { \
    int ii_, jj_; \
    if (DIAG) { \
      for (jj_ = 0; jj_ < (CN); ++jj_) { \
        int istart_, iend_; \
        INTERNAL_SYRK_IRANGE(UPPER, jj_, IB, JB, CM, istart_, iend_); \
        for (ii_ = istart_; ii_ < iend_; ++ii_) { \
          const TYPE val_ = (SYM) \
            ? ((const TYPE*)(T1))[ii_ + (size_t)jj_ * (LDT)] \
              + ((const TYPE*)(T1))[jj_ + (size_t)ii_ * (LDT)] \
            : ((const TYPE*)(T1))[ii_ + (size_t)jj_ * (LDT)] \
              + ((const TYPE*)(T2))[ii_ + (size_t)jj_ * (LDT)]; \
          ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] = \
            (BETA) * ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] \
            + (ALPHA) * val_; \
        } \
      } \
    } \
    else { \
      for (jj_ = 0; jj_ < (CN); ++jj_) { \
        for (ii_ = 0; ii_ < (CM); ++ii_) { \
          ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] = \
            (BETA) * ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] \
            + (ALPHA) * (((const TYPE*)(T1))[ii_ + (size_t)jj_ * (LDT)] \
                       + ((const TYPE*)(T2))[ii_ + (size_t)jj_ * (LDT)]); \
        } \
      } \
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


LIBXS_APIVAR_DEFINE(internal_libxs_dsyrk_t internal_libxs_dsyrk_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_ssyrk_t internal_libxs_ssyrk_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_dsyr2k_t internal_libxs_dsyr2k_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_ssyr2k_t internal_libxs_ssyr2k_blas);

static LIBXS_TLS void* internal_libxs_syrk_buffer;
static LIBXS_TLS size_t internal_libxs_syrk_buffer_size;


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


/* smallest panel width that takes one call of full k rather than the k-blocking */
LIBXS_API_INLINE int internal_libxs_syrk_kfull(void)
{
  return (NULL == internal_libxs_mkl_nthreads
    && NULL != internal_libxs_openblas_parallel)
    ? LIBXS_SYRK_KFULL_OPENBLAS : LIBXS_SYRK_KFULL;
}


/**
 * Predicts that the call side runs the BLAS SYRK, which reads the shape of the
 * config but never its kernel (see libxs_syrk_task and libxs_syr2k_task). The
 * decision rests on the block size and the resolved entry points, both of which
 * are settled at this point and no longer change. Requiring the SYRK and the
 * SYR2K entry point conservatively covers either caller, because the dispatch
 * is shared and cannot tell them apart.
 */
LIBXS_API_INLINE int internal_libxs_syrk_blas_due(
  libxs_data_t datatype, int n, int k)
{
  int result = 0;
  if (n > internal_libxs_gemm_bm || n > internal_libxs_gemm_bn
    || k > internal_libxs_gemm_bk)
  {
    if (LIBXS_DATATYPE_F64 == datatype) {
      result = (NULL != internal_libxs_dsyrk_blas
        && NULL != internal_libxs_dsyr2k_blas);
    }
    else if (LIBXS_DATATYPE_F32 == datatype) {
      result = (NULL != internal_libxs_ssyrk_blas
        && NULL != internal_libxs_ssyr2k_blas);
    }
  }
  return result;
}


LIBXS_API_INTERN libxs_gemm_config_t* internal_libxs_syr2k_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  const libxs_gemm_backend_t* backend, void* registry,
  libxs_gemm_config_t* own);
LIBXS_API_INTERN libxs_gemm_config_t* internal_libxs_syr2k_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  const libxs_gemm_backend_t* backend, void* registry,
  libxs_gemm_config_t* own)
{
  libxs_gemm_config_t* result = NULL;
  libxs_gemm_shape_t shape;
  /* the tile is derived from the block size, hence the state must be settled */
  internal_libxs_gemm_init();
  LIBXS_MEMZERO(&shape);
  shape.datatype = datatype;
  shape.transa = 'N'; shape.transb = 'T';
  shape.m = n; shape.n = n; shape.k = k;
  shape.lda = lda; shape.ldb = ldb; shape.ldc = ldc;
  shape.alpha = 1.0; shape.beta = 0.0;
  if (NULL != own && 0 != internal_libxs_syrk_blas_due(datatype, n, k)) {
    /* an unused kernel is worth neither the JIT nor an entry */
    LIBXS_MEMZERO(own);
    own->shape = shape;
    internal_libxs_gemm_blas_init(own, backend,
      INTERNAL_GEMM_BACKEND_BLAS >= internal_libxs_gemm_backend);
    result = own;
  }
  else {
    const int km = LIBXS_MIN(n, internal_libxs_gemm_bm);
    const int kn = LIBXS_MIN(n, internal_libxs_gemm_bn);
    const int kk = LIBXS_MIN(k, internal_libxs_gemm_bk);
    libxs_gemm_shape_t kshape;
    LIBXS_MEMZERO(&kshape);
    kshape.datatype = datatype;
    kshape.transa = 'N'; kshape.transb = 'T';
    kshape.m = km; kshape.n = kn; kshape.k = kk;
    kshape.lda = lda; kshape.ldb = ldb; kshape.ldc = km;
    kshape.alpha = 1.0; kshape.beta = 1.0;
    result = internal_libxs_gemm_dispatch(
      &shape, &kshape, backend, registry, own);
  }
  return result;
}


LIBXS_API libxs_gemm_config_t* libxs_syr2k_dispatch_rt(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  const libxs_gemm_backend_t* backend, void* registry)
{
  return internal_libxs_syr2k_dispatch(
    datatype, n, k, lda, ldb, ldc, backend, registry, NULL);
}


LIBXS_API libxs_gemm_config_t* libxs_syrk_dispatch_rt(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  const libxs_gemm_backend_t* backend, void* registry)
{
  return libxs_syr2k_dispatch_rt(datatype, n, k, lda, lda, ldc,
    backend, registry);
}


LIBXS_API int libxs_syr2k_dispatch_cpy_rt(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  const libxs_gemm_backend_t* backend, void* registry)
{
  int result = 0;
  if (NULL != config) {
    libxs_gemm_config_t *const cfg = internal_libxs_syr2k_dispatch(
      datatype, n, k, lda, ldb, ldc, backend, registry, config);
    result = (cfg == config) ? 1 : libxs_gemm_config_cpy(config, cfg);
  }
  return result;
}


LIBXS_API int libxs_syrk_dispatch_cpy_rt(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  const libxs_gemm_backend_t* backend, void* registry)
{
  return libxs_syr2k_dispatch_cpy_rt(config,
    datatype, n, k, lda, lda, ldc, backend, registry);
}


LIBXS_API_INTERN void* internal_libxs_syrk_scratch(size_t need);
LIBXS_API_INTERN void* internal_libxs_syrk_scratch(size_t need)
{
  if (need > internal_libxs_syrk_buffer_size) {
    libxs_free(internal_libxs_syrk_buffer);
    /* pooled: the scratch is a kernel operand, hence aligned and accounted */
    internal_libxs_syrk_buffer = libxs_malloc(
      libxs_default_pool(), need, LIBXS_CACHELINE);
    internal_libxs_syrk_buffer_size = (NULL != internal_libxs_syrk_buffer) ? need : 0;
  }
  return internal_libxs_syrk_buffer;
}


LIBXS_API_INLINE int internal_libxs_syrk_partition_lower(
  int t, int nb, int ntasks)
{
  int result;
  if (0 >= t) {
    result = 0;
  }
  else if (t >= ntasks) {
    result = nb;
  }
  else {
    const unsigned long long s = (unsigned long long)nb * (nb + 1) / 2;
    const unsigned long long ct = ((unsigned long long)t * s + ((unsigned int)ntasks >> 1)) / (unsigned int)ntasks;
    const unsigned long long two_nb_plus_1 = 2 * (unsigned long long)nb + 1;
    const unsigned long long d = two_nb_plus_1 * two_nb_plus_1 - 8 * ct;
    const unsigned int q = libxs_isqrt_u64(0 < d ? d : 1);
    const int j0 = (int)((two_nb_plus_1 - q) / 2);
    const unsigned long long fj0 = (0 < j0
      ? (unsigned long long)j0 * nb - (unsigned long long)j0 * (j0 - 1) / 2
      : 0);
    const unsigned long long e0 = (fj0 < ct ? ct - fj0 : fj0 - ct);
    result = j0;
    if (j0 + 1 <= nb) {
      const int j1 = j0 + 1;
      const unsigned long long fj1 = (unsigned long long)j1 * nb - (unsigned long long)j1 * (j1 - 1) / 2;
      const unsigned long long e1 = (fj1 < ct ? ct - fj1 : fj1 - ct);
      if (e1 < e0) {
        result = j1;
      }
    }
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_syrk_partition(
  int tid, int ntasks, int nb, int upper, int* begin, int* end)
{
  LIBXS_ASSERT(NULL != begin && NULL != end);
  if (0 == upper) {
    *begin = internal_libxs_syrk_partition_lower(tid, nb, ntasks);
    *end = internal_libxs_syrk_partition_lower(tid + 1, nb, ntasks);
  }
  else {
    *begin = nb - internal_libxs_syrk_partition_lower(ntasks - tid, nb, ntasks);
    *end = nb - internal_libxs_syrk_partition_lower(ntasks - (tid + 1), nb, ntasks);
  }
}


LIBXS_API void libxs_syr2k_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c,
  int tid, int ntasks)
{
  LIBXS_ASSERT(NULL != config && NULL != a && NULL != b && NULL != c);
  LIBXS_ASSERT(0 <= tid && tid < ntasks);
  LIBXS_ASSERT_MSG(LIBXS_DATATYPE_F64 == config->shape.datatype
    || LIBXS_DATATYPE_F32 == config->shape.datatype, "unsupported datatype");
  {
    const size_t elemsize = LIBXS_TYPESIZE(config->shape.datatype);
    const int upper = ('U' == uplo || 'u' == uplo);
    const int n = config->shape.m, k = config->shape.k;
    const int lda = config->shape.lda;
    const int ldb = config->shape.ldb;
    const int ldc = config->shape.ldc;
    if  (n <= internal_libxs_gemm_bm
      && n <= internal_libxs_gemm_bn
      && k <= internal_libxs_gemm_bk)
    {
      if (0 == tid) {
        const size_t need = (size_t)n * (size_t)n * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (NULL != scratch) {
          memset(scratch, 0, need);
          if (NULL != config->xgemm || NULL != config->dgemm_jit || NULL != config->sgemm_jit) {
            libxs_gemm_call(config, a, b, scratch);
          }
          else {
            internal_libxs_gemm_blas(config, a, b, scratch,
              n, n, k, lda, ldb, n, 1.0, 0.0);
          }
          if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
            INTERNAL_SYR2K_SCATTER(double, c, ldc, scratch, scratch,
              n, 0, 0, n, n, upper, 1, 1, alpha, beta);
          }
          else {
            INTERNAL_SYR2K_SCATTER(float, c, ldc, scratch, scratch,
              n, 0, 0, n, n, upper, 1, 1, (float)alpha, (float)beta);
          }
        }
      }
    }
    /* the BLAS entry point cannot honor a split, hence it serves one task */
    else if (1 >= ntasks
      && LIBXS_DATATYPE_F64 == config->shape.datatype
      && NULL != internal_libxs_dsyr2k_blas)
    {
      if (0 == tid) {
#if defined(LIBXS_GEMM_PRINT)
        { static int interval = -1;
          if (-1 == interval) {
            const char *const env = getenv("LIBXS_SYRK_PRINT");
            interval = (NULL != env ? atoi(env) : 0);
          }
          if (0 < interval) {
            fprintf(stderr, "LIBXS INFO[%u]: dsyr2k uplo=%c n=%i k=%i"
              " lda=%i ldb=%i ldc=%i alpha=%g beta=%g upper=%i\n",
              internal_libxs_gemm_origin(),
              uplo, n, k, lda, ldb, ldc, alpha, beta, upper);
          }
        }
#endif
        internal_libxs_dsyr2k_blas(&uplo, "N", &n, &k,
          (const double*)&alpha, (const double*)a, &lda,
          (const double*)b, &ldb,
          (const double*)&beta, (double*)c, &ldc, 1, 1);
      }
    }
    else if (1 >= ntasks
      && LIBXS_DATATYPE_F32 == config->shape.datatype
      && NULL != internal_libxs_ssyr2k_blas)
    {
      if (0 == tid) {
        const float fa = (float)alpha, fb = (float)beta;
        internal_libxs_ssyr2k_blas(&uplo, "N", &n, &k,
          &fa, (const float*)a, &lda,
          (const float*)b, &ldb,
          &fb, (float*)c, &ldc, 1, 1);
      }
    }
    else {
      const int bm = internal_libxs_gemm_bm;
      const int bn = internal_libxs_gemm_bn;
      const int bk = internal_libxs_gemm_bk;
      /* one panel owns its columns, hence the packed operand is not duplicated:
       * the width follows the task count, with two panels per task to balance */
      const int np = LIBXS_MAX(internal_libxs_gemm_panel,
        LIBXS_UPDIV(n, 2 * ntasks));
      const int nb_p = LIBXS_UPDIV(n, np);
      const int f64 = (LIBXS_DATATYPE_F64 == config->shape.datatype);
      const internal_libxs_dsyr2k_t dsyr2k_fn = (0 != f64
        ? internal_libxs_dsyr2k_blas : NULL);
      const internal_libxs_ssyr2k_t ssyr2k_fn = (0 == f64
        ? internal_libxs_ssyr2k_blas : NULL);
      int p_begin, p_end;
      internal_libxs_syrk_partition(tid, ntasks, nb_p, upper, &p_begin, &p_end);
      if (p_begin < p_end) {
        const size_t need = (size_t)bm * bn * 2 * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (NULL != scratch) {
          void* scratch2 = (char*)scratch + (size_t)bm * bn * elemsize;
          int p;
          for (p = p_begin; p < p_end; ++p) {
            const int pb = p * np;
            const int pn = LIBXS_MIN(np, n - pb);
            /* the panel beside the diagonal block lies inside the requested
             * triangle, hence BLAS accumulates into C without scratch */
            const int pm = (0 == upper ? (n - pb - pn) : pb);
            const int pi = (0 == upper ? (pb + pn) : 0);
            int j;
            if (0 < pm) {
              const int kc = (internal_libxs_syrk_kfull() <= pn ? k : bk);
              void *const ct = (char*)c + ((size_t)pb * ldc + pi) * elemsize;
              double bc = beta;
              int kb;
              for (kb = 0; kb < k; kb += kc) {
                const int ck = LIBXS_MIN(kc, k - kb);
                internal_libxs_gemm_blas(config,
                  (const char*)a + ((size_t)pi + (size_t)kb * lda) * elemsize,
                  (const char*)b + ((size_t)pb + (size_t)kb * ldb) * elemsize,
                  ct, pm, pn, ck, lda, ldb, ldc, alpha, bc);
                internal_libxs_gemm_blas(config,
                  (const char*)b + ((size_t)pi + (size_t)kb * ldb) * elemsize,
                  (const char*)a + ((size_t)pb + (size_t)kb * lda) * elemsize,
                  ct, pm, pn, ck, ldb, lda, ldc, alpha, 1.0);
                bc = 1.0;
              }
            }
            /* the diagonal block straddles the triangle: a BLAS SYR2K writes it,
             * otherwise the tiles below do, which keeps a kernel reachable */
            if (NULL != dsyr2k_fn) {
              internal_libxs_dsyr2k_blas(&uplo, "N", &pn, &k,
                (const double*)&alpha, (const double*)a + pb, &lda,
                (const double*)b + pb, &ldb,
                (const double*)&beta, (double*)c + ((size_t)pb * ldc + pb), &ldc, 1, 1);
              continue;
            }
            else if (NULL != ssyr2k_fn) {
              const float fa = (float)alpha, fb = (float)beta;
              internal_libxs_ssyr2k_blas(&uplo, "N", &pn, &k,
                &fa, (const float*)a + pb, &lda, (const float*)b + pb, &ldb,
                &fb, (float*)c + ((size_t)pb * ldc + pb), &ldc, 1, 1);
              continue;
            }
            for (j = 0; j < pn; j += bn) {
              const int jb = pb + j;
              const int cn = LIBXS_MIN(bn, pn - j);
              const int i_begin = (0 == upper ? ((jb - pb) / bm) : 0);
              const int i_end = (0 == upper ? LIBXS_UPDIV(pn, bm)
                : LIBXS_MIN(LIBXS_UPDIV(pn, bm), (jb - pb + cn - 1) / bm + 1));
              int i;
              for (i = i_begin; i < i_end; ++i) {
                const int ib = pb + i * bm;
                const int cm = LIBXS_MIN(bm, pb + pn - ib);
                const int diag = (ib < jb + cn && jb < ib + cm);
                const int sym = (diag && ib == jb && cm == cn);
                const int full = (cm == bm && cn == bn);
                const size_t clear = sym
                  ? (size_t)bm * bn * elemsize
                  : need;
                int kb;
                memset(scratch, 0, clear);
                for (kb = 0; kb < k; kb += bk) {
                  const int ck = LIBXS_MIN(bk, k - kb);
                  if (full && ck == bk && (NULL != config->xgemm || NULL != config->dgemm_jit || NULL != config->sgemm_jit)) {
                    const size_t aoff = ((size_t)ib + (size_t)kb * lda) * elemsize;
                    const size_t boff = ((size_t)jb + (size_t)kb * ldb) * elemsize;
                    libxs_gemm_call(config,
                      (const char*)a + aoff,
                      (const char*)b + boff, scratch);
                    if (0 == sym) {
                      const size_t bioff = ((size_t)ib + (size_t)kb * ldb) * elemsize;
                      const size_t ajoff = ((size_t)jb + (size_t)kb * lda) * elemsize;
                      libxs_gemm_call(config,
                        (const char*)b + bioff,
                        (const char*)a + ajoff, scratch2);
                    }
                  }
                  else {
                    const size_t aoff = ((size_t)ib + (size_t)kb * lda) * elemsize;
                    const size_t boff = ((size_t)jb + (size_t)kb * ldb) * elemsize;
                    internal_libxs_gemm_blas(config,
                      (const char*)a + aoff,
                      (const char*)b + boff, scratch,
                      cm, cn, ck, lda, ldb, bm, 1.0, 1.0);
                    if (0 == sym) {
                      const size_t bioff = ((size_t)ib + (size_t)kb * ldb) * elemsize;
                      const size_t ajoff = ((size_t)jb + (size_t)kb * lda) * elemsize;
                      internal_libxs_gemm_blas(config,
                        (const char*)b + bioff,
                        (const char*)a + ajoff, scratch2,
                        cm, cn, ck, ldb, lda, bm, 1.0, 1.0);
                    }
                  }
                }
                if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
                  INTERNAL_SYR2K_SCATTER(double, c, ldc, scratch, scratch2,
                    bm, ib, jb, cm, cn, upper, diag, sym, alpha, beta);
                }
                else {
                  INTERNAL_SYR2K_SCATTER(float, c, ldc, scratch, scratch2,
                    bm, ib, jb, cm, cn, upper, diag, sym,
                    (float)alpha, (float)beta);
                }
              }
            }
          }
        }
      }
    }
  }
}


LIBXS_API void libxs_syr2k(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c)
{
  libxs_syr2k_task(config, uplo, alpha, beta, a, b, c, 0, 1);
}


LIBXS_API void libxs_syrk_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c,
  int tid, int ntasks)
{
  LIBXS_ASSERT(NULL != config && NULL != a && NULL != c);
  LIBXS_ASSERT(0 <= tid && tid < ntasks);
  LIBXS_ASSERT_MSG(LIBXS_DATATYPE_F64 == config->shape.datatype
    || LIBXS_DATATYPE_F32 == config->shape.datatype, "unsupported datatype");
  {
    const int n = config->shape.m;
    const int k = config->shape.k;
    const int lda = config->shape.lda;
    const int ldc = config->shape.ldc;
    const int upper = ('U' == uplo || 'u' == uplo);
    const size_t elemsize = LIBXS_TYPESIZE(config->shape.datatype);
    if (n <= internal_libxs_gemm_bm && n <= internal_libxs_gemm_bn
      && k <= internal_libxs_gemm_bk)
    {
      if (0 == tid) {
        const size_t need = (size_t)n * (size_t)n * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (NULL != scratch) {
          memset(scratch, 0, need);
          if (NULL != config->xgemm || NULL != config->dgemm_jit || NULL != config->sgemm_jit) {
            libxs_gemm_call(config, a, a, scratch);
          }
          else {
            internal_libxs_gemm_blas(config, a, a, scratch,
              n, n, k, lda, lda, n, 1.0, 0.0);
          }
          if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
            INTERNAL_SYRK_SCATTER(double, c, ldc, scratch, n,
              0, 0, n, n, upper, 1, alpha, beta);
          }
          else {
            INTERNAL_SYRK_SCATTER(float, c, ldc, scratch, n,
              0, 0, n, n, upper, 1, (float)alpha, (float)beta);
          }
        }
      }
    }
    /* the BLAS entry point cannot honor a split, hence it serves one task */
    else if (1 >= ntasks
      && LIBXS_DATATYPE_F64 == config->shape.datatype
      && NULL != internal_libxs_dsyrk_blas)
    {
      if (0 == tid) {
#if defined(LIBXS_GEMM_PRINT)
        { static int interval = -1;
          if (-1 == interval) {
            const char *const env = getenv("LIBXS_SYRK_PRINT");
            interval = (NULL != env ? atoi(env) : 0);
          }
          if (0 < interval) {
            fprintf(stderr, "LIBXS INFO[%u]: dsyrk uplo=%c n=%i k=%i"
              " lda=%i ldc=%i alpha=%g beta=%g upper=%i\n",
              internal_libxs_gemm_origin(),
              uplo, n, k, lda, ldc, alpha, beta, upper);
          }
        }
#endif
        internal_libxs_dsyrk_blas(&uplo, "N", &n, &k,
          (const double*)&alpha, (const double*)a, &lda,
          (const double*)&beta, (double*)c, &ldc, 1, 1);
      }
    }
    else if (1 >= ntasks
      && LIBXS_DATATYPE_F32 == config->shape.datatype
      && NULL != internal_libxs_ssyrk_blas)
    {
      if (0 == tid) {
        const float fa = (float)alpha, fb = (float)beta;
        internal_libxs_ssyrk_blas(&uplo, "N", &n, &k,
          &fa, (const float*)a, &lda,
          &fb, (float*)c, &ldc, 1, 1);
      }
    }
    else {
      const int bm = internal_libxs_gemm_bm;
      const int bn = internal_libxs_gemm_bn;
      const int bk = internal_libxs_gemm_bk;
      /* one panel owns its columns, hence the packed operand is not duplicated:
       * the width follows the task count, with two panels per task to balance */
      const int np = LIBXS_MAX(internal_libxs_gemm_panel,
        LIBXS_UPDIV(n, 2 * ntasks));
      const int nb_p = LIBXS_UPDIV(n, np);
      const int f64 = (LIBXS_DATATYPE_F64 == config->shape.datatype);
      const internal_libxs_dsyrk_t dsyrk_fn = (0 != f64
        ? internal_libxs_dsyrk_blas : NULL);
      const internal_libxs_ssyrk_t ssyrk_fn = (0 == f64
        ? internal_libxs_ssyrk_blas : NULL);
      int p_begin, p_end;
      internal_libxs_syrk_partition(tid, ntasks, nb_p, upper, &p_begin, &p_end);
      if (p_begin < p_end) {
        const size_t need = (size_t)bm * bn * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (NULL != scratch) {
          int p;
          for (p = p_begin; p < p_end; ++p) {
            const int pb = p * np;
            const int pn = LIBXS_MIN(np, n - pb);
            /* the panel beside the diagonal block lies inside the requested
             * triangle, hence BLAS accumulates into C without scratch */
            const int pm = (0 == upper ? (n - pb - pn) : pb);
            const int pi = (0 == upper ? (pb + pn) : 0);
            int j;
            if (0 < pm) {
              const int kc = (internal_libxs_syrk_kfull() <= pn ? k : bk);
              void *const ct = (char*)c + ((size_t)pb * ldc + pi) * elemsize;
              double bc = beta;
              int kb;
              for (kb = 0; kb < k; kb += kc) {
                const int ck = LIBXS_MIN(kc, k - kb);
                internal_libxs_gemm_blas(config,
                  (const char*)a + ((size_t)pi + (size_t)kb * lda) * elemsize,
                  (const char*)a + ((size_t)pb + (size_t)kb * lda) * elemsize,
                  ct, pm, pn, ck, lda, lda, ldc, alpha, bc);
                bc = 1.0;
              }
            }
            /* the diagonal block straddles the triangle: a BLAS SYRK writes it,
             * otherwise the tiles below do, which keeps a kernel reachable */
            if (NULL != dsyrk_fn) {
              internal_libxs_dsyrk_blas(&uplo, "N", &pn, &k,
                (const double*)&alpha, (const double*)a + pb, &lda,
                (const double*)&beta, (double*)c + ((size_t)pb * ldc + pb), &ldc, 1, 1);
              continue;
            }
            else if (NULL != ssyrk_fn) {
              const float fa = (float)alpha, fb = (float)beta;
              internal_libxs_ssyrk_blas(&uplo, "N", &pn, &k,
                &fa, (const float*)a + pb, &lda,
                &fb, (float*)c + ((size_t)pb * ldc + pb), &ldc, 1, 1);
              continue;
            }
            for (j = 0; j < pn; j += bn) {
              const int jb = pb + j;
              const int cn = LIBXS_MIN(bn, pn - j);
              const int i_begin = (0 == upper ? ((jb - pb) / bm) : 0);
              const int i_end = (0 == upper ? LIBXS_UPDIV(pn, bm)
                : LIBXS_MIN(LIBXS_UPDIV(pn, bm), (jb - pb + cn - 1) / bm + 1));
              int i;
              for (i = i_begin; i < i_end; ++i) {
                const int ib = pb + i * bm;
                const int cm = LIBXS_MIN(bm, pb + pn - ib);
                const int diag = (ib < jb + cn && jb < ib + cm);
                const int full = (cm == bm && cn == bn);
                int kb;
                memset(scratch, 0, need);
                for (kb = 0; kb < k; kb += bk) {
                  const int ck = LIBXS_MIN(bk, k - kb);
                  if (full && ck == bk && (NULL != config->xgemm || NULL != config->dgemm_jit || NULL != config->sgemm_jit)) {
                    const size_t aoff = ((size_t)ib + (size_t)kb * lda) * elemsize;
                    const size_t ajoff = ((size_t)jb + (size_t)kb * lda) * elemsize;
                    libxs_gemm_call(config,
                      (const char*)a + aoff,
                      (const char*)a + ajoff, scratch);
                  }
                  else {
                    const size_t aoff = ((size_t)ib + (size_t)kb * lda) * elemsize;
                    const size_t ajoff = ((size_t)jb + (size_t)kb * lda) * elemsize;
                    internal_libxs_gemm_blas(config,
                      (const char*)a + aoff,
                      (const char*)a + ajoff, scratch,
                      cm, cn, ck, lda, lda, bm, 1.0, 1.0);
                  }
                }
                if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
                  INTERNAL_SYRK_SCATTER(double, c, ldc, scratch, bm,
                    ib, jb, cm, cn, upper, diag, alpha, beta);
                }
                else {
                  INTERNAL_SYRK_SCATTER(float, c, ldc, scratch, bm,
                    ib, jb, cm, cn, upper, diag,
                    (float)alpha, (float)beta);
                }
              }
            }
          }
        }
      }
    }
  }
}


LIBXS_API void libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c)
{
  libxs_syrk_task(config, uplo, alpha, beta, a, c, 0, 1);
}


LIBXS_API int libxs_syrk_ntasks(const libxs_gemm_config_t* config, int nthreads)
{
  int result = 1;
  if (NULL != config && 1 < nthreads) {
    const int n = config->shape.m, k = config->shape.k;
    internal_libxs_gemm_init(); /* a caller-built config may precede any dispatch */
    if (n > internal_libxs_gemm_bm || n > internal_libxs_gemm_bn
      || k > internal_libxs_gemm_bk)
    {
      /* a split runs at ~2/3 of a sequential BLAS SYRK per core, and loses to a threaded one */
      if (0 == internal_libxs_syrk_blas_due(config->shape.datatype, n, k)
        || (LIBXS_SYRK_MINTASKS <= nthreads && 0 == internal_libxs_blas_threaded()))
      {
        /* a panel is the unit of work, and its width bottoms out at the floor */
        const int npanel = LIBXS_UPDIV(n, internal_libxs_gemm_panel);
        result = LIBXS_MIN(nthreads, npanel);
      }
    }
  }
  return result;
}
