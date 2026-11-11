/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_gemm.h>
#include <libxs/libxs_math.h>
#include <libxs/libxs_malloc.h>
#include "libxs_gemm_internal.h"
#include "libxs_crc32.h"

#if !defined(LIBXS_GEMM_BM)
# define LIBXS_GEMM_BM 192
#endif
#if !defined(LIBXS_GEMM_BN)
# define LIBXS_GEMM_BN 32
#endif
#if !defined(LIBXS_GEMM_BK)
# define LIBXS_GEMM_BK 48
#endif
/* narrower panels give more tasks, at the cost of the GEMM they leave behind;
 * zero takes the column block size, which is the finest the tiles ever were */
#if !defined(LIBXS_GEMM_PANEL)
# define LIBXS_GEMM_PANEL 0
#endif
#if !defined(INTERNAL_GEMM_NLOCKS)
# define INTERNAL_GEMM_NLOCKS 16
#endif
/** Number of JIT warm-up counters (POT); shapes share slots by hash. */
#if !defined(LIBXS_GEMM_NWARMUP)
# define LIBXS_GEMM_NWARMUP 4096
#endif
#if 0 != (LIBXS_GEMM_NWARMUP & (LIBXS_GEMM_NWARMUP - 1))
# error LIBXS_GEMM_NWARMUP must be a power of two
#endif

/**
 * A slot packs a 24-bit identity tag (bits 31..8) and an 8-bit state (bits
 * 7..0): 0 is unseen, 1 to threshold-1 counts dispatches. DONE is terminal
 * (the JIT was attempted, successfully or not), and NOJIT is terminal and
 * stronger, because the arithmetic-intensity gate reads the kernel shape
 * alone and no backend can change its answer. A tag mismatch reclaims the
 * slot rather than inheriting foreign state.
 */
#define INTERNAL_GEMM_WARMUP_DONE 255
#define INTERNAL_GEMM_WARMUP_NOJIT 254
#define INTERNAL_GEMM_WARMUP_TAG(HASH) (((HASH) >> 8) & 0xFFFFFF)
#define INTERNAL_GEMM_WARMUP_SLOT(TAG, STATE) \
  (((unsigned int)(TAG) << 8) | (unsigned int)(STATE))
#define INTERNAL_GEMM_WARMUP_STATE(SLOT) ((SLOT) & 0xFF)
/* LIBXS_GEMM_FLAG_STREAM can change the verdict, hence it settles in a slot of its own */
#define INTERNAL_GEMM_WARMUP_HASH(HASH, STREAM) \
  (0 != (STREAM) ? ((HASH) ^ 0x9E3779B9u) : (HASH))

/* bits of internal_libxs_gemm_select */
#define INTERNAL_GEMM_SELECT_JIT 1
#define INTERNAL_GEMM_SELECT_XGEMM 2
#define INTERNAL_GEMM_SELECT_BLAS 4
#define INTERNAL_GEMM_SELECT_KERNEL 8
/* MKL JIT addresses an operand with signed 32-bit byte offsets, and faults beyond */
#define INTERNAL_GEMM_MKLJIT_MAXSPAN ((size_t)INT_MAX)

#define INTERNAL_GEMM_NOTRANS(C) ('N' == (C) || 'n' == (C))

#define INTERNAL_GEMM_LOCKIDX(PTR) \
  ((int)LIBXS_MOD2(LIBXS_CRCPTR(1975, PTR), INTERNAL_GEMM_NLOCKS))

#define INTERNAL_GEMM_LOCKFWD(CPTR, LOCKIDX) do { \
  const int internal_libxs_gemm_li_ = INTERNAL_GEMM_LOCKIDX(CPTR); \
  if (internal_libxs_gemm_li_ != (LOCKIDX)) { \
    if (0 <= (LOCKIDX)) \
      LIBXS_LOCK_RELEASE(LIBXS_LOCK, internal_libxs_gemm_locks + (LOCKIDX)); \
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, internal_libxs_gemm_locks + internal_libxs_gemm_li_); \
    (LOCKIDX) = internal_libxs_gemm_li_; \
  } \
} while(0)

#define INTERNAL_GEMM_LOCKFWD_IDX(IDX, LOCKIDX) do { \
  const int internal_libxs_gemm_li_ = \
    (int)LIBXS_MOD2((IDX), INTERNAL_GEMM_NLOCKS); \
  if (internal_libxs_gemm_li_ != (LOCKIDX)) { \
    if (0 <= (LOCKIDX)) \
      LIBXS_LOCK_RELEASE(LIBXS_LOCK, internal_libxs_gemm_locks + (LOCKIDX)); \
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, internal_libxs_gemm_locks + internal_libxs_gemm_li_); \
    (LOCKIDX) = internal_libxs_gemm_li_; \
  } \
} while(0)

#define INTERNAL_GEMM_UNLOCK(LOCKIDX) do { \
  if (0 <= (LOCKIDX)) \
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, internal_libxs_gemm_locks + (LOCKIDX)); \
} while(0)


LIBXS_APIVAR_DEFINE(LIBXS_LOCK_TYPE(LIBXS_LOCK) internal_libxs_gemm_locks[INTERNAL_GEMM_NLOCKS]);
LIBXS_APIVAR_DEFINE(libxs_registry_t* internal_libxs_gemm_registry);

LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_bm);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_bn);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_bk);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_panel);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_jit_max);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_jit_warmup);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_backend);
LIBXS_APIVAR_DEFINE(unsigned int
  internal_libxs_gemm_warmup[LIBXS_GEMM_NWARMUP]);
LIBXS_APIVAR_DEFINE(unsigned int internal_libxs_gemm_nshape);

LIBXS_APIVAR_DEFINE(libxs_gemm_dblas_t internal_libxs_dgemm_blas);
LIBXS_APIVAR_DEFINE(libxs_gemm_sblas_t internal_libxs_sgemm_blas);

LIBXS_APIVAR_DEFINE(internal_libxs_blas_query_t internal_libxs_mkl_nthreads);
LIBXS_APIVAR_DEFINE(internal_libxs_blas_query_t internal_libxs_openblas_parallel);
LIBXS_APIVAR_DEFINE(internal_libxs_blas_query_t internal_libxs_openblas_nthreads);
LIBXS_APIVAR_DEFINE(libxs_blas_nthreads_t internal_libxs_blas_nthreads);

LIBXS_APIVAR_DEFINE(libxs_jit_create_dgemm_t internal_libxs_jit_create_dgemm);
LIBXS_APIVAR_DEFINE(libxs_jit_get_dgemm_t internal_libxs_jit_get_dgemm);
LIBXS_APIVAR_DEFINE(libxs_jit_create_sgemm_t internal_libxs_jit_create_sgemm);
LIBXS_APIVAR_DEFINE(libxs_jit_get_sgemm_t internal_libxs_jit_get_sgemm);
LIBXS_APIVAR_DEFINE(libxs_xgemm_dispatch_t internal_libxs_xgemm_dispatch);


LIBXS_API_INTERN void internal_libxs_gemm_init(void)
{
  static int internal_libxs_gemm_init_once = 0;
  if (0 == internal_libxs_gemm_init_once) {
    const char *const gemm_bm_env = getenv("LIBXS_GEMM_BM");
    const char *const gemm_bn_env = getenv("LIBXS_GEMM_BN");
    const char *const gemm_bk_env = getenv("LIBXS_GEMM_BK");
    const char *const gemm_panel_env = getenv("LIBXS_GEMM_PANEL");
    const char *const gemm_jit_max_env = getenv("LIBXS_GEMM_JIT_MAX");
    const char *const gemm_jit_warmup_env = getenv("LIBXS_GEMM_JIT_WARMUP");
    const char *const gemm_backend_env = getenv("LIBXS_GEMM_BACKEND");
#if defined(LIBXS_INTERCEPT_DYNAMIC)
    void* dl;
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(dgemm)));
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(libxs_gemm_dblas_t, internal_libxs_dgemm_blas, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(sgemm)));
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(libxs_gemm_sblas_t, internal_libxs_sgemm_blas, dl);
    }
    internal_libxs_syrk_init();
    /* layered MKL exports the lowercase name only (interface layer), libmkl_rt both */
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, "mkl_get_max_threads");
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_blas_query_t, internal_libxs_mkl_nthreads, dl);
    }
    else {
      dlerror();
      dl = dlsym(LIBXS_RTLD_NEXT, "MKL_Get_Max_Threads");
      if (NULL == dlerror() && NULL != dl) {
        LIBXS_FPTR_FROM_VPTR(internal_libxs_blas_query_t, internal_libxs_mkl_nthreads, dl);
      }
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, "openblas_get_parallel");
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_blas_query_t, internal_libxs_openblas_parallel, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, "openblas_get_num_threads");
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_blas_query_t, internal_libxs_openblas_nthreads, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, "mkl_cblas_jit_create_dgemm");
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(libxs_jit_create_dgemm_t, internal_libxs_jit_create_dgemm, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, "mkl_jit_get_dgemm_ptr");
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(libxs_jit_get_dgemm_t, internal_libxs_jit_get_dgemm, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, "mkl_cblas_jit_create_sgemm");
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(libxs_jit_create_sgemm_t, internal_libxs_jit_create_sgemm, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, "mkl_jit_get_sgemm_ptr");
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(libxs_jit_get_sgemm_t, internal_libxs_jit_get_sgemm, dl);
    }
    dlerror();
    dl = dlsym(LIBXS_RTLD_NEXT, "libxsmm_dispatch_gemm");
    if (NULL == dlerror() && NULL != dl) {
      LIBXS_FPTR_FROM_VPTR(libxs_xgemm_dispatch_t, internal_libxs_xgemm_dispatch, dl);
    }
#endif
    internal_libxs_gemm_bm = (NULL == gemm_bm_env ? LIBXS_GEMM_BM : atoi(gemm_bm_env));
    internal_libxs_gemm_bn = (NULL == gemm_bn_env ? LIBXS_GEMM_BN : atoi(gemm_bn_env));
    internal_libxs_gemm_bk = (NULL == gemm_bk_env ? LIBXS_GEMM_BK : atoi(gemm_bk_env));
    internal_libxs_gemm_panel = (NULL == gemm_panel_env
      ? LIBXS_GEMM_PANEL : atoi(gemm_panel_env));
    if (internal_libxs_gemm_panel <= 0) { /* zero means unspecified */
      internal_libxs_gemm_panel = internal_libxs_gemm_bn;
    }
    internal_libxs_gemm_jit_max = (NULL == gemm_jit_max_env
      ? 7 /*default: AI of ~80x80x80*/ : atoi(gemm_jit_max_env));
    internal_libxs_gemm_jit_warmup = (NULL == gemm_jit_warmup_env
      ? 8 /*default*/ : atoi(gemm_jit_warmup_env));
    /* the counting states must stay below the terminal ones */
    if (INTERNAL_GEMM_WARMUP_NOJIT <= internal_libxs_gemm_jit_warmup) {
      internal_libxs_gemm_jit_warmup = INTERNAL_GEMM_WARMUP_NOJIT - 1;
    }
    internal_libxs_gemm_backend = (NULL == gemm_backend_env)
      ? INTERNAL_GEMM_BACKEND_AUTO : atoi(gemm_backend_env);
    if (INTERNAL_GEMM_BACKEND_AUTO > internal_libxs_gemm_backend
      || INTERNAL_GEMM_BACKEND_DEFAULT < internal_libxs_gemm_backend)
    {
      internal_libxs_gemm_backend = INTERNAL_GEMM_BACKEND_AUTO;
    }
    internal_libxs_gemm_registry = libxs_registry_create();
    internal_libxs_gemm_init_once = 1;
  }
}


LIBXS_API_INLINE unsigned int internal_libxs_gemm_warmup_state(
  unsigned int hash)
{
  const unsigned int *const slot = internal_libxs_gemm_warmup
    + (hash & (LIBXS_GEMM_NWARMUP - 1));
  const unsigned int cur = LIBXS_ATOMIC_LOAD(slot, LIBXS_ATOMIC_RELAXED);
  const unsigned int s = INTERNAL_GEMM_WARMUP_STATE(cur);
  /* a foreign tag reclaims the slot rather than inheriting its state */
  return (INTERNAL_GEMM_WARMUP_SLOT(INTERNAL_GEMM_WARMUP_TAG(hash), s) == cur)
    ? s : 0;
}


LIBXS_API_INLINE int internal_libxs_gemm_warmup_due(
  unsigned int hash, unsigned int* state)
{
  const int threshold = internal_libxs_gemm_jit_warmup;
  unsigned int *const slot = internal_libxs_gemm_warmup
    + (hash & (LIBXS_GEMM_NWARMUP - 1));
  const unsigned int tag = INTERNAL_GEMM_WARMUP_TAG(hash);
  unsigned int s = internal_libxs_gemm_warmup_state(hash);
  int result = 0;
  if (INTERNAL_GEMM_WARMUP_DONE != s && INTERNAL_GEMM_WARMUP_NOJIT != s) {
    if (1 >= threshold) {
      result = 1;
    }
    else if (0 == s && 1 == (unsigned int)LIBXS_ATOMIC_ADD_FETCH(
      &internal_libxs_gemm_nshape, 1, LIBXS_ATOMIC_RELAXED))
    { /* the first shape cannot prove reuse: dispatch once, call often */
      s = (unsigned int)threshold;
      result = 1;
      LIBXS_ATOMIC_STORE(slot,
        INTERNAL_GEMM_WARMUP_SLOT(tag, s), LIBXS_ATOMIC_RELAXED);
    }
    else {
      if ((unsigned int)threshold <= ++s) result = 1;
      LIBXS_ATOMIC_STORE(slot,
        INTERNAL_GEMM_WARMUP_SLOT(tag, s), LIBXS_ATOMIC_RELAXED);
    }
  }
  LIBXS_ASSERT(NULL != state);
  *state = s;
  return result;
}


LIBXS_API_INLINE void internal_libxs_gemm_warmup_term(
  unsigned int hash, unsigned int state)
{
  unsigned int *const slot = internal_libxs_gemm_warmup
    + (hash & (LIBXS_GEMM_NWARMUP - 1));
  LIBXS_ASSERT(INTERNAL_GEMM_WARMUP_DONE == state
    || INTERNAL_GEMM_WARMUP_NOJIT == state);
  LIBXS_ATOMIC_STORE(slot, INTERNAL_GEMM_WARMUP_SLOT(
    INTERNAL_GEMM_WARMUP_TAG(hash), state), LIBXS_ATOMIC_RELAXED);
}


LIBXS_API_INTERN void internal_libxs_dgemm_default(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc,
  size_t transa_len, size_t transb_len);
LIBXS_API_INTERN void internal_libxs_dgemm_default(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc,
  size_t transa_len, size_t transb_len)
{
  const int mm = *m, nn = *n, kk = *k;
  const int llda = *lda, lldb = *ldb, lldc = *ldc;
  const double dalpha = (NULL != alpha ? *alpha : 1.0);
  const double dbeta = (NULL != beta ? *beta : 0.0);
  int i, j, p;
  LIBXS_UNUSED(transa_len);
  LIBXS_UNUSED(transb_len);
  LIBXS_ASSERT(NULL != transa && NULL != transb);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  for (j = 0; j < nn; ++j) {
    for (i = 0; i < mm; ++i) {
      double sum = 0.0;
      for (p = 0; p < kk; ++p) {
        const double aval = INTERNAL_GEMM_NOTRANS(*transa)
          ? a[i + p * llda] : a[p + i * llda];
        const double bval = INTERNAL_GEMM_NOTRANS(*transb)
          ? b[p + j * lldb] : b[j + p * lldb];
        sum += aval * bval;
      }
      /* BLAS does not read C when beta is zero (C may hold NaN, e.g., fresh memory) */
      c[i + j * lldc] = (0.0 == dbeta ? dalpha * sum : (dalpha * sum + dbeta * c[i + j * lldc]));
    }
  }
}


LIBXS_API_INTERN void internal_libxs_sgemm_default(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc,
  size_t transa_len, size_t transb_len);
LIBXS_API_INTERN void internal_libxs_sgemm_default(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc,
  size_t transa_len, size_t transb_len)
{
  const int mm = *m, nn = *n, kk = *k;
  const int llda = *lda, lldb = *ldb, lldc = *ldc;
  const float falpha = (NULL != alpha ? *alpha : 1.f);
  const float fbeta = (NULL != beta ? *beta : 0.f);
  int i, j, p;
  LIBXS_UNUSED(transa_len);
  LIBXS_UNUSED(transb_len);
  LIBXS_ASSERT(NULL != transa && NULL != transb);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  for (j = 0; j < nn; ++j) {
    for (i = 0; i < mm; ++i) {
      float sum = 0.f;
      for (p = 0; p < kk; ++p) {
        const float aval = INTERNAL_GEMM_NOTRANS(*transa)
          ? a[i + p * llda] : a[p + i * llda];
        const float bval = INTERNAL_GEMM_NOTRANS(*transb)
          ? b[p + j * lldb] : b[j + p * lldb];
        sum += aval * bval;
      }
      c[i + j * lldc] = (0.f == fbeta ? falpha * sum : (falpha * sum + fbeta * c[i + j * lldc]));
    }
  }
}


LIBXS_API_INLINE void internal_libxs_gemm_print_registry(const libxs_registry_t* registry)
{
#if defined(LIBXS_GEMM_PRINT)
  if (NULL != registry && 0 != libxs_verbosity) {
    const char *const env = getenv("LIBXS_GEMM_PRINT");
    if (NULL != env && 0 == atoi(env)) {
      libxs_registry_info_t info;
      LIBXS_MEMZERO(&info);
      if (EXIT_SUCCESS == libxs_registry_info(registry, &info) && 0 < info.size) {
        const void* key = NULL;
        size_t cursor = 0;
        unsigned long nf64 = 0, nf32 = 0, njit = 0, nxgemm = 0, nblas = 0, nfallback = 0;
        const char* backend = "0:auto";
        const libxs_gemm_config_t* config = (const libxs_gemm_config_t*)
          libxs_registry_begin(registry, &key, &cursor);
        if (INTERNAL_GEMM_BACKEND_MKL_JIT == internal_libxs_gemm_backend) backend = "1:mkl-jit";
        else if (INTERNAL_GEMM_BACKEND_LIBXSMM == internal_libxs_gemm_backend) backend = "2:libxsmm";
        else if (INTERNAL_GEMM_BACKEND_BLAS == internal_libxs_gemm_backend) backend = "3:blas";
        else if (INTERNAL_GEMM_BACKEND_DEFAULT == internal_libxs_gemm_backend) backend = "4:fallback";
        while (NULL != config && NULL != key) {
          const libxs_gemm_shape_t* shape = (const libxs_gemm_shape_t*)key;
          if (LIBXS_DATATYPE_F64 == shape->datatype) ++nf64;
          else if (LIBXS_DATATYPE_F32 == shape->datatype) ++nf32;
          if (NULL != config->dgemm_jit || NULL != config->sgemm_jit) ++njit;
          else if (NULL != config->xgemm) ++nxgemm;
          else if ((LIBXS_DATATYPE_F64 == shape->datatype && internal_libxs_dgemm_default != config->dgemm_blas)
            || (LIBXS_DATATYPE_F32 == shape->datatype && internal_libxs_sgemm_default != config->sgemm_blas))
          {
            ++nblas;
          }
          else ++nfallback;
          config = (const libxs_gemm_config_t*)
            libxs_registry_next(registry, &key, &cursor);
        }
        fprintf(stderr, "LIBXS INFO[%u]: GEMM registry"
          " entries=%lu capacity=%lu nbytes=%lu backend=%s\n",
          internal_libxs_gemm_origin(), (unsigned long)info.size,
          (unsigned long)info.capacity, (unsigned long)info.nbytes, backend);
        fprintf(stderr, "LIBXS INFO[%u]: GEMM histogram"
          " f64=%lu f32=%lu mkl-jit=%lu libxsmm=%lu blas=%lu fallback=%lu\n",
          internal_libxs_gemm_origin(),
          nf64, nf32, njit, nxgemm, nblas, nfallback);
        { /* kernel-less shapes own no entry, hence the counters report them */
          unsigned long nwarm = 0, nojit = 0, i;
          for (i = 0; i < LIBXS_GEMM_NWARMUP; ++i) {
            const unsigned int s = INTERNAL_GEMM_WARMUP_STATE(
              LIBXS_ATOMIC_LOAD(internal_libxs_gemm_warmup + i,
                LIBXS_ATOMIC_RELAXED));
            if (0 != s) ++nwarm;
            if (INTERNAL_GEMM_WARMUP_NOJIT == s) ++nojit;
          }
          fprintf(stderr, "LIBXS INFO[%u]:"
            " GEMM warm-up slots=%lu/%lu nojit=%lu\n",
            internal_libxs_gemm_origin(),
            nwarm, (unsigned long)LIBXS_GEMM_NWARMUP, nojit);
        }
      }
    }
  }
#else
  LIBXS_UNUSED(registry);
#endif
}


LIBXS_API void libxs_gemm_release_registry(libxs_registry_t* registry)
{
  if (NULL != registry) {
    const void* key = NULL;
    size_t cursor = 0;
    libxs_gemm_config_t* config;
    internal_libxs_gemm_print_registry(registry);
    config = (libxs_gemm_config_t*)libxs_registry_begin(registry, &key, &cursor);
    while (NULL != config) {
      libxs_gemm_release(config);
      config = (libxs_gemm_config_t*)libxs_registry_next(registry, &key, &cursor);
    }
    libxs_registry_destroy(registry);
  }
}


LIBXS_API_INTERN void internal_libxs_gemm_finalize(void)
{
  if (NULL != internal_libxs_gemm_registry) {
    libxs_gemm_release_registry(internal_libxs_gemm_registry);
    internal_libxs_gemm_registry = NULL;
  }
}


/**
 * Which kinds of kernel a kernel shape may receive (INTERNAL_GEMM_SELECT_*),
 * warm-up aside; KERNEL tells whether generating one pays (LIBXS_GEMM_JIT_MAX).
 * A NULL shape selects what any shape may receive.
 */
LIBXS_API_INLINE int internal_libxs_gemm_select(
  const libxs_gemm_shape_t* kernel_shape, int stream)
{
  const int gemm_backend = internal_libxs_gemm_backend;
  int strided = 0, xgemm_ok = 1, kernel_ok = (0 < internal_libxs_gemm_jit_max);
  int mkljit_ok = 1, result = 0;
  if (NULL != kernel_shape) {
    const int ta = ('N' != kernel_shape->transa && 'n' != kernel_shape->transa);
    const int tb = ('N' != kernel_shape->transb && 'n' != kernel_shape->transb);
    const int km = kernel_shape->m, kn = kernel_shape->n, kk = kernel_shape->k;
    const size_t elemsize = LIBXS_TYPESIZE(kernel_shape->datatype);
    const size_t kflops = (size_t)km * kn * kk * 2;
    const size_t kbytes = elemsize *
      ((size_t)km * kk + (size_t)kk * kn + (size_t)km * kn);
    const size_t span_a = elemsize * kernel_shape->lda * (size_t)(0 != ta ? km : kk);
    const size_t span_b = elemsize * kernel_shape->ldb * (size_t)(0 != tb ? kk : kn);
    const size_t span_c = elemsize * kernel_shape->ldc * (size_t)kn;
    /* a leading dimension beyond the operand's own extent means a window
     * into a larger matrix: generated kernels neither pack nor prefetch
     * and then lose to BLAS, whereas resident operands are their domain;
     * a caller streaming such windows on purpose says so (stream) */
    strided = (0 == stream && (kernel_shape->lda > (0 != ta ? kk : km)
      || kernel_shape->ldb > (0 != tb ? kn : kk)));
    xgemm_ok = (1.0 == kernel_shape->alpha
      && (0.0 == kernel_shape->beta || 1.0 == kernel_shape->beta));
    kernel_ok = (0 != kernel_ok
      && kflops < (size_t)internal_libxs_gemm_jit_max * kbytes);
    /* even when requested (LIBXS_GEMM_BACKEND=1), since LIBXSMM or BLAS take over */
    mkljit_ok = (INTERNAL_GEMM_MKLJIT_MAXSPAN >= span_a
      && INTERNAL_GEMM_MKLJIT_MAXSPAN >= span_b
      && INTERNAL_GEMM_MKLJIT_MAXSPAN >= span_c);
  }
  if (0 != mkljit_ok && (INTERNAL_GEMM_BACKEND_MKL_JIT == gemm_backend
    || (INTERNAL_GEMM_BACKEND_AUTO == gemm_backend && 0 == strided)))
  {
    result |= INTERNAL_GEMM_SELECT_JIT;
  }
  if (INTERNAL_GEMM_BACKEND_LIBXSMM >= gemm_backend && 0 != xgemm_ok
    && (INTERNAL_GEMM_BACKEND_AUTO != gemm_backend || 0 == strided))
  {
    result |= INTERNAL_GEMM_SELECT_XGEMM;
  }
  if (INTERNAL_GEMM_BACKEND_BLAS >= gemm_backend) result |= INTERNAL_GEMM_SELECT_BLAS;
  if (0 != kernel_ok) result |= INTERNAL_GEMM_SELECT_KERNEL;
  return result;
}


LIBXS_API libxs_gemm_kind_t libxs_gemm_backend_kind(
  const libxs_gemm_backend_t* backend, const libxs_gemm_shape_t* shape)
{
  libxs_gemm_kind_t result = LIBXS_GEMM_KIND_DEFAULT;
  const int f64 = (NULL == shape || LIBXS_DATATYPE_F64 == shape->datatype);
  const int f32 = (NULL == shape || LIBXS_DATATYPE_F32 == shape->datatype);
  if (0 != f64 || 0 != f32) {
    int select;
    internal_libxs_gemm_init();
    select = internal_libxs_gemm_select(shape, 0/*stream*/);
    { const libxs_jit_create_dgemm_t jcd = (NULL != backend && NULL != backend->jit_create_dgemm)
        ? backend->jit_create_dgemm : internal_libxs_jit_create_dgemm;
      const libxs_jit_get_dgemm_t jgd = (NULL != backend && NULL != backend->jit_get_dgemm)
        ? backend->jit_get_dgemm : internal_libxs_jit_get_dgemm;
      const libxs_jit_create_sgemm_t jcs = (NULL != backend && NULL != backend->jit_create_sgemm)
        ? backend->jit_create_sgemm : internal_libxs_jit_create_sgemm;
      const libxs_jit_get_sgemm_t jgs = (NULL != backend && NULL != backend->jit_get_sgemm)
        ? backend->jit_get_sgemm : internal_libxs_jit_get_sgemm;
      const libxs_xgemm_dispatch_t xdisp = (NULL != backend && NULL != backend->xgemm_dispatch)
        ? backend->xgemm_dispatch : internal_libxs_xgemm_dispatch;
      const libxs_gemm_dblas_t dblas = (NULL != backend && NULL != backend->dgemm_blas)
        ? backend->dgemm_blas : internal_libxs_dgemm_blas;
      const libxs_gemm_sblas_t sblas = (NULL != backend && NULL != backend->sgemm_blas)
        ? backend->sgemm_blas : internal_libxs_sgemm_blas;
      const int jit = (0 != (INTERNAL_GEMM_SELECT_JIT & select)
        && ((0 != f64 && NULL != jcd && NULL != jgd)
         || (0 != f32 && NULL != jcs && NULL != jgs)));
      const int xgemm = (0 != (INTERNAL_GEMM_SELECT_XGEMM & select) && NULL != xdisp);
      if (0 != (INTERNAL_GEMM_SELECT_KERNEL & select) && (0 != jit || 0 != xgemm)) {
        result = LIBXS_GEMM_KIND_JIT;
      }
      else if (0 != (INTERNAL_GEMM_SELECT_BLAS & select)
        && ((0 != f64 && NULL != dblas) || (0 != f32 && NULL != sblas)))
      {
        result = LIBXS_GEMM_KIND_BLAS;
      }
    }
  }
  return result;
}


/* own is the caller's config; a kernel-less config then owns no entry */
LIBXS_API_INTERN libxs_gemm_config_t* internal_libxs_gemm_dispatch(
  const libxs_gemm_shape_t* shape,
  const libxs_gemm_shape_t* kernel_shape,
  const libxs_gemm_backend_t* backend,
  void* registry, libxs_gemm_config_t* own);
LIBXS_API_INTERN libxs_gemm_config_t* internal_libxs_gemm_dispatch(
  const libxs_gemm_shape_t* shape,
  const libxs_gemm_shape_t* kernel_shape,
  const libxs_gemm_backend_t* backend,
  void* registry, libxs_gemm_config_t* own)
{
  libxs_gemm_config_t* result = NULL;
  libxs_registry_t* reg = NULL;
  libxs_gemm_shape_t key, kkey;
  unsigned int whash = 0, shash = 0, khash = 0, wstate = 0;
  /* hints arrive in the caller's config, which dispatch then overwrites */
  const int hint = (NULL != own ? (int)own->flags : 0);
  const int stream = (0 != (LIBXS_GEMM_FLAG_STREAM & hint));
  const int reuse = (0 != (LIBXS_GEMM_FLAG_REUSE & hint));
  int jit_due = 0;
  LIBXS_ASSERT(NULL != shape);
  LIBXS_MEMZERO(&key);
  key.datatype = shape->datatype;
  key.transa = shape->transa; key.transb = shape->transb;
  key.m = shape->m; key.n = shape->n; key.k = shape->k;
  key.lda = shape->lda; key.ldb = shape->ldb; key.ldc = shape->ldc;
  key.alpha = shape->alpha; key.beta = shape->beta;
  if (NULL == kernel_shape) kernel_shape = shape;
  if (kernel_shape != shape) {
    LIBXS_MEMZERO(&kkey);
    kkey.datatype = kernel_shape->datatype;
    kkey.transa = kernel_shape->transa; kkey.transb = kernel_shape->transb;
    kkey.m = kernel_shape->m; kkey.n = kernel_shape->n; kkey.k = kernel_shape->k;
    kkey.lda = kernel_shape->lda; kkey.ldb = kernel_shape->ldb; kkey.ldc = kernel_shape->ldc;
    kkey.alpha = kernel_shape->alpha; kkey.beta = kernel_shape->beta;
    kernel_shape = &kkey;
  }
  else {
    kernel_shape = &key;
  }
  shape = &key;
  if (LIBXS_DATATYPE_F64 == shape->datatype
   || LIBXS_DATATYPE_F32 == shape->datatype)
  {
    internal_libxs_gemm_init();
    reg = (NULL != registry)
      ? (libxs_registry_t*)registry : internal_libxs_gemm_registry;
    if (NULL != reg) {
      /* one hash per dispatch, shared by the registry and the warm-up table */
      whash = libxs_registry_hash(
        (const libxs_registry_t*)reg, shape, sizeof(*shape));
      shash = INTERNAL_GEMM_WARMUP_HASH(whash, stream);
      /* a NOJIT shape owns no entry when the caller owns the config */
      if (NULL == own || INTERNAL_GEMM_WARMUP_NOJIT
        != internal_libxs_gemm_warmup_state(shash))
      {
        result = (libxs_gemm_config_t*)libxs_registry_get_hashed(
          (const libxs_registry_t*)reg, shape, sizeof(*shape),
          whash, libxs_registry_lock(reg));
      }
    }
    /* a config without kernel is provisional: reuse decides when to JIT */
    if (NULL != reg && (NULL == result
      || (NULL == result->dgemm_jit && NULL == result->sgemm_jit
        && NULL == result->xgemm)))
    {
      jit_due = internal_libxs_gemm_warmup_due(shash, &wstate);
      /* a caller vouching for reuse needs no warm-up, but a settled verdict stands */
      if (0 == jit_due && 0 != reuse && INTERNAL_GEMM_WARMUP_DONE != wstate
        && INTERNAL_GEMM_WARMUP_NOJIT != wstate)
      {
        jit_due = 1;
      }
      if (0 != jit_due) result = NULL;
    }
    if (NULL == result && NULL != reg) {
      const int jit_allowed = jit_due;
      const int tiled = (0 != memcmp(shape, kernel_shape, sizeof(*shape)));
      const libxs_gemm_config_t* kernel = NULL;
      const libxs_gemm_config_t* cached = NULL;
      libxs_gemm_config_t config;
      int gate = 0;
      LIBXS_MEMZERO(&config);
      /* snapshot of the shared warm-up counter (introspection only) */
      config.warmup = (int)wstate;
      if (0 != tiled) {
        khash = libxs_registry_hash((const libxs_registry_t*)reg,
          kernel_shape, sizeof(*kernel_shape));
        cached = (const libxs_gemm_config_t*)libxs_registry_get_hashed(
          (const libxs_registry_t*)reg, kernel_shape, sizeof(*kernel_shape),
          khash, libxs_registry_lock(reg));
        /* a warm-up entry carries no kernel, hence it is not reused */
        if (NULL != cached && (NULL != cached->dgemm_jit
          || NULL != cached->sgemm_jit || NULL != cached->xgemm))
        {
          kernel = cached;
        }
      }
      config.shape = *shape;
      if (NULL != kernel) {
        config.dgemm_jit = kernel->dgemm_jit;
        config.sgemm_jit = kernel->sgemm_jit;
        config.xgemm = kernel->xgemm;
        config.jitter = kernel->jitter;
        config.dgemm_blas = kernel->dgemm_blas;
        config.sgemm_blas = kernel->sgemm_blas;
      }
      else {
        const int ta = ('N' != kernel_shape->transa && 'n' != kernel_shape->transa);
        const int tb = ('N' != kernel_shape->transb && 'n' != kernel_shape->transb);
        const int km = kernel_shape->m, kn = kernel_shape->n, kk = kernel_shape->k;
        const int klda = kernel_shape->lda, kldb = kernel_shape->ldb;
        const int kldc = kernel_shape->ldc;
        const int select = internal_libxs_gemm_select(kernel_shape, stream);
        const int use_jit = (0 != (INTERNAL_GEMM_SELECT_JIT & select));
        const int use_xgemm = (0 != (INTERNAL_GEMM_SELECT_XGEMM & select));
        const int use_blas = (0 != (INTERNAL_GEMM_SELECT_BLAS & select));
        const int use_kernel = (0 != jit_allowed
          && 0 != (INTERNAL_GEMM_SELECT_KERNEL & select));
        const libxs_jit_create_dgemm_t jcd =
          (NULL != backend && NULL != backend->jit_create_dgemm)
          ? backend->jit_create_dgemm : internal_libxs_jit_create_dgemm;
        const libxs_jit_get_dgemm_t jgd =
          (NULL != backend && NULL != backend->jit_get_dgemm)
          ? backend->jit_get_dgemm : internal_libxs_jit_get_dgemm;
        const libxs_jit_create_sgemm_t jcs =
          (NULL != backend && NULL != backend->jit_create_sgemm)
          ? backend->jit_create_sgemm : internal_libxs_jit_create_sgemm;
        const libxs_jit_get_sgemm_t jgs =
          (NULL != backend && NULL != backend->jit_get_sgemm)
          ? backend->jit_get_sgemm : internal_libxs_jit_get_sgemm;
        const libxs_xgemm_dispatch_t xdisp =
          (NULL != backend && NULL != backend->xgemm_dispatch)
          ? backend->xgemm_dispatch : internal_libxs_xgemm_dispatch;
        gate = use_kernel; /* the gate reads the kernel shape alone */
        if (0 != use_jit && 0 != use_kernel
          && NULL != jcd && NULL != jgd
          && LIBXS_DATATYPE_F64 == kernel_shape->datatype)
        {
          const int mkl_ta = (0 == ta) ? 111 : 112;
          const int mkl_tb = (0 == tb) ? 111 : 112;
          void* jitter = NULL;
          if (2 != jcd(&jitter, 102, mkl_ta, mkl_tb,
            km, kn, kk, kernel_shape->alpha, klda, kldb,
            kernel_shape->beta, kldc) && NULL != jitter)
          {
            void* fn = jgd(jitter);
            if (NULL != fn) LIBXS_FPTR_FROM_VPTR(libxs_gemm_djit_t, config.dgemm_jit, fn);
            config.jitter = jitter;
            config.flags = (libxs_gemm_flags_t)
              (config.flags | LIBXS_GEMM_FLAG_OWNJIT);
          }
        }
        else if (0 != use_jit && 0 != use_kernel
          && NULL != jcs && NULL != jgs
          && LIBXS_DATATYPE_F32 == kernel_shape->datatype)
        {
          const int mkl_ta = (0 == ta) ? 111 : 112;
          const int mkl_tb = (0 == tb) ? 111 : 112;
          void* jitter = NULL;
          if (2 != jcs(&jitter, 102, mkl_ta, mkl_tb,
            km, kn, kk, (float)kernel_shape->alpha, klda, kldb,
            (float)kernel_shape->beta, kldc) && NULL != jitter)
          {
            void* fn = jgs(jitter);
            if (NULL != fn) LIBXS_FPTR_FROM_VPTR(libxs_gemm_sjit_t, config.sgemm_jit, fn);
            config.jitter = jitter;
            config.flags = (libxs_gemm_flags_t)
              (config.flags | LIBXS_GEMM_FLAG_OWNJIT);
          }
        }
        if (NULL == config.dgemm_jit && NULL == config.sgemm_jit
          && 0 != use_xgemm && 0 != use_kernel && NULL != xdisp)
        {
          unsigned int xflags = 0;
          int xsmm_ok = 0;
          if (0 != ta) xflags |= 1;
          if (0 != tb) xflags |= 2;
          if (1.0 == kernel_shape->alpha) {
            if (0.0 == kernel_shape->beta) { xflags |= 4; xsmm_ok = 1; }
            else if (1.0 == kernel_shape->beta) xsmm_ok = 1;
          }
          if (0 != xsmm_ok) {
            libxs_xgemm_shape_t xs;
            const int xtype = (LIBXS_DATATYPE_F64 == kernel_shape->datatype) ? 0
              : ((LIBXS_DATATYPE_F32 == kernel_shape->datatype) ? 1 : -1);
            xs.m = km; xs.n = kn; xs.k = kk;
            xs.lda = klda; xs.ldb = kldb; xs.ldc = kldc;
            xs.a_in_type = xtype;
            xs.b_in_type = xtype;
            xs.out_type = xtype;
            xs.comp_type = xtype;
            if (0 <= xtype) {
              const libxs_gemm_xfn_t fn = xdisp(xs, xflags, 0);
              if (NULL != fn) config.xgemm = fn;
            }
          }
        }
        internal_libxs_gemm_blas_init(&config, backend, use_blas);
        if (0 != tiled
          && (NULL != config.dgemm_jit || NULL != config.sgemm_jit
            || NULL != config.xgemm))
        {
          libxs_gemm_config_t kconfig;
          LIBXS_MEMZERO(&kconfig);
          kconfig.shape = *kernel_shape;
          kconfig.dgemm_jit = config.dgemm_jit;
          kconfig.sgemm_jit = config.sgemm_jit;
          kconfig.xgemm = config.xgemm;
          kconfig.jitter = config.jitter;
          kconfig.flags = config.flags;
          kconfig.dgemm_blas = config.dgemm_blas;
          kconfig.sgemm_blas = config.sgemm_blas;
          /* the kernel entry owns the handle shared with this config */
          if (NULL != libxs_registry_set_hashed(reg,
            kernel_shape, sizeof(*kernel_shape), khash,
            &kconfig, sizeof(kconfig), libxs_registry_lock(reg)))
          {
            config.flags = (libxs_gemm_flags_t)
              (config.flags & ~LIBXS_GEMM_FLAG_OWNJIT);
          }
        }
      }
      { const int kernel_ok = (NULL != config.dgemm_jit
          || NULL != config.sgemm_jit || NULL != config.xgemm);
        if (0 != jit_allowed) { /* attempted once, hence never again */
          const unsigned int term = (0 == kernel_ok
            && NULL == kernel && 0 == gate)
            ? INTERNAL_GEMM_WARMUP_NOJIT : INTERNAL_GEMM_WARMUP_DONE;
          internal_libxs_gemm_warmup_term(shash, term);
        }
        if (0 == kernel_ok && NULL != own) {
          *own = config; /* caller-owned, hence not registered */
          result = own;
        }
        else {
          result = (libxs_gemm_config_t*)libxs_registry_set_hashed(
            reg, shape, sizeof(*shape), whash,
            &config, sizeof(config), libxs_registry_lock(reg));
          if (NULL == result) {
            static LIBXS_TLS libxs_gemm_config_t fallback;
            fallback = config;
            result = &fallback;
          }
        }
      }
      LIBXS_ASSERT(LIBXS_DATATYPE_F64 != shape->datatype
        || NULL != result->dgemm_blas);
      LIBXS_ASSERT(LIBXS_DATATYPE_F32 != shape->datatype
        || NULL != result->sgemm_blas);
    }
#if defined(LIBXS_GEMM_PRINT)
    { static int interval = -1;
      if (-1 == interval) {
        const char *const env = getenv("LIBXS_GEMM_PRINT");
        interval = (NULL != env ? atoi(env) : 0);
      }
      if (0 < interval) {
        static int counter = 0;
        if (0 == (++counter % interval)) {
          libxs_registry_info_t info;
          LIBXS_MEMZERO(&info);
          LIBXS_EXPECT(EXIT_SUCCESS == libxs_registry_info(reg, &info));
          LIBXS_ASSERT((NULL != result));
          fprintf(stderr, "LIBXS INFO[%u]: "
            "gemm=%s trans=%c%c mnk=%ix%ix%i ld=%ix%ix%i alpha=%g beta=%g regsize=%lu jit=%i\n",
            internal_libxs_gemm_origin(),
            libxs_typename(shape->datatype), shape->transa, shape->transb, shape->m, shape->n, shape->k,
            shape->lda, shape->ldb, shape->ldc, shape->alpha, shape->beta, (unsigned long)info.size,
            NULL != result->dgemm_jit || NULL != result->sgemm_jit || NULL != result->xgemm);
        }
      }
    }
#endif
  }
  return result;
}


LIBXS_API libxs_gemm_config_t* libxs_gemm_dispatch_rt(
  const libxs_gemm_shape_t* shape,
  const libxs_gemm_shape_t* kernel_shape,
  const libxs_gemm_backend_t* backend,
  void* registry)
{
  return internal_libxs_gemm_dispatch(
    shape, kernel_shape, backend, registry, NULL);
}


LIBXS_API int libxs_gemm_dispatch_cpy_rt(
  libxs_gemm_config_t* config,
  const libxs_gemm_shape_t* shape,
  const libxs_gemm_shape_t* kernel_shape,
  const libxs_gemm_backend_t* backend,
  void* registry)
{
  int result = 0;
  if (NULL != config) {
    libxs_gemm_config_t *const cfg = internal_libxs_gemm_dispatch(
      shape, kernel_shape, backend, registry, config);
    /* cfg == config when the impl filled it, else it is registry-owned */
    result = (cfg == config) ? 1 : libxs_gemm_config_cpy(config, cfg);
  }
  return result;
}


LIBXS_API void libxs_gemm_batch_task(
  const void* a_array[], const void* b_array[], void* c_array[],
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks)
{
  const int size = LIBXS_ABS(batchsize);
  const int nsplit = LIBXS_MIN(size, ntasks);
  LIBXS_ASSERT(NULL != config);
  LIBXS_ASSERT(0 <= tid);
  if (0 < nsplit && tid < nsplit) {
    const int need_lock = (1 < ntasks
      && 0 == (config->flags & LIBXS_GEMM_FLAG_NOLOCK));
    const int tasksize = LIBXS_UPDIV(size, nsplit);
    const int begin = tid * tasksize;
    int end = begin + tasksize;
    int lockidx = -1, i;
    if (end > size) end = size;
    if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
      if (NULL != config->dgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          config->dgemm_jit(config->jitter,
            (const double*)a_array[i],
            (const double*)b_array[i],
            (double*)c_array[i]);
        }
      }
      else if (NULL != config->xgemm) {
        libxs_gemm_param_t xparam;
        memset(&xparam, 0, sizeof(xparam));
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          xparam.a[0] = a_array[i];
          xparam.b[0] = b_array[i];
          xparam.c[0] = c_array[i];
          config->xgemm(&xparam);
        }
      }
      else {
        const int m = config->shape.m, n = config->shape.n, k = config->shape.k;
        const int lda = config->shape.lda, ldb = config->shape.ldb, ldc = config->shape.ldc;
        const double dalpha = config->shape.alpha, dbeta = config->shape.beta;
        const libxs_gemm_dblas_t dgemm_blas = (NULL != config->dgemm_blas
          ? config->dgemm_blas : internal_libxs_dgemm_default);
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          dgemm_blas(&config->shape.transa, &config->shape.transb, &m, &n, &k,
            &dalpha, (const double*)a_array[i], &lda,
            (const double*)b_array[i], &ldb,
            &dbeta, (double*)c_array[i], &ldc, 1, 1);
        }
      }
    }
    else if (LIBXS_DATATYPE_F32 == config->shape.datatype) {
      if (NULL != config->sgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          config->sgemm_jit(config->jitter,
            (const float*)a_array[i],
            (const float*)b_array[i],
            (float*)c_array[i]);
        }
      }
      else if (NULL != config->xgemm) {
        libxs_gemm_param_t xparam;
        memset(&xparam, 0, sizeof(xparam));
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          xparam.a[0] = a_array[i];
          xparam.b[0] = b_array[i];
          xparam.c[0] = c_array[i];
          config->xgemm(&xparam);
        }
      }
      else {
        const int m = config->shape.m, n = config->shape.n, k = config->shape.k;
        const int lda = config->shape.lda, ldb = config->shape.ldb, ldc = config->shape.ldc;
        const float falpha = (float)config->shape.alpha, fbeta = (float)config->shape.beta;
        const libxs_gemm_sblas_t sgemm_blas = (NULL != config->sgemm_blas
          ? config->sgemm_blas : internal_libxs_sgemm_default);
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          sgemm_blas(&config->shape.transa, &config->shape.transb, &m, &n, &k,
            &falpha, (const float*)a_array[i], &lda,
            (const float*)b_array[i], &ldb,
            &fbeta, (float*)c_array[i], &ldc, 1, 1);
        }
      }
    }
    else {
      LIBXS_ASSERT_MSG(0, "unsupported datatype");
    }
    INTERNAL_GEMM_UNLOCK(lockidx);
  }
}


LIBXS_API void libxs_gemm_batch(
  const void* a_array[], const void* b_array[], void* c_array[],
  int batchsize, const libxs_gemm_config_t* config)
{
  libxs_gemm_batch_task(a_array, b_array, c_array,
    batchsize, config, 0, 1);
}


LIBXS_API void libxs_gemm_index_task(
  const void* a, const int stride_a[],
  const void* b, const int stride_b[],
        void* c, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks)
{
  const int size = LIBXS_ABS(batchsize);
  const int nsplit = LIBXS_MIN(size, ntasks);
  LIBXS_ASSERT(NULL != config);
  LIBXS_ASSERT(NULL != stride_a && NULL != stride_b && NULL != stride_c);
  LIBXS_ASSERT(0 <= index_stride);
  LIBXS_ASSERT(0 <= tid);
  if (0 < nsplit && tid < nsplit) {
    const size_t elemsize = LIBXS_TYPESIZE(config->shape.datatype);
    const int need_lock = (1 < ntasks
      && 0 == (config->flags & LIBXS_GEMM_FLAG_NOLOCK));
    const int tasksize = LIBXS_UPDIV(size, nsplit);
    const int begin = tid * tasksize;
    int end = begin + tasksize;
    int lockidx = -1, i;
    if (end > size) end = size;
#define INTERNAL_GEMM_INDEX(I, STRIDE) \
    (0 != index_stride \
      ? (*(const int*)((const char*)(STRIDE) + (size_t)(I) * index_stride) \
          - index_base) \
      : (*(STRIDE) * (I)))
    if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
      if (NULL != config->dgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          const int ci_idx = INTERNAL_GEMM_INDEX(i, stride_c);
          double* ci = (double*)((char*)c + (size_t)ci_idx * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD_IDX(ci_idx, lockidx);
          config->dgemm_jit(config->jitter,
            (const double*)((const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize),
            (const double*)((const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize), ci);
        }
      }
      else if (NULL != config->xgemm) {
        libxs_gemm_param_t xparam;
        memset(&xparam, 0, sizeof(xparam));
        for (i = begin; i < end; ++i) {
          const int ci_idx = INTERNAL_GEMM_INDEX(i, stride_c);
          char *const ci = (char*)c + (size_t)ci_idx * elemsize;
          if (need_lock) INTERNAL_GEMM_LOCKFWD_IDX(ci_idx, lockidx);
          xparam.a[0] = (const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize;
          xparam.b[0] = (const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize;
          xparam.c[0] = ci;
          config->xgemm(&xparam);
        }
      }
      else {
        const int m = config->shape.m, n = config->shape.n, k = config->shape.k;
        const int lda = config->shape.lda, ldb = config->shape.ldb, ldc = config->shape.ldc;
        const double dalpha = config->shape.alpha, dbeta = config->shape.beta;
        const libxs_gemm_dblas_t dgemm_blas = (NULL != config->dgemm_blas
          ? config->dgemm_blas : internal_libxs_dgemm_default);
        for (i = begin; i < end; ++i) {
          const int ci_idx = INTERNAL_GEMM_INDEX(i, stride_c);
          double* ci = (double*)((char*)c + (size_t)ci_idx * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD_IDX(ci_idx, lockidx);
          dgemm_blas(&config->shape.transa, &config->shape.transb, &m, &n, &k, &dalpha,
            (const double*)((const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize), &lda,
            (const double*)((const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize), &ldb,
            &dbeta, ci, &ldc, 1, 1);
        }
      }
    }
    else if (LIBXS_DATATYPE_F32 == config->shape.datatype) {
      if (NULL != config->sgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          const int ci_idx = INTERNAL_GEMM_INDEX(i, stride_c);
          float* ci = (float*)((char*)c + (size_t)ci_idx * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD_IDX(ci_idx, lockidx);
          config->sgemm_jit(config->jitter,
            (const float*)((const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize),
            (const float*)((const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize), ci);
        }
      }
      else if (NULL != config->xgemm) {
        libxs_gemm_param_t xparam;
        memset(&xparam, 0, sizeof(xparam));
        for (i = begin; i < end; ++i) {
          const int ci_idx = INTERNAL_GEMM_INDEX(i, stride_c);
          char* ci = (char*)c + (size_t)ci_idx * elemsize;
          if (need_lock) INTERNAL_GEMM_LOCKFWD_IDX(ci_idx, lockidx);
          xparam.a[0] = (const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize;
          xparam.b[0] = (const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize;
          xparam.c[0] = ci;
          config->xgemm(&xparam);
        }
      }
      else {
        const int m = config->shape.m, n = config->shape.n, k = config->shape.k;
        const int lda = config->shape.lda, ldb = config->shape.ldb, ldc = config->shape.ldc;
        const float falpha = (float)config->shape.alpha, fbeta = (float)config->shape.beta;
        const libxs_gemm_sblas_t sgemm_blas = (NULL != config->sgemm_blas
          ? config->sgemm_blas : internal_libxs_sgemm_default);
        for (i = begin; i < end; ++i) {
          const int ci_idx = INTERNAL_GEMM_INDEX(i, stride_c);
          float* ci = (float*)((char*)c + (size_t)ci_idx * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD_IDX(ci_idx, lockidx);
          sgemm_blas(&config->shape.transa, &config->shape.transb, &m, &n, &k, &falpha,
            (const float*)((const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize), &lda,
            (const float*)((const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize), &ldb,
            &fbeta, ci, &ldc, 1, 1);
        }
      }
    }
    else {
      LIBXS_ASSERT_MSG(0, "unsupported datatype");
    }
#undef INTERNAL_GEMM_INDEX
    INTERNAL_GEMM_UNLOCK(lockidx);
  }
}


LIBXS_API void libxs_gemm_index(
  const void* a, const int stride_a[],
  const void* b, const int stride_b[],
        void* c, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config)
{
  libxs_gemm_index_task(a, stride_a, b, stride_b, c, stride_c,
    index_stride, index_base, batchsize, config, 0, 1);
}


LIBXS_API int libxs_gemm_ntasks(const libxs_gemm_config_t* config,
  int batchsize, int nthreads)
{
  int result = 1;
  const int size = LIBXS_ABS(batchsize);
  if (NULL != config && 1 < nthreads && 1 < size) {
    const int kernel = (NULL != config->dgemm_jit || NULL != config->sgemm_jit
      || NULL != config->xgemm);
    internal_libxs_gemm_init(); /* a caller-built config may precede any dispatch */
    result = LIBXS_MIN(nthreads, size); /* more tasks than elements idle */
    /* a threaded BLAS gives every call the whole team, which a split of fewer elements cannot */
    if (0 == kernel && size < nthreads && 0 != internal_libxs_blas_threaded()) {
      result = 1;
    }
  }
  return result;
}


LIBXS_API_INTERN void internal_libxs_gemm_blas(
  const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c,
  int m, int n, int k,
  int lda, int ldb, int ldc,
  double alpha, double beta);
LIBXS_API_INTERN void internal_libxs_gemm_blas(
  const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c,
  int m, int n, int k,
  int lda, int ldb, int ldc,
  double alpha, double beta)
{
  if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
    const libxs_gemm_dblas_t fn = (NULL != config->dgemm_blas)
      ? config->dgemm_blas : (NULL != internal_libxs_dgemm_blas)
      ? internal_libxs_dgemm_blas : internal_libxs_dgemm_default;
    fn(&config->shape.transa, &config->shape.transb, &m, &n, &k,
      &alpha, (const double*)a, &lda,
      (const double*)b, &ldb, &beta, (double*)c, &ldc, 1, 1);
  }
  else if (LIBXS_DATATYPE_F32 == config->shape.datatype) {
    const float falpha = (float)alpha, fbeta = (float)beta;
    const libxs_gemm_sblas_t fn = (NULL != config->sgemm_blas)
      ? config->sgemm_blas : (NULL != internal_libxs_sgemm_blas)
      ? internal_libxs_sgemm_blas : internal_libxs_sgemm_default;
    fn(&config->shape.transa, &config->shape.transb, &m, &n, &k,
      &falpha, (const float*)a, &lda,
      (const float*)b, &ldb, &fbeta, (float*)c, &ldc, 1, 1);
  }
}


#if defined(LIBXS_BUILD) && !defined(LIBXS_NOFORTRAN)

LIBXS_API void libxs_gemm_call_f(const libxs_gemm_config_t*,
  const void*, const void*, void*);
LIBXS_API void libxs_gemm_call_f(const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c)
{
  libxs_gemm_call(config, a, b, c);
}

#endif /*defined(LIBXS_BUILD) && !defined(LIBXS_NOFORTRAN)*/
