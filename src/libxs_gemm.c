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
#include "libxs_main.h"
#include "libxs_crc32.h"

#if !defined(LIBXS_GEMM_PRINT) && 1
# define LIBXS_GEMM_PRINT
#endif

#if !defined(LIBXS_GEMM_BM)
# define LIBXS_GEMM_BM 24
#endif
#if !defined(LIBXS_GEMM_BN)
# define LIBXS_GEMM_BN 48
#endif
#if !defined(LIBXS_GEMM_BK)
# define LIBXS_GEMM_BK 128
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

#define INTERNAL_GEMM_BACKEND_AUTO 0
#define INTERNAL_GEMM_BACKEND_MKL_JIT 1
#define INTERNAL_GEMM_BACKEND_LIBXSMM 2
#define INTERNAL_GEMM_BACKEND_BLAS 3
#define INTERNAL_GEMM_BACKEND_DEFAULT 4

#define INTERNAL_GEMM_NOTRANS(C) ('N' == (C) || 'n' == (C))

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


typedef void (*internal_libxs_dsyrk_t)(const char*, const char*, const int*, const int*,
  const double*, const double*, const int*, const double*, double*, const int*);
typedef void (*internal_libxs_ssyrk_t)(const char*, const char*, const int*, const int*,
  const float*, const float*, const int*, const float*, float*, const int*);
typedef void (*internal_libxs_dsyr2k_t)(const char*, const char*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
typedef void (*internal_libxs_ssyr2k_t)(const char*, const char*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);


LIBXS_APIVAR_DEFINE(LIBXS_LOCK_TYPE(LIBXS_LOCK) internal_libxs_gemm_locks[INTERNAL_GEMM_NLOCKS]);
LIBXS_APIVAR_DEFINE(libxs_registry_t* internal_libxs_gemm_registry);

LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_bm);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_bn);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_bk);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_jit_max);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_jit_warmup);
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_backend);
LIBXS_APIVAR_DEFINE(unsigned int
  internal_libxs_gemm_warmup[LIBXS_GEMM_NWARMUP]);
LIBXS_APIVAR_DEFINE(unsigned int internal_libxs_gemm_nshape);

static LIBXS_TLS void* internal_libxs_syrk_buffer;
static LIBXS_TLS size_t internal_libxs_syrk_buffer_size;

LIBXS_APIVAR_DEFINE(libxs_gemm_dblas_t internal_libxs_dgemm_blas);
LIBXS_APIVAR_DEFINE(libxs_gemm_sblas_t internal_libxs_sgemm_blas);

LIBXS_APIVAR_DEFINE(internal_libxs_dsyrk_t internal_libxs_dsyrk_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_ssyrk_t internal_libxs_ssyrk_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_dsyr2k_t internal_libxs_dsyr2k_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_ssyr2k_t internal_libxs_ssyr2k_blas);

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
    const char *const gemm_jit_max_env = getenv("LIBXS_GEMM_JIT_MAX");
    const char *const gemm_jit_warmup_env = getenv("LIBXS_GEMM_JIT_WARMUP");
    const char *const gemm_backend_env = getenv("LIBXS_GEMM_BACKEND");
#if defined(LIBXS_INTERCEPT_DYNAMIC)
    const char *const env = getenv("LIBXS_SYRK_BLAS");
    const int syrk_blas = (NULL == env ? 1/*default*/ : atoi(env));
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
  const double* beta,        double* c, const int* ldc);
LIBXS_API_INTERN void internal_libxs_dgemm_default(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc)
{
  const int mm = *m, nn = *n, kk = *k;
  const int llda = *lda, lldb = *ldb, lldc = *ldc;
  const double dalpha = (NULL != alpha ? *alpha : 1.0);
  const double dbeta = (NULL != beta ? *beta : 0.0);
  int i, j, p;
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
      c[i + j * lldc] = dalpha * sum + dbeta * c[i + j * lldc];
    }
  }
}


LIBXS_API_INTERN void internal_libxs_sgemm_default(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc);
LIBXS_API_INTERN void internal_libxs_sgemm_default(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc)
{
  const int mm = *m, nn = *n, kk = *k;
  const int llda = *lda, lldb = *ldb, lldc = *ldc;
  const float falpha = (NULL != alpha ? *alpha : 1.f);
  const float fbeta = (NULL != beta ? *beta : 0.f);
  int i, j, p;
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
      c[i + j * lldc] = falpha * sum + fbeta * c[i + j * lldc];
    }
  }
}


/**
 * Identifies the process behind a line, because ranks share one stream and
 * their lines interleave. The rank is preferred where it can be determined
 * (libxs_rid), and the value is stable for the lifetime of the process.
 */
LIBXS_API_INLINE unsigned int internal_libxs_gemm_origin(void)
{
  static unsigned int origin = 0xFFFFFFFF;
  if (0xFFFFFFFF == origin) origin = libxs_rid();
  return origin;
}


LIBXS_API_INLINE void internal_libxs_gemm_print_registry(const libxs_registry_t* registry)
{
#if defined(LIBXS_GEMM_PRINT)
  if (NULL != registry) {
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


/** Assigns the BLAS entry points, either the backend's or the library's. */
LIBXS_API_INLINE void internal_libxs_gemm_blas_init(
  libxs_gemm_config_t* config,
  const libxs_gemm_backend_t* backend, int use_blas)
{
  if (0 != use_blas) {
    config->dgemm_blas = (NULL != backend && NULL != backend->dgemm_blas)
      ? backend->dgemm_blas : (NULL != internal_libxs_dgemm_blas)
      ? internal_libxs_dgemm_blas : internal_libxs_dgemm_default;
    config->sgemm_blas = (NULL != backend && NULL != backend->sgemm_blas)
      ? backend->sgemm_blas : (NULL != internal_libxs_sgemm_blas)
      ? internal_libxs_sgemm_blas : internal_libxs_sgemm_default;
  }
  else {
    config->dgemm_blas = internal_libxs_dgemm_default;
    config->sgemm_blas = internal_libxs_sgemm_default;
  }
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
  unsigned int whash = 0, khash = 0, wstate = 0;
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
      /* a NOJIT shape owns no entry when the caller owns the config */
      if (NULL == own || INTERNAL_GEMM_WARMUP_NOJIT
        != internal_libxs_gemm_warmup_state(whash))
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
      jit_due = internal_libxs_gemm_warmup_due(whash, &wstate);
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
        const int gemm_backend = internal_libxs_gemm_backend;
        /* MKL JIT assumes resident operands: a tile streams and loses ~2x to BLAS */
        const int use_jit = (INTERNAL_GEMM_BACKEND_MKL_JIT == gemm_backend
          || (INTERNAL_GEMM_BACKEND_AUTO == gemm_backend && 0 == tiled));
        const int use_xgemm = (INTERNAL_GEMM_BACKEND_LIBXSMM >= gemm_backend);
        const int use_blas = (INTERNAL_GEMM_BACKEND_BLAS >= gemm_backend);
        const size_t elemsize = LIBXS_TYPESIZE(kernel_shape->datatype);
        const size_t kflops = (size_t)km * kn * kk * 2;
        const size_t kbytes = elemsize *
          ((size_t)km * kk + (size_t)kk * kn + (size_t)km * kn);
        const int use_kernel = (0 != jit_allowed
          && 0 < internal_libxs_gemm_jit_max
          && kflops < (size_t)internal_libxs_gemm_jit_max * kbytes);
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
          internal_libxs_gemm_warmup_term(whash, term);
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
            &dbeta, (double*)c_array[i], &ldc);
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
            &fbeta, (float*)c_array[i], &ldc);
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
            &dbeta, ci, &ldc);
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
            &fbeta, ci, &ldc);
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
      (const double*)b, &ldb, &beta, (double*)c, &ldc);
  }
  else if (LIBXS_DATATYPE_F32 == config->shape.datatype) {
    const float falpha = (float)alpha, fbeta = (float)beta;
    const libxs_gemm_sblas_t fn = (NULL != config->sgemm_blas)
      ? config->sgemm_blas : (NULL != internal_libxs_sgemm_blas)
      ? internal_libxs_sgemm_blas : internal_libxs_sgemm_default;
    fn(&config->shape.transa, &config->shape.transb, &m, &n, &k,
      &falpha, (const float*)a, &lda,
      (const float*)b, &ldb, &fbeta, (float*)c, &ldc);
  }
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
    else if (LIBXS_DATATYPE_F64 == config->shape.datatype
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
          (const double*)&beta, (double*)c, &ldc);
      }
    }
    else if (LIBXS_DATATYPE_F32 == config->shape.datatype
      && NULL != internal_libxs_ssyr2k_blas)
    {
      if (0 == tid) {
        const float fa = (float)alpha, fb = (float)beta;
        internal_libxs_ssyr2k_blas(&uplo, "N", &n, &k,
          &fa, (const float*)a, &lda,
          (const float*)b, &ldb,
          &fb, (float*)c, &ldc);
      }
    }
    else {
      const int bm = internal_libxs_gemm_bm;
      const int bn = internal_libxs_gemm_bn;
      const int bk = internal_libxs_gemm_bk;
      const int nb_m = LIBXS_UPDIV(n, bm);
      const int nb_n = LIBXS_UPDIV(n, bn);
      int j_begin, j_end;
      internal_libxs_syrk_partition(tid, ntasks, nb_n, upper, &j_begin, &j_end);
      if (j_begin < j_end) {
        const size_t need = (size_t)bm * bn * 2 * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (NULL != scratch) {
          void* scratch2 = (char*)scratch + (size_t)bm * bn * elemsize;
          int j;
          for (j = j_begin; j < j_end; ++j) {
            const int jb = j * bn;
            const int cn = LIBXS_MIN(bn, n - jb);
            const int i_begin = (0 == upper ? (jb / bm) : 0);
            const int i_end = (0 == upper ? nb_m : LIBXS_MIN(nb_m, (jb + cn - 1) / bm + 1));
            int i;
            for (i = i_begin; i < i_end; ++i) {
              const int ib = i * bm;
              const int cm = LIBXS_MIN(bm, n - ib);
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
    else if (LIBXS_DATATYPE_F64 == config->shape.datatype
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
          (const double*)&beta, (double*)c, &ldc);
      }
    }
    else if (LIBXS_DATATYPE_F32 == config->shape.datatype
      && NULL != internal_libxs_ssyrk_blas)
    {
      if (0 == tid) {
        const float fa = (float)alpha, fb = (float)beta;
        internal_libxs_ssyrk_blas(&uplo, "N", &n, &k,
          &fa, (const float*)a, &lda,
          &fb, (float*)c, &ldc);
      }
    }
    else {
      const int bm = internal_libxs_gemm_bm;
      const int bn = internal_libxs_gemm_bn;
      const int bk = internal_libxs_gemm_bk;
      const int nb_m = LIBXS_UPDIV(n, bm);
      const int nb_n = LIBXS_UPDIV(n, bn);
      int j_begin, j_end;
      internal_libxs_syrk_partition(tid, ntasks, nb_n, upper, &j_begin, &j_end);
      if (j_begin < j_end) {
        const size_t need = (size_t)bm * bn * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (NULL != scratch) {
          int j;
          for (j = j_begin; j < j_end; ++j) {
            const int jb = j * bn;
            const int cn = LIBXS_MIN(bn, n - jb);
            const int i_begin = (0 == upper ? (jb / bm) : 0);
            const int i_end = (0 == upper ? nb_m : LIBXS_MIN(nb_m, (jb + cn - 1) / bm + 1));
            int i;
            for (i = i_begin; i < i_end; ++i) {
              const int ib = i * bm;
              const int cm = LIBXS_MIN(bm, n - ib);
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


LIBXS_API void libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c)
{
  libxs_syrk_task(config, uplo, alpha, beta, a, c, 0, 1);
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
