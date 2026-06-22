/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_gemm.h>
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
LIBXS_APIVAR_DEFINE(int internal_libxs_gemm_backend);

LIBXS_APIVAR_DEFINE(LIBXS_TLS void* internal_libxs_syrk_buffer);
LIBXS_APIVAR_DEFINE(LIBXS_TLS size_t internal_libxs_syrk_buffer_size);

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


LIBXS_API_INLINE void internal_libxs_gemm_print_registry(const libxs_registry_t* registry)
{
#if defined(LIBXS_GEMM_PRINT)
  if (NULL != registry) {
    const char *const env = getenv("LIBXS_GEMM_PRINT");
    if (NULL != env && 0 == atoi(env)) {
      libxs_registry_info_t info = { 0 };
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
            || (LIBXS_DATATYPE_F32 == shape->datatype && internal_libxs_sgemm_default != config->sgemm_blas)) ++nblas;
          else ++nfallback;
          config = (const libxs_gemm_config_t*)
            libxs_registry_next(registry, &key, &cursor);
        }
        fprintf(stderr, "LIBXS INFO: GEMM registry entries=%lu capacity=%lu nbytes=%lu backend=%s\n",
          (unsigned long)info.size, (unsigned long)info.capacity, (unsigned long)info.nbytes, backend);
        fprintf(stderr, "LIBXS INFO: GEMM histogram f64=%lu f32=%lu mkl-jit=%lu libxsmm=%lu blas=%lu fallback=%lu\n",
          nf64, nf32, njit, nxgemm, nblas, nfallback);
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


LIBXS_API libxs_gemm_config_t* libxs_gemm_dispatch_rt(
  const libxs_gemm_shape_t* shape,
  const libxs_gemm_shape_t* kernel_shape,
  const libxs_gemm_backend_t* backend,
  void* registry)
{
  libxs_gemm_config_t* result = NULL;
  libxs_registry_t* reg = NULL;
  libxs_gemm_shape_t key, kkey;
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
      result = (libxs_gemm_config_t*)libxs_registry_get(
        (const libxs_registry_t*)reg,
        shape, sizeof(*shape), libxs_registry_lock(reg));
    }
    if (NULL == result && NULL != reg) {
      libxs_gemm_config_t config;
      const libxs_gemm_config_t* kernel = NULL;
      if (0 != memcmp(shape, kernel_shape, sizeof(*shape))) {
        kernel = (const libxs_gemm_config_t*)libxs_registry_get(
          (const libxs_registry_t*)reg,
          kernel_shape, sizeof(*kernel_shape), libxs_registry_lock(reg));
      }
      LIBXS_MEMZERO(&config);
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
        const int use_jit = (INTERNAL_GEMM_BACKEND_AUTO == gemm_backend
          || INTERNAL_GEMM_BACKEND_MKL_JIT == gemm_backend);
        const int use_xgemm = (INTERNAL_GEMM_BACKEND_LIBXSMM >= gemm_backend);
        const int use_blas = (INTERNAL_GEMM_BACKEND_BLAS >= gemm_backend);
        const size_t elemsize = LIBXS_TYPESIZE(kernel_shape->datatype);
        const size_t kflops = (size_t)km * kn * kk * 2;
        const size_t kbytes = elemsize *
          ((size_t)km * kk + (size_t)kk * kn + (size_t)km * kn);
        const int use_kernel = (0 < internal_libxs_gemm_jit_max
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
        if (0 != use_jit && 0 != use_kernel
          && NULL != jcd && NULL != jgd
          && LIBXS_DATATYPE_F64 == kernel_shape->datatype)
        {
          const int mkl_ta = (0 == ta) ? 111 : 112;
          const int mkl_tb = (0 == tb) ? 111 : 112;
          void* jitter = NULL;
          if (2 != jcd(&jitter, 102, mkl_ta, mkl_tb,
            km, kn, kk, kernel_shape->alpha, klda, kldb,
            kernel_shape->beta, kldc))
          {
            void* fn = jgd(jitter);
            if (NULL != fn) LIBXS_FPTR_FROM_VPTR(libxs_gemm_djit_t, config.dgemm_jit, fn);
            config.jitter = jitter;
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
            (float)kernel_shape->beta, kldc))
          {
            void* fn = jgs(jitter);
            if (NULL != fn) LIBXS_FPTR_FROM_VPTR(libxs_gemm_sjit_t, config.sgemm_jit, fn);
            config.jitter = jitter;
          }
        }
        if (NULL == config.dgemm_jit && NULL == config.sgemm_jit
          && 0 != use_xgemm && 0 != use_kernel && NULL != xdisp) {
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
        if (0 != use_blas) {
          config.dgemm_blas = (NULL != backend && NULL != backend->dgemm_blas)
            ? backend->dgemm_blas : (NULL != internal_libxs_dgemm_blas)
            ? internal_libxs_dgemm_blas : internal_libxs_dgemm_default;
          config.sgemm_blas = (NULL != backend && NULL != backend->sgemm_blas)
            ? backend->sgemm_blas : (NULL != internal_libxs_sgemm_blas)
            ? internal_libxs_sgemm_blas : internal_libxs_sgemm_default;
        }
        else {
          config.dgemm_blas = internal_libxs_dgemm_default;
          config.sgemm_blas = internal_libxs_sgemm_default;
        }
        if (0 != memcmp(shape, kernel_shape, sizeof(*shape))) {
          libxs_gemm_config_t kconfig;
          LIBXS_MEMZERO(&kconfig);
          kconfig.shape = *kernel_shape;
          kconfig.dgemm_jit = config.dgemm_jit;
          kconfig.sgemm_jit = config.sgemm_jit;
          kconfig.xgemm = config.xgemm;
          kconfig.jitter = config.jitter;
          kconfig.dgemm_blas = config.dgemm_blas;
          kconfig.sgemm_blas = config.sgemm_blas;
          libxs_registry_set(reg, kernel_shape, sizeof(*kernel_shape),
            &kconfig, sizeof(kconfig), libxs_registry_lock(reg));
        }
      }
      result = (libxs_gemm_config_t*)libxs_registry_set(
        reg, shape, sizeof(*shape),
        &config, sizeof(config), libxs_registry_lock(reg));
      if (NULL == result) {
        static LIBXS_TLS libxs_gemm_config_t fallback;
        fallback = config;
        result = &fallback;
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
          libxs_registry_info_t info = { 0 };
          LIBXS_EXPECT(EXIT_SUCCESS == libxs_registry_info(reg, &info));
          LIBXS_ASSERT((NULL != result));
          fprintf(stderr, "LIBXS INFO: "
            "gemm=%s trans=%c%c mnk=%ix%ix%i ld=%ix%ix%i alpha=%g beta=%g regsize=%lu jit=%i\n",
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


LIBXS_API libxs_gemm_config_t* libxs_syr2k_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  const libxs_gemm_backend_t* backend, void* registry)
{
  const int km = LIBXS_MIN(n, internal_libxs_gemm_bm);
  const int kn = LIBXS_MIN(n, internal_libxs_gemm_bn);
  const int kk = LIBXS_MIN(k, internal_libxs_gemm_bk);
  libxs_gemm_shape_t shape, kshape;
  LIBXS_MEMZERO(&shape);
  shape.datatype = datatype;
  shape.transa = 'N'; shape.transb = 'T';
  shape.m = n; shape.n = n; shape.k = k;
  shape.lda = lda; shape.ldb = ldb; shape.ldc = ldc;
  shape.alpha = 1.0; shape.beta = 0.0;
  LIBXS_MEMZERO(&kshape);
  kshape.datatype = datatype;
  kshape.transa = 'N'; kshape.transb = 'T';
  kshape.m = km; kshape.n = kn; kshape.k = kk;
  kshape.lda = lda; kshape.ldb = ldb; kshape.ldc = km;
  kshape.alpha = 1.0; kshape.beta = 1.0;
  return libxs_gemm_dispatch_rt(&shape, &kshape, backend, registry);
}


LIBXS_API libxs_gemm_config_t* libxs_syrk_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  const libxs_gemm_backend_t* backend, void* registry)
{
  return libxs_syr2k_dispatch(datatype, n, k, lda, lda, ldc,
    backend, registry);
}


LIBXS_API_INTERN void* internal_libxs_syrk_scratch(size_t need);
LIBXS_API_INTERN void* internal_libxs_syrk_scratch(size_t need)
{
  if (need > internal_libxs_syrk_buffer_size) {
    free(internal_libxs_syrk_buffer);
    internal_libxs_syrk_buffer = malloc(need);
    internal_libxs_syrk_buffer_size = (NULL != internal_libxs_syrk_buffer) ? need : 0;
  }
  return internal_libxs_syrk_buffer;
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
    else if (0 == tid && LIBXS_DATATYPE_F64 == config->shape.datatype
      && NULL != internal_libxs_dsyr2k_blas)
    {
#if defined(LIBXS_GEMM_PRINT)
      { static int interval = -1;
        if (-1 == interval) {
          const char *const env = getenv("LIBXS_SYRK_PRINT");
          interval = (NULL != env ? atoi(env) : 0);
        }
        if (0 < interval) {
          fprintf(stderr, "LIBXS INFO: dsyr2k uplo=%c n=%i k=%i lda=%i ldb=%i ldc=%i"
            " alpha=%g beta=%g upper=%i\n", uplo, n, k, lda, ldb, ldc, alpha, beta, upper);
        }
      }
#endif
      internal_libxs_dsyr2k_blas(&uplo, "N", &n, &k,
        (const double*)&alpha, (const double*)a, &lda,
        (const double*)b, &ldb,
        (const double*)&beta, (double*)c, &ldc);
    }
    else if (0 == tid && LIBXS_DATATYPE_F32 == config->shape.datatype
      && NULL != internal_libxs_ssyr2k_blas)
    {
      const float fa = (float)alpha, fb = (float)beta;
      internal_libxs_ssyr2k_blas(&uplo, "N", &n, &k,
        &fa, (const float*)a, &lda,
        (const float*)b, &ldb,
        &fb, (float*)c, &ldc);
    }
    else {
      const int bm = internal_libxs_gemm_bm;
      const int bn = internal_libxs_gemm_bn;
      const int bk = internal_libxs_gemm_bk;
      const int nb_m = LIBXS_UPDIV(n, bm);
      const int nb_n = LIBXS_UPDIV(n, bn);
      const int nblocks = nb_m * nb_n;
      const int nsplit = LIBXS_MIN(nblocks, ntasks);
      if (tid < nsplit) {
        const int tasksize = LIBXS_UPDIV(nblocks, nsplit);
        const int begin = tid * tasksize;
        int end = begin + tasksize;
        const size_t need = (size_t)bm * bn * 2 * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (end > nblocks) end = nblocks;
        if (NULL != scratch) {
          void* scratch2 = (char*)scratch + (size_t)bm * bn * elemsize;
          int idx;
          for (idx = begin; idx < end; ++idx) {
            const int jb = (idx / nb_m) * bn;
            const int ib = (idx % nb_m) * bm;
            const int cn = LIBXS_MIN(bn, n - jb);
            const int cm = LIBXS_MIN(bm, n - ib);
            const int skip = upper
              ? (ib > jb + cn - 1) : (ib + cm - 1 < jb);
            if (0 == skip) {
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
    else if (0 == tid && LIBXS_DATATYPE_F64 == config->shape.datatype
      && NULL != internal_libxs_dsyrk_blas)
    {
#if defined(LIBXS_GEMM_PRINT)
      { static int interval = -1;
        if (-1 == interval) {
          const char *const env = getenv("LIBXS_SYRK_PRINT");
          interval = (NULL != env ? atoi(env) : 0);
        }
        if (0 < interval) {
          fprintf(stderr, "LIBXS INFO: dsyrk uplo=%c n=%i k=%i lda=%i ldc=%i"
            " alpha=%g beta=%g upper=%i\n", uplo, n, k, lda, ldc, alpha, beta, upper);
        }
      }
#endif
      internal_libxs_dsyrk_blas(&uplo, "N", &n, &k,
        (const double*)&alpha, (const double*)a, &lda,
        (const double*)&beta, (double*)c, &ldc);
    }
    else if (0 == tid && LIBXS_DATATYPE_F32 == config->shape.datatype
      && NULL != internal_libxs_ssyrk_blas)
    {
      const float fa = (float)alpha, fb = (float)beta;
      internal_libxs_ssyrk_blas(&uplo, "N", &n, &k,
        &fa, (const float*)a, &lda,
        &fb, (float*)c, &ldc);
    }
    else {
      const int bm = internal_libxs_gemm_bm;
      const int bn = internal_libxs_gemm_bn;
      const int bk = internal_libxs_gemm_bk;
      const int nb_m = LIBXS_UPDIV(n, bm);
      const int nb_n = LIBXS_UPDIV(n, bn);
      const int nblocks = nb_m * nb_n;
      const int nsplit = LIBXS_MIN(nblocks, ntasks);
      if (tid < nsplit) {
        const int tasksize = LIBXS_UPDIV(nblocks, nsplit);
        const int begin = tid * tasksize;
        int end = begin + tasksize;
        const size_t need = (size_t)bm * bn * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (end > nblocks) end = nblocks;
        if (NULL != scratch) {
          int idx;
          for (idx = begin; idx < end; ++idx) {
            const int jb = (idx / nb_m) * bn;
            const int ib = (idx % nb_m) * bm;
            const int cn = LIBXS_MIN(bn, n - jb);
            const int cm = LIBXS_MIN(bm, n - ib);
            const int skip = upper
              ? (ib > jb + cn - 1) : (ib + cm - 1 < jb);
            if (0 == skip) {
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
