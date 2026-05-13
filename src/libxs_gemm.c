/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_gemm.h>
#include <libxs_mem.h>
#include "libxs_main.h"
#include "libxs_crc32.h"

#if !defined(LIBXS_GEMM_PRINT) && 1
# define LIBXS_GEMM_PRINT
#endif

#if !defined(LIBXS_GEMM_BLOCK_M)
# define LIBXS_GEMM_BLOCK_M 32
#endif
#if !defined(LIBXS_GEMM_BLOCK_N)
# define LIBXS_GEMM_BLOCK_N LIBXS_GEMM_BLOCK_M
#endif
#if !defined(LIBXS_GEMM_BLOCK_K)
# define LIBXS_GEMM_BLOCK_K 128
#endif
#if !defined(INTERNAL_GEMM_NLOCKS)
# define INTERNAL_GEMM_NLOCKS 16
#endif

#define INTERNAL_GEMM_NOTRANS(C) ('N' == (C) || 'n' == (C))


#define INTERNAL_SYRK_IRANGE(UPPER, JJ, CM, ISTART, IEND) \
  do { \
    (ISTART) = (UPPER) ? 0 : (JJ); \
    (IEND)   = (UPPER) ? (JJ) + 1 : (CM); \
  } while(0)

#define INTERNAL_SYRK_SCATTER(TYPE, CC, LDC, T, LDT, \
  IB, JB, CM, CN, UPPER, DIAG, ALPHA, BETA) \
  do { \
    int ii_, jj_; \
    if (DIAG) { \
      for (jj_ = 0; jj_ < (CN); ++jj_) { \
        int istart_, iend_; \
        INTERNAL_SYRK_IRANGE(UPPER, jj_, CM, istart_, iend_); \
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
  IB, JB, CM, CN, UPPER, DIAG, ALPHA, BETA) \
  do { \
    int ii_, jj_; \
    if (DIAG) { \
      for (jj_ = 0; jj_ < (CN); ++jj_) { \
        int istart_, iend_; \
        INTERNAL_SYRK_IRANGE(UPPER, jj_, CM, istart_, iend_); \
        for (ii_ = istart_; ii_ < iend_; ++ii_) { \
          ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] = \
            (BETA) * ((TYPE*)(CC))[(IB) + ii_ + (size_t)((JB) + jj_) * (LDC)] \
            + (ALPHA) * (((const TYPE*)(T1))[ii_ + (size_t)jj_ * (LDT)] \
                       + ((const TYPE*)(T1))[jj_ + (size_t)ii_ * (LDT)]); \
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

LIBXS_APIVAR_DEFINE(LIBXS_TLS void* internal_libxs_syrk_buffer);
LIBXS_APIVAR_DEFINE(LIBXS_TLS size_t internal_libxs_syrk_buffer_size);

LIBXS_APIVAR_DEFINE(libxs_gemm_dblas_t internal_libxs_dgemm_blas);
LIBXS_APIVAR_DEFINE(libxs_gemm_sblas_t internal_libxs_sgemm_blas);

LIBXS_APIVAR_DEFINE(internal_libxs_dsyrk_t internal_libxs_dsyrk_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_ssyrk_t internal_libxs_ssyrk_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_dsyr2k_t internal_libxs_dsyr2k_blas);
LIBXS_APIVAR_DEFINE(internal_libxs_ssyr2k_t internal_libxs_ssyr2k_blas);


LIBXS_API_INTERN void internal_libxs_gemm_init(void)
{
  if (NULL == internal_libxs_gemm_registry) {
    internal_libxs_gemm_registry = libxs_registry_create();
  }
  if (NULL == internal_libxs_dgemm_blas) {
    union { const void* pin; libxs_gemm_dblas_t pout; } wd;
    union { const void* pin; libxs_gemm_sblas_t pout; } ws;
    union { const void* pin; internal_libxs_dsyrk_t pout; } wdk;
    union { const void* pin; internal_libxs_ssyrk_t pout; } wsk;
    union { const void* pin; internal_libxs_dsyr2k_t pout; } wd2k;
    union { const void* pin; internal_libxs_ssyr2k_t pout; } ws2k;
    dlerror();
    wd.pin = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(dgemm)));
    if (NULL == dlerror() && NULL != wd.pout) internal_libxs_dgemm_blas = wd.pout;
    dlerror();
    ws.pin = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(sgemm)));
    if (NULL == dlerror() && NULL != ws.pout) internal_libxs_sgemm_blas = ws.pout;
    dlerror();
    wdk.pin = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(dsyrk)));
    if (NULL == dlerror() && NULL != wdk.pout) internal_libxs_dsyrk_blas = wdk.pout;
    dlerror();
    wsk.pin = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(ssyrk)));
    if (NULL == dlerror() && NULL != wsk.pout) internal_libxs_ssyrk_blas = wsk.pout;
    dlerror();
    wd2k.pin = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(dsyr2k)));
    if (NULL == dlerror() && NULL != wd2k.pout) internal_libxs_dsyr2k_blas = wd2k.pout;
    dlerror();
    ws2k.pin = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(LIBXS_FSYMBOL(ssyr2k)));
    if (NULL == dlerror() && NULL != ws2k.pout) internal_libxs_ssyr2k_blas = ws2k.pout;
  }
}


LIBXS_API_INTERN void internal_libxs_gemm_finalize(void)
{
  if (NULL != internal_libxs_gemm_registry) {
    libxs_gemm_release_registry(internal_libxs_gemm_registry);
    internal_libxs_gemm_registry = NULL;
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


LIBXS_API libxs_gemm_config_t* libxs_gemm_dispatch_rt(
  const libxs_gemm_shape_t* shape,
  const libxs_gemm_shape_t* kernel_shape,
  const libxs_gemm_backend_t* backend,
  void* registry)
{
  libxs_gemm_config_t* result = NULL;
  libxs_registry_t* reg = NULL;
  if (NULL == kernel_shape) kernel_shape = shape;
  if (NULL != shape
    && (LIBXS_DATATYPE_F64 == shape->datatype
     || LIBXS_DATATYPE_F32 == shape->datatype))
  {
    if (NULL != registry) {
      reg = (libxs_registry_t*)registry;
    }
    else {
      if (NULL == internal_libxs_gemm_registry) {
        internal_libxs_gemm_registry = libxs_registry_create();
      }
      reg = internal_libxs_gemm_registry;
    }
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
        if (NULL != backend && NULL != backend->jit_create_dgemm
            && NULL != backend->jit_get_dgemm
            && LIBXS_DATATYPE_F64 == kernel_shape->datatype) {
          const int mkl_ta = (0 == ta) ? 111 : 112;
          const int mkl_tb = (0 == tb) ? 111 : 112;
          void* jitter = NULL;
          if (2 != backend->jit_create_dgemm(&jitter, 102, mkl_ta, mkl_tb,
            km, kn, kk, kernel_shape->alpha, klda, kldb,
            kernel_shape->beta, kldc))
          {
            void* fn = backend->jit_get_dgemm(jitter);
            if (NULL != fn) LIBXS_VALUE_ASSIGN(config.dgemm_jit, fn);
            config.jitter = jitter;
          }
        }
        else if (NULL != backend && NULL != backend->jit_create_sgemm
            && NULL != backend->jit_get_sgemm
            && LIBXS_DATATYPE_F32 == kernel_shape->datatype) {
          const int mkl_ta = (0 == ta) ? 111 : 112;
          const int mkl_tb = (0 == tb) ? 111 : 112;
          void* jitter = NULL;
          if (2 != backend->jit_create_sgemm(&jitter, 101, mkl_ta, mkl_tb,
            km, kn, kk, (float)kernel_shape->alpha, klda, kldb,
            (float)kernel_shape->beta, kldc))
          {
            void* fn = backend->jit_get_sgemm(jitter);
            if (NULL != fn) LIBXS_VALUE_ASSIGN(config.sgemm_jit, fn);
            config.jitter = jitter;
          }
        }
        if (0 == libxs_gemm_ready(&config)
            && NULL != backend && NULL != backend->xgemm_dispatch) {
          int xflags = 0, xsmm_ok = 0;
          if (0 != ta) xflags |= 1;
          if (0 != tb) xflags |= 2;
          if (1.0 == kernel_shape->alpha) {
            if (0.0 == kernel_shape->beta) { xflags |= 4; xsmm_ok = 1; }
            else if (1.0 == kernel_shape->beta) xsmm_ok = 1;
          }
          if (0 != xsmm_ok) {
            libxs_gemm_xfn_t fn = backend->xgemm_dispatch(
              kernel_shape->datatype, xflags, km, kn, kk, klda, kldb, kldc);
            if (NULL != fn) config.xgemm = fn;
          }
        }
        if (NULL != backend) {
          config.dgemm_blas = backend->dgemm_blas;
          config.sgemm_blas = backend->sgemm_blas;
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
    }
#if defined(LIBXS_GEMM_PRINT)
    {
      static int interval = -1;
      if (-1 == interval) {
        const char *const env = getenv("LIBXS_GEMM_PRINT");
        interval = (NULL != env ? atoi(env) : 0);
      }
      if (0 < interval) {
        static int counter = 0;
        if (0 == (++counter % interval)) {
          const int ready = (NULL != internal_libxs_dsyr2k_blas && NULL != internal_libxs_ssyr2k_blas
              && NULL != internal_libxs_dgemm_blas && NULL != internal_libxs_sgemm_blas
              && NULL != internal_libxs_dsyrk_blas && NULL != internal_libxs_ssyrk_blas)
            ? 1 : libxs_gemm_ready(result);
          libxs_registry_info_t info;
          LIBXS_EXPECT(EXIT_SUCCESS == libxs_registry_info(reg, &info));
          LIBXS_ASSERT((NULL != result));
          fprintf(stderr, "LIBXS INFO: "
            "gemm=%s trans=%c%c mnk=%ix%ix%i ld=%ix%ix%i alpha=%g beta=%g regsize=%lu ready=%i\n",
            libxs_typename(shape->datatype), shape->transa, shape->transb,
            shape->m, shape->n, shape->k, shape->lda, shape->ldb, shape->ldc,
            shape->alpha, shape->beta, (unsigned long)info.size, ready);
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
  if (NULL != config && 0 < nsplit && 0 <= tid && tid < nsplit) {
    const int need_lock = (1 < ntasks
      && 0 == (config->flags & LIBXS_GEMM_FLAG_NOLOCK));
    const int tasksize = (size + nsplit - 1) / nsplit;
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
  if (NULL != config && 0 < nsplit && 0 <= tid && tid < nsplit
    && NULL != stride_a && NULL != stride_b && NULL != stride_c
    && 0 <= index_stride)
  {
    const size_t elemsize = LIBXS_TYPESIZE(config->shape.datatype);
    const int need_lock = (1 < ntasks
      && 0 == (config->flags & LIBXS_GEMM_FLAG_NOLOCK));
    const int tasksize = (size + nsplit - 1) / nsplit;
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
  const int km = LIBXS_MIN(n, LIBXS_GEMM_BLOCK_M);
  const int kn = LIBXS_MIN(n, LIBXS_GEMM_BLOCK_N);
  const int kk = LIBXS_MIN(k, LIBXS_GEMM_BLOCK_K);
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


LIBXS_API int libxs_syr2k_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c,
  int tid, int ntasks)
{
  int result = EXIT_FAILURE;
  if (NULL != config && NULL != a && NULL != b && NULL != c
    && 0 <= tid && tid < ntasks
    && (LIBXS_DATATYPE_F64 == config->shape.datatype
     || LIBXS_DATATYPE_F32 == config->shape.datatype))
  {
    const size_t elemsize = LIBXS_TYPESIZE(config->shape.datatype);
    const int upper = ('U' == uplo || 'u' == uplo);
    const int n = config->shape.m, k = config->shape.k;
    const int lda = config->shape.lda;
    const int ldb = config->shape.ldb;
    const int ldc = config->shape.ldc;
    if (n <= LIBXS_GEMM_BLOCK_M && n <= LIBXS_GEMM_BLOCK_N
      && k <= LIBXS_GEMM_BLOCK_K)
    {
      if (0 == tid) {
        const size_t need = (size_t)n * (size_t)n * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (NULL != scratch) {
          memset(scratch, 0, need);
          if (EXIT_SUCCESS != libxs_gemm_call(config, a, b, scratch)) {
            internal_libxs_gemm_blas(config, a, b, scratch,
              n, n, k, lda, ldb, n, 1.0, 0.0);
          }
          if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
            INTERNAL_SYR2K_SCATTER(double, c, ldc, scratch, scratch,
              n, 0, 0, n, n, upper, 1, alpha, beta);
          }
          else {
            INTERNAL_SYR2K_SCATTER(float, c, ldc, scratch, scratch,
              n, 0, 0, n, n, upper, 1, (float)alpha, (float)beta);
          }
          result = EXIT_SUCCESS;
        }
      }
      result = EXIT_SUCCESS;
    }
    else if (0 == tid && LIBXS_DATATYPE_F64 == config->shape.datatype
      && NULL != internal_libxs_dsyr2k_blas)
    {
      internal_libxs_dsyr2k_blas(&uplo, "N", &n, &k,
        (const double*)&alpha, (const double*)a, &lda,
        (const double*)b, &ldb,
        (const double*)&beta, (double*)c, &ldc);
      result = EXIT_SUCCESS;
    }
    else if (0 == tid && LIBXS_DATATYPE_F32 == config->shape.datatype
      && NULL != internal_libxs_ssyr2k_blas)
    {
      const float fa = (float)alpha, fb = (float)beta;
      internal_libxs_ssyr2k_blas(&uplo, "N", &n, &k,
        &fa, (const float*)a, &lda,
        (const float*)b, &ldb,
        &fb, (float*)c, &ldc);
      result = EXIT_SUCCESS;
    }
    else {
      const int bm = LIBXS_GEMM_BLOCK_M;
      const int bn = LIBXS_GEMM_BLOCK_N;
      const int bk = LIBXS_GEMM_BLOCK_K;
      const int nb_m = (n + bm - 1) / bm;
      const int nb_n = (n + bn - 1) / bn;
      const int nblocks = nb_m * nb_n;
      const int nsplit = LIBXS_MIN(nblocks, ntasks);
      if (tid < nsplit) {
        const int tasksize = (nblocks + nsplit - 1) / nsplit;
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
            int skip, kb;
            if (upper) {
              skip = (ib > jb + cn - 1);
            }
            else {
              skip = (ib + cm - 1 < jb);
            }
            if (0 == skip) {
              const int diag = (ib == jb);
              const int full = (cm == bm && cn == bn);
              const size_t clear = diag
                ? (size_t)bm * bn * elemsize
                : need;
              memset(scratch, 0, clear);
              for (kb = 0; kb < k; kb += bk) {
                const int ck = LIBXS_MIN(bk, k - kb);
                if (full && ck == bk && 0 != libxs_gemm_ready(config)) {
                  const size_t aoff = ((size_t)ib + (size_t)kb * lda) * elemsize;
                  const size_t boff = ((size_t)jb + (size_t)kb * ldb) * elemsize;
                  const size_t bioff = ((size_t)ib + (size_t)kb * ldb) * elemsize;
                  const size_t ajoff = ((size_t)jb + (size_t)kb * lda) * elemsize;
                  libxs_gemm_call(config,
                    (const char*)a + aoff,
                    (const char*)b + boff, scratch);
                  if (0 == diag) {
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
                  if (0 == diag) {
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
                  bm, ib, jb, cm, cn, upper, diag, alpha, beta);
              }
              else {
                INTERNAL_SYR2K_SCATTER(float, c, ldc, scratch, scratch2,
                  bm, ib, jb, cm, cn, upper, diag,
                  (float)alpha, (float)beta);
              }
            }
          }
          result = EXIT_SUCCESS;
        }
      }
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


LIBXS_API int libxs_syr2k(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c)
{
  return libxs_syr2k_task(config, uplo, alpha, beta, a, b, c, 0, 1);
}


LIBXS_API int libxs_syrk_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c,
  int tid, int ntasks)
{
  int result = EXIT_FAILURE;
  if (NULL != config && NULL != a && NULL != c
    && 0 <= tid && tid < ntasks
    && (LIBXS_DATATYPE_F64 == config->shape.datatype
     || LIBXS_DATATYPE_F32 == config->shape.datatype))
  {
    const int n = config->shape.m;
    const int k = config->shape.k;
    const int lda = config->shape.lda;
    const int ldc = config->shape.ldc;
    const int upper = ('U' == uplo || 'u' == uplo);
    const size_t elemsize = LIBXS_TYPESIZE(config->shape.datatype);
    if (n <= LIBXS_GEMM_BLOCK_M && n <= LIBXS_GEMM_BLOCK_N
      && k <= LIBXS_GEMM_BLOCK_K)
    {
      if (0 == tid) {
        const size_t need = (size_t)n * (size_t)n * elemsize;
        void* scratch = internal_libxs_syrk_scratch(need);
        if (NULL != scratch) {
          memset(scratch, 0, need);
          if (EXIT_SUCCESS != libxs_gemm_call(config, a, a, scratch)) {
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
          result = EXIT_SUCCESS;
        }
      }
      result = EXIT_SUCCESS;
    }
    else if (0 == tid && LIBXS_DATATYPE_F64 == config->shape.datatype
      && NULL != internal_libxs_dsyrk_blas)
    {
      internal_libxs_dsyrk_blas(&uplo, "N", &n, &k,
        (const double*)&alpha, (const double*)a, &lda,
        (const double*)&beta, (double*)c, &ldc);
      result = EXIT_SUCCESS;
    }
    else if (0 == tid && LIBXS_DATATYPE_F32 == config->shape.datatype
      && NULL != internal_libxs_ssyrk_blas)
    {
      const float fa = (float)alpha, fb = (float)beta;
      internal_libxs_ssyrk_blas(&uplo, "N", &n, &k,
        &fa, (const float*)a, &lda,
        &fb, (float*)c, &ldc);
      result = EXIT_SUCCESS;
    }
    else {
      const int bm = LIBXS_GEMM_BLOCK_M;
      const int bn = LIBXS_GEMM_BLOCK_N;
      const int bk = LIBXS_GEMM_BLOCK_K;
      const int nb_m = (n + bm - 1) / bm;
      const int nb_n = (n + bn - 1) / bn;
      const int nblocks = nb_m * nb_n;
      const int nsplit = LIBXS_MIN(nblocks, ntasks);
      if (tid < nsplit) {
        const int tasksize = (nblocks + nsplit - 1) / nsplit;
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
            int skip, kb;
            if (upper) {
              skip = (ib > jb + cn - 1);
            }
            else {
              skip = (ib + cm - 1 < jb);
            }
            if (0 == skip) {
              const int diag = (ib == jb);
              const int full = (cm == bm && cn == bn);
              memset(scratch, 0, need);
              for (kb = 0; kb < k; kb += bk) {
                const int ck = LIBXS_MIN(bk, k - kb);
                if (full && ck == bk && 0 != libxs_gemm_ready(config)) {
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
          result = EXIT_SUCCESS;
        }
      }
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


LIBXS_API int libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c)
{
  return libxs_syrk_task(config, uplo, alpha, beta, a, c, 0, 1);
}


#if defined(LIBXS_BUILD) && !defined(LIBXS_NOFORTRAN)

LIBXS_API int libxs_gemm_ready_f(const libxs_gemm_config_t*);
LIBXS_API int libxs_gemm_ready_f(const libxs_gemm_config_t* config)
{
  return libxs_gemm_ready(config);
}


LIBXS_API int libxs_gemm_call_f(const libxs_gemm_config_t*,
  const void*, const void*, void*);
LIBXS_API int libxs_gemm_call_f(const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c)
{
  return libxs_gemm_call(config, a, b, c);
}




#endif /*defined(LIBXS_BUILD) && !defined(LIBXS_NOFORTRAN)*/
