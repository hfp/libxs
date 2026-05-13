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

#if !defined(LIBXS_GEMM_PRINT) && 0
# define LIBXS_GEMM_PRINT
#endif

#if !defined(LIBXS_GEMM_BLOCK_M)
# define LIBXS_GEMM_BLOCK_M 64
#endif
#if !defined(LIBXS_GEMM_BLOCK_N)
# define LIBXS_GEMM_BLOCK_N LIBXS_GEMM_BLOCK_M
#endif
#if !defined(LIBXS_GEMM_BLOCK_K)
# define LIBXS_GEMM_BLOCK_K LIBXS_GEMM_BLOCK_M
#endif

#define INTERNAL_GEMM_NOTRANS(C) ('N' == (C) || 'n' == (C))
#define INTERNAL_GEMM_NLOCKS 16
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

LIBXS_APIVAR_DEFINE(LIBXS_TLS void* internal_libxs_syrk_buffer);
LIBXS_APIVAR_DEFINE(LIBXS_TLS size_t internal_libxs_syrk_buffer_size);


LIBXS_API_INTERN void internal_libxs_gemm_init(void);
LIBXS_API_INTERN void internal_libxs_gemm_init(void)
{
  if (NULL == internal_libxs_gemm_registry) {
    internal_libxs_gemm_registry = libxs_registry_create();
  }
}


LIBXS_API_INTERN void internal_libxs_gemm_finalize(void);
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


#if defined(LIBXS_GEMM_PRINT)
LIBXS_API_INTERN void internal_libxs_gemm_print(FILE* ostream,
  const libxs_gemm_config_t* config, const libxs_registry_t* reg);
LIBXS_API_INTERN void internal_libxs_gemm_print(FILE* ostream,
  const libxs_gemm_config_t* config, const libxs_registry_t* reg)
{
  static int interval = -1;
  if (-1 == interval) {
    const char *const env = getenv("LIBXS_GEMM_PRINT");
    interval = (NULL != env ? atoi(env) : 0);
  }
  if (0 < interval) {
    static int counter = 0;
    if (0 == (++counter % interval) && NULL != config) {
      libxs_registry_info_t info;
      LIBXS_EXPECT(EXIT_SUCCESS == libxs_registry_info(reg, &info));
      fprintf(ostream, "LIBXS INFO: gemm=%s trans=%c%c mnk=%ix%ix%i ld=%ix%ix%i alpha=%g beta=%g regsize=%lu ready=%i\n",
        libxs_typename(config->shape.datatype), config->shape.transa, config->shape.transb,
        config->shape.m, config->shape.n, config->shape.k,
        config->shape.lda, config->shape.ldb, config->shape.ldc,
        config->shape.alpha, config->shape.beta,
        (unsigned long)info.size, libxs_gemm_ready(config));
    }
  }
}
#endif


LIBXS_API libxs_gemm_config_t* libxs_gemm_dispatch_rt(
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta,
  libxs_jit_create_dgemm_t jit_create_dgemm,
  libxs_jit_get_dgemm_t jit_get_dgemm,
  libxs_jit_create_sgemm_t jit_create_sgemm,
  libxs_jit_get_sgemm_t jit_get_sgemm,
  libxs_xgemm_dispatch_t  xgemm_dispatch,
  libxs_gemm_dblas_t dgemm_blas,
  libxs_gemm_sblas_t sgemm_blas,
  void* registry)
{
  libxs_gemm_config_t* result = NULL;
  libxs_gemm_config_t config;
  libxs_gemm_shape_t shape;
  libxs_registry_t* reg = NULL;
  LIBXS_MEMZERO(&shape);
  shape.datatype = datatype;
  shape.transa = transa;
  shape.transb = transb;
  shape.m = m; shape.n = n; shape.k = k;
  shape.lda = lda; shape.ldb = ldb; shape.ldc = ldc;
  if (LIBXS_DATATYPE_F64 == datatype) {
    shape.alpha = (NULL != alpha ? *(const double*)alpha : 1.0);
    shape.beta  = (NULL != beta  ? *(const double*)beta  : 0.0);
  }
  else if (LIBXS_DATATYPE_F32 == datatype) {
    shape.alpha = (NULL != alpha ? (double)*(const float*)alpha : 1.0);
    shape.beta  = (NULL != beta  ? (double)*(const float*)beta  : 0.0);
  }
  if (LIBXS_DATATYPE_F64 == datatype || LIBXS_DATATYPE_F32 == datatype) {
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
      result = (libxs_gemm_config_t*)
        libxs_registry_get((const libxs_registry_t*)reg,
          &shape, sizeof(shape), libxs_registry_lock(reg));
    }
  }
  if (NULL == result && NULL != reg) {
    const int ta = ('N' != transa && 'n' != transa);
    const int tb = ('N' != transb && 'n' != transb);
    LIBXS_MEMZERO(&config);
    config.shape = shape;
    /* MKL JIT (dgemm) */
    if (NULL != jit_create_dgemm && NULL != jit_get_dgemm
        && LIBXS_DATATYPE_F64 == datatype) {
      const int mkl_ta = (0 == ta) ? 111 : 112;
      const int mkl_tb = (0 == tb) ? 111 : 112;
      void* jitter = NULL;
      if (2 != jit_create_dgemm(&jitter, 102, mkl_ta, mkl_tb, m, n, k,
        NULL != alpha ? *(const double*)alpha : 1.0, lda, ldb,
        NULL != beta ? *(const double*)beta : 0.0, ldc))
      {
        void* fn = jit_get_dgemm(jitter);
        if (NULL != fn) LIBXS_VALUE_ASSIGN(config.dgemm_jit, fn);
        config.jitter = jitter;
      }
    }
    /* MKL JIT (sgemm) */
    else if (NULL != jit_create_sgemm && NULL != jit_get_sgemm
        && LIBXS_DATATYPE_F32 == datatype) {
      const int mkl_ta = (0 == ta) ? 111 : 112;
      const int mkl_tb = (0 == tb) ? 111 : 112;
      void* jitter = NULL;
      if (2 != jit_create_sgemm(&jitter, 101, mkl_ta, mkl_tb, m, n, k,
        NULL != alpha ? *(const float*)alpha : 1.0f, lda, ldb,
        NULL != beta ? *(const float*)beta : 0.0f, ldc))
      {
        void* fn = jit_get_sgemm(jitter);
        if (NULL != fn) LIBXS_VALUE_ASSIGN(config.sgemm_jit, fn);
        config.jitter = jitter;
      }
    }
    /* LIBXSMM-style xgemm dispatch */
    if (0 == libxs_gemm_ready(&config) && NULL != xgemm_dispatch) {
      int xflags = 0, xsmm_ok = 0;
      if (0 != ta) xflags |= 1;
      if (0 != tb) xflags |= 2;
      if (LIBXS_DATATYPE_F64 == datatype) {
        const double a1 = (NULL != alpha ? *(const double*)alpha : 1.0);
        const double b1 = (NULL != beta ? *(const double*)beta : 0.0);
        if (1.0 == a1) {
          if (0.0 == b1) { xflags |= 4; xsmm_ok = 1; }
          else if (1.0 == b1) xsmm_ok = 1;
        }
      }
      else if (LIBXS_DATATYPE_F32 == datatype) {
        const float a1 = (NULL != alpha ? *(const float*)alpha : 1.0f);
        const float b1 = (NULL != beta ? *(const float*)beta : 0.0f);
        if (1.0f == a1) {
          if (0.0f == b1) { xflags |= 4; xsmm_ok = 1; }
          else if (1.0f == b1) xsmm_ok = 1;
        }
      }
      if (0 != xsmm_ok) {
        libxs_gemm_xfn_t fn = xgemm_dispatch(
          datatype, xflags, m, n, k, lda, ldb, ldc);
        if (NULL != fn) config.xgemm = fn;
      }
    }
    /* BLAS fallback pointers */
    config.dgemm_blas = dgemm_blas;
    config.sgemm_blas = sgemm_blas;
    if (NULL != reg) {
      result = (libxs_gemm_config_t*)
        libxs_registry_set(reg, &shape, sizeof(shape),
          &config, sizeof(config), libxs_registry_lock(reg));
    }
    if (NULL == result) {
      static LIBXS_TLS libxs_gemm_config_t fallback;
      fallback = config;
      result = &fallback;
    }
  }
#if defined(LIBXS_GEMM_PRINT)
  internal_libxs_gemm_print(stderr, result, reg);
#endif
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
  int lda, int ldb, int ldc,
  double alpha, double beta);
LIBXS_API_INTERN void internal_libxs_gemm_blas(
  const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c,
  int lda, int ldb, int ldc,
  double alpha, double beta)
{
  const int m = config->shape.m, n = config->shape.n, k = config->shape.k;
  if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
    const libxs_gemm_dblas_t fn = (NULL != config->dgemm_blas
      ? config->dgemm_blas : internal_libxs_dgemm_default);
    fn(&config->shape.transa, &config->shape.transb, &m, &n, &k,
      &alpha, (const double*)a, &lda,
      (const double*)b, &ldb, &beta, (double*)c, &ldc);
  }
  else if (LIBXS_DATATYPE_F32 == config->shape.datatype) {
    const float falpha = (float)alpha, fbeta = (float)beta;
    const libxs_gemm_sblas_t fn = (NULL != config->sgemm_blas)
      ? config->sgemm_blas : internal_libxs_sgemm_default;
    fn(&config->shape.transa, &config->shape.transb, &m, &n, &k,
      &falpha, (const float*)a, &lda,
      (const float*)b, &ldb, &fbeta, (float*)c, &ldc);
  }
}


LIBXS_API libxs_gemm_config_t* libxs_syr2k_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  void* registry)
{
  const double one = 1.0, zero = 0.0;
  libxs_gemm_config_t* result;
  if (n <= LIBXS_GEMM_BLOCK_M && n <= LIBXS_GEMM_BLOCK_N
    && k <= LIBXS_GEMM_BLOCK_K)
  {
    result = libxs_gemm_dispatch_rt(
      datatype, 'N', 'T', n, n, k, lda, ldb, ldc, &one, &zero,
      NULL, NULL, NULL, NULL, NULL, NULL, NULL, registry);
  }
  else {
    libxs_gemm_config_t config;
    libxs_gemm_shape_t shape;
    libxs_registry_t* reg;
    result = NULL;
    LIBXS_MEMZERO(&shape);
    shape.datatype = datatype;
    shape.transa = 'N'; shape.transb = 'T';
    shape.m = n; shape.n = n; shape.k = k;
    shape.lda = lda; shape.ldb = ldb; shape.ldc = ldc;
    shape.alpha = 1.0; shape.beta = 0.0;
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
        &shape, sizeof(shape), libxs_registry_lock(reg));
    }
    if (NULL == result && NULL != reg) {
      libxs_gemm_config_t* kernel = libxs_gemm_dispatch_rt(
        datatype, 'N', 'T', LIBXS_GEMM_BLOCK_M, LIBXS_GEMM_BLOCK_N, LIBXS_GEMM_BLOCK_K,
        LIBXS_GEMM_BLOCK_M, LIBXS_GEMM_BLOCK_N, LIBXS_GEMM_BLOCK_M, &one, &zero,
        NULL, NULL, NULL, NULL, NULL, NULL, NULL, registry);
      LIBXS_MEMZERO(&config);
      config.shape = shape;
      if (NULL != kernel) {
        config.dgemm_jit = kernel->dgemm_jit;
        config.sgemm_jit = kernel->sgemm_jit;
        config.xgemm = kernel->xgemm;
        config.jitter = kernel->jitter;
        config.dgemm_blas = kernel->dgemm_blas;
        config.sgemm_blas = kernel->sgemm_blas;
      }
      result = (libxs_gemm_config_t*)libxs_registry_set(
        reg, &shape, sizeof(shape),
        &config, sizeof(config), libxs_registry_lock(reg));
    }
  }
  return result;
}


LIBXS_API libxs_gemm_config_t* libxs_syrk_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  void* registry)
{
  return libxs_syr2k_dispatch(datatype, n, k, lda, lda, ldc, registry);
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


LIBXS_API int libxs_syr2k(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c)
{
  int result = EXIT_FAILURE;
  if (NULL != config && NULL != a && NULL != b && NULL != c
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
      const size_t need = (size_t)ldc * (size_t)n * elemsize;
      void* scratch = internal_libxs_syrk_scratch(need);
      if (NULL != scratch) {
        if (EXIT_SUCCESS != libxs_gemm_call(config, a, b, scratch)) {
          internal_libxs_gemm_blas(config, a, b, scratch,
            lda, ldb, ldc, 1.0, 0.0);
        }
        if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
          double* cc = (double*)c;
          const double* t = (const double*)scratch;
          int i, j;
          if (upper) {
            for (j = 0; j < n; ++j) {
              for (i = 0; i <= j; ++i) {
                cc[i + (size_t)j * ldc] = beta * cc[i + (size_t)j * ldc]
                  + alpha * (t[i + (size_t)j * ldc] + t[j + (size_t)i * ldc]);
              }
            }
          }
          else {
            for (j = 0; j < n; ++j) {
              for (i = j; i < n; ++i) {
                cc[i + (size_t)j * ldc] = beta * cc[i + (size_t)j * ldc]
                  + alpha * (t[i + (size_t)j * ldc] + t[j + (size_t)i * ldc]);
              }
            }
          }
        }
        else {
          float* cc = (float*)c;
          const float* t = (const float*)scratch;
          const float fa = (float)alpha, fb = (float)beta;
          int i, j;
          if (upper) {
            for (j = 0; j < n; ++j) {
              for (i = 0; i <= j; ++i) {
                cc[i + (size_t)j * ldc] = fb * cc[i + (size_t)j * ldc]
                  + fa * (t[i + (size_t)j * ldc] + t[j + (size_t)i * ldc]);
              }
            }
          }
          else {
            for (j = 0; j < n; ++j) {
              for (i = j; i < n; ++i) {
                cc[i + (size_t)j * ldc] = fb * cc[i + (size_t)j * ldc]
                  + fa * (t[i + (size_t)j * ldc] + t[j + (size_t)i * ldc]);
              }
            }
          }
        }
        result = EXIT_SUCCESS;
      }
    }
    else {
      const int bm = LIBXS_GEMM_BLOCK_M;
      const int bn = LIBXS_GEMM_BLOCK_N;
      const int bk = LIBXS_GEMM_BLOCK_K;
      const size_t need = (size_t)bm * bn * 2 * elemsize;
      void* scratch = internal_libxs_syrk_scratch(need);
      if (NULL != scratch) {
        void* scratch2 = (char*)scratch + (size_t)bm * bn * elemsize;
        int jb, ib, kb;
        for (jb = 0; jb < n; jb += bn) {
          const int cn = LIBXS_MIN(bn, n - jb);
          for (ib = 0; ib < n; ib += bm) {
            const int cm = LIBXS_MIN(bm, n - ib);
            int skip;
            if (upper) {
              skip = (ib > jb + cn - 1);
            }
            else {
              skip = (ib + cm - 1 < jb);
            }
            if (0 == skip) {
              memset(scratch, 0, need);
              for (kb = 0; kb < k; kb += bk) {
                const int ck = LIBXS_MIN(bk, k - kb);
                if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
                  const double* ai = (const double*)a + ib + (size_t)kb * lda;
                  const double* bj = (const double*)b + jb + (size_t)kb * ldb;
                  const double* bi = (const double*)b + ib + (size_t)kb * ldb;
                  const double* aj = (const double*)a + jb + (size_t)kb * lda;
                  double* t1 = (double*)scratch;
                  double* t2 = (double*)scratch2;
                  int ii, jj, pp;
                  for (jj = 0; jj < cn; ++jj) {
                    for (ii = 0; ii < cm; ++ii) {
                      double s1 = 0.0, s2 = 0.0;
                      for (pp = 0; pp < ck; ++pp) {
                        s1 += ai[ii + (size_t)pp * lda]
                            * bj[jj + (size_t)pp * ldb];
                        s2 += bi[ii + (size_t)pp * ldb]
                            * aj[jj + (size_t)pp * lda];
                      }
                      t1[ii + (size_t)jj * bm] += s1;
                      t2[ii + (size_t)jj * bm] += s2;
                    }
                  }
                }
                else {
                  const float* ai = (const float*)a + ib + (size_t)kb * lda;
                  const float* bj = (const float*)b + jb + (size_t)kb * ldb;
                  const float* bi = (const float*)b + ib + (size_t)kb * ldb;
                  const float* aj = (const float*)a + jb + (size_t)kb * lda;
                  float* t1 = (float*)scratch;
                  float* t2 = (float*)scratch2;
                  int ii, jj, pp;
                  for (jj = 0; jj < cn; ++jj) {
                    for (ii = 0; ii < cm; ++ii) {
                      float s1 = 0.f, s2 = 0.f;
                      for (pp = 0; pp < ck; ++pp) {
                        s1 += ai[ii + (size_t)pp * lda]
                            * bj[jj + (size_t)pp * ldb];
                        s2 += bi[ii + (size_t)pp * ldb]
                            * aj[jj + (size_t)pp * lda];
                      }
                      t1[ii + (size_t)jj * bm] += s1;
                      t2[ii + (size_t)jj * bm] += s2;
                    }
                  }
                }
              }
              if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
                double* cc = (double*)c;
                const double* t1 = (const double*)scratch;
                const double* t2 = (const double*)scratch2;
                int ii, jj;
                for (jj = 0; jj < cn; ++jj) {
                  const int istart = (upper && ib == jb) ? 0
                    : (upper ? 0 : (ib == jb ? jj : 0));
                  const int iend = (upper && ib == jb) ? jj + 1
                    : (upper ? cm : (ib == jb ? cm : cm));
                  for (ii = istart; ii < iend; ++ii) {
                    cc[(ib + ii) + (size_t)(jb + jj) * ldc] =
                      beta * cc[(ib + ii) + (size_t)(jb + jj) * ldc]
                      + alpha * (t1[ii + (size_t)jj * bm]
                               + t2[ii + (size_t)jj * bm]);
                  }
                }
              }
              else {
                float* cc = (float*)c;
                const float* t1 = (const float*)scratch;
                const float* t2 = (const float*)scratch2;
                const float fa = (float)alpha, fb = (float)beta;
                int ii, jj;
                for (jj = 0; jj < cn; ++jj) {
                  const int istart = (upper && ib == jb) ? 0
                    : (upper ? 0 : (ib == jb ? jj : 0));
                  const int iend = (upper && ib == jb) ? jj + 1
                    : (upper ? cm : (ib == jb ? cm : cm));
                  for (ii = istart; ii < iend; ++ii) {
                    cc[(ib + ii) + (size_t)(jb + jj) * ldc] =
                      fb * cc[(ib + ii) + (size_t)(jb + jj) * ldc]
                      + fa * (t1[ii + (size_t)jj * bm]
                             + t2[ii + (size_t)jj * bm]);
                  }
                }
              }
            }
          }
        }
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


LIBXS_API int libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c)
{
  int result = EXIT_FAILURE;
  if (NULL != config && NULL != a && NULL != c
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
      const size_t need = (size_t)ldc * (size_t)n * elemsize;
      void* scratch = internal_libxs_syrk_scratch(need);
      if (NULL != scratch) {
        if (EXIT_SUCCESS != libxs_gemm_call(config, a, a, scratch)) {
          internal_libxs_gemm_blas(config, a, a, scratch,
            lda, lda, ldc, 1.0, 0.0);
        }
        if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
          double* cc = (double*)c;
          const double* t = (const double*)scratch;
          int i, j;
          if (upper) {
            for (j = 0; j < n; ++j) {
              for (i = 0; i <= j; ++i) {
                cc[i + (size_t)j * ldc] = beta * cc[i + (size_t)j * ldc]
                  + alpha * t[i + (size_t)j * ldc];
              }
            }
          }
          else {
            for (j = 0; j < n; ++j) {
              for (i = j; i < n; ++i) {
                cc[i + (size_t)j * ldc] = beta * cc[i + (size_t)j * ldc]
                  + alpha * t[i + (size_t)j * ldc];
              }
            }
          }
        }
        else {
          float* cc = (float*)c;
          const float* t = (const float*)scratch;
          const float fa = (float)alpha, fb = (float)beta;
          int i, j;
          if (upper) {
            for (j = 0; j < n; ++j) {
              for (i = 0; i <= j; ++i) {
                cc[i + (size_t)j * ldc] = fb * cc[i + (size_t)j * ldc]
                  + fa * t[i + (size_t)j * ldc];
              }
            }
          }
          else {
            for (j = 0; j < n; ++j) {
              for (i = j; i < n; ++i) {
                cc[i + (size_t)j * ldc] = fb * cc[i + (size_t)j * ldc]
                  + fa * t[i + (size_t)j * ldc];
              }
            }
          }
        }
        result = EXIT_SUCCESS;
      }
    }
    else {
      const int bm = LIBXS_GEMM_BLOCK_M;
      const int bn = LIBXS_GEMM_BLOCK_N;
      const int bk = LIBXS_GEMM_BLOCK_K;
      const size_t need = (size_t)bm * bn * elemsize;
      void* scratch = internal_libxs_syrk_scratch(need);
      if (NULL != scratch) {
        int jb, ib, kb;
        for (jb = 0; jb < n; jb += bn) {
          const int cn = LIBXS_MIN(bn, n - jb);
          for (ib = 0; ib < n; ib += bm) {
            const int cm = LIBXS_MIN(bm, n - ib);
            int skip;
            if (upper) {
              skip = (ib > jb + cn - 1);
            }
            else {
              skip = (ib + cm - 1 < jb);
            }
            if (0 == skip) {
              memset(scratch, 0, need);
              for (kb = 0; kb < k; kb += bk) {
                const int ck = LIBXS_MIN(bk, k - kb);
                if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
                  const double* ai = (const double*)a + ib + (size_t)kb * lda;
                  const double* aj = (const double*)a + jb + (size_t)kb * lda;
                  double* t = (double*)scratch;
                  int ii, jj, pp;
                  for (jj = 0; jj < cn; ++jj) {
                    for (ii = 0; ii < cm; ++ii) {
                      double sum = 0.0;
                      for (pp = 0; pp < ck; ++pp) {
                        sum += ai[ii + (size_t)pp * lda]
                             * aj[jj + (size_t)pp * lda];
                      }
                      t[ii + (size_t)jj * bm] += sum;
                    }
                  }
                }
                else {
                  const float* ai = (const float*)a + ib + (size_t)kb * lda;
                  const float* aj = (const float*)a + jb + (size_t)kb * lda;
                  float* t = (float*)scratch;
                  int ii, jj, pp;
                  for (jj = 0; jj < cn; ++jj) {
                    for (ii = 0; ii < cm; ++ii) {
                      float sum = 0.f;
                      for (pp = 0; pp < ck; ++pp) {
                        sum += ai[ii + (size_t)pp * lda]
                             * aj[jj + (size_t)pp * lda];
                      }
                      t[ii + (size_t)jj * bm] += sum;
                    }
                  }
                }
              }
              if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
                double* cc = (double*)c;
                const double* t = (const double*)scratch;
                int ii, jj;
                for (jj = 0; jj < cn; ++jj) {
                  const int istart = (upper && ib == jb) ? 0
                    : (upper ? 0 : (ib == jb ? jj : 0));
                  const int iend = (upper && ib == jb) ? jj + 1
                    : (upper ? cm : (ib == jb ? cm : cm));
                  for (ii = istart; ii < iend; ++ii) {
                    cc[(ib + ii) + (size_t)(jb + jj) * ldc] =
                      beta * cc[(ib + ii) + (size_t)(jb + jj) * ldc]
                      + alpha * t[ii + (size_t)jj * bm];
                  }
                }
              }
              else {
                float* cc = (float*)c;
                const float* t = (const float*)scratch;
                const float fa = (float)alpha, fb = (float)beta;
                int ii, jj;
                for (jj = 0; jj < cn; ++jj) {
                  const int istart = (upper && ib == jb) ? 0
                    : (upper ? 0 : (ib == jb ? jj : 0));
                  const int iend = (upper && ib == jb) ? jj + 1
                    : (upper ? cm : (ib == jb ? cm : cm));
                  for (ii = istart; ii < iend; ++ii) {
                    cc[(ib + ii) + (size_t)(jb + jj) * ldc] =
                      fb * cc[(ib + ii) + (size_t)(jb + jj) * ldc]
                      + fa * t[ii + (size_t)jj * bm];
                  }
                }
              }
            }
          }
        }
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
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


LIBXS_API libxs_gemm_config_t* libxs_gemm_dispatch_f(
  int datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta, void* registry);
LIBXS_API libxs_gemm_config_t* libxs_gemm_dispatch_f(
  int datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta, void* registry)
{
  return libxs_gemm_dispatch((libxs_data_t)datatype, transa, transb,
    m, n, k, lda, ldb, ldc, alpha, beta, registry);
}

#endif /*defined(LIBXS_BUILD) && !defined(LIBXS_NOFORTRAN)*/
