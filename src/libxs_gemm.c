/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_gemm.h>
#include "libxs_main.h"
#include "libxs_crc32.h"

#if !defined(LIBXS_SYRK_TRACE) && 1
# define LIBXS_SYRK_TRACE
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
  int j;
  LIBXS_ASSERT(NULL != transa && NULL != transb);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  for (j = 0; j < nn; ++j) {
    int i;
    for (i = 0; i < mm; ++i) {
      double sum = 0.0;
      int p;
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
  int j;
  LIBXS_ASSERT(NULL != transa && NULL != transb);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  for (j = 0; j < nn; ++j) {
    int i;
    for (i = 0; i < mm; ++i) {
      float sum = 0.f;
      int p;
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
        const char transa = (char)config->shape.transa;
        const char transb = (char)config->shape.transb;
        int m = config->shape.m, n = config->shape.n, k = config->shape.k;
        int lda = config->shape.lda, ldb = config->shape.ldb, ldc = config->shape.ldc;
        const double dalpha = config->shape.alpha, dbeta = config->shape.beta;
        const libxs_gemm_dblas_t dgemm_blas = NULL != config->dgemm_blas
          ? config->dgemm_blas : internal_libxs_dgemm_default;
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          dgemm_blas(&transa, &transb, &m, &n, &k,
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
        const char transa = (char)config->shape.transa;
        const char transb = (char)config->shape.transb;
        int m = config->shape.m, n = config->shape.n, k = config->shape.k;
        int lda = config->shape.lda, ldb = config->shape.ldb, ldc = config->shape.ldc;
        const float falpha = (float)config->shape.alpha, fbeta = (float)config->shape.beta;
        const libxs_gemm_sblas_t sgemm_blas = NULL != config->sgemm_blas
          ? config->sgemm_blas : internal_libxs_sgemm_default;
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          sgemm_blas(&transa, &transb, &m, &n, &k,
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
    const int need_lock = (1 < ntasks
      && 0 == (config->flags & LIBXS_GEMM_FLAG_NOLOCK));
    const int tasksize = (size + nsplit - 1) / nsplit;
    const int begin = tid * tasksize;
    int end = begin + tasksize;
    const size_t elemsize = LIBXS_TYPESIZE(config->shape.datatype);
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
          char* ci = (char*)c + (size_t)ci_idx * elemsize;
          if (need_lock) INTERNAL_GEMM_LOCKFWD_IDX(ci_idx, lockidx);
          xparam.a[0] = (const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize;
          xparam.b[0] = (const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize;
          xparam.c[0] = ci;
          config->xgemm(&xparam);
        }
      }
      else {
        const char transa = (char)config->shape.transa;
        const char transb = (char)config->shape.transb;
        int m = config->shape.m, n = config->shape.n, k = config->shape.k;
        int lda = config->shape.lda, ldb = config->shape.ldb, ldc = config->shape.ldc;
        const double dalpha = config->shape.alpha, dbeta = config->shape.beta;
        const libxs_gemm_dblas_t dgemm_blas = NULL != config->dgemm_blas
          ? config->dgemm_blas : internal_libxs_dgemm_default;
        for (i = begin; i < end; ++i) {
          const int ci_idx = INTERNAL_GEMM_INDEX(i, stride_c);
          double* ci = (double*)((char*)c + (size_t)ci_idx * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD_IDX(ci_idx, lockidx);
          dgemm_blas(&transa, &transb, &m, &n, &k,
            &dalpha,
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
        const char transa = (char)config->shape.transa;
        const char transb = (char)config->shape.transb;
        int m = config->shape.m, n = config->shape.n, k = config->shape.k;
        int lda = config->shape.lda, ldb = config->shape.ldb, ldc = config->shape.ldc;
        const float falpha = (float)config->shape.alpha, fbeta = (float)config->shape.beta;
        const libxs_gemm_sblas_t sgemm_blas = NULL != config->sgemm_blas
          ? config->sgemm_blas : internal_libxs_sgemm_default;
        for (i = begin; i < end; ++i) {
          const int ci_idx = INTERNAL_GEMM_INDEX(i, stride_c);
          float* ci = (float*)((char*)c + (size_t)ci_idx * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD_IDX(ci_idx, lockidx);
          sgemm_blas(&transa, &transb, &m, &n, &k,
            &falpha,
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
  const char ta = (char)config->shape.transa;
  const char tb = (char)config->shape.transb;
  int m = config->shape.m, n = config->shape.n, k = config->shape.k;
  if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
    const libxs_gemm_dblas_t fn = (NULL != config->dgemm_blas)
      ? config->dgemm_blas : internal_libxs_dgemm_default;
    fn(&ta, &tb, &m, &n, &k,
      &alpha, (const double*)a, &lda,
      (const double*)b, &ldb, &beta, (double*)c, &ldc);
  }
  else if (LIBXS_DATATYPE_F32 == config->shape.datatype) {
    const float falpha = (float)alpha, fbeta = (float)beta;
    const libxs_gemm_sblas_t fn = (NULL != config->sgemm_blas)
      ? config->sgemm_blas : internal_libxs_sgemm_default;
    fn(&ta, &tb, &m, &n, &k,
      &falpha, (const float*)a, &lda,
      (const float*)b, &ldb, &fbeta, (float*)c, &ldc);
  }
}


#if defined(LIBXS_SYRK_TRACE)
LIBXS_API_INTERN void internal_libxs_syrk_trace(
  const char* name, void* registry);
LIBXS_API_INTERN void internal_libxs_syrk_trace(
  const char* name, void* registry)
{
  static int interval = -1;
  static int counter = 0;
  if (-1 == interval) {
    const char* env = getenv("LIBXS_SYRK");
    interval = (NULL != env) ? atoi(env) : 0;
  }
  if (0 < interval && NULL != registry) {
    if (0 == (++counter % interval)) {
      libxs_registry_info_t info;
      if (0 == libxs_registry_info((libxs_registry_t*)registry, &info)) {
        fprintf(stderr, "LIBXS %s [%d]: registry size=%lu capacity=%lu"
          " nbytes=%lu\n", name, counter,
          (unsigned long)info.size, (unsigned long)info.capacity,
          (unsigned long)info.nbytes);
      }
    }
  }
}
#endif


LIBXS_API int libxs_syr2k_dispatch(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  size_t* scratch_size, void* registry)
{
  const double one = 1.0, zero = 0.0;
  int result;
  if (NULL != scratch_size) {
    *scratch_size = (size_t)n * (size_t)n * (size_t)LIBXS_TYPESIZE(datatype);
  }
  result = libxs_gemm_dispatch(config, datatype, 'N', 'T',
    n, n, k, lda, ldb, n, &one, &zero, registry);
  if (NULL != config) {
    config->shape.ldc = ldc;
  }
#if defined(LIBXS_SYRK_TRACE)
  internal_libxs_syrk_trace("syr2k", registry);
#endif
  return result;
}


LIBXS_API int libxs_syrk_dispatch(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  size_t* scratch_size, void* registry)
{
  return libxs_syr2k_dispatch(config, datatype,
    n, k, lda, lda, ldc, scratch_size, registry);
}


LIBXS_API int libxs_syr2k(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c,
  void* scratch)
{
  int n, ldc, upper, j;
  if (NULL == config || NULL == a || NULL == b || NULL == c) {
    return EXIT_FAILURE;
  }
  n = config->shape.m;
  ldc = config->shape.ldc;
  upper = ('U' == uplo || 'u' == uplo);
  if (LIBXS_DATATYPE_F64 == config->shape.datatype
    || LIBXS_DATATYPE_F32 == config->shape.datatype)
  {
    const size_t elemsize = (size_t)LIBXS_TYPESIZE(config->shape.datatype);
    if (NULL != scratch) {
      /* Scratch path: T = A*B^T via JIT (alpha=1,beta=0 baked in),
       * then C := beta*C + alpha*(T + T^T). */
      if (EXIT_SUCCESS != libxs_gemm_call(config, a, b, scratch)) {
        internal_libxs_gemm_blas(config, a, b, scratch,
          config->shape.lda, config->shape.ldb, n, 1.0, 0.0);
      }
      if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
        double* cc = (double*)c;
        const double* t = (const double*)scratch;
        int i;
        if (upper) {
          for (j = 0; j < n; ++j)
            for (i = 0; i <= j; ++i)
              cc[i + (size_t)j * ldc] = beta * cc[i + (size_t)j * ldc]
                + alpha * (t[i + (size_t)j * n] + t[j + (size_t)i * n]);
        }
        else {
          for (j = 0; j < n; ++j)
            for (i = j; i < n; ++i)
              cc[i + (size_t)j * ldc] = beta * cc[i + (size_t)j * ldc]
                + alpha * (t[i + (size_t)j * n] + t[j + (size_t)i * n]);
        }
      }
      else {
        float* cc = (float*)c;
        const float* t = (const float*)scratch;
        const float fa = (float)alpha, fb = (float)beta;
        int i;
        if (upper) {
          for (j = 0; j < n; ++j)
            for (i = 0; i <= j; ++i)
              cc[i + (size_t)j * ldc] = fb * cc[i + (size_t)j * ldc]
                + fa * (t[i + (size_t)j * n] + t[j + (size_t)i * n]);
        }
        else {
          for (j = 0; j < n; ++j)
            for (i = j; i < n; ++i)
              cc[i + (size_t)j * ldc] = fb * cc[i + (size_t)j * ldc]
                + fa * (t[i + (size_t)j * n] + t[j + (size_t)i * n]);
        }
      }
    }
    else {
      /* No-scratch path: two BLAS calls directly into C. */
      internal_libxs_gemm_blas(config, a, b, c,
        config->shape.lda, config->shape.ldb, ldc, alpha, beta);
      internal_libxs_gemm_blas(config, b, a, c,
        config->shape.ldb, config->shape.lda, ldc, alpha, 1.0);
      if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
        double* cc = (double*)c;
        int i;
        if (upper) {
          for (j = 0; j < n; ++j)
            for (i = j + 1; i < n; ++i)
              cc[i + (size_t)j * ldc] = cc[j + (size_t)i * ldc];
        }
        else {
          for (j = 0; j < n; ++j)
            for (i = 0; i < j; ++i)
              cc[i + (size_t)j * ldc] = cc[j + (size_t)i * ldc];
        }
      }
      else {
        float* cc = (float*)c;
        int i;
        if (upper) {
          for (j = 0; j < n; ++j)
            for (i = j + 1; i < n; ++i)
              cc[i + (size_t)j * ldc] = cc[j + (size_t)i * ldc];
        }
        else {
          for (j = 0; j < n; ++j)
            for (i = 0; i < j; ++i)
              cc[i + (size_t)j * ldc] = cc[j + (size_t)i * ldc];
        }
      }
    }
    LIBXS_UNUSED(elemsize);
  }
  else {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


LIBXS_API int libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c,
  void* scratch)
{
  int n, ldc, upper, j;
  if (NULL == config || NULL == a || NULL == c) {
    return EXIT_FAILURE;
  }
  n = config->shape.m;
  ldc = config->shape.ldc;
  upper = ('U' == uplo || 'u' == uplo);
  if (LIBXS_DATATYPE_F64 == config->shape.datatype
    || LIBXS_DATATYPE_F32 == config->shape.datatype)
  {
    if (NULL != scratch) {
      if (EXIT_SUCCESS != libxs_gemm_call(config, a, a, scratch)) {
        internal_libxs_gemm_blas(config, a, a, scratch,
          config->shape.lda, config->shape.ldb, n, 1.0, 0.0);
      }
      if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
        double* cc = (double*)c;
        const double* t = (const double*)scratch;
        int i;
        if (upper) {
          for (j = 0; j < n; ++j)
            for (i = 0; i <= j; ++i)
              cc[i + (size_t)j * ldc] = beta * cc[i + (size_t)j * ldc]
                + alpha * t[i + (size_t)j * n];
        }
        else {
          for (j = 0; j < n; ++j)
            for (i = j; i < n; ++i)
              cc[i + (size_t)j * ldc] = beta * cc[i + (size_t)j * ldc]
                + alpha * t[i + (size_t)j * n];
        }
      }
      else {
        float* cc = (float*)c;
        const float* t = (const float*)scratch;
        const float fa = (float)alpha, fb = (float)beta;
        int i;
        if (upper) {
          for (j = 0; j < n; ++j)
            for (i = 0; i <= j; ++i)
              cc[i + (size_t)j * ldc] = fb * cc[i + (size_t)j * ldc]
                + fa * t[i + (size_t)j * n];
        }
        else {
          for (j = 0; j < n; ++j)
            for (i = j; i < n; ++i)
              cc[i + (size_t)j * ldc] = fb * cc[i + (size_t)j * ldc]
                + fa * t[i + (size_t)j * n];
        }
      }
    }
    else {
      /* A*A^T is symmetric -- single BLAS call + mirror */
      internal_libxs_gemm_blas(config, a, a, c,
        config->shape.lda, config->shape.lda, ldc, alpha, beta);
      if (LIBXS_DATATYPE_F64 == config->shape.datatype) {
        double* cc = (double*)c;
        int i;
        if (upper) {
          for (j = 0; j < n; ++j)
            for (i = j + 1; i < n; ++i)
              cc[i + (size_t)j * ldc] = cc[j + (size_t)i * ldc];
        }
        else {
          for (j = 0; j < n; ++j)
            for (i = 0; i < j; ++i)
              cc[i + (size_t)j * ldc] = cc[j + (size_t)i * ldc];
        }
      }
      else {
        float* cc = (float*)c;
        int i;
        if (upper) {
          for (j = 0; j < n; ++j)
            for (i = j + 1; i < n; ++i)
              cc[i + (size_t)j * ldc] = cc[j + (size_t)i * ldc];
        }
        else {
          for (j = 0; j < n; ++j)
            for (i = 0; i < j; ++i)
              cc[i + (size_t)j * ldc] = cc[j + (size_t)i * ldc];
        }
      }
    }
  }
  else {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
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
