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
#include "libxs_hash.h"

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


LIBXS_API void libxs_gemm_strided_task(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, int stride_a,
                     const void* b, int ldb, int stride_b,
  const void* beta,        void* c, int ldc, int stride_c,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks)
{
  const int size = LIBXS_ABS(batchsize);
  const int nsplit = LIBXS_MIN(size, ntasks);
  if (0 < nsplit && 0 <= tid && tid < nsplit) {
    const int need_lock = (1 < ntasks
      && (NULL == config || 0 == (config->flags & LIBXS_GEMM_FLAG_NOLOCK)));
    const int tasksize = (size + nsplit - 1) / nsplit;
    const int begin = tid * tasksize;
    int end = begin + tasksize;
    const size_t da = stride_a * LIBXS_TYPESIZE(datatype);
    const size_t db = stride_b * LIBXS_TYPESIZE(datatype);
    const size_t dc = stride_c * LIBXS_TYPESIZE(datatype);
    int lockidx = -1, i;
    if (end > size) end = size;
    if (LIBXS_DATATYPE_F64 == datatype) {
      if (NULL != config && NULL != config->dgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          double* ci = (double*)((char*)c + i * dc);
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          config->dgemm_jit(config->jitter,
            (const double*)((const char*)a + i * da),
            (const double*)((const char*)b + i * db), ci);
        }
      }
      else if (NULL != config && NULL != config->xgemm) {
        libxs_gemm_param_t xparam;
        memset(&xparam, 0, sizeof(xparam));
        for (i = begin; i < end; ++i) {
          char* ci = (char*)c + i * dc;
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          xparam.a[0] = (const void*)((uintptr_t)a + i * da);
          xparam.b[0] = (const void*)((uintptr_t)b + i * db);
          xparam.c[0] = ci;
          config->xgemm(&xparam);
        }
      }
      else {
        const libxs_gemm_dblas_t dgemm_blas = (NULL != config && NULL != config->dgemm_blas)
          ? config->dgemm_blas : internal_libxs_dgemm_default;
        for (i = begin; i < end; ++i) {
          double* ci = (double*)((char*)c + i * dc);
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          dgemm_blas(transa, transb, &m, &n, &k,
            (const double*)alpha, (const double*)((const char*)a + i * da), &lda,
            (const double*)((const char*)b + i * db), &ldb,
            (const double*)beta, ci, &ldc);
        }
      }
    }
    else if (LIBXS_DATATYPE_F32 == datatype) {
      if (NULL != config && NULL != config->sgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          float* ci = (float*)((char*)c + i * dc);
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          config->sgemm_jit(config->jitter,
            (const float*)((const char*)a + i * da),
            (const float*)((const char*)b + i * db), ci);
        }
      }
      else if (NULL != config && NULL != config->xgemm) {
        libxs_gemm_param_t xparam;
        memset(&xparam, 0, sizeof(xparam));
        for (i = begin; i < end; ++i) {
          char* ci = (char*)c + i * dc;
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          xparam.a[0] = (const void*)((uintptr_t)a + i * da);
          xparam.b[0] = (const void*)((uintptr_t)b + i * db);
          xparam.c[0] = ci;
          config->xgemm(&xparam);
        }
      }
      else {
        const libxs_gemm_sblas_t sgemm_blas = (NULL != config && NULL != config->sgemm_blas)
          ? config->sgemm_blas : internal_libxs_sgemm_default;
        for (i = begin; i < end; ++i) {
          float* ci = (float*)((char*)c + i * dc);
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          sgemm_blas(transa, transb, &m, &n, &k,
            (const float*)alpha, (const float*)((const char*)a + i * da), &lda,
            (const float*)((const char*)b + i * db), &ldb,
            (const float*)beta, ci, &ldc);
        }
      }
    }
    INTERNAL_GEMM_UNLOCK(lockidx);
  }
}


LIBXS_API void libxs_gemm_strided(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, int stride_a,
                     const void* b, int ldb, int stride_b,
  const void* beta,        void* c, int ldc, int stride_c,
  int batchsize, const libxs_gemm_config_t* config)
{
  libxs_gemm_strided_task(datatype, transa, transb, m, n, k,
    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
    batchsize, config, 0, 1);
}


LIBXS_API void libxs_gemm_batch_task(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a_array[], int lda,
                     const void* b_array[], int ldb,
  const void* beta,        void* c_array[], int ldc,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks)
{
  const int size = LIBXS_ABS(batchsize);
  const int nsplit = LIBXS_MIN(size, ntasks);
  if (0 < nsplit && 0 <= tid && tid < nsplit) {
    const int need_lock = (1 < ntasks
      && (NULL == config || 0 == (config->flags & LIBXS_GEMM_FLAG_NOLOCK)));
    const int tasksize = (size + nsplit - 1) / nsplit;
    const int begin = tid * tasksize;
    int end = begin + tasksize;
    int lockidx = -1, i;
    if (end > size) end = size;
    if (LIBXS_DATATYPE_F64 == datatype) {
      if (NULL != config && NULL != config->dgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          config->dgemm_jit(config->jitter,
            (const double*)a_array[i],
            (const double*)b_array[i],
            (double*)c_array[i]);
        }
      }
      else if (NULL != config && NULL != config->xgemm) {
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
        const libxs_gemm_dblas_t dgemm_blas = (NULL != config && NULL != config->dgemm_blas)
          ? config->dgemm_blas : internal_libxs_dgemm_default;
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          dgemm_blas(transa, transb, &m, &n, &k,
            (const double*)alpha, (const double*)a_array[i], &lda,
            (const double*)b_array[i], &ldb,
            (const double*)beta, (double*)c_array[i], &ldc);
        }
      }
    }
    else if (LIBXS_DATATYPE_F32 == datatype) {
      if (NULL != config && NULL != config->sgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          config->sgemm_jit(config->jitter,
            (const float*)a_array[i],
            (const float*)b_array[i],
            (float*)c_array[i]);
        }
      }
      else if (NULL != config && NULL != config->xgemm) {
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
        const libxs_gemm_sblas_t sgemm_blas = (NULL != config && NULL != config->sgemm_blas)
          ? config->sgemm_blas : internal_libxs_sgemm_default;
        for (i = begin; i < end; ++i) {
          if (need_lock) INTERNAL_GEMM_LOCKFWD(c_array[i], lockidx);
          sgemm_blas(transa, transb, &m, &n, &k,
            (const float*)alpha, (const float*)a_array[i], &lda,
            (const float*)b_array[i], &ldb,
            (const float*)beta, (float*)c_array[i], &ldc);
        }
      }
    }
    INTERNAL_GEMM_UNLOCK(lockidx);
  }
}


LIBXS_API void libxs_gemm_batch(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a_array[], int lda,
                     const void* b_array[], int ldb,
  const void* beta,        void* c_array[], int ldc,
  int batchsize, const libxs_gemm_config_t* config)
{
  libxs_gemm_batch_task(datatype, transa, transb, m, n, k,
    alpha, a_array, lda, b_array, ldb, beta, c_array, ldc,
    batchsize, config, 0, 1);
}


LIBXS_API void libxs_gemm_index_task(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, const int stride_a[],
                     const void* b, int ldb, const int stride_b[],
  const void* beta,        void* c, int ldc, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks)
{
  const int size = LIBXS_ABS(batchsize);
  const int nsplit = LIBXS_MIN(size, ntasks);
  if (0 < nsplit && 0 <= tid && tid < nsplit
    && NULL != stride_a && NULL != stride_b && NULL != stride_c
    && 0 < index_stride)
  {
    const int need_lock = (1 < ntasks
      && (NULL == config || 0 == (config->flags & LIBXS_GEMM_FLAG_NOLOCK)));
    const int tasksize = (size + nsplit - 1) / nsplit;
    const int begin = tid * tasksize;
    int end = begin + tasksize;
    const size_t elemsize = LIBXS_TYPESIZE(datatype);
    int lockidx = -1, i;
    if (end > size) end = size;
#define INTERNAL_GEMM_INDEX(I, STRIDE) \
    (*(const int*)((const char*)(STRIDE) + (size_t)(I) * index_stride) \
      - index_base)
    if (LIBXS_DATATYPE_F64 == datatype) {
      if (NULL != config && NULL != config->dgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          double* ci = (double*)((char*)c + (size_t)INTERNAL_GEMM_INDEX(i, stride_c) * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          config->dgemm_jit(config->jitter,
            (const double*)((const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize),
            (const double*)((const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize), ci);
        }
      }
      else if (NULL != config && NULL != config->xgemm) {
        libxs_gemm_param_t xparam;
        memset(&xparam, 0, sizeof(xparam));
        for (i = begin; i < end; ++i) {
          char* ci = (char*)c + (size_t)INTERNAL_GEMM_INDEX(i, stride_c) * elemsize;
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          xparam.a[0] = (const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize;
          xparam.b[0] = (const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize;
          xparam.c[0] = ci;
          config->xgemm(&xparam);
        }
      }
      else {
        const libxs_gemm_dblas_t dgemm_blas = (NULL != config && NULL != config->dgemm_blas)
          ? config->dgemm_blas : internal_libxs_dgemm_default;
        for (i = begin; i < end; ++i) {
          double* ci = (double*)((char*)c + (size_t)INTERNAL_GEMM_INDEX(i, stride_c) * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          dgemm_blas(transa, transb, &m, &n, &k,
            (const double*)alpha,
            (const double*)((const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize), &lda,
            (const double*)((const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize), &ldb,
            (const double*)beta, ci, &ldc);
        }
      }
    }
    else if (LIBXS_DATATYPE_F32 == datatype) {
      if (NULL != config && NULL != config->sgemm_jit && NULL != config->jitter) {
        for (i = begin; i < end; ++i) {
          float* ci = (float*)((char*)c + (size_t)INTERNAL_GEMM_INDEX(i, stride_c) * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          config->sgemm_jit(config->jitter,
            (const float*)((const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize),
            (const float*)((const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize), ci);
        }
      }
      else if (NULL != config && NULL != config->xgemm) {
        libxs_gemm_param_t xparam;
        memset(&xparam, 0, sizeof(xparam));
        for (i = begin; i < end; ++i) {
          char* ci = (char*)c + (size_t)INTERNAL_GEMM_INDEX(i, stride_c) * elemsize;
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          xparam.a[0] = (const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize;
          xparam.b[0] = (const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize;
          xparam.c[0] = ci;
          config->xgemm(&xparam);
        }
      }
      else {
        const libxs_gemm_sblas_t sgemm_blas = (NULL != config && NULL != config->sgemm_blas)
          ? config->sgemm_blas : internal_libxs_sgemm_default;
        for (i = begin; i < end; ++i) {
          float* ci = (float*)((char*)c + (size_t)INTERNAL_GEMM_INDEX(i, stride_c) * elemsize);
          if (need_lock) INTERNAL_GEMM_LOCKFWD(ci, lockidx);
          sgemm_blas(transa, transb, &m, &n, &k,
            (const float*)alpha,
            (const float*)((const char*)a + (size_t)INTERNAL_GEMM_INDEX(i, stride_a) * elemsize), &lda,
            (const float*)((const char*)b + (size_t)INTERNAL_GEMM_INDEX(i, stride_b) * elemsize), &ldb,
            (const float*)beta, ci, &ldc);
        }
      }
    }
#undef INTERNAL_GEMM_INDEX
    INTERNAL_GEMM_UNLOCK(lockidx);
  }
}


LIBXS_API void libxs_gemm_index(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, const int stride_a[],
                     const void* b, int ldb, const int stride_b[],
  const void* beta,        void* c, int ldc, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config)
{
  libxs_gemm_index_task(datatype, transa, transb, m, n, k,
    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
    index_stride, index_base, batchsize, config, 0, 1);
}


LIBXS_API void libxs_gemm_groups(
  libxs_data_t datatype, const char transa_array[], const char transb_array[],
  const int m_array[], const int n_array[], const int k_array[],
  const void* alpha_array, const void* a_array[], const int lda_array[],
                           const void* b_array[], const int ldb_array[],
  const void* beta_array,        void* c_array[], const int ldc_array[],
  int ngroups, const int batchsize[],
  const libxs_gemm_config_t* config)
{
  const int ng = LIBXS_ABS(ngroups);
  int i, j = 0;
  for (i = 0; i < ng; ++i) {
    const int size = LIBXS_ABS(batchsize[i]);
    int s;
    if (LIBXS_DATATYPE_F64 == datatype) {
      const libxs_gemm_dblas_t dgemm_blas = (NULL != config && NULL != config->dgemm_blas)
        ? config->dgemm_blas : internal_libxs_dgemm_default;
      const double *const palpha = (const double*)alpha_array + i;
      const double *const pbeta = (const double*)beta_array + i;
      for (s = 0; s < size; ++s) {
        dgemm_blas(transa_array + i, transb_array + i,
          m_array + i, n_array + i, k_array + i, palpha,
          (const double*)a_array[j + s], lda_array + i,
          (const double*)b_array[j + s], ldb_array + i,
          pbeta, (double*)c_array[j + s], ldc_array + i);
      }
    }
    else if (LIBXS_DATATYPE_F32 == datatype) {
      const libxs_gemm_sblas_t sgemm_blas = (NULL != config && NULL != config->sgemm_blas)
        ? config->sgemm_blas : internal_libxs_sgemm_default;
      const float *const palpha = (const float*)alpha_array + i;
      const float *const pbeta = (const float*)beta_array + i;
      for (s = 0; s < size; ++s) {
        sgemm_blas(transa_array + i, transb_array + i,
          m_array + i, n_array + i, k_array + i, palpha,
          (const float*)a_array[j + s], lda_array + i,
          (const float*)b_array[j + s], ldb_array + i,
          pbeta, (float*)c_array[j + s], ldc_array + i);
      }
    }
    j += size;
  }
}


#if defined(LIBXS_BUILD) && !defined(LIBXS_NOFORTRAN)

LIBXS_API int libxs_gemm_dispatch_f(libxs_gemm_config_t* config,
  libxs_data_t, char, char, int, int, int, int, int, int,
  const void*, const void*);
LIBXS_API int libxs_gemm_dispatch_f(libxs_gemm_config_t* config,
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta)
{
  return libxs_gemm_dispatch(config, datatype, transa, transb,
    m, n, k, lda, ldb, ldc, alpha, beta);
}


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


LIBXS_API void libxs_gemm_release_f(libxs_gemm_config_t*);
LIBXS_API void libxs_gemm_release_f(libxs_gemm_config_t* config)
{
  libxs_gemm_release(config);
}

#endif /*defined(LIBXS_BUILD) && !defined(LIBXS_NOFORTRAN)*/
