/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"
#include <libxs_malloc.h>
#include <libxs_sync.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(MAX_NUM_BITS)
# define MAX_NUM_BITS 8
#endif
#if !defined(MAX_BLOCK_M)
# define MAX_BLOCK_M 16
#endif
#if !defined(MAX_BLOCK_N)
# define MAX_BLOCK_N 16
#endif
#if !defined(MAX_PENDING)
# define MAX_PENDING 3
#endif


LIBXS_APIVAR_PRIVATE_DEF(gemm_function_t gemm_original);
LIBXS_APIVAR_DEFINE(int gemm_initialized);


/**
 * The alternative functionality is implemented here.
 * The original function can be called as fallback.
 */
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void GEMM_WRAP(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  const libxs_datatype datatype = LIBXS_DATATYPE(GEMM_REAL_TYPE);
  GEMM_REAL_TYPE *aerr = NULL, *berr = NULL, *cerr = NULL;
  LIBXS_ASSERT(NULL != lda && NULL != ldb && NULL != ldc);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  LIBXS_ASSERT(NULL != m && NULL != n && NULL != k);
  LIBXS_ASSERT(NULL != transa && NULL != transb);

  if (0 == gemm_initialized) {
    static volatile LIBXS_ATOMIC_LOCKTYPE lock = 0;
    LIBXS_ATOMIC_ACQUIRE(&lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
    if (0 == gemm_initialized) {
#if defined(_OPENMP)
      const int max_nthreads = omp_get_max_threads();
#else
      const int max_nthreads = 1;
#endif
      libxs_malloc_pool(max_nthreads, MAX_PENDING);
      gemm_initialized = 1;
    }
    LIBXS_ATOMIC_RELEASE(&lock, LIBXS_ATOMIC_SEQ_CST);
  }
  LIBXS_ASSERT(0 != gemm_initialized);

  /* allocate scratch memory to hold error analysis */
  aerr = libxs_malloc(LIBXS_TYPESIZE(datatype) * (*m) * (*k), 0/*auto*/);
  berr = libxs_malloc(LIBXS_TYPESIZE(datatype) * (*k) * (*n), 0/*auto*/);
  cerr = libxs_malloc(LIBXS_TYPESIZE(datatype) * (*m) * (*n), 0/*auto*/);
#if 0
  const int max_num = (1 << (MAX_NUM_BITS - 1)); /* POT */
  GEMM_INT_TYPE i, j, imax, vmax;
  LIBXS_ASSERT(MAX_NUM_BITS <= (sizeof(max_num) * 8));

  /* determine scaling factor and error for A-matrix */
  for (i = 0; i < *k; ++i) {
    GEMM_REAL_TYPE amax = LIBXS_ABS(a[i * (*lda)]), u;
    for (j = 1; j < *m; ++j) {
      const GEMM_REAL_TYPE aj = a[i * (*lda) + j];
      amax = LIBXS_MAX(amax, LIBXS_ABS(aj));
    }
    imax = LIBXS_ROUNDX(int, max_num / amax);
    vmax = 1 << LIBXS_INTRINSICS_BITSCANBWD32(imax);
    u = vmax / amax;
    for (j = 0; j < *m; ++j) {
      const GEMM_REAL_TYPE aj = a[i * (*lda) + j];
      const GEMM_REAL_TYPE v = u * aj;
      LIBXS_ASSERT(v <= vmax);
      aerr[i * (*m) + j] = v - LIBXS_ROUNDX(int, v);
    }
  }

  /* determine scaling factor and error for B-matrix */
  for (j = 0; j < *n; ++j) {
    GEMM_REAL_TYPE bmax = LIBXS_ABS(b[j * (*ldb)]), u;
    for (i = 1; i < *k; ++i) {
      const GEMM_REAL_TYPE bi = b[j * (*ldb) + i];
      bmax = LIBXS_MAX(bmax, LIBXS_ABS(bi));
    }
    imax = LIBXS_ROUNDX(int, max_num / bmax);
    vmax = 1 << LIBXS_INTRINSICS_BITSCANBWD32(imax);
    u = vmax / bmax;
    for (i = 0; i < *k; ++i) {
      const GEMM_REAL_TYPE bi = b[j * (*ldb) + i];
      const GEMM_REAL_TYPE v = u * bi;
      LIBXS_ASSERT(v <= vmax);
      berr[j * (*k) + i] = v - LIBXS_ROUNDX(int, v);
    }
  }
#endif
  if (NULL != gemm_original) {
    gemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else {
    GEMM_REAL(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  libxs_free(aerr);
  libxs_free(berr);
  libxs_free(cerr);
}
