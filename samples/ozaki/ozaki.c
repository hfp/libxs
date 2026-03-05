/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"
#include <libxs_sync.h>


LIBXS_APIVAR_PUBLIC_DEF(libxs_matdiff_info_t gemm_diff);
LIBXS_APIVAR_PUBLIC_DEF(int ozaki_verbose);
LIBXS_APIVAR_PUBLIC_DEF(int ozaki);
LIBXS_APIVAR_PUBLIC_DEF(int ozaki_stat);

LIBXS_APIVAR_PRIVATE_DEF(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
LIBXS_APIVAR_PRIVATE_DEF(gemm_function_t gemm_original);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_target_arch);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_flags);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_trim);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_n);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_exit);
LIBXS_APIVAR_PRIVATE_DEF(double ozaki_eps);
LIBXS_APIVAR_PRIVATE_DEF(double ozaki_rsq);
LIBXS_APIVAR_PRIVATE_DEF(libxs_malloc_pool_t* gemm_pool);
LIBXS_TLS int gemm_dump_inhibit;
#if defined(__LIBXSTREAM)
LIBXS_APIVAR_PRIVATE_DEF(void* ozaki_gpu_handle);
#endif


LIBXS_API_INTERN void gemm_atexit(void);
LIBXS_API_INTERN void gemm_atexit(void)
{
  if (0 != ozaki_verbose && 0 < gemm_diff.r) {
    print_diff(stderr, &gemm_diff);
  }
#if defined(__LIBXSTREAM)
  ozaki_gpu_release(ozaki_gpu_handle);
  ozaki_gpu_handle = NULL;
  ozaki_gpu_finalize();
#endif
  libxs_free_pool(gemm_pool);
  gemm_pool = NULL;
  libxs_finalize();
}


#if defined(__LIBXSTREAM)
/**
 * GPU diff function matching the CPU _diff signature.
 * Computes GEMM result on GPU via ozaki_gemm, then optionally
 * runs reference BLAS on CPU and computes matdiff.
 */
LIBXS_API_INLINE void gemm_oz_gpu_diff(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
  unsigned int diff_abc, libxs_matdiff_info_t* diff)
{
  const size_t c_size = (size_t)*ldc * (size_t)*n * sizeof(GEMM_REAL_TYPE);
  GEMM_REAL_TYPE* c_ref = NULL;
  /* Save C for reference comparison (before GPU modifies it) */
  if (NULL != diff && 0 == (diff_abc % 3)) {
    c_ref = (GEMM_REAL_TYPE*)libxs_malloc(gemm_pool, c_size, 0);
    if (NULL != c_ref) memcpy(c_ref, c, c_size);
  }
  /* Compute result on GPU */
  ozaki_gpu_dgemm(ozaki_gpu_handle,
    *transa, *transb, *m, *n, *k,
    (double)*alpha, a, *lda, b, *ldb,
    (double)*beta, c, *ldc);
  /* Reference BLAS and diff comparison */
  if (NULL != c_ref) {
    if (NULL != gemm_original) {
      gemm_original(transa, transb, m, n, k, alpha, a, lda,
                    b, ldb, beta, c_ref, ldc);
    }
    else {
      GEMM_REAL(transa, transb, m, n, k, alpha, a, lda,
                b, ldb, beta, c_ref, ldc);
    }
    libxs_matdiff(diff, LIBXS_DATATYPE(GEMM_REAL_TYPE),
      *m, *n, c_ref, c, ldc, ldc);
    libxs_free(c_ref);
  }
}
#endif


/** Function gemm_oz1 is called here with the original GEMM as fallback and for comparison. */
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void GEMM_WRAP(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  static volatile int gemm_initialized = 0;
  LIBXS_ASSERT(NULL != lda && NULL != ldb && NULL != ldc);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  LIBXS_ASSERT(NULL != m && NULL != n && NULL != k);
  LIBXS_ASSERT(NULL != transa && NULL != transb);

  if (0 == gemm_initialized) {
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    if (0 == gemm_initialized) {
      const union { uint32_t raw; float value; } inf = { 0x7F800000U };
      const char *const ozaki_stat_env = getenv("OZAKI_STAT");
      const char *const ozaki_exit_env = getenv("OZAKI_EXIT");
      const char *const ozaki_verbose_env = getenv("OZAKI_VERBOSE");
      const char *const ozaki_flags_env = getenv("OZAKI_FLAGS");
      const char *const ozaki_trim_env = getenv("OZAKI_TRIM");
      const char *const ozaki_env = getenv("OZAKI");
      const char *const ozaki_n_env = getenv("OZAKI_N");
      const char *const ozaki_eps_env = getenv("OZAKI_EPS");
      const char *const ozaki_rsq_env = getenv("OZAKI_RSQ");
      libxs_init(); /*libxs_malloc_pool()*/
      gemm_pool = libxs_malloc_pool(NULL, NULL);
      libxs_matdiff_clear(&gemm_diff);
      ozaki_flags = (NULL == ozaki_flags_env ? OZ1_DEFAULT : atoi(ozaki_flags_env));
      ozaki_trim = (NULL == ozaki_trim_env ? 0/*exact*/ : atoi(ozaki_trim_env));
      ozaki = (NULL == ozaki_env ? 1/*default*/ : atoi(ozaki_env));
      ozaki_exit = (NULL == ozaki_exit_env ? 1/*default*/ : atoi(ozaki_exit_env));
      if (NULL != ozaki_stat_env) ozaki_stat = atoi(ozaki_stat_env);
      if (NULL != ozaki_verbose_env) ozaki_verbose = atoi(ozaki_verbose_env);
      else if (0 != ozaki_stat) ozaki_verbose = 1;
      if (2 == ozaki || 4 == ozaki) { /* Scheme 2/4: CRT primes */
        ozaki_n = LIBXS_CLMP(NULL == ozaki_n_env
          ? OZ2_NPRIMES_DEFAULT : atoi(ozaki_n_env), 1, OZ2_NPRIMES_MAX);
      }
      else { /* Scheme 1/3: mantissa slices */
        ozaki_n = LIBXS_CLMP(NULL == ozaki_n_env
          ? NSLICES_DEFAULT : atoi(ozaki_n_env), 1, MAX_NSLICES);
      }
      if (NULL == ozaki_eps_env) ozaki_eps = inf.value;
      else {
        if (0 == ozaki_verbose) ozaki_verbose = 1;
        ozaki_eps = atof(ozaki_eps_env);
      }
      if (NULL == ozaki_rsq_env) ozaki_rsq = 0;
      else {
        if (0 == ozaki_verbose) ozaki_verbose = 1;
        ozaki_rsq = atof(ozaki_rsq_env);
      }
      ozaki_target_arch = libxs_cpuid(NULL);
#if defined(__LIBXSTREAM)
      /* Initialize GPU Ozaki context (schemes 1 and 3 only) */
      if (1 == ozaki || 3 == ozaki) {
        ozaki_gpu_handle = ozaki_gpu_create(
          GEMM_IS_DOUBLE, ozaki,
          (0 < ozaki_verbose ? 1 : 0),
          ozaki_n, 0, ozaki_flags, ozaki_trim);
      }
#endif
      LIBXS_EXPECT(EXIT_SUCCESS == atexit(gemm_atexit));
      gemm_initialized = 1;
    }
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
  }
  LIBXS_ASSERT(0 != gemm_initialized);

#if defined(__LIBXSTREAM)
  if (NULL != ozaki_gpu_handle) {
    OZAKI_GEMM_WRAPPER(gemm_oz_gpu_diff)
  }
  else
#endif
  if (1 == ozaki) { /* slice-based LP-GEMM (Scheme 1, default) */
    gemm_oz1(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else if (2 == ozaki) { /* CRT-based LP-GEMM (Scheme 2) */
    gemm_oz2(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else if (3 == ozaki) { /* BF16 slice-based LP-GEMM (Scheme 3) */
    gemm_oz3(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else if (4 == ozaki) { /* CRT-based LP-GEMM with BF16 dot products (Scheme 4) */
    gemm_oz4(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else { /* only run original GEMM right away */
    if (NULL != gemm_original) {
      gemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    else {
      GEMM_REAL(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    if (0 != ozaki_verbose) {
      LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
      ++gemm_diff.r;
      LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
    }
  }
}
