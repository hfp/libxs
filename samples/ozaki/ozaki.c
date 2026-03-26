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


LIBXS_APIVAR_PUBLIC_DEF(libxs_matdiff_t gemm_diff);
LIBXS_APIVAR_PUBLIC_DEF(gemm_function_t gemm_original);
LIBXS_APIVAR_PUBLIC_DEF(int ozaki_verbose);
LIBXS_APIVAR_PUBLIC_DEF(int ozaki_stat);
LIBXS_APIVAR_PUBLIC_DEF(int ozaki);

LIBXS_APIVAR_PRIVATE_DEF(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
LIBXS_APIVAR_PRIVATE_DEF(libxs_malloc_pool_t* gemm_pool);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_target_arch);
LIBXS_APIVAR_PRIVATE_DEF(double ozaki_eps);
LIBXS_APIVAR_PRIVATE_DEF(double ozaki_rsq);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_flags);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_trim);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_exit);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_n);
LIBXS_TLS int gemm_dump_inhibit;
#if defined(__LIBXSTREAM)
LIBXS_APIVAR_PRIVATE_DEF(void* ozaki_ocl_handle);
#endif


LIBXS_API_INTERN void gemm_atexit(void);
LIBXS_API_INTERN void gemm_atexit(void)
{
  if (0 != ozaki_verbose && 0 < gemm_diff.r) {
    print_diff(stderr, &gemm_diff);
  }
#if defined(__LIBXSTREAM)
  ozaki_ocl_release(ozaki_ocl_handle);
  ozaki_ocl_handle = NULL;
  ozaki_ocl_finalize();
#endif
  libxs_free_pool(gemm_pool);
  gemm_pool = NULL;
  libxs_finalize();
}


#if defined(__LIBXSTREAM)
/**
 * OpenCL diff function matching the CPU _diff signature.
 * Computes GEMM result on GPU via ozaki_gemm, then optionally
 * runs reference BLAS on CPU and computes matdiff.
 */
LIBXS_API_INLINE void gemm_oz_ocl_diff(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
  unsigned int diff_abc, libxs_matdiff_t* diff)
{
  GEMM_REAL_TYPE* c_ref = NULL;
  /* Save C for reference comparison (before OpenCL modifies it) */
  if (NULL != diff && 0 == (diff_abc % 3)) {
    const size_t c_size = (size_t)*ldc * (size_t)*n * sizeof(GEMM_REAL_TYPE);
    c_ref = (GEMM_REAL_TYPE*)libxs_malloc(gemm_pool, c_size, 0);
    if (NULL != c_ref) memcpy(c_ref, c, c_size);
  }
  /* Compute result on OpenCL device */
  ozaki_ocl_gemm(ozaki_ocl_handle,
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
  static int gemm_threshold = 0;
  LIBXS_ASSERT(NULL != lda && NULL != ldb && NULL != ldc);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  LIBXS_ASSERT(NULL != m && NULL != n && NULL != k);
  LIBXS_ASSERT(NULL != transa && NULL != transb);

  if (0 == gemm_initialized) {
    const char *const ozaki_env = getenv("OZAKI");
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    ozaki = (NULL == ozaki_env ? 1/*default*/ : atoi(ozaki_env));
    if (0 == ozaki) gemm_initialized = 1;
    if (0 == gemm_initialized) {
      const union { uint32_t raw; float value; } inf = { 0x7F800000U };
      const char *const threshold_env = getenv("OZAKI_THRESHOLD");
      const char *const ozaki_stat_env = getenv("OZAKI_STAT");
      const char *const ozaki_exit_env = getenv("OZAKI_EXIT");
      const char *const ozaki_verbose_env = getenv("OZAKI_VERBOSE");
      const char *const ozaki_flags_env = getenv("OZAKI_FLAGS");
      const char *const ozaki_trim_env = getenv("OZAKI_TRIM");
      const char *const ozaki_eps_env = getenv("OZAKI_EPS");
      const char *const ozaki_rsq_env = getenv("OZAKI_RSQ");
      const char *const ozaki_n_env = getenv("OZAKI_N");
#if defined(__LIBXSTREAM)
      const char *const ozaki_ocl_env = getenv("OZAKI_OCL");
      const char *const ozaki_tm_env = getenv("OZAKI_TM");
      const char *const ozaki_tn_env = getenv("OZAKI_TN");
      const char *const ozaki_groups_env = getenv("OZAKI_GROUPS");
      const int ozaki_ocl = (NULL == ozaki_ocl_env ? 1/*default*/ : atoi(ozaki_ocl_env));
#endif
      libxs_init(); /*libxs_malloc_pool()*/
      libxs_matdiff_clear(&gemm_diff);
      gemm_pool = libxs_malloc_pool(NULL, NULL);
      /* consider threshold measured as arithmetic intensity */
      gemm_threshold = (NULL == threshold_env
#if defined(NDEBUG)
        ? 12/*default*/
#else
        ? 0/*default*/
#endif
        : atoi(threshold_env));
      ozaki_flags = (NULL == ozaki_flags_env ? OZ1_DEFAULT : atoi(ozaki_flags_env));
      ozaki_trim = (NULL == ozaki_trim_env ? 0/*exact*/ : atoi(ozaki_trim_env));
      ozaki_exit = (NULL == ozaki_exit_env ? 1/*default*/ : atoi(ozaki_exit_env));
      if (NULL != ozaki_stat_env) ozaki_stat = atoi(ozaki_stat_env);
      if (NULL != ozaki_verbose_env) ozaki_verbose = atoi(ozaki_verbose_env);
      else if (0 != ozaki_stat) ozaki_verbose = 1;
      if (2 == ozaki) { /* Scheme 2: CRT primes */
        ozaki_n = LIBXS_CLMP(NULL == ozaki_n_env
          ? OZ2_NPRIMES_DEFAULT : atoi(ozaki_n_env), 1, OZ2_NPRIMES_MAX);
      }
      else { /* Scheme 1: mantissa slices */
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
      /* Initialize OpenCL Ozaki context (schemes 1, 2, and 3). */
      if (0 != ozaki_ocl && (0 < ozaki && 2 >= ozaki)) {
        const int ocl_tm = (NULL != ozaki_tm_env ? atoi(ozaki_tm_env) : 0);
        const int ocl_tn = (NULL != ozaki_tn_env ? atoi(ozaki_tn_env) : 0);
        const int ocl_groups = (NULL != ozaki_groups_env
          ? atoi(ozaki_groups_env) : 0);
        ozaki_ocl_handle = ozaki_ocl_create(
          GEMM_IS_DOUBLE, ozaki, ozaki_verbose,
          ocl_tm, ocl_tn, ozaki_n, ozaki_flags, ozaki_trim,
          ocl_groups);
      }
#endif
      LIBXS_EXPECT(EXIT_SUCCESS == atexit(gemm_atexit));
      gemm_initialized = 1;
    }
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
  }
  LIBXS_ASSERT(0 != gemm_initialized);

  { /* consider threshold */
    int run_ozaki = 0;
    if (0 != ozaki) {
      const size_t size = (size_t)(*m) * (*k) + (*k) * (*n) + (*m) * (*n);
      const size_t flops = (size_t)(*m) * (*n) * (*k) * 2;
      const size_t bytes = sizeof(GEMM_REAL_TYPE) * size;
      if ((bytes * LIBXS_MAX(gemm_threshold, 0)) <= flops) {
        run_ozaki = ozaki;
      }
    }
    if (0 != run_ozaki) {
#if defined(__LIBXSTREAM)
      if (NULL != ozaki_ocl_handle) {
        OZAKI_GEMM_WRAPPER(gemm_oz_ocl_diff)
      }
      else
#endif
      if (1 == run_ozaki) { /* slice-based LP-GEMM (Scheme 1, default) */
        gemm_oz1(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
      else /*if (2 == run_ozaki)*/ { /* CRT-based LP-GEMM (Scheme 2) */
        gemm_oz2(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
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
}
