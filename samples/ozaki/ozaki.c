/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"
#include <libxs_hist.h>
#include <libxs_sync.h>
#include <signal.h>


OZAKI_APIVAR_PUBLIC_DEF(libxs_matdiff_t gemm_diff);
OZAKI_APIVAR_PUBLIC_DEF(gemm_function_t gemm_original);
OZAKI_APIVAR_PUBLIC_DEF(int ozaki_verbose);
OZAKI_APIVAR_PUBLIC_DEF(int ozaki_stat);
OZAKI_APIVAR_PUBLIC_DEF(int ozaki);
OZAKI_APIVAR_PUBLIC_DEF(int ozaki_complex);
OZAKI_APIVAR_PUBLIC_DEF(int ozaki_maxk);

OZAKI_APIVAR_PRIVATE_DEF(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
OZAKI_APIVAR_PRIVATE_DEF(libxs_malloc_pool_t* gemm_pool);
OZAKI_APIVAR_PRIVATE_DEF(int ozaki_target_arch);
OZAKI_APIVAR_PRIVATE_DEF(int ozaki_idx);
OZAKI_APIVAR_PRIVATE_DEF(double ozaki_eps);
OZAKI_APIVAR_PRIVATE_DEF(double ozaki_rsq);
OZAKI_APIVAR_PRIVATE_DEF(int ozaki_flags);
OZAKI_APIVAR_PRIVATE_DEF(int ozaki_trim);
OZAKI_APIVAR_PRIVATE_DEF(int ozaki_dump);
OZAKI_APIVAR_PRIVATE_DEF(int ozaki_exit);
OZAKI_APIVAR_PRIVATE_DEF(int ozaki_n);
OZAKI_APIVAR_PRIVATE_DEF(int ozaki_profile);
OZAKI_APIVAR_PRIVATE_DEF(libxs_hist_t* ozaki_hist);
OZAKI_APIVAR_PRIVATE_DEF(int gemm_threshold);
#if GEMM_IS_DOUBLE /* single definition across both precision builds */
LIBXS_TLS int gemm_nozaki;
LIBXS_TLS int gemm_dump_inhibit;
#endif
#if defined(__LIBXSTREAM)
OZAKI_APIVAR_PRIVATE_DEF(void* ozaki_ocl_handle);
#endif


OZAKI_API_INTERN void gemm_atexit(void);
OZAKI_API_INTERN void gemm_atexit(void)
{
  static volatile sig_atomic_t once = 0;
  if (0 == once) {
    once = 1;
    if (0 != ozaki_verbose && 0 < gemm_diff.r) {
      print_diff(stderr, NULL /*GEMM*/, ozaki_stat, &gemm_diff);
    }
    if (NULL != ozaki_hist) {
      const char* const kind = GEMM_IS_DOUBLE ? "DP" : "SP";
      double gflops = 0;
      int ngemms = 0; /* number of int8 GEMMs per FP GEMM */
      libxs_hist_get_median(NULL /*lock*/, ozaki_hist, &gflops);
      if (1 == ozaki) { /* Scheme 1: slice pairs */
        const int cutoff = 2 * (ozaki_n - 1) - ozaki_trim;
        int sa, sb;
        for (sa = 0; sa < ozaki_n && sa <= cutoff; ++sa) {
          const int sb_start = (0 != (ozaki_flags & OZ1_TRIANGULAR)) ? sa : 0;
          const int sb_end = LIBXS_MIN(ozaki_n, cutoff + 1 - sa);
          for (sb = sb_start; sb < sb_end; ++sb) {
            ++ngemms;
            if (0 != (ozaki_flags & OZ1_SYMMETRIZE) && sa != sb) ++ngemms;
          }
        }
      }
      else if (3 == ozaki) { /* Scheme 3: Foeppl SBP (remainder GEMM) */
        ngemms = 1;
      }
      else { /* Scheme 2: one int8 GEMM per prime */
        ngemms = ozaki_n;
      }
      fprintf(stderr, "OZAKI PROF: %.0f %s-GFLOPS/s", gflops, kind);
      if (0 < ngemms) {
        const double tops = gflops * ngemms * 1E-3;
        fprintf(stderr, " (%.1f INT8-TOPS/s, %dx)", tops, ngemms);
      }
      fprintf(stderr, "\n");
      libxs_hist_destroy(ozaki_hist);
      ozaki_hist = NULL;
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
}


OZAKI_API_INTERN void gemm_signal_handler(int sig);
OZAKI_API_INTERN void gemm_signal_handler(int sig)
{
  gemm_atexit();
  signal(sig, SIG_DFL);
  raise(sig);
}


#if defined(__LIBXSTREAM)
/**
 * OpenCL diff function matching the CPU _diff signature.
 * Computes GEMM result on GPU via ozaki_gemm, then optionally
 * runs reference BLAS on CPU and computes matdiff.
 */
LIBXS_API_INLINE void gemm_oz_ocl_diff(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
  libxs_matdiff_t* diff)
{
  GEMM_REAL_TYPE* c_ref = NULL;
  size_t c_size = 0;
  /* Save C for reference comparison (before OpenCL modifies it) */
  if (NULL != diff) {
    c_size = (size_t)*ldc * (size_t)*n * sizeof(GEMM_REAL_TYPE);
    c_ref = (GEMM_REAL_TYPE*)libxs_malloc(gemm_pool, c_size, 0);
    if (NULL != c_ref) memcpy(c_ref, c, c_size);
  }
  /* Compute result on OpenCL device */
  ozaki_ocl_gemm(ozaki_ocl_handle, *transa, *transb, *m, *n, *k, (double)*alpha, a, *lda, b, *ldb, (double)*beta, c, *ldc,
    ozaki_hist, ozaki_profile);
  /* Reference BLAS and diff comparison */
  if (NULL != c_ref) {
    ozaki_diff_reference(GEMM_ARGPASS, c_ref, c_size, diff);
    libxs_free(c_ref);
  }
}
#endif


OZAKI_API_INTERN void gemm_init(void)
{
  static volatile int gemm_initialized = 0;
  if (0 == gemm_initialized) {
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    if (0 == gemm_initialized) {
      const char* const ozaki_env = getenv("OZAKI");
      const char* const ozaki_stat_env = getenv("OZAKI_STAT");
      const char* const ozaki_maxk_env = getenv("OZAKI_MAXK");
      const char* const ozaki_verbose_env = getenv("OZAKI_VERBOSE");
      const char* const ozaki_complex_env = getenv("OZAKI_COMPLEX");
      ozaki = (NULL == ozaki_env ? 1 /*default*/ : atoi(ozaki_env));
      /* OZAKI_MAXK: max K per preprocessing pass (0=no grouping).
       * Default: K_GRP (compile-time, typically 32768). */
      ozaki_maxk = (NULL != ozaki_maxk_env ? atoi(ozaki_maxk_env) : K_GRP);
      /* OZAKI_COMPLEX: 0=original BLAS, 1=CPU, 2=GPU,
       * 3=original ZGEMM + lock out real GEMM during ZGEMM.
       * Default: 0 if OZAKI=0, else 2 (GPU preferred, CPU fallback). */
      ozaki_complex = (NULL != ozaki_complex_env ? atoi(ozaki_complex_env) : (0 != ozaki ? 2 : 0));
      if (NULL != ozaki_stat_env) ozaki_stat = atoi(ozaki_stat_env);
      if (NULL != ozaki_verbose_env) ozaki_verbose = atoi(ozaki_verbose_env);
      else if (0 != ozaki_stat) ozaki_verbose = 1;
      if (0 != ozaki) {
        const union {
          uint32_t raw;
          float value;
        } inf = {0x7F800000U};
        const char* const threshold_env = getenv("OZAKI_THRESHOLD");
        const char* const ozaki_dump_env = getenv("OZAKI_DUMP");
        const char* const ozaki_exit_env = getenv("OZAKI_EXIT");
        const char* const ozaki_flags_env = getenv("OZAKI_FLAGS");
        const char* const ozaki_trim_env = getenv("OZAKI_TRIM");
        const char* const ozaki_amx_env = getenv("OZAKI_AMX");
        const char* const ozaki_idx_env = getenv("OZAKI_IDX");
        const char* const ozaki_eps_env = getenv("OZAKI_EPS");
        const char* const ozaki_rsq_env = getenv("OZAKI_RSQ");
        const char* const ozaki_n_env = getenv("OZAKI_N");
#if defined(__LIBXSTREAM)
        const char* const ozaki_groups_env = getenv("OZAKI_GROUPS");
        const char* const ozaki_ocl_env = getenv("OZAKI_OCL");
        const char* const ozaki_tm_env = getenv("OZAKI_TM");
        const char* const ozaki_tn_env = getenv("OZAKI_TN");
        const int ozaki_ocl = (NULL == ozaki_ocl_env ? 0 /*default*/ : atoi(ozaki_ocl_env));
#endif
        const int ozaki_amx = (NULL == ozaki_amx_env ? 0 /*default*/ : atoi(ozaki_amx_env));
        libxs_init(); /*libxs_malloc_pool()*/
        libxs_matdiff_clear(&gemm_diff);
        gemm_pool = libxs_malloc_pool(NULL, NULL);
        /* consider threshold measured as arithmetic intensity */
        gemm_threshold = (NULL == threshold_env
#if defined(NDEBUG)
                            ? 12 /*default*/
#else
                            ? 0 /*default*/
#endif
                            : atoi(threshold_env));
        ozaki_flags = (NULL == ozaki_flags_env ? OZ1_DEFAULT : atoi(ozaki_flags_env));
        ozaki_trim = (NULL == ozaki_trim_env ? 0 /*exact*/ : atoi(ozaki_trim_env));
        ozaki_exit = (NULL == ozaki_exit_env ? 1 /*default*/ : atoi(ozaki_exit_env));
        ozaki_idx = (NULL == ozaki_idx_env ? 0 : atoi(ozaki_idx_env));
        if (2 == ozaki) { /* Scheme 2: CRT primes */
          ozaki_n = LIBXS_CLMP(NULL == ozaki_n_env ? OZ2_NPRIMES_DEFAULT : atoi(ozaki_n_env), 1, OZ2_NPRIMES_MAX);
        }
        else if (3 == ozaki) { /* Scheme 3: Foeppl SBP order */
          ozaki_n = LIBXS_CLMP(NULL == ozaki_n_env ? 4 : atoi(ozaki_n_env), 1, 16);
        }
        else { /* Scheme 1: mantissa slices */
          ozaki_n = LIBXS_CLMP(NULL == ozaki_n_env ? NSLICES_DEFAULT : atoi(ozaki_n_env), 1, MAX_NSLICES);
        }
        if (NULL == ozaki_dump_env) ozaki_dump = 0;
        else {
          if (0 == ozaki_verbose) ozaki_verbose = 1;
          ozaki_dump = atoi(ozaki_dump_env);
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
        if (0 != ozaki_amx) {
          ozaki_target_arch = libxs_cpuid(NULL);
          if (LIBXS_X86_AVX512_AMX <= ozaki_target_arch) {
            if (EXIT_SUCCESS != libxs_cpuid_amx_enable()) {
              ozaki_target_arch = LIBXS_X86_AVX512;
            }
          }
        }
        { /* Profiling: create histogram if requested */
          const char *const env_prof = getenv("OZAKI_PROFILE");
          ozaki_profile = (NULL == env_prof ? 0 : atoi(env_prof));
          if (0 != ozaki_profile) {
            const libxs_hist_update_t update[] = {libxs_hist_update_avg};
            ozaki_hist = libxs_hist_create(3, 1, update);
            if (NULL == ozaki_hist) ozaki_profile = 0;
          }
        }
#if defined(__LIBXSTREAM)
        /* initialize OpenCL Ozaki context */
        if (0 != ozaki_ocl && (0 < ozaki && 2 >= ozaki)) {
          const int ocl_tm = (NULL != ozaki_tm_env ? atoi(ozaki_tm_env) : 0);
          const int ocl_tn = (NULL != ozaki_tn_env ? atoi(ozaki_tn_env) : 0);
          const int ocl_groups = (NULL != ozaki_groups_env ? atoi(ozaki_groups_env) : 0);
          ozaki_ocl_handle = ozaki_ocl_create(
            GEMM_IS_DOUBLE, ozaki, ozaki_verbose, ocl_tm, ocl_tn, ozaki_n, ozaki_flags, ozaki_trim, ocl_groups, ozaki_maxk,
            0 != ozaki_profile);
        }
#endif
        atexit(gemm_atexit);
        signal(SIGABRT, gemm_signal_handler);
        signal(SIGTERM, gemm_signal_handler);
        signal(SIGINT, gemm_signal_handler);
#if defined(SIGHUP)
        signal(SIGHUP, gemm_signal_handler);
#endif
      }
      gemm_initialized = 1;
    }
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
  }
}


/** Function gemm_oz1 is called here with the original GEMM as fallback and for comparison. */
OZAKI_API_INTERN LIBXS_ATTRIBUTE_WEAK void GEMM_WRAP(const char* transa, const char* transb, const GEMM_INT_TYPE* m,
  const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
  const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  int run_ozaki = 0;
  LIBXS_ASSERT(NULL != lda && NULL != ldb && NULL != ldc);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  LIBXS_ASSERT(NULL != m && NULL != n && NULL != k);
  LIBXS_ASSERT(NULL != transa && NULL != transb);

  gemm_init();

  /* Quick-return for degenerate dimensions (BLAS spec: valid no-op).
   * Some BLAS implementations reject lda=0 even when m*n*k=0. */
  if (*m > 0 && *n > 0 && *k > 0) {
    /* Bypass Ozaki: fall through to original BLAS.
     * Used when the reference ZGEMM may internally call sgemm_,
     * which --wrap redirects back into GEMM_WRAP. Without bypass,
     * the "reference" result would itself be Ozaki-approximate. */
    if (0 != gemm_nozaki) {
      if (NULL != gemm_original) {
        gemm_original(GEMM_ARGPASS);
      }
      else {
        GEMM_REAL(GEMM_ARGPASS);
      }
    }
    else {
      if (0 != ozaki) { /* consider threshold */
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
          OZAKI_GEMM_WRAPPER(gemm_oz_ocl_diff, GEMM_LABEL, 1)
        }
        else
#endif
          if (1 == run_ozaki)
        { /* slice-based LP-GEMM (Scheme 1, default) */
          gemm_oz1(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        else if (2 == run_ozaki) { /* CRT-based LP-GEMM (Scheme 2) */
          gemm_oz2(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        else { /* summation by parts (Scheme 3) */
          gemm_oz3(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
      }
      else { /* only run original GEMM right away */
        if (NULL != gemm_original) {
          gemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        else {
          GEMM_REAL(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
        if (0 != ozaki_verbose && 0 > ozaki_stat) {
          LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
          if (1 < ozaki_verbose || 0 > ozaki_verbose) {
            const int nth = (0 < ozaki_verbose ? ozaki_verbose : 1);
            if (0 == (gemm_diff.r % nth)) {
              const int id = libxs_rid();
              fprintf(stderr, GEMM_LABEL "[%i.%i]: ", gemm_diff.r, id);
              print_gemm(stderr, 2 /*compact*/, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
          }
          LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
        }
#if defined(__LIBXSTREAM)
        /* Invalidate cache entries matching output C: the CPU path just wrote C,
         * so any cached preprocessed data keyed by C's pointer is now stale
         * (C's address may be reused as A or B in a subsequent GEMM call). */
        if (NULL != ozaki_ocl_handle) {
          ozaki_ocl_invalidate_cache(ozaki_ocl_handle, c, c);
        }
#endif
      }
    }
  }
}
