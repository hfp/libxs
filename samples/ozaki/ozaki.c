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

/* Runtime flag-set controlling the Ozaki scheme (GEMM_OZFLAGS env var).
 * Bit 0 (1): TRIANGULAR  - drop symmetric contributions (speed for accuracy)
 * Bit 1 (2): SYMMETRIZE  - double off-diagonal upper-triangle terms
 * Bit 2 (4): REVERSE_PASS - recover most significant lower-triangle terms
 * Bit 3 (8): TRIM_FORWARD - limit forward pass to slice_a < S/2
 * Default 15 = all enabled. */
#define OZ1_TRIANGULAR   1
#define OZ1_SYMMETRIZE   2
#define OZ1_REVERSE_PASS 4
#define OZ1_TRIM_FORWARD 8
#define OZ1_DEFAULT (OZ1_TRIANGULAR | OZ1_SYMMETRIZE | OZ1_REVERSE_PASS | OZ1_TRIM_FORWARD)
/* IEEE-754 format parameters derived from GEMM_REAL_TYPE */
#if GEMM_IS_DOUBLE
# define OZ1_MANT_BITS  52
# define OZ1_EXP_BIAS   1023
#else /* single-precision */
# define OZ1_MANT_BITS  23
# define OZ1_EXP_BIAS   127
#endif
#define OZ1_BIAS_PLUS_MANT (OZ1_EXP_BIAS + OZ1_MANT_BITS)
#if !defined(BLOCK_M)
# define BLOCK_M 16
#endif
#if !defined(BLOCK_N)
# define BLOCK_N 16
#endif
#if !defined(BLOCK_K)
# define BLOCK_K 16
#endif
#if !defined(MAX_NSLICES)
# if GEMM_IS_DOUBLE
#   define MAX_NSLICES 16
# else
#   define MAX_NSLICES 8
# endif
#endif
#if !defined(NSLICES_DEFAULT)
# if GEMM_IS_DOUBLE
#   define NSLICES_DEFAULT 5
# else
#   define NSLICES_DEFAULT 4
# endif
#endif
/* Scheme 2 (CRT) limits */
#if GEMM_IS_DOUBLE
# define OZ2_MAX_NPRIMES     16
# define OZ2_NPRIMES_DEFAULT 15
#else
# define OZ2_MAX_NPRIMES     10
# define OZ2_NPRIMES_DEFAULT  7
#endif


LIBXS_APIVAR_PUBLIC_DEF(libxs_matdiff_info_t gemm_diff);
LIBXS_APIVAR_PUBLIC_DEF(int gemm_verbose);

LIBXS_APIVAR_PRIVATE_DEF(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
LIBXS_APIVAR_PRIVATE_DEF(gemm_function_t gemm_original);
LIBXS_APIVAR_PRIVATE_DEF(int gemm_ozn);
LIBXS_APIVAR_PRIVATE_DEF(int gemm_ozflags);
LIBXS_APIVAR_PRIVATE_DEF(int gemm_diff_abc);
LIBXS_APIVAR_PRIVATE_DEF(double gemm_eps);
LIBXS_APIVAR_PRIVATE_DEF(double gemm_rsq);
LIBXS_APIVAR_PRIVATE_DEF(int ozaki_target_arch);
LIBXS_TLS int gemm_dump_inhibit;


LIBXS_API_INTERN void print_diff_atexit(void);
LIBXS_API_INTERN void print_diff_atexit(void)
{
  if (0 != gemm_verbose && 0 < gemm_diff.r) {
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    print_diff(stderr, &gemm_diff);
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
  }
}


/** Function gemm_oz1 is called here with the original GEMM as fallback and for comparison. */
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void GEMM_WRAP(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  static volatile int gemm_initialized = 0;
  static int gemm_ozaki = 1;
  LIBXS_ASSERT(NULL != lda && NULL != ldb && NULL != ldc);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  LIBXS_ASSERT(NULL != m && NULL != n && NULL != k);
  LIBXS_ASSERT(NULL != transa && NULL != transb);

  if (0 == gemm_initialized) {
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    if (0 == gemm_initialized) {
      const union { uint32_t raw; float value; } inf = { 0x7F800000U };
      const char *const gemm_diff_abc_env = getenv("GEMM_DIFF");
      const char *const gemm_verbose_env = getenv("GEMM_VERBOSE");
      const char *const gemm_ozaki_env = getenv("GEMM_OZAKI");
      const char *const gemm_ozn_env = getenv("GEMM_OZN");
      const char *const gemm_ozflags_env = getenv("GEMM_OZFLAGS");
      const char *const gemm_eps_env = getenv("GEMM_EPS");
      const char *const gemm_rsq_env = getenv("GEMM_RSQ");
      libxs_matdiff_clear(&gemm_diff);
      gemm_diff_abc = (NULL == gemm_diff_abc_env ? 0 : atoi(gemm_diff_abc_env));
      gemm_verbose = (NULL == gemm_verbose_env ? 0 : atoi(gemm_verbose_env));
      gemm_ozaki = (NULL == gemm_ozaki_env ? 1 : atoi(gemm_ozaki_env));
      gemm_ozflags = (NULL == gemm_ozflags_env
        ? OZ1_DEFAULT : atoi(gemm_ozflags_env));
      if (2 == gemm_ozaki) { /* Scheme 2: CRT primes */
        gemm_ozn = LIBXS_CLMP((NULL == gemm_ozn_env
          ? OZ2_NPRIMES_DEFAULT : atoi(gemm_ozn_env)), 1, OZ2_MAX_NPRIMES);
      }
      else { /* Scheme 1: mantissa slices */
        gemm_ozn = LIBXS_CLMP((NULL == gemm_ozn_env
          ? NSLICES_DEFAULT : atoi(gemm_ozn_env)), 1, MAX_NSLICES);
      }
      if (NULL == gemm_eps_env) gemm_eps = inf.value;
      else {
        if (0 == gemm_verbose) gemm_verbose = 1;
        gemm_eps = atof(gemm_eps_env);
      }
      if (NULL == gemm_rsq_env) gemm_rsq = 0;
      else {
        if (0 == gemm_verbose) gemm_verbose = 1;
        gemm_rsq = atof(gemm_rsq_env);
      }
      ozaki_target_arch = libxs_cpuid(NULL);
      LIBXS_EXPECT(EXIT_SUCCESS == atexit(print_diff_atexit));
      gemm_initialized = 1;
    }
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
  }
  LIBXS_ASSERT(0 != gemm_initialized);

  if (0 == gemm_ozaki) { /* only run original GEMM right away */
    if (NULL != gemm_original) {
      gemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    else {
      GEMM_REAL(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    if (0 != gemm_verbose) {
      LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
      ++gemm_diff.r;
      LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
    }
  }
  else if (2 == gemm_ozaki) { /* CRT-based LP-GEMM (Scheme 2) */
    gemm_oz2(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else { /* slice-based LP-GEMM (Scheme 1, default) */
    gemm_oz1(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}
