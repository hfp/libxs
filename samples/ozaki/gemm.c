/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"
#include <libxs_timer.h>
#include <libxs_rng.h>

/* Weak references: gemm-blas.x links without the Ozaki library,
 * so these symbols may be undefined. CHECK should not be used
 * with gemm-blas.x (the variables resolve to zero-address). */
LIBXS_PRAGMA_WEAK(gemm_verbose)
LIBXS_PRAGMA_WEAK(gemm_diff)


int main(int argc, char* argv[])
{
  const char *const nrepeat_env = getenv("NREPEAT");
  const char *const env_check = getenv("CHECK");
  const double check = (NULL == env_check || 0 == *env_check) ? 0 : atof(env_check);
  const int nrep = (NULL == nrepeat_env ? 1 : atoi(nrepeat_env));
  const int nrepeat = (0 < nrep ? nrep : 1);
  GEMM_INT_TYPE m = (1 < argc ? atoi(argv[1]) : 257);
  GEMM_INT_TYPE n = (2 < argc ? atoi(argv[2]) : m);
  GEMM_INT_TYPE k = (3 < argc ? atoi(argv[3]) : m);
  const int ta = (4 < argc ? atoi(argv[4]) : 0);
  const int tb = (5 < argc ? atoi(argv[5]) : 0);
  GEMM_REAL_TYPE alpha = (6 < argc ? atof(argv[6]) : 1);
  GEMM_REAL_TYPE beta = (7 < argc ? atof(argv[7]) : 1);
  GEMM_INT_TYPE lda = (8 < argc ? atoi(argv[8]) : (0 == ta ? m : k));
  GEMM_INT_TYPE ldb = (9 < argc ? atoi(argv[9]) : (0 == tb ? k : n));
  GEMM_INT_TYPE ldc = (10 < argc ? atoi(argv[10]) : m);
  char transa = (0 == ta ? 'N' : 'T'), transb = (0 == tb ? 'N' : 'T');
  const GEMM_REAL_TYPE scale = (1 < nrepeat ? (1.0 / nrepeat) : 1);
  int result = EXIT_SUCCESS, file_input = 0, complex_input = 0, i;
  GEMM_REAL_TYPE complex_alpha[2] = { 0 }, complex_beta[2] = { 0 };
  GEMM_REAL_TYPE *a = NULL, *b = NULL, *c = NULL;
  GEMM_INT_TYPE a_rows, a_cols, b_rows, b_cols;
  libxs_matdiff_info_t diff;

  libxs_init();

  if (2 < argc && 0 == m) { /* Indicate filename(s) */
    GEMM_REAL_TYPE scalar[2] = { 0 };
    GEMM_INT_TYPE dim0, dim1;
    size_t ncomp = 0;
    gemm_mhd_settings_t settings_a;
    if (EXIT_SUCCESS == gemm_mhd_read(argv[1],
      &dim0, &dim1, &transa, &lda, scalar, &ncomp, &settings_a, NULL))
    {
      m = dim0; k = dim1;
      ldc = (0 < settings_a.ldc) ? settings_a.ldc : dim0;
      alpha = scalar[0];
      if (2 == ncomp) {
        complex_alpha[0] = scalar[0]; complex_alpha[1] = scalar[1];
        complex_input = 1;
      }
      file_input |= 0x1;
    }
    if (0 == n) {
      size_t ncomp_b = 0;
      const int b_read = gemm_mhd_read(argv[2],
        &dim0, &dim1, &transb, &ldb, scalar, &ncomp_b, NULL, NULL);
      if (EXIT_SUCCESS == b_read && k == dim0 && ncomp_b == ncomp)
      {
        n = dim1;
        beta = scalar[0];
        if (2 == ncomp_b) {
          complex_beta[0] = scalar[0]; complex_beta[1] = scalar[1];
        }
        file_input |= 0x2;
      }
      else if (EXIT_SUCCESS == b_read) {
        fprintf(stderr, "Mismatched files: A implies k=%i but B has k=%i\n",
          (int)k, (int)dim0);
      }
    }
  }

  /* Compute physical (stored) matrix dimensions.
   * For file input, the MHD writer stores A as (m, k) and B as (k, n),
   * so a_cols = k and b_cols = n regardless of transpose. */
  a_rows = ('N' == transa || 'n' == transa) ? m : k;
  a_cols = (0x1 & file_input) ? k : (('N' == transa || 'n' == transa) ? k : m);
  b_rows = ('N' == transb || 'n' == transb) ? k : n;
  b_cols = (0x2 & file_input) ? n : (('N' == transb || 'n' == transb) ? n : k);

  if (1 > m || 1 > n || 1 > k || lda < a_rows || ldb < b_rows || ldc < m) {
    fprintf(stderr, "Invalid dimensions: m=%i n=%i k=%i lda=%i(>=%i) ldb=%i(>=%i) ldc=%i(>=%i)\n",
      (int)m, (int)n, (int)k, (int)lda, (int)a_rows, (int)ldb, (int)b_rows, (int)ldc, (int)m);
    result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) { /* Allocate matrices */
    const size_t nc = (0 != complex_input ? 2 : 1);
    a = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * nc * lda * a_cols);
    b = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * nc * ldb * b_cols);
    c = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * nc * ldc * n);
    if (NULL != a && NULL != b && NULL != c) {
      if (0 == file_input || 0 == beta) {
        LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, c, m, n, ldc, scale);
      }
      else memset(c, 0, sizeof(GEMM_REAL_TYPE) * nc * ldc * n);
    }
    else result = EXIT_FAILURE;
  }

  /* Print requested GEMM arguments (regardless of result code) */
  print_gemm(stdout, &transa, &transb, &m, &n, &k,
    &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

  if (EXIT_SUCCESS == result) { /* Initialize A-matrix */
    if (0x1 & file_input) {
      result = gemm_mhd_read(argv[1], NULL, NULL, NULL, NULL, NULL, NULL, NULL, a);
    }
    else {
      LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, a, a_rows, a_cols, lda, scale);
    }
  }

  if (EXIT_SUCCESS == result) { /* Initialize B-matrix */
    if (0x2 & file_input) {
      result = gemm_mhd_read(argv[2], NULL, NULL, NULL, NULL, NULL, NULL, NULL, b);
    }
    else {
      LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, b, b_rows, b_cols, ldb, scale);
    }
  }

  if (EXIT_SUCCESS == result) { /* Call GEMM */
    const GEMM_REAL_TYPE* const ga = (0 != complex_input) ? complex_alpha : &alpha;
    const GEMM_REAL_TYPE* const gb = (0 != complex_input) ? complex_beta : &beta;
    libxs_timer_tick_t start;
    int ncalls = nrepeat;
    if (1 < nrepeat) { /* Peel one warmup call */
      if (0 != complex_input) ZGEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
      else GEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
      --ncalls;
    }
    start = libxs_timer_tick();
    for (i = 0; i < ncalls; ++i) {
      if (0 != complex_input) ZGEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
      else GEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
    }
    printf("Called %i times (%f s/call).\n", nrepeat,
      libxs_timer_duration(start, libxs_timer_tick()) / ncalls);
  }

  if (EXIT_SUCCESS == result) { /* Calculate final checksum */
    const size_t nc = (0 != complex_input ? 2 : 1);
    const int ldtst = (int)(nc * ldc);
    result = libxs_matdiff(&diff, LIBXS_DATATYPE(GEMM_REAL_TYPE), (int)(nc * m), n,
        NULL/*ref*/, c/*tst*/, NULL/*ldref*/, &ldtst);
  }

  if (EXIT_SUCCESS == result) {
    printf("\n%f (check)\n", diff.l1_tst);
  }

  if (EXIT_SUCCESS == result && 0 != check) { /* Accuracy validation */
    const double epsilon = libxs_matdiff_epsilon(&gemm_diff);
    const double threshold = (0 < check) ? check
      : (sizeof(double) == sizeof(GEMM_REAL_TYPE) ? 1.0E-10 : 1.0E-3);
    if (threshold < epsilon) {
      fprintf(stderr, "CHECK: eps=%g exceeds threshold=%g\n",
        epsilon, threshold);
      result = EXIT_FAILURE;
    }
    else {
      fprintf(stderr, "CHECK: eps=%g (threshold=%g)\n",
        epsilon, threshold);
    }
  }

  free(a);
  free(b);
  free(c);

  return result;
}
