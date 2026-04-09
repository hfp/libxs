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
LIBXS_PRAGMA_WEAK(gemm_original)
LIBXS_PRAGMA_WEAK(ozaki_verbose)
LIBXS_PRAGMA_WEAK(gemm_diff)
LIBXS_PRAGMA_WEAK(GEMM_REAL)


int main(int argc, char* argv[])
{
  const char* const nrepeat_env = getenv("NREPEAT");
  const char* const env_check = getenv("CHECK");
  const double check = (NULL == env_check || 0 == *env_check) ? 0 : atof(env_check);
  const int nrep = (NULL == nrepeat_env ? 3 : atoi(nrepeat_env));
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
  int result = EXIT_SUCCESS, file_input = 0, i;
#if defined(GEMM_COMPLEX)
  int complex_input = 1;
#else
  int complex_input = 0;
#endif
  GEMM_REAL_TYPE complex_alpha[2] = {0}, complex_beta[2] = {0};
  GEMM_REAL_TYPE *a = NULL, *b = NULL, *c = NULL, *c_ref = NULL;
  GEMM_INT_TYPE a_rows, a_cols, b_rows, b_cols;
  libxs_matdiff_t diff;

  libxs_init();

#if defined(GEMM_COMPLEX)
  /* Complex mode: alpha and beta are [real, imag] pairs */
  complex_alpha[0] = alpha;
  complex_alpha[1] = 0.0;
  complex_beta[0] = beta;
  complex_beta[1] = 0.0;
#endif

  if (2 < argc && 0 == m) { /* Indicate filename(s) */
    GEMM_REAL_TYPE scalar[2] = {0};
    GEMM_INT_TYPE dim0, dim1;
    size_t ncomp = 0;
    gemm_mhd_settings_t settings_a;
    if (EXIT_SUCCESS == gemm_mhd_read(argv[1], &dim0, &dim1, &transa, &lda, scalar, &ncomp, &settings_a, NULL)) {
      /* MHD stores physical layout: trans='N' is (m,k), trans='C'/'T' is (k,m) */
      if ('N' == transa || 'n' == transa) {
        m = dim0;
        if (3 >= argc) k = dim1;
        else k = atoi(argv[3]);
      }
      else {
        m = dim1;
        if (3 >= argc) k = dim0;
        else k = atoi(argv[3]);
      }
      if (4 >= argc) { /*transa from file*/
      }
      else transa = (0 == ta ? 'N' : 'T');
      if (6 >= argc) alpha = scalar[0];
      else alpha = atof(argv[6]);
      if (8 >= argc) { /*lda from file*/
      }
      else lda = atoi(argv[8]);
      if (10 >= argc) {
        ldc = (0 < settings_a.ldc) ? settings_a.ldc : m;
      }
      if (2 == ncomp) {
        complex_alpha[0] = scalar[0];
        complex_alpha[1] = scalar[1];
        complex_input = 1;
      }
      file_input |= 0x1;
    }
    if (0 == n) {
      size_t ncomp_b = 0;
      const int b_read = gemm_mhd_read(argv[2], &dim0, &dim1, &transb, &ldb, scalar, &ncomp_b, NULL, NULL);
      /* MHD stores physical layout: transb='N' is (k,n), transb='C'/'T' is (n,k) */
      if (EXIT_SUCCESS == b_read) {
        const GEMM_INT_TYPE bk = ('N' == transb || 'n' == transb) ? dim0 : dim1;
        const GEMM_INT_TYPE bn = ('N' == transb || 'n' == transb) ? dim1 : dim0;
        if (k == bk && ncomp_b == ncomp) {
          n = bn;
          if (5 >= argc) { /*transb from file*/
          }
          else transb = (0 == tb ? 'N' : 'T');
          if (7 >= argc) beta = scalar[0];
          else beta = atof(argv[7]);
          if (9 >= argc) { /*ldb from file*/
          }
          else ldb = atoi(argv[9]);
          if (2 == ncomp_b) {
            complex_beta[0] = scalar[0];
            complex_beta[1] = scalar[1];
          }
          file_input |= 0x2;
        }
        else {
          fprintf(stderr, "Mismatched files: A implies k=%i but B has k=%i\n", (int)k, (int)bk);
        }
      }
    }
  }

  /* Compute physical (stored) matrix dimensions. */
  a_rows = ('N' == transa || 'n' == transa) ? m : k;
  a_cols = ('N' == transa || 'n' == transa) ? k : m;
  b_rows = ('N' == transb || 'n' == transb) ? k : n;
  b_cols = ('N' == transb || 'n' == transb) ? n : k;

  if (1 > m || 1 > n || 1 > k || lda < a_rows || ldb < b_rows || ldc < m) {
    fprintf(stderr, "Invalid dimensions: m=%i n=%i k=%i lda=%i(>=%i) ldb=%i(>=%i) ldc=%i(>=%i)\n", (int)m, (int)n, (int)k, (int)lda,
      (int)a_rows, (int)ldb, (int)b_rows, (int)ldc, (int)m);
    result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) { /* Allocate matrices */
    const size_t nc = (0 != complex_input ? 2 : 1);
    a = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * nc * lda * a_cols);
    b = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * nc * ldb * b_cols);
    c = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * nc * ldc * n);
    c_ref = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * nc * ldc * n);
    if (NULL != a && NULL != b && NULL != c && NULL != c_ref) {
      if (0 == file_input || 0 == beta) {
        LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, c, m, n, ldc, scale);
      }
      else memset(c, 0, sizeof(GEMM_REAL_TYPE) * nc * ldc * n);
      memcpy(c_ref, c, sizeof(GEMM_REAL_TYPE) * nc * ldc * n);
    }
    else result = EXIT_FAILURE;
  }

  /* Print requested GEMM arguments (regardless of result code) */
  print_gemm(stdout, 0, &transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

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
    const double gflops = (0 != complex_input ? 8.0 : 2.0) * m * n * k * 1E-9;
    libxs_timer_tick_t start;
    double duration;
    /* Warmup: untimed call to trigger lazy initialization (JIT, etc.) */
    if (0 != complex_input) ZGEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
    else GEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
    start = libxs_timer_tick();
    for (i = 0; i < nrepeat; ++i) {
      if (0 != complex_input) ZGEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
      else GEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c, &ldc);
    }
    duration = libxs_timer_duration(start, libxs_timer_tick()) / nrepeat;
    printf("OZAKI GEMM: %.1f ms (%.1f GFLOPS/s)\n", 1E3 * duration, gflops / duration);
  }

  if (EXIT_SUCCESS == result) { /* Reference BLAS GEMM + diff */
    const GEMM_REAL_TYPE* const ga = (0 != complex_input) ? complex_alpha : &alpha;
    const GEMM_REAL_TYPE* const gb = (0 != complex_input) ? complex_beta : &beta;
    /* gemm_original: resolved via dlsym (LD_PRELOAD); GEMM_REAL: static --wrap */
    const gemm_function_t ref_gemm = (NULL != &gemm_original && NULL != gemm_original) ? gemm_original
                                                                                       : (NULL != &GEMM_REAL ? GEMM_REAL : NULL);
    if (NULL != ref_gemm) {
      const double gflops = (0 != complex_input ? 8.0 : 2.0) * m * n * k * 1E-9;
      libxs_timer_tick_t start;
      double duration;
      /* Warmup */
      if (0 != complex_input) ZGEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c_ref, &ldc);
      else ref_gemm(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c_ref, &ldc);
      start = libxs_timer_tick();
      for (i = 0; i < nrepeat; ++i) {
        if (0 != complex_input) ZGEMM(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c_ref, &ldc);
        else ref_gemm(&transa, &transb, &m, &n, &k, ga, a, &lda, b, &ldb, gb, c_ref, &ldc);
      }
      duration = libxs_timer_duration(start, libxs_timer_tick()) / nrepeat;
      printf("BLAS GEMM:  %.1f ms (%.1f GFLOPS/s)\n", 1E3 * duration, gflops / duration);
      {
        const size_t nc = (0 != complex_input ? 2 : 1);
        const int ldref = (int)(nc * ldc), ldtst = ldref;
        result = libxs_matdiff(&diff, LIBXS_DATATYPE(GEMM_REAL_TYPE), (int)(nc * m), n, c_ref, c, &ldref, &ldtst);
      }
      if (EXIT_SUCCESS == result) {
        diff.r = nrepeat;
        print_diff(stdout, &diff);
      }
    }
    else { /* fallback: checksum only (no reference GEMM available) */
      const size_t nc = (0 != complex_input ? 2 : 1);
      const int ldtst = (int)(nc * ldc);
      result = libxs_matdiff(
        &diff, LIBXS_DATATYPE(GEMM_REAL_TYPE), (int)(nc * m), n, NULL /*ref*/, c /*tst*/, NULL /*ldref*/, &ldtst);
      if (EXIT_SUCCESS == result) {
        printf("l1_tst=%f ncalls=%i\n", diff.l1_tst, nrepeat);
      }
    }
  }

  if (EXIT_SUCCESS == result && 0 != check) { /* Accuracy validation */
    const double epsilon = libxs_matdiff_epsilon(&gemm_diff);
    const double threshold = (0 < check) ? check : (sizeof(double) == sizeof(GEMM_REAL_TYPE) ? 1.0E-10 : 1.0E-3);
    if (threshold < epsilon) {
      fprintf(stderr, "CHECK: eps=%g exceeds threshold=%g\n", epsilon, threshold);
      result = EXIT_FAILURE;
    }
    else {
      fprintf(stderr, "CHECK: eps=%g (threshold=%g)\n", epsilon, threshold);
    }
  }

  libxs_finalize();
  free(c_ref);
  free(c);
  free(b);
  free(a);

  return result;
}
