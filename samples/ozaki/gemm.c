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
#include <libxs_mhd.h>
#include <libxs_rng.h>

#if !defined(ALPHA)
# define ALPHA 1
#endif
#if !defined(BETA)
# define BETA 1
#endif


int main(int argc, char* argv[])
{
  const char *const nrepeat_env = getenv("NREPEAT");
  const int nrep = (NULL == nrepeat_env ? 1 : atoi(nrepeat_env));
  const int nrepeat = (0 < nrep ? nrep : 1);
  GEMM_INT_TYPE m = (1 < argc ? atoi(argv[1]) : 257);
  GEMM_INT_TYPE n = (2 < argc ? atoi(argv[2]) : m);
  GEMM_INT_TYPE k = (3 < argc ? atoi(argv[3]) : m);
  const int ta = (4 < argc ? atoi(argv[4]) : 0);
  const int tb = (5 < argc ? atoi(argv[5]) : 0);
  GEMM_REAL_TYPE alpha = (6 < argc ? atof(argv[6]) : (ALPHA));
  GEMM_REAL_TYPE beta = (7 < argc ? atof(argv[7]) : (BETA));
  GEMM_INT_TYPE lda = (8 < argc ? atoi(argv[8]) : (0 == ta ? m : k));
  GEMM_INT_TYPE ldb = (9 < argc ? atoi(argv[9]) : (0 == tb ? k : n));
  GEMM_INT_TYPE ldc = (10 < argc ? atoi(argv[10]) : m);
  char transa = (0 == ta ? 'N' : 'T'), transb = (0 == tb ? 'N' : 'T');
  const GEMM_REAL_TYPE scale = (1 < nrepeat ? (1.0 / nrepeat) : 1);
  int result = EXIT_SUCCESS, file_input = 0, i;
  libxs_mhd_info_t info_a = { 2, 0, LIBXS_DATATYPE_UNKNOWN, 0 };
  libxs_mhd_info_t info_b = { 2, 0, LIBXS_DATATYPE_UNKNOWN, 0 };
  GEMM_REAL_TYPE *a = NULL, *b = NULL, *c = NULL;
  GEMM_INT_TYPE a_rows, a_cols, b_rows, b_cols;
  libxs_matdiff_info_t diff;

  if (2 < argc && 0 == m) { /* Indicate filename(s) */
    char extension[
      sizeof(char) /*trans*/ +
      sizeof(GEMM_INT_TYPE) /*ld*/+
      sizeof(GEMM_REAL_TYPE) /* alpha/beta */
    ];
    size_t size[2], extension_size = sizeof(extension);
    result |= libxs_mhd_read_header(argv[1], strlen(argv[1]),
      argv[1], &info_a, size, extension, &extension_size);
    if (EXIT_SUCCESS == result
      && 2 == info_a.ndims && 1 == info_a.ncomponents
      && LIBXS_DATATYPE(GEMM_REAL_TYPE) == info_a.type
      && sizeof(extension) == extension_size)
    {
      m = ldc = (int)size[0]; k = (int)size[1];
      transa = *(const char*)extension;
      lda = *(const GEMM_INT_TYPE*)(extension + sizeof(char));
      alpha = *(const GEMM_REAL_TYPE*)(extension
        + sizeof(char) + sizeof(GEMM_INT_TYPE));
      file_input |= 0x1;
    }
    if (0 == n) {
      result |= libxs_mhd_read_header(argv[2], strlen(argv[2]),
        argv[2], &info_b, size, extension, &extension_size);
      if (EXIT_SUCCESS == result && k == (int)size[0]
        && 2 == info_b.ndims && 1 == info_b.ncomponents
        && LIBXS_DATATYPE(GEMM_REAL_TYPE) == info_b.type
        && sizeof(extension) == extension_size)
      {
        n = (int)size[1]; transb = *(char*)extension;
        ldb = *(const GEMM_INT_TYPE*)(extension + sizeof(char));
        beta = *(const GEMM_REAL_TYPE*)(extension
          + sizeof(char) + sizeof(GEMM_INT_TYPE));
        file_input |= 0x2;
      }
    }
  }

  /* Compute physical (stored) matrix dimensions */
  a_rows = ('N' == transa || 'n' == transa) ? m : k;
  a_cols = ('N' == transa || 'n' == transa) ? k : m;
  b_rows = ('N' == transb || 'n' == transb) ? k : n;
  b_cols = ('N' == transb || 'n' == transb) ? n : k;

  if (1 > m || 1 > n || 1 > k || lda < a_rows || ldb < b_rows || ldc < m) {
    result = EXIT_FAILURE;
  }

  if (EXIT_SUCCESS == result) { /* Allocate matrices */
    a = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * lda * a_cols);
    b = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * ldb * b_cols);
    c = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * ldc * n);
    if (NULL != a && NULL != b && NULL != c) {
      if (0 == file_input) {
        LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, c, m, n, ldc, scale);
      }
      else memset(c, 0, sizeof(GEMM_REAL_TYPE) * ldc * n);
    }
    else result = EXIT_FAILURE;
  }

  /* Print requested GEMM arguments (regardless of result code) */
  print_gemm(stdout, &transa, &transb, &m, &n, &k,
    &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

  if (EXIT_SUCCESS == result) { /* Initialize A-matrix */
    if (0x1 & file_input) {
      size_t size[2], ld[2];
      size[0] = a_rows; size[1] = a_cols;
      ld[0] = lda; ld[1] = a_cols;
      result = libxs_mhd_read(argv[1], NULL/*offset*/, size, ld,
        &info_a, a, NULL/*handler_info*/, NULL/*handler*/);
    }
    else {
      LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, a, a_rows, a_cols, lda, scale);
    }
  }

  if (EXIT_SUCCESS == result) { /* Initialize B-matrix */
    if (0x2 & file_input) {
      size_t size[2], ld[2];
      size[0] = b_rows; size[1] = b_cols;
      ld[0] = ldb; ld[1] = b_cols;
      result = libxs_mhd_read(argv[2], NULL/*offset*/, size, ld,
        &info_b, b, NULL/*handler_info*/, NULL/*handler*/);
    }
    else {
      LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, b, b_rows, b_cols, ldb, scale);
    }
  }

  if (EXIT_SUCCESS == result) { /* Call GEMM */
    libxs_timer_tick_t start;
    int ncalls = nrepeat;
    if (1 < nrepeat) { /* Peel one warmup call */
      GEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      --ncalls;
    }
    start = libxs_timer_tick();
    for (i = 0; i < ncalls; ++i) {
      GEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
    printf("Called %i times (%f s/call).\n", nrepeat,
      libxs_timer_duration(start, libxs_timer_tick()) / ncalls);
  }

  if (EXIT_SUCCESS == result) { /* Calculate final checksum */
    const int ldtst = (int)ldc;
    result = libxs_matdiff(&diff, LIBXS_DATATYPE(GEMM_REAL_TYPE), m, n,
        NULL/*ref*/, c/*tst*/, NULL/*ldref*/, &ldtst);
  }

  if (EXIT_SUCCESS == result) {
    printf("\n%f (check)\n", diff.l1_tst);
  }

  free(a);
  free(b);
  free(c);

  return result;
}
