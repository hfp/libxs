/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_rng.h>
#include <libxs_gemm.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#if !defined(ELEMTYPE)
# define ELEMTYPE double
#endif

#define EPSILON(T) LIBXS_CONCATENATE(EPSILON_, T)
#define EPSILON_double 1E-10
#define EPSILON_float 1E-4

#define N 4
#define K 3


static int check_upper(const char* label, const ELEMTYPE* c, int ldc,
  const ELEMTYPE* ref, int ldref, int n, double tol)
{
  int i, j;
  for (j = 0; j < n; ++j) {
    for (i = 0; i <= j; ++i) {
      const double diff = fabs((double)c[i + (size_t)j * ldc]
        - (double)ref[i + (size_t)j * ldref]);
      if (diff > tol) {
        fprintf(stderr, "%s: MISMATCH at (%d,%d): got %g expect %g\n",
          label, i, j, (double)c[i + (size_t)j * ldc],
          (double)ref[i + (size_t)j * ldref]);
        return EXIT_FAILURE;
      }
    }
  }
  return EXIT_SUCCESS;
}


static void ref_syr2k(int n, int k,
  double alpha, const ELEMTYPE* a, int lda,
  const ELEMTYPE* b, int ldb,
  double beta, ELEMTYPE* c, int ldc)
{
  int i, j, p;
  for (j = 0; j < n; ++j) {
    for (i = 0; i <= j; ++i) {
      double sum = 0.0;
      for (p = 0; p < k; ++p) {
        sum += (double)a[i + (size_t)p * lda] * (double)b[j + (size_t)p * ldb]
             + (double)b[i + (size_t)p * ldb] * (double)a[j + (size_t)p * lda];
      }
      c[i + (size_t)j * ldc] = (ELEMTYPE)(beta * (double)c[i + (size_t)j * ldc]
        + alpha * sum);
    }
  }
}


static void ref_syrk(int n, int k,
  double alpha, const ELEMTYPE* a, int lda,
  double beta, ELEMTYPE* c, int ldc)
{
  int i, j, p;
  for (j = 0; j < n; ++j) {
    for (i = 0; i <= j; ++i) {
      double sum = 0.0;
      for (p = 0; p < k; ++p) {
        sum += (double)a[i + (size_t)p * lda] * (double)a[j + (size_t)p * lda];
      }
      c[i + (size_t)j * ldc] = (ELEMTYPE)(beta * (double)c[i + (size_t)j * ldc]
        + alpha * sum);
    }
  }
}


int main(void)
{
  int result = EXIT_SUCCESS;
  const libxs_gemm_config_t* cfg;
  ELEMTYPE a[N * K], b[N * K];
  ELEMTYPE c[N * N], cref[N * N];
  const double alpha = 0.5, beta = 0.25;

  LIBXS_MATRNG(int, ELEMTYPE, 0, a, N, K, N, 1.0);
  LIBXS_MATRNG(int, ELEMTYPE, 0, b, N, K, N, 1.0);

  /* --- Test 1: libxs_syr2k --- */
  cfg = libxs_syr2k_dispatch(LIBXS_DATATYPE(ELEMTYPE),
    N, K, N, N, N, NULL);

  LIBXS_MATRNG(int, ELEMTYPE, 0, c, N, N, N, 0.5);
  memcpy(cref, c, sizeof(c));
  ref_syr2k(N, K, alpha, a, N, b, N, beta, cref, N);
  libxs_syr2k(cfg, 'U', alpha, beta, a, b, c);

  if (EXIT_SUCCESS != check_upper("syr2k", c, N, cref, N, N, EPSILON(ELEMTYPE))) {
    result = EXIT_FAILURE;
  }

  /* --- Test 2: libxs_syrk --- */
  cfg = libxs_syrk_dispatch(LIBXS_DATATYPE(ELEMTYPE),
    N, K, N, N, NULL);

  LIBXS_MATRNG(int, ELEMTYPE, 0, c, N, N, N, 0.3);
  memcpy(cref, c, sizeof(c));
  ref_syrk(N, K, alpha, a, N, beta, cref, N);
  libxs_syrk(cfg, 'U', alpha, beta, a, c);

  if (EXIT_SUCCESS != check_upper("syrk", c, N, cref, N, N, EPSILON(ELEMTYPE))) {
    result = EXIT_FAILURE;
  }

  /* --- Test 3: beta=0 (fresh output) --- */
  LIBXS_MATRNG(int, ELEMTYPE, 0, c, N, N, N, 99.0);
  memset(cref, 0, sizeof(cref));
  ref_syr2k(N, K, 1.0, a, N, b, N, 0.0, cref, N);

  cfg = libxs_syr2k_dispatch(LIBXS_DATATYPE(ELEMTYPE),
    N, K, N, N, N, NULL);
  libxs_syr2k(cfg, 'U', 1.0, 0.0, a, b, c);

  if (EXIT_SUCCESS != check_upper("syr2k-beta0", c, N, cref, N, N, EPSILON(ELEMTYPE))) {
    result = EXIT_FAILURE;
  }

  /* --- Test 4: non-square leading dimensions --- */
  { const int lda = N + 2, ldb = N + 1, ldc = N + 3;
    ELEMTYPE abig[lda * K], bbig[ldb * K];
    ELEMTYPE cbig[ldc * N], crefbig[ldc * N];

    LIBXS_MATRNG(int, ELEMTYPE, 0, abig, N, K, lda, 1.0);
    LIBXS_MATRNG(int, ELEMTYPE, 0, bbig, N, K, ldb, 1.0);
    LIBXS_MATRNG(int, ELEMTYPE, 0, cbig, N, N, ldc, 0.4);
    memcpy(crefbig, cbig, sizeof(cbig));
    ref_syr2k(N, K, alpha, abig, lda, bbig, ldb, beta, crefbig, ldc);

    cfg = libxs_syr2k_dispatch(LIBXS_DATATYPE(ELEMTYPE),
      N, K, lda, ldb, ldc, NULL);
    libxs_syr2k(cfg, 'U', alpha, beta, abig, bbig, cbig);

    if (EXIT_SUCCESS != check_upper("syr2k-ld", cbig, ldc, crefbig, ldc, N, EPSILON(ELEMTYPE))) {
      result = EXIT_FAILURE;
    }
  }

  if (EXIT_SUCCESS == result) {
    fprintf(stdout, "PASS (%s)\n",
      LIBXS_STRINGIFY(LIBXS_TYPESYMBOL(ELEMTYPE)));
  }
  return result;
}
