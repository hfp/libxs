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
  const GEMM_INT_TYPE m = (1 < argc ? atoi(argv[1]) : 257);
  const GEMM_INT_TYPE k = (2 < argc ? atoi(argv[2]) : m);
  const GEMM_INT_TYPE n = (3 < argc ? atoi(argv[3]) : k);
  const int ta = (4 < argc ? atoi(argv[4]) : 0);
  const int tb = (5 < argc ? atoi(argv[5]) : 0);
  const GEMM_REAL_TYPE alpha = (6 < argc ? atof(argv[6]) : (ALPHA));
  const GEMM_REAL_TYPE beta = (7 < argc ? atof(argv[7]) : (BETA));
  const GEMM_INT_TYPE lda = (8 < argc ? atoi(argv[8]) : m);
  const GEMM_INT_TYPE ldb = (9 < argc ? atoi(argv[9]) : k);
  const GEMM_INT_TYPE ldc = (10 < argc ? atoi(argv[10]) : m);
  const char transa = (0 == ta ? 'N' : 'T'), transb = (0 == tb ? 'N' : 'T');
  GEMM_REAL_TYPE *const a = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * lda * k);
  GEMM_REAL_TYPE *const b = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * ldb * n);
  GEMM_REAL_TYPE *const c = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * ldc * n);
  const GEMM_REAL_TYPE scale = (1 < nrepeat ? (1.0 / nrepeat) : 1);
  int result = EXIT_SUCCESS, i;
  libxs_matdiff_info_t diff;

  assert(NULL != a && NULL != b && NULL != c);
  printf(
    "gemm('%c', '%c', %lli/*m*/, %lli/*n*/, %lli/*k*/,\n"
    "     %g/*alpha*/, %p/*a*/, %lli/*lda*/,\n"
    "                 %p/*b*/, %lli/*ldb*/,\n"
    "      %g/*beta*/, %p/*c*/, %lli/*ldc*/)\n",
    transa, transb, (long long int)m, (long long int)n, (long long int)k,
      alpha, (const void*)a, (long long int)lda,
             (const void*)b, (long long int)ldb,
       beta, (const void*)c, (long long int)ldc);

  LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, a, m, k, lda, scale);
  LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, b, k, n, ldb, scale);
  LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 0, c, m, n, ldc, scale);

  { /* Call GEMM */
    libxs_timer_tick_t start;
    int ncalls = nrepeat;
    if (1 < nrepeat) { /* peel one warmup call */
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

  { /* calculate final checksum */
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
