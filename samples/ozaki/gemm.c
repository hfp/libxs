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
  const int arg1 = (2 == argc ? atoi(argv[1]) : 0);
  int nrepeat = (0 < arg1 ? arg1 : 500);
  const GEMM_INT_TYPE m = (2 < argc ? atoi(argv[1]) : 23);
  const GEMM_INT_TYPE k = (3 < argc ? atoi(argv[3]) : m);
  const GEMM_INT_TYPE n = (2 < argc ? atoi(argv[2]) : k);
  const GEMM_INT_TYPE lda = (4 < argc ? atoi(argv[4]) : m);
  const GEMM_INT_TYPE ldb = (5 < argc ? atoi(argv[5]) : k);
  const GEMM_INT_TYPE ldc = (6 < argc ? atoi(argv[6]) : m);
  const GEMM_REAL_TYPE alpha = (7 < argc ? atof(argv[7]) : (ALPHA));
  const GEMM_REAL_TYPE beta = (8 < argc ? atof(argv[8]) : (BETA));
  const char transa = 'N', transb = 'N';
  const GEMM_INT_TYPE na = lda * k, nb = ldb * n, nc = ldc * n;
  GEMM_REAL_TYPE *const a = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * na);
  GEMM_REAL_TYPE *const b = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * nb);
  GEMM_REAL_TYPE *const c = (GEMM_REAL_TYPE*)malloc(sizeof(GEMM_REAL_TYPE) * nc);
  GEMM_REAL_TYPE scale = 1.0;
  int result = EXIT_SUCCESS, i;
  libxs_matdiff_info_t diff;

  assert(NULL != a && NULL != b && NULL != c);
  if (9 < argc) nrepeat = atoi(argv[9]);
  if (0 < nrepeat) scale /= nrepeat;

  printf(
    "gemm('%c', '%c', %i/*m*/, %i/*n*/, %i/*k*/,\n"
    "     %g/*alpha*/, %p/*a*/, %i/*lda*/,\n"
    "                 %p/*b*/, %i/*ldb*/,\n"
    "      %g/*beta*/, %p/*c*/, %i/*ldc*/)\n",
    transa, transb, m, n, k, alpha, (const void*)a, lda,
                                    (const void*)b, ldb,
                              beta, (const void*)c, ldc);

  LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 42, a, m, k, lda, scale);
  LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE, 24, b, k, n, ldb, scale);
  LIBXS_MATRNG(GEMM_INT_TYPE, GEMM_REAL_TYPE,  0, c, m, n, ldc, scale);

  { /* Call GEMM */
    const libxs_timer_tick_t start = libxs_timer_tick();
    for (i = 0; i < nrepeat; ++i) {
      GEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
    if (0 < nrepeat) {
      printf("Called %i times (%f s).\n", nrepeat,
        libxs_timer_duration(start, libxs_timer_tick()));
    }
    else fprintf(stderr, "Not executed!\n");
  }

  /* calculate final checksum */
  result = libxs_matdiff(&diff, LIBXS_DATATYPE(GEMM_REAL_TYPE), m, n,
      NULL/*ref*/, c/*tst*/, NULL/*ldref*/, &ldc);

  if (EXIT_SUCCESS == result) {
    printf("\n%f (check)\n", diff.l1_tst);
  }

  free(a);
  free(b);
  free(c);

  return result;
}
