/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#if !defined(GEMM)
# define GEMM dgemm_
#endif
#if !defined(ALPHA)
# define ALPHA 1
#endif
#if !defined(BETA)
# define BETA 1
#endif


void init(int seed, GEMM_REAL_TYPE* dst, GEMM_INT_TYPE nrows, GEMM_INT_TYPE ncols, GEMM_INT_TYPE ld, GEMM_REAL_TYPE scale);
GEMM_REAL_TYPE checksum(const GEMM_REAL_TYPE* src, GEMM_INT_TYPE nrows, GEMM_INT_TYPE ncols, GEMM_INT_TYPE ld);


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
  const GEMM_REAL_TYPE scale = 1.0;
  int i;

  assert(NULL != a && NULL != b && NULL != c);
  if (9 < argc) nrepeat = atoi(argv[9]);

  printf(
    "gemm('%c', '%c', %i/*m*/, %i/*n*/, %i/*k*/,\n"
    "     %g/*alpha*/, %p/*a*/, %i/*lda*/,\n"
    "                  %p/*b*/, %i/*ldb*/,\n"
    "      %g/*beta*/, %p/*c*/, %i/*ldc*/)\n",
    transa, transb, m, n, k, alpha, (const void*)a, lda,
                                    (const void*)b, ldb,
                              beta, (const void*)c, ldc);

  init(42, a, m, k, lda, scale);
  init(24, b, k, n, ldb, scale);
  init( 0, c, m, n, ldc, scale);

  { /* Call GEMM */
#if defined(_OPENMP)
    const double start = omp_get_wtime();
#endif
    for (i = 0; i < nrepeat; ++i) {
      GEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
#if defined(_OPENMP)
    if (0 < nrepeat) printf("Called %i times (%f s).\n", nrepeat, omp_get_wtime() - start);
#else
    if (0 < nrepeat) printf("Called %i times.\n", nrepeat);
#endif
    else fprintf(stderr, "Not executed!\n");
  }

  /* calculate final checksum */
  printf("\n%f (check)\n", checksum(c, m, n, ldc));

  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}


void init(int seed, GEMM_REAL_TYPE* dst, GEMM_INT_TYPE nrows, GEMM_INT_TYPE ncols, GEMM_INT_TYPE ld, GEMM_REAL_TYPE scale)
{
  const GEMM_REAL_TYPE seed1 = scale * (seed + 1);
  GEMM_INT_TYPE i = 0;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    GEMM_INT_TYPE j = 0;
    for (; j < nrows; ++j) {
      const GEMM_INT_TYPE k = i * ld + j;
      dst[k] = (GEMM_REAL_TYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const GEMM_INT_TYPE k = i * ld + j;
      dst[k] = (GEMM_REAL_TYPE)seed;
    }
  }
}


GEMM_REAL_TYPE checksum(const GEMM_REAL_TYPE* src, GEMM_INT_TYPE nrows, GEMM_INT_TYPE ncols, GEMM_INT_TYPE ld)
{
  GEMM_INT_TYPE i, j;
  GEMM_REAL_TYPE result = 0, comp = 0;
  for (i = 0; i < ncols; ++i) {
    for (j = 0; j < nrows; ++j) {
      const GEMM_REAL_TYPE v = src[i * ld + j];
      const GEMM_REAL_TYPE x = (0 <= v ? v : -v) - comp;
      const GEMM_REAL_TYPE y = result + x;
      comp = (y - result) - x;
      result = y;
    }
  }
  return result;
}
