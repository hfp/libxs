/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
#include <libxs.h>
#include <stdlib.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif
#if !defined(REFERENCE_BLAS)
# define REFERENCE_BLAS LIBXS_GEMM_SYMBOL
#endif
#if !defined(LIBXS_BLAS)
# define LIBXS_BLAS LIBXS_XGEMM_SYMBOL
/*# define LIBXS_BLAS LIBXS_YGEMM_SYMBOL*/
#endif


LIBXS_GEMM_SYMBOL_DECL(LIBXS_GEMM_CONST, REAL_TYPE);


LIBXS_INLINE LIBXS_RETARGETABLE void init(libxs_blasint seed, REAL_TYPE *LIBXS_RESTRICT dst,
  libxs_blasint nrows, libxs_blasint ncols, libxs_blasint ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    libxs_blasint j = 0;
    for (; j < nrows; ++j) {
      const libxs_blasint k = i * ld + j;
      dst[k] = (REAL_TYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxs_blasint k = i * ld + j;
      dst[k] = (REAL_TYPE)seed;
    }
  }
}


int main(void)
{
#if !defined(__BLAS) || (0 != __BLAS)
  libxs_blasint m[]                   = { 0, 0, 1, 1, 3, 3, 1,   64,  64,    16,    16, 350, 350, 350, 350, 350,  5, 10, 12, 20,   32,    9 };
  libxs_blasint n[]                   = { 0, 1, 1, 1, 3, 1, 3,    8, 239, 13824, 65792,  16,   1,  25,   4,   9, 13,  1, 10,  6,   33,    9 };
  libxs_blasint k[]                   = { 0, 1, 1, 1, 3, 2, 2,   64,  64,    16,    16,  20,   1,  35,   4,  10, 70,  1, 12,  6,  192, 1742 };
  libxs_blasint lda[]                 = { 0, 1, 1, 1, 3, 3, 1,   64,  64,    16,    16, 350, 350, 350, 350, 350,  5, 22, 22, 22,   32,    9 };
  libxs_blasint ldb[]                 = { 0, 1, 1, 1, 3, 2, 2, 9216, 240,    16,    16,  35,  35,  35,  35,  35, 70,  1, 20,  8, 2048, 1742 };
  libxs_blasint ldc[]                 = { 0, 1, 0, 1, 3, 3, 1, 4096, 240,    16,    16, 350, 350, 350, 350, 350,  5, 22, 12, 20, 2048,    9 };
  LIBXS_GEMM_CONST REAL_TYPE alpha[]  = { 1, 1, 1, 1, 1, 1, 1,    1,   1,     1,     1,   1,   1,   1,   1,   1,  1,  1,  1,  1,    1,    1 };
  LIBXS_GEMM_CONST REAL_TYPE beta[]   = { 0, 1, 0, 1, 1, 0, 0,    0,   1,     0,     0,   0,   0,   0,   0,   0,  0,  0,  0,  0,    0,    0 };
  LIBXS_GEMM_CONST char transa = 'N', transb = 'N';
  const int begin = 3, end = sizeof(m) / sizeof(*m);
  libxs_blasint max_size_a = 0, max_size_b = 0, max_size_c = 0;
  REAL_TYPE *a = 0, *b = 0, *c = 0, *d = 0;
  libxs_matdiff_info diff = { 0 };
  int test;

  for (test = begin; test < end; ++test) {
    const libxs_blasint size_a = lda[test] * k[test], size_b = ldb[test] * n[test], size_c = ldc[test] * n[test];
    assert(m[test] <= lda[test] && k[test] <= ldb[test] && m[test] <= ldc[test]);
    max_size_a = LIBXS_MAX(max_size_a, size_a);
    max_size_b = LIBXS_MAX(max_size_b, size_b);
    max_size_c = LIBXS_MAX(max_size_c, size_c);
  }

  a = (REAL_TYPE*)libxs_malloc((size_t)(max_size_a * sizeof(REAL_TYPE)));
  b = (REAL_TYPE*)libxs_malloc((size_t)(max_size_b * sizeof(REAL_TYPE)));
  c = (REAL_TYPE*)libxs_malloc((size_t)(max_size_c * sizeof(REAL_TYPE)));
  d = (REAL_TYPE*)libxs_malloc((size_t)(max_size_c * sizeof(REAL_TYPE)));
  assert(0 != a && 0 != b && 0 != c && 0 != d);

  init(42, a, max_size_a, 1, max_size_a, 1.0);
  init(24, b, max_size_b, 1, max_size_b, 1.0);
  init( 0, c, max_size_c, 1, max_size_c, 1.0);
  init( 0, d, max_size_c, 1, max_size_c, 1.0);

  for (test = begin; test < end; ++test) {
    libxs_matdiff_info diff_test;

    LIBXS_BLAS(REAL_TYPE)(&transa, &transb, m + test, n + test, k + test,
      alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);

    REFERENCE_BLAS(REAL_TYPE)(&transa, &transb, m + test, n + test, k + test,
      alpha + test, a, lda + test, b, ldb + test, beta + test, d, ldc + test);

    if (EXIT_SUCCESS == libxs_matdiff(LIBXS_DATATYPE(REAL_TYPE), m[test], n[test], d, c, ldc + test, ldc + test, &diff_test)) {
      libxs_matdiff_reduce(&diff, &diff_test);
    }
  }

  libxs_free(a);
  libxs_free(b);
  libxs_free(c);
  libxs_free(d);

  if (1000.0 * diff.normf_rel <= 1.0) {
    return EXIT_SUCCESS;
  }
  else {
# if defined(_DEBUG)
    fprintf(stderr, "diff: L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
# endif
    return EXIT_FAILURE;
  }
#else
# if defined(_DEBUG)
  fprintf(stderr, "Warning: skipped the test due to missing BLAS support!\n");
# endif
  return EXIT_SUCCESS;
#endif
}

