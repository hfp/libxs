/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_gemm.h>
#include <libxs/libxs_rng.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

/* panels of a tall A, i.e., windows (lda > m) as a tall-and-skinny GEMM cuts them */
#define TEST_M 256
#define TEST_N 16
#define TEST_K 16
#define TEST_NPANELS 512
/* a race shows up by chance, hence the batch runs more than once */
#define TEST_NROUNDS 8


/**
 * A batch split over threads must equal the same batch run as one task, bit
 * for bit, since each element is the same call on the same data either way.
 * The BLAS entry point serves every element here: a BLAS that is unsafe to
 * call concurrently (a sequential build without locking) corrupts results
 * under the split only, which a single-threaded check never sees.
 */
static int test_blas_split(void)
{
  const int m = TEST_M * TEST_NPANELS, lda = m, ldb = TEST_K, ldc = m;
  const int stride_a = TEST_M, stride_b = 0, stride_c = TEST_M;
  const size_t na = (size_t)lda * TEST_K, nb = (size_t)ldb * TEST_N;
  const size_t nc = (size_t)ldc * TEST_N;
  const double alpha = 1.0, beta = 1.0;
  double *const a = (double*)malloc(na * sizeof(double));
  double *const b = (double*)malloc(nb * sizeof(double));
  double *const c = (double*)malloc(nc * sizeof(double));
  double *const ref = (double*)malloc(nc * sizeof(double));
  double *const tst = (double*)malloc(nc * sizeof(double));
  libxs_gemm_config_t config;
  int result = EXIT_SUCCESS, nthreads = 1, r;
  if (NULL == a || NULL == b || NULL == c || NULL == ref || NULL == tst) {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    LIBXS_MEMZERO(&config);
    if (0 == libxs_gemm_dispatch_cpy(&config, LIBXS_DATATYPE_F64, 'N', 'N',
      TEST_M, TEST_N, TEST_K, lda, ldb, ldc, &alpha, &beta, NULL))
    {
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    /* the BLAS entry point is under test, whichever kernel dispatch found */
    config.dgemm_jit = NULL; config.sgemm_jit = NULL;
    config.xgemm = NULL; config.jitter = NULL;
    config.flags = LIBXS_GEMM_FLAG_NOLOCK;
    LIBXS_MATRNG(int, double, 0, a, m, TEST_K, lda, 1.0);
    LIBXS_MATRNG(int, double, 0, b, TEST_K, TEST_N, ldb, 1.0);
    LIBXS_MATRNG(int, double, 0, c, m, TEST_N, ldc, 1.0);
    memcpy(ref, c, nc * sizeof(double));
    libxs_gemm_index(a, &stride_a, b, &stride_b, ref, &stride_c,
      0/*index_stride*/, 0/*index_base*/, TEST_NPANELS, &config);
#if defined(_OPENMP)
    nthreads = omp_get_max_threads();
#endif
  }
  for (r = 0; r < TEST_NROUNDS && EXIT_SUCCESS == result; ++r) {
    size_t i, ndiff = 0;
    memcpy(tst, c, nc * sizeof(double));
#if defined(_OPENMP)
#   pragma omp parallel num_threads(nthreads)
    { libxs_gemm_index_task(a, &stride_a, b, &stride_b, tst, &stride_c,
        0/*index_stride*/, 0/*index_base*/, TEST_NPANELS, &config,
        omp_get_thread_num(), omp_get_num_threads());
    }
#else
    libxs_gemm_index(a, &stride_a, b, &stride_b, tst, &stride_c,
      0/*index_stride*/, 0/*index_base*/, TEST_NPANELS, &config);
#endif
    for (i = 0; i < nc; ++i) ndiff += (ref[i] != tst[i] ? 1 : 0);
    if (0 != ndiff) {
      fprintf(stderr, "FAIL: %i threads, round %i: %lu of %lu elements"
        " differ from one task (BLAS unsafe to call concurrently?)\n",
        nthreads, r, (unsigned long)ndiff, (unsigned long)nc);
      result = EXIT_FAILURE;
    }
  }
  free(a); free(b); free(c); free(ref); free(tst);
  return result;
}


int main(void)
{
  int result;
  libxs_init(); /* header-only: no constructor, which warns about such a BLAS */
  result = test_blas_split();
  libxs_finalize();
  return result;
}
