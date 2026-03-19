/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_gemm.h>
#include <libxs_timer.h>
#include <libxs_rng.h>

#if defined(_OPENMP)
# include <omp.h>
#endif
#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) \
  && defined(LIBXS_PLATFORM_X86)
# include <mkl.h>
#endif


int main(int argc, char* argv[])
{
  const int m = (1 < argc ? atoi(argv[1]) : 23);
  const int n = (2 < argc ? atoi(argv[2]) : m);
  const int k = (3 < argc ? atoi(argv[3]) : m);
  const int batchsize = (4 < argc ? atoi(argv[4]) : 30000);
  const int nrepeat = (5 < argc ? atoi(argv[5]) : 3);
  const char *const env_check = getenv("CHECK");
  const double check = (NULL == env_check || 0 == *env_check) ? 0 : atof(env_check);
  const int lda = m, ldb = k, ldc = m;
  const int stride_a = lda * k, stride_b = ldb * n, stride_c = ldc * n;
  const double alpha = 1.0, beta = 1.0;
  const double gflops_per_op = 2.0 * m * n * k * 1E-9;
  double *a = NULL, *b = NULL, *c = NULL;
  libxs_gemm_config_t config;
  libxs_matdiff_t check_diff;
  double duration = 0;
  libxs_timer_tick_t t0, t1;
  int result = EXIT_SUCCESS, r;
#if defined(mkl_jit_create_dgemm)
  void* jitter = NULL;
  dgemm_jit_kernel_t jit_kernel = NULL;
#endif

  libxs_init();
  printf("gemm_strided: M=%d N=%d K=%d batch=%d nrepeat=%d\n",
    m, n, k, batchsize, nrepeat);

  a = (double*)malloc((size_t)stride_a * batchsize * sizeof(double));
  b = (double*)malloc((size_t)stride_b * batchsize * sizeof(double));
  c = (double*)malloc((size_t)stride_c * batchsize * sizeof(double));
  if (NULL == a || NULL == b || NULL == c) {
    fprintf(stderr, "ERROR: memory allocation failed\n");
    free(a); free(b); free(c);
    return EXIT_FAILURE;
  }

  LIBXS_MATRNG(int, double, 1.0, a, lda, (size_t)k * batchsize, lda, 1.0);
  LIBXS_MATRNG(int, double, 2.0, b, ldb, (size_t)n * batchsize, ldb, 1.0);
  LIBXS_MATRNG(int, double, 0.5, c, ldc, (size_t)n * batchsize, ldc, 1.0);

  memset(&config, 0, sizeof(config));
  config.flags = LIBXS_GEMM_FLAG_NOLOCK;
#if defined(mkl_jit_create_dgemm)
  if (MKL_JIT_SUCCESS == mkl_cblas_jit_create_dgemm(&jitter,
    MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS,
    m, n, k, alpha, lda, ldb, beta, ldc))
  {
    jit_kernel = mkl_jit_get_dgemm_ptr(jitter);
    config.dgemm_jit = (libxs_dgemm_jit_t)jit_kernel;
    config.jitter = jitter;
    printf("  MKL JIT kernel enabled\n");
  }
#endif

  /* warmup */
  libxs_gemm_strided(LIBXS_DATATYPE(double), "N", "N", m, n, k,
    &alpha, a, lda, stride_a, b, ldb, stride_b,
    &beta, c, ldc, stride_c, batchsize, &config);

  t0 = libxs_timer_tick();
  for (r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#   pragma omp parallel
    { const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      libxs_gemm_strided_task(LIBXS_DATATYPE(double), "N", "N", m, n, k,
        &alpha, a, lda, stride_a, b, ldb, stride_b,
        &beta, c, ldc, stride_c, batchsize, &config, tid, nthreads);
    }
#else
    libxs_gemm_strided(LIBXS_DATATYPE(double), "N", "N", m, n, k,
      &alpha, a, lda, stride_a, b, ldb, stride_b,
      &beta, c, ldc, stride_c, batchsize, &config);
#endif
  }
  t1 = libxs_timer_tick();
  duration = libxs_timer_duration(t0, t1);

  if (0 < duration) {
    printf("Total time : %.3f s (%d repeats)\n", duration, nrepeat);
    printf("Performance: %.1f GFLOPS/s\n",
      gflops_per_op * batchsize * nrepeat / duration);
  }

  if (0 != check) {
    libxs_matdiff(&check_diff, LIBXS_DATATYPE(double),
      m, n, NULL/*ref*/, c, NULL/*ldref*/, &ldc);
    printf("CHECK: l1_tst=%f\n", check_diff.l1_tst);
  }

#if defined(mkl_jit_create_dgemm)
  mkl_jit_destroy(jitter);
#endif
  free(a); free(b); free(c);
  libxs_finalize();
  return result;
}
