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


int main(int argc, char* argv[])
{
  const int m = (1 < argc ? atoi(argv[1]) : 23);
  const int n = (2 < argc ? atoi(argv[2]) : m);
  const int k = (3 < argc ? atoi(argv[3]) : m);
  const int batchsize = (4 < argc ? atoi(argv[4]) : 30000);
  const int nrepeat = (5 < argc ? atoi(argv[5]) : 3);
  const char *const env_check = getenv("CHECK");
  const double check = (NULL == env_check || 0 == *env_check) ? 0 : atof(env_check);
  const double alpha = 1.0, beta = (6 < argc ? atof(argv[6]) : 1.0);
  const int pad = (7 < argc ? atoi(argv[7]) : 0);
  const int lda = m + pad, ldb = k + pad, ldc = m + pad;
  const int stride_a = lda * k, stride_b = ldb * n, stride_c = ldc * n;
  const double gflops_per_op = 2.0 * m * n * k * 1E-9;
  double *a = NULL, *b = NULL, *c = NULL;
  int *ia = NULL, *ib = NULL, *ic = NULL;
  libxs_gemm_config_t config;
  libxs_matdiff_t check_diff;
  double duration = 0;
  libxs_timer_tick_t t0, t1;
  int result = EXIT_SUCCESS, r, i;

  libxs_init();
  printf("gemm_index: M=%d N=%d K=%d batch=%d nrepeat=%d\n",
    m, n, k, batchsize, nrepeat);

  a = (double*)malloc((size_t)stride_a * batchsize * sizeof(double));
  b = (double*)malloc((size_t)stride_b * batchsize * sizeof(double));
  c = (double*)malloc((size_t)stride_c * batchsize * sizeof(double));
  ia = (int*)malloc((size_t)batchsize * sizeof(int));
  ib = (int*)malloc((size_t)batchsize * sizeof(int));
  ic = (int*)malloc((size_t)batchsize * sizeof(int));
  if (NULL == a || NULL == b || NULL == c
    || NULL == ia || NULL == ib || NULL == ic)
  {
    fprintf(stderr, "ERROR: memory allocation failed\n");
    free(a); free(b); free(c);
    free(ia); free(ib); free(ic);
    return EXIT_FAILURE;
  }

  /* A: m rows filled per column, ld-padding rows set to SEED (sentinel) */
  LIBXS_MATRNG(int, double, 1.0, a, m, (size_t)k * batchsize, lda, 1.0);
  /* B: k rows filled per column, ld-padding rows set to SEED (sentinel) */
  LIBXS_MATRNG(int, double, 2.0, b, k, (size_t)n * batchsize, ldb, 1.0);
  /* C: m rows filled per column, ld-padding rows set to SEED (sentinel) */
  LIBXS_MATRNG(int, double, 0.5, c, m, (size_t)n * batchsize, ldc, 1.0);

  /* populate zero-based element-offset index arrays */
  for (i = 0; i < batchsize; ++i) {
    ia[i] = i * stride_a;
    ib[i] = i * stride_b;
    ic[i] = i * stride_c;
  }

  /* dispatch JIT kernel (MKL JIT or XSMM); falls through to BLAS/default */
  memset(&config, 0, sizeof(config));
  config.flags = LIBXS_GEMM_FLAG_NOLOCK;
  if (EXIT_SUCCESS == libxs_gemm_dispatch(&config,
    LIBXS_DATATYPE(double), 'N', 'N', m, n, k, lda, ldb, ldc,
    &alpha, &beta))
  {
    printf("  JIT kernel dispatched\n");
  }

  /* warmup */
  libxs_gemm_index(LIBXS_DATATYPE(double), "N", "N", m, n, k,
    &alpha, a, lda, ia, b, ldb, ib,
    &beta, c, ldc, ic,
    (int)sizeof(int)/*index_stride*/, 0/*index_base*/,
    batchsize, &config);

  t0 = libxs_timer_tick();
  for (r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#   pragma omp parallel
    { const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      libxs_gemm_index_task(LIBXS_DATATYPE(double), "N", "N", m, n, k,
        &alpha, a, lda, ia, b, ldb, ib,
        &beta, c, ldc, ic,
        (int)sizeof(int), 0/*index_base*/,
        batchsize, &config, tid, nthreads);
    }
#else
    libxs_gemm_index(LIBXS_DATATYPE(double), "N", "N", m, n, k,
      &alpha, a, lda, ia, b, ldb, ib,
      &beta, c, ldc, ic,
      (int)sizeof(int), 0/*index_base*/,
      batchsize, &config);
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
    /* verify ld-padding in first C-matrix is untouched */
    if (ldc > m) {
      int ci, cj;
      for (ci = 0; ci < n && EXIT_SUCCESS == result; ++ci) {
        for (cj = m; cj < ldc; ++cj) {
          if (0.5 != c[(size_t)ci * ldc + cj]) {
            fprintf(stderr, "FAILED: C ld-padding overwritten"
              " (col=%d row=%d)\n", ci, cj);
            result = EXIT_FAILURE;
          }
        }
      }
    }
    libxs_matdiff(&check_diff, LIBXS_DATATYPE(double),
      m, n, NULL/*ref*/, c, NULL/*ldref*/, &ldc);
    if (1 < batchsize) {
      libxs_matdiff_t d;
      libxs_matdiff(&d, LIBXS_DATATYPE(double),
        m, n, NULL/*ref*/, c + (size_t)(batchsize - 1) * stride_c,
        NULL/*ldref*/, &ldc);
      libxs_matdiff_combine(&check_diff, &d);
    }
    printf("checksum=%f\n", check_diff.l1_ref + check_diff.l1_tst);
  }

  libxs_gemm_release(&config);
  free(a); free(b); free(c);
  free(ia); free(ib); free(ic);
  libxs_finalize();
  return result;
}
