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

/* Fisher-Yates shuffle of a pointer array (in-place). */
#define SHUFFLE_PTRS(ARRAY, COUNT) do { \
  size_t shuffle_i_; \
  for (shuffle_i_ = (COUNT); 1 < shuffle_i_; --shuffle_i_) { \
    const size_t shuffle_j_ = libxs_rng_u32((unsigned int)shuffle_i_); \
    void* shuffle_t_ = (ARRAY)[shuffle_i_ - 1]; \
    (ARRAY)[shuffle_i_ - 1] = (ARRAY)[shuffle_j_]; \
    (ARRAY)[shuffle_j_] = shuffle_t_; \
  } \
} while(0)


int main(int argc, char* argv[])
{
  const int m = (1 < argc ? atoi(argv[1]) : 23);
  const int n = (2 < argc ? atoi(argv[2]) : m);
  const int k = (3 < argc ? atoi(argv[3]) : m);
  const int batchsize = (4 < argc ? atoi(argv[4]) : 30000);
  const int nrepeat = (5 < argc ? atoi(argv[5]) : 3);
  const int dup_mode = (6 < argc ? atoi(argv[6]) : 0);
  const char *const env_check = getenv("CHECK");
  const double check = (NULL == env_check || 0 == *env_check) ? 0 : atof(env_check);
  const double alpha = 1.0, beta = (7 < argc ? atof(argv[7]) : 1.0);
  const int pad = (8 < argc ? atoi(argv[8]) : 0);
  const int lda = m + pad, ldb = k + pad, ldc = m + pad;
  const size_t asize = (size_t)lda * k;
  const size_t bsize = (size_t)ldb * n;
  const size_t csize = (size_t)ldc * n;
  const double gflops_per_op = 2.0 * m * n * k * 1E-9;
  int nunique = batchsize, r;
  double *a_data = NULL, *b_data = NULL, *c_data = NULL;
  const void **a_ptrs = NULL, **b_ptrs = NULL;
  void **c_ptrs = NULL;
  libxs_gemm_config_t config;
  libxs_matdiff_t check_diff;
  double duration = 0;
  libxs_timer_tick_t t0, t1;
  int result = EXIT_SUCCESS;

  libxs_init();

  /* duplicate mode: 0=none, 1=half-sorted, 2=half-shuffled */
  if (1 <= dup_mode) {
    nunique = (batchsize + 1) / 2;
    if (0 >= nunique) nunique = 1;
  }

  printf("gemm_batch: M=%d N=%d K=%d batch=%d (unique_C=%d) nrepeat=%d dup=%d\n",
    m, n, k, batchsize, nunique, nrepeat, dup_mode);

  a_data = (double*)malloc(asize * nunique * sizeof(double));
  b_data = (double*)malloc(bsize * nunique * sizeof(double));
  c_data = (double*)malloc(csize * nunique * sizeof(double));
  a_ptrs = (const void**)malloc((size_t)batchsize * sizeof(void*));
  b_ptrs = (const void**)malloc((size_t)batchsize * sizeof(void*));
  c_ptrs = (void**)malloc((size_t)batchsize * sizeof(void*));
  if (NULL == a_data || NULL == b_data || NULL == c_data
    || NULL == a_ptrs || NULL == b_ptrs || NULL == c_ptrs)
  {
    fprintf(stderr, "ERROR: memory allocation failed\n");
    free(a_data); free(b_data); free(c_data);
    free(a_ptrs); free(b_ptrs); free(c_ptrs);
    return EXIT_FAILURE;
  }

  /* A: m rows filled per column, ld-padding rows set to SEED (sentinel) */
  LIBXS_MATRNG(int, double, 1.0, a_data, m, (size_t)k * nunique, lda, 1.0);
  /* B: k rows filled per column, ld-padding rows set to SEED (sentinel) */
  LIBXS_MATRNG(int, double, 2.0, b_data, k, (size_t)n * nunique, ldb, 1.0);
  /* C: m rows filled per column, ld-padding rows set to SEED (sentinel) */
  LIBXS_MATRNG(int, double, 0.5, c_data, m, (size_t)n * nunique, ldc, 1.0);

  { int i;
    for (i = 0; i < batchsize; ++i) {
      const int idx = i % nunique;
      a_ptrs[i] = a_data + idx * asize;
      b_ptrs[i] = b_data + idx * bsize;
      c_ptrs[i] = c_data + idx * csize;
    }
  }

  /* shuffle C-pointers to stress the lock-forward path */
  if (2 <= dup_mode) {
    SHUFFLE_PTRS(c_ptrs, (size_t)batchsize);
    printf("  C-pointers shuffled (lock-forward stress test)\n");
  }
  else if (1 <= dup_mode) {
    printf("  C-pointers sorted (duplicates are consecutive)\n");
  }

  /* configure locking: enable locks when duplicates exist */
  memset(&config, 0, sizeof(config));
  config.flags = (0 < dup_mode)
    ? LIBXS_GEMM_FLAGS_DEFAULT : LIBXS_GEMM_FLAG_NOLOCK;
  if (EXIT_SUCCESS == libxs_gemm_dispatch(&config,
    LIBXS_DATATYPE(double), 'N', 'N', m, n, k, lda, ldb, ldc,
    &alpha, &beta))
  {
    printf("  JIT kernel dispatched\n");
  }

  /* warmup */
  libxs_gemm_batch(LIBXS_DATATYPE(double), "N", "N", m, n, k,
    &alpha, a_ptrs, lda, b_ptrs, ldb,
    &beta, c_ptrs, ldc, batchsize, &config);

  t0 = libxs_timer_tick();
  for (r = 0; r < nrepeat; ++r) {
#if defined(_OPENMP)
#   pragma omp parallel
    { const int tid = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      libxs_gemm_batch_task(LIBXS_DATATYPE(double), "N", "N", m, n, k,
        &alpha, a_ptrs, lda, b_ptrs, ldb,
        &beta, c_ptrs, ldc, batchsize, &config, tid, nthreads);
    }
#else
    libxs_gemm_batch(LIBXS_DATATYPE(double), "N", "N", m, n, k,
      &alpha, a_ptrs, lda, b_ptrs, ldb,
      &beta, c_ptrs, ldc, batchsize, &config);
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
          if (0.5 != c_data[(size_t)ci * ldc + cj]) {
            fprintf(stderr, "FAILED: C ld-padding overwritten"
              " (col=%d row=%d)\n", ci, cj);
            result = EXIT_FAILURE;
          }
        }
      }
    }
    libxs_matdiff(&check_diff, LIBXS_DATATYPE(double),
      m, n, NULL/*ref*/, c_data, NULL/*ldref*/, &ldc);
    if (1 < nunique) {
      libxs_matdiff_t d;
      libxs_matdiff(&d, LIBXS_DATATYPE(double),
        m, n, NULL/*ref*/, c_data + (nunique - 1) * csize,
        NULL/*ldref*/, &ldc);
      libxs_matdiff_combine(&check_diff, &d);
    }
    printf("checksum=%f\n", check_diff.l1_ref + check_diff.l1_tst);
  }

  libxs_gemm_release(&config);
  free(a_data); free(b_data); free(c_data);
  free(a_ptrs); free(b_ptrs); free(c_ptrs);
  libxs_finalize();
  return result;
}
