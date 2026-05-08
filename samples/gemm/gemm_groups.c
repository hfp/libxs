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

#define MAX_GROUPS 4


int main(int argc, char* argv[])
{
  const int ngroups = (1 < argc ? atoi(argv[1]) : 2);
  const int batch_per_group = (2 < argc ? atoi(argv[2]) : 30000);
  const int nrepeat = (3 < argc ? atoi(argv[3]) : 3);
  const int ng = (0 < ngroups && ngroups <= MAX_GROUPS) ? ngroups : 2;
  const char *const env_check = getenv("CHECK");
  const double check = (NULL == env_check || 0 == *env_check) ? 0 : atof(env_check);
  /* per-group shapes: grow by 4 per group for illustration */
  const int base_m = (4 < argc ? atoi(argv[4]) : 8);
  const double beta = (5 < argc ? atof(argv[5]) : 1.0);
  const int pad = (6 < argc ? atoi(argv[6]) : 0);
  int m_array[MAX_GROUPS], n_array[MAX_GROUPS], k_array[MAX_GROUPS];
  int lda_array[MAX_GROUPS], ldb_array[MAX_GROUPS], ldc_array[MAX_GROUPS];
  int batchsize[MAX_GROUPS];
  double alpha_array[MAX_GROUPS], beta_array[MAX_GROUPS];
  libxs_gemm_config_t configs[MAX_GROUPS];
  const void **a_ptrs = NULL, **b_ptrs = NULL;
  void **c_ptrs = NULL;
  double *storage = NULL;
  size_t total_ptrs = 0, total_elems = 0;
  double total_gflops = 0, duration = 0;
  libxs_matdiff_t check_diff;
  libxs_timer_tick_t t0, t1;
  int result = EXIT_SUCCESS, g, r;

  libxs_init();

  /* configure groups and dispatch per-group kernels */
  for (g = 0; g < ng; ++g) {
    const int dim = base_m + g * 4;
    m_array[g] = dim; n_array[g] = dim; k_array[g] = dim;
    lda_array[g] = dim + pad; ldb_array[g] = dim + pad; ldc_array[g] = dim + pad;
    batchsize[g] = batch_per_group;
    alpha_array[g] = 1.0; beta_array[g] = beta;
    total_ptrs += (size_t)batch_per_group;
    total_elems += (size_t)batch_per_group * (
      (size_t)lda_array[g] * dim + (size_t)ldb_array[g] * dim
      + (size_t)ldc_array[g] * dim);
    total_gflops += 2.0 * dim * dim * dim * batch_per_group * 1E-9;

    /* dispatch a JIT kernel per group (each group has its own shape) */
    { libxs_gemm_config_t* cfg = libxs_gemm_dispatch(
        LIBXS_DATATYPE(double), 'N', 'N', dim, dim, dim,
        lda_array[g], ldb_array[g], ldc_array[g],
        alpha_array + g, beta_array + g, NULL);
      if (NULL != cfg) {
        configs[g] = *cfg;
        printf("  group %d: JIT kernel dispatched (M=%d)\n", g, dim);
      }
      else {
        memset(configs + g, 0, sizeof(configs[g]));
      }
      configs[g].flags = LIBXS_GEMM_FLAG_NOLOCK;
    }
  }

  printf("gemm_groups: ngroups=%d batch/group=%d nrepeat=%d base_m=%d\n",
    ng, batch_per_group, nrepeat, base_m);
  for (g = 0; g < ng; ++g) {
    printf("  group %d: M=%d N=%d K=%d batch=%d\n",
      g, m_array[g], n_array[g], k_array[g], batchsize[g]);
  }

  /* allocate concatenated pointer arrays and backing storage */
  a_ptrs = (const void**)malloc(total_ptrs * sizeof(void*));
  b_ptrs = (const void**)malloc(total_ptrs * sizeof(void*));
  c_ptrs = (void**)malloc(total_ptrs * sizeof(void*));
  storage = (double*)malloc(total_elems * sizeof(double));
  if (NULL == a_ptrs || NULL == b_ptrs || NULL == c_ptrs || NULL == storage) {
    fprintf(stderr, "ERROR: memory allocation failed\n");
    free(a_ptrs); free(b_ptrs); free(c_ptrs); free(storage);
    return EXIT_FAILURE;
  }

  /* distribute storage across groups and set up pointer arrays */
  { size_t offset = 0, pidx = 0;
    for (g = 0; g < ng; ++g) {
      const size_t asize = (size_t)lda_array[g] * k_array[g];
      const size_t bsize = (size_t)ldb_array[g] * n_array[g];
      const size_t csize = (size_t)ldc_array[g] * n_array[g];
      int i;
      for (i = 0; i < batchsize[g]; ++i) {
        double *pa = storage + offset;
        double *pb, *pc;
        LIBXS_MATRNG(int, double, 0,
          pa, m_array[g], k_array[g], lda_array[g], 1.0);
        a_ptrs[pidx] = pa; offset += asize;
        pb = storage + offset;
        LIBXS_MATRNG(int, double, 0,
          pb, k_array[g], n_array[g], ldb_array[g], 1.0);
        b_ptrs[pidx] = pb; offset += bsize;
        pc = storage + offset;
        LIBXS_MATRNG(int, double, 0,
          pc, m_array[g], n_array[g], ldc_array[g], 1.0);
        c_ptrs[pidx] = pc; offset += csize;
        ++pidx;
      }
    }
  }

  /* warmup: loop over groups, call batch per group */
  { size_t j = 0;
    for (g = 0; g < ng; ++g) {
      libxs_gemm_batch(a_ptrs + j, b_ptrs + j, c_ptrs + j,
        batchsize[g], configs + g);
      j += (size_t)batchsize[g];
    }
  }

  t0 = libxs_timer_tick();
  for (r = 0; r < nrepeat; ++r) {
    size_t j = 0;
    for (g = 0; g < ng; ++g) {
      libxs_gemm_batch(a_ptrs + j, b_ptrs + j, c_ptrs + j,
        batchsize[g], configs + g);
      j += (size_t)batchsize[g];
    }
  }
  t1 = libxs_timer_tick();
  duration = libxs_timer_duration(t0, t1);

  if (0 < duration) {
    printf("Total time : %.3f s (%d repeats)\n", duration, nrepeat);
    printf("Performance: %.1f GFLOPS/s\n",
      total_gflops * nrepeat / duration);
  }

  if (0 != check) {
    /* verify ld-padding in first C-matrix per group */
    { size_t pidx = 0;
      for (g = 0; g < ng && EXIT_SUCCESS == result; ++g) {
        if (ldc_array[g] > m_array[g]) {
          const double *cm = (const double*)c_ptrs[pidx];
          int ci, cj;
          for (ci = 0; ci < n_array[g] && EXIT_SUCCESS == result; ++ci) {
            for (cj = m_array[g]; cj < ldc_array[g]; ++cj) {
              if (0 != cm[(size_t)ci * ldc_array[g] + cj]) {
                fprintf(stderr, "FAILED: C ld-padding overwritten"
                  " (group=%d col=%d row=%d)\n", g, ci, cj);
                result = EXIT_FAILURE;
              }
            }
          }
        }
        pidx += (size_t)batchsize[g];
      }
    }
    libxs_matdiff(&check_diff, LIBXS_DATATYPE(double),
      m_array[0], n_array[0], NULL/*ref*/,
      c_ptrs[0], NULL/*ldref*/, ldc_array);
    if (1 < ng) {
      libxs_matdiff_t d;
      size_t pidx = 0;
      for (g = 0; g < ng - 1; ++g) pidx += (size_t)batchsize[g];
      libxs_matdiff(&d, LIBXS_DATATYPE(double),
        m_array[ng - 1], n_array[ng - 1], NULL/*ref*/,
        c_ptrs[pidx], NULL/*ldref*/, ldc_array + ng - 1);
      libxs_matdiff_combine(&check_diff, &d);
    }
    printf("checksum=%.6e\n", check_diff.l1_ref + check_diff.l1_tst);
  }

  for (g = 0; g < ng; ++g) libxs_gemm_release(configs + g);
  free(a_ptrs); free(b_ptrs); free(c_ptrs); free(storage);
  libxs_finalize();
  return result;
}
