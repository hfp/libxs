/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_tas.h>
#include <libxs/libxs_timer.h>
#include <libxs/libxs_rng.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

/* largest relative difference (Frobenius norm) to the BLAS call that passes CHECK */
#if !defined(GEMM_TAS_TOLERANCE_F64)
# define GEMM_TAS_TOLERANCE_F64 1E-12
#endif
#if !defined(GEMM_TAS_TOLERANCE_F32)
# define GEMM_TAS_TOLERANCE_F32 1E-4
#endif

#define GEMM_TAS_NOTRANS(C) ('N' == (C) || 'n' == (C))


static void gemm_tas_init(libxs_data_t datatype, void* data, int nrows, int ncols, int ld)
{
  if (LIBXS_DATATYPE_F64 == datatype) {
    LIBXS_MATRNG_OMP(int, double, 0, data, nrows, ncols, ld, 1.0);
  }
  else {
    LIBXS_MATRNG_OMP(int, float, 0, data, nrows, ncols, ld, 1.0);
  }
}


/* one call of the whole GEMM, threaded if the BLAS is */
static void gemm_tas_blas(const libxs_tas_t* plan, const void* a, const void* b, void* c)
{
  const libxs_gemm_shape_t *const s = &plan->shape;
  if (LIBXS_DATATYPE_F64 == s->datatype) {
    plan->panel.dgemm_blas(&s->transa, &s->transb, &s->m, &s->n, &s->k,
      &s->alpha, (const double*)a, &s->lda, (const double*)b, &s->ldb,
      &s->beta, (double*)c, &s->ldc, 1, 1);
  }
  else {
    const float alpha = (float)s->alpha, beta = (float)s->beta;
    plan->panel.sgemm_blas(&s->transa, &s->transb, &s->m, &s->n, &s->k,
      &alpha, (const float*)a, &s->lda, (const float*)b, &s->ldb,
      &beta, (float*)c, &s->ldc, 1, 1);
  }
}


static void gemm_tas_run(const libxs_tas_t* plan, const void* a, const void* b, void* c)
{
#if defined(_OPENMP)
  const int ntasks = libxs_tas_ntasks(plan, omp_get_max_threads());
# pragma omp parallel num_threads(ntasks)
  { /* the team may be smaller than requested, and every task of it must call */
    libxs_tas_task(plan, a, b, c, omp_get_thread_num(), omp_get_num_threads());
  }
#else
  libxs_tas(plan, a, b, c);
#endif
}


int main(int argc, char* argv[])
{
  const int m = (1 < argc ? atoi(argv[1]) : 1000000);
  const int n = (2 < argc ? atoi(argv[2]) : 16);
  const int k = (3 < argc ? atoi(argv[3]) : n);
  const int panel = (4 < argc ? atoi(argv[4]) : 0);
  const int nrepeat = LIBXS_MAX(5 < argc ? atoi(argv[5]) : 3, 1);
  const double alpha = 1.0, beta = (6 < argc ? atof(argv[6]) : 1.0);
  const libxs_data_t datatype = ((7 < argc && ('s' == *argv[7] || 'S' == *argv[7]))
    ? LIBXS_DATATYPE_F32 : LIBXS_DATATYPE_F64);
  const char transa = (8 < argc ? *argv[8] : 'N'), transb = (9 < argc ? *argv[9] : 'N');
  const char *const env_check = getenv("CHECK");
  const int check = (NULL == env_check || 0 == *env_check) ? 0 : atoi(env_check);
  const int ta = !GEMM_TAS_NOTRANS(transa), tb = !GEMM_TAS_NOTRANS(transb);
  const int lda = (0 != ta ? k : m), ldb = (0 != tb ? n : k), ldc = m;
  const size_t esize = LIBXS_TYPESIZE(datatype);
  const size_t na = (size_t)lda * (0 != ta ? m : k), nb = (size_t)ldb * (0 != tb ? k : n);
  const size_t nc = (size_t)ldc * n;
  /* bytes a GEMM has to move at least: A and B once, C written (and read) once */
  const double nbytes = (double)esize * ((double)na + nb + nc * (0.0 != beta ? 2 : 1));
  const double gflops = 2.0 * m * n * k * 1E-9;
  const float falpha = (float)alpha, fbeta = (float)beta;
  void *a = NULL, *b = NULL, *c = NULL, *d = NULL;
  libxs_timer_tick_t t0, t1;
  libxs_tas_t plan;
  int result = EXIT_SUCCESS, r;

  libxs_init();
  LIBXS_MEMZERO(&plan); /* released in any case */
  if (0 >= m || 0 >= n || 0 >= k
    || 0 == libxs_tas_dispatch(&plan, datatype, transa, transb, m, n, k, lda, ldb, ldc,
      LIBXS_DATATYPE_F64 == datatype ? (const void*)&alpha : (const void*)&falpha,
      LIBXS_DATATYPE_F64 == datatype ? (const void*)&beta : (const void*)&fbeta, panel, NULL))
  {
    fprintf(stderr, "ERROR: invalid shape\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    printf("gemm_tas: %s M=%d N=%d K=%d trans=%c%c nrepeat=%d\n",
      libxs_typename(datatype), m, n, k, transa, transb, nrepeat);
    if (0 != plan.dim) {
      const libxs_gemm_shape_t *const s = &plan.panel.shape;
      printf("  along %c: %d panels %ix%ix%i%s, %s\n", plan.dim, plan.npanels,
        s->m, s->n, s->k, 0 < plan.rest.shape.m ? " and a remainder" : "",
        (NULL != plan.panel.dgemm_jit || NULL != plan.panel.sgemm_jit
          || NULL != plan.panel.xgemm) ? "kernel" : "BLAS");
    }
    else printf("  one call (not tall and skinny)\n");
    a = malloc(na * esize); b = malloc(nb * esize);
    c = malloc(nc * esize); d = malloc(nc * esize);
    if (NULL == a || NULL == b || NULL == c || NULL == d) {
      fprintf(stderr, "ERROR: memory allocation failed\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    /* parallel initialization places the pages next to the threads using them */
    gemm_tas_init(datatype, a, 0 != ta ? k : m, 0 != ta ? m : k, lda);
    gemm_tas_init(datatype, b, 0 != tb ? n : k, 0 != tb ? k : n, ldb);
    gemm_tas_init(datatype, c, m, n, ldc);
    memcpy(d, c, nc * esize);
  }
  if (EXIT_SUCCESS == result) {
    double duration;
    gemm_tas_blas(&plan, a, b, d);
    t0 = libxs_timer_tick();
    for (r = 0; r < nrepeat; ++r) gemm_tas_blas(&plan, a, b, d);
    t1 = libxs_timer_tick();
    duration = libxs_timer_duration(t0, t1) / nrepeat;
    if (0 < duration) {
      printf("BLAS call  : %.3f ms %.1f GB/s %.1f GFLOPS/s\n",
        1E3 * duration, 1E-9 * nbytes / duration, gflops / duration);
    }
    gemm_tas_run(&plan, a, b, c);
    t0 = libxs_timer_tick();
    for (r = 0; r < nrepeat; ++r) gemm_tas_run(&plan, a, b, c);
    t1 = libxs_timer_tick();
    duration = libxs_timer_duration(t0, t1) / nrepeat;
    if (0 < duration) {
      printf("TAS        : %.3f ms %.1f GB/s %.1f GFLOPS/s\n",
        1E3 * duration, 1E-9 * nbytes / duration, gflops / duration);
    }
  }
  if (EXIT_SUCCESS == result && 0 != check) {
    const double tolerance = (LIBXS_DATATYPE_F64 == datatype
      ? GEMM_TAS_TOLERANCE_F64 : GEMM_TAS_TOLERANCE_F32);
    libxs_matdiff_t diff;
    if (EXIT_SUCCESS == libxs_matdiff(&diff, datatype, m, n, d, c, &ldc, &ldc)) {
      printf("diff: normf_rel=%.3e\n", diff.normf_rel);
      printf("checksum=%.6e\n", diff.l1_tst);
      if (tolerance < diff.normf_rel) {
        fprintf(stderr, "FAILED: TAS differs from the BLAS call\n");
        result = EXIT_FAILURE;
      }
    }
  }
  libxs_tas_release(&plan);
  free(a); free(b); free(c); free(d);
  libxs_finalize();
  return result;
}
