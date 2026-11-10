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
#include <libxs/libxs_timer.h>
#include <libxs/libxs_rng.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

/* default extent of a panel along M or N, and of a chunk along K */
#if !defined(GEMM_TAS_PANEL)
# define GEMM_TAS_PANEL 256
#endif
#if !defined(GEMM_TAS_CHUNK)
# define GEMM_TAS_CHUNK 4096
#endif
/* largest relative difference to the BLAS call that passes CHECK */
#if !defined(GEMM_TAS_TOLERANCE)
# define GEMM_TAS_TOLERANCE 1E-12
#endif


/**
 * One column-major C := alpha * A * B + beta * C, planned as a constant-stride
 * batch of panels along its longest dimension, plus one remainder call. Along
 * K, each chunk writes a partial C of its own (scratch), which are summed in
 * a fixed order: the result does not depend on the number of threads.
 */
typedef struct gemm_tas_plan_t {
  /* kernel shapes of a full panel and of the remainder (rest.m is zero if none) */
  libxs_gemm_shape_t panel, rest;
  /* element strides between panels, and element offsets of the remainder */
  int stride_a, stride_b, stride_c;
  size_t rest_a, rest_b, rest_c;
  /* number of full panels, and number of partial C (zero unless along K) */
  int npanels, nscratch;
  char dim;
} gemm_tas_plan_t;


static void gemm_tas_shape(libxs_gemm_shape_t* shape,
  int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta)
{
  LIBXS_MEMZERO(shape);
  shape->datatype = LIBXS_DATATYPE_F64;
  shape->transa = shape->transb = 'N';
  shape->m = m; shape->n = n; shape->k = k;
  shape->lda = lda; shape->ldb = ldb; shape->ldc = ldc;
  shape->alpha = alpha; shape->beta = beta;
}


/* zero panel takes the default of the dimension the plan runs along */
static void gemm_tas_plan(gemm_tas_plan_t* plan,
  int m, int n, int k, int lda, int ldb, int ldc,
  double alpha, double beta, int panel)
{
  int w, nrest;
  LIBXS_MEMZERO(plan);
  if (m >= n && m >= k) {
    w = LIBXS_CLMP(0 < panel ? panel : GEMM_TAS_PANEL, 1, m);
    plan->dim = 'M';
    plan->npanels = m / w;
    nrest = m - plan->npanels * w;
    gemm_tas_shape(&plan->panel, w, n, k, lda, ldb, ldc, alpha, beta);
    plan->stride_a = w; plan->stride_b = 0; plan->stride_c = w;
    if (0 < nrest) {
      gemm_tas_shape(&plan->rest, nrest, n, k, lda, ldb, ldc, alpha, beta);
    }
  }
  else if (n >= k) {
    w = LIBXS_CLMP(0 < panel ? panel : GEMM_TAS_PANEL, 1, n);
    plan->dim = 'N';
    plan->npanels = n / w;
    nrest = n - plan->npanels * w;
    gemm_tas_shape(&plan->panel, m, w, k, lda, ldb, ldc, alpha, beta);
    plan->stride_a = 0; plan->stride_b = w * ldb; plan->stride_c = w * ldc;
    if (0 < nrest) {
      gemm_tas_shape(&plan->rest, m, nrest, k, lda, ldb, ldc, alpha, beta);
    }
  }
  else { /* partial C are tight (ld = m), and alpha/beta apply in the reduction */
    w = LIBXS_CLMP(0 < panel ? panel : GEMM_TAS_CHUNK, 1, k);
    plan->dim = 'K';
    plan->npanels = k / w;
    nrest = k - plan->npanels * w;
    gemm_tas_shape(&plan->panel, m, n, w, lda, ldb, m, 1.0, 0.0);
    plan->stride_a = w * lda; plan->stride_b = w; plan->stride_c = m * n;
    plan->nscratch = plan->npanels + (0 < nrest ? 1 : 0);
    if (0 < nrest) {
      gemm_tas_shape(&plan->rest, m, n, nrest, lda, ldb, m, 1.0, 0.0);
    }
  }
  plan->rest_a = (size_t)plan->stride_a * plan->npanels;
  plan->rest_b = (size_t)plan->stride_b * plan->npanels;
  plan->rest_c = (size_t)plan->stride_c * plan->npanels;
}


/* the lower layer decides the kernel, hinted that panels stream and are called often */
static int gemm_tas_dispatch(libxs_gemm_config_t* config,
  const libxs_gemm_shape_t* shape, int hint, const char* label)
{
  int result = EXIT_SUCCESS;
  LIBXS_MEMZERO(config);
  if (0 < shape->m) {
    if (0 != hint) {
      config->flags = (libxs_gemm_flags_t)(LIBXS_GEMM_FLAG_STREAM | LIBXS_GEMM_FLAG_REUSE);
    }
    if (0 != libxs_gemm_dispatch_cpy(config, shape->datatype,
      shape->transa, shape->transb, shape->m, shape->n, shape->k,
      shape->lda, shape->ldb, shape->ldc, &shape->alpha, &shape->beta, NULL))
    {
      config->flags = LIBXS_GEMM_FLAG_NOLOCK; /* panels never share a C */
      printf("  %s %ix%ix%i ld=%ix%ix%i: %s\n", label,
        shape->m, shape->n, shape->k, shape->lda, shape->ldb, shape->ldc,
        (NULL != config->dgemm_jit || NULL != config->xgemm) ? "kernel" : "BLAS");
    }
    else result = EXIT_FAILURE;
  }
  return result;
}


static void gemm_tas_reduce(const gemm_tas_plan_t* plan, const double* scratch,
  int m, int n, double alpha, double beta, double* c, int ldc)
{
  const size_t tile = (size_t)m * n;
  int j;
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (j = 0; j < n; ++j) {
    int i, t;
    for (i = 0; i < m; ++i) {
      const size_t ij = (size_t)j * m + i;
      double sum = scratch[ij];
      for (t = 1; t < plan->nscratch; ++t) sum += scratch[t * tile + ij];
      /* BLAS does not read C when beta is zero (C may hold NaN, e.g., fresh memory) */
      c[(size_t)j * ldc + i] = (0.0 == beta ? alpha * sum
        : (alpha * sum + beta * c[(size_t)j * ldc + i]));
    }
  }
}


static void gemm_tas_exec(const gemm_tas_plan_t* plan,
  const libxs_gemm_config_t* panel, const libxs_gemm_config_t* rest,
  const double* a, const double* b, double* c, int ldc, double* scratch,
  double alpha, double beta)
{
  double *const cp = (0 != plan->nscratch ? scratch : c);
#if defined(_OPENMP)
# pragma omp parallel
  { const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    libxs_gemm_index_task(a, &plan->stride_a, b, &plan->stride_b, cp, &plan->stride_c,
      0/*index_stride*/, 0/*index_base*/, plan->npanels, panel, tid, nthreads);
  }
#else
  libxs_gemm_index(a, &plan->stride_a, b, &plan->stride_b, cp, &plan->stride_c,
    0/*index_stride*/, 0/*index_base*/, plan->npanels, panel);
#endif
  if (0 < plan->rest.m) {
    libxs_gemm_call(rest, a + plan->rest_a, b + plan->rest_b, cp + plan->rest_c);
  }
  if (0 != plan->nscratch) {
    gemm_tas_reduce(plan, scratch, plan->panel.m, plan->panel.n, alpha, beta, c, ldc);
  }
}


int main(int argc, char* argv[])
{
  const int m = (1 < argc ? atoi(argv[1]) : 1000000);
  const int n = (2 < argc ? atoi(argv[2]) : 16);
  const int k = (3 < argc ? atoi(argv[3]) : n);
  const int panel = (4 < argc ? atoi(argv[4]) : 0);
  const int nrepeat = LIBXS_MAX(5 < argc ? atoi(argv[5]) : 3, 1);
  const double alpha = 1.0, beta = (6 < argc ? atof(argv[6]) : 1.0);
  const int hint = (7 < argc ? atoi(argv[7]) : 1);
  const char *const env_check = getenv("CHECK");
  const int check = (NULL == env_check || 0 == *env_check) ? 0 : atoi(env_check);
  const int lda = m, ldb = k, ldc = m;
  const size_t na = (size_t)lda * k, nb = (size_t)ldb * n, nc = (size_t)ldc * n;
  /* bytes a GEMM has to move at least: A and B once, C written (and read) once */
  const double nbytes = sizeof(double) * ((double)na + nb + nc * (0.0 != beta ? 2 : 1));
  const double gflops = 2.0 * m * n * k * 1E-9;
  double *a = NULL, *b = NULL, *c = NULL, *d = NULL, *scratch = NULL;
  libxs_gemm_config_t config_panel, config_rest;
  libxs_timer_tick_t t0, t1;
  gemm_tas_plan_t plan;
  int result = EXIT_SUCCESS, r;

  libxs_init();
  /* the batch addresses elements with int offsets */
  if (0 >= m || 0 >= n || 0 >= k || INT_MAX < na || INT_MAX < nb || INT_MAX < nc) {
    fprintf(stderr, "ERROR: invalid or too large shape\n");
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    gemm_tas_plan(&plan, m, n, k, lda, ldb, ldc, alpha, beta, panel);
    if ((size_t)INT_MAX < (size_t)plan.nscratch * m * n) {
      fprintf(stderr, "ERROR: too many partial C, increase the chunk\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    printf("gemm_tas: M=%d N=%d K=%d along %c: %d panels of %d%s nrepeat=%d\n",
      m, n, k, plan.dim, plan.npanels,
      'M' == plan.dim ? plan.panel.m : ('N' == plan.dim ? plan.panel.n : plan.panel.k),
      0 < plan.rest.m ? " and a remainder" : "", nrepeat);
    a = (double*)malloc(na * sizeof(double));
    b = (double*)malloc(nb * sizeof(double));
    c = (double*)malloc(nc * sizeof(double));
    d = (double*)malloc(nc * sizeof(double));
    if (0 != plan.nscratch) {
      scratch = (double*)malloc((size_t)plan.nscratch * m * n * sizeof(double));
    }
    if (NULL == a || NULL == b || NULL == c || NULL == d
      || (0 != plan.nscratch && NULL == scratch))
    {
      fprintf(stderr, "ERROR: memory allocation failed\n");
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    /* parallel initialization places the pages next to the threads using them */
    LIBXS_MATRNG_OMP(int, double, 0, a, m, k, lda, 1.0);
    LIBXS_MATRNG_OMP(int, double, 0, b, k, n, ldb, 1.0);
    LIBXS_MATRNG_OMP(int, double, 0, c, m, n, ldc, 1.0);
    memcpy(d, c, nc * sizeof(double));
    result = gemm_tas_dispatch(&config_panel, &plan.panel, hint, "panel");
  }
  if (EXIT_SUCCESS == result) {
    result = gemm_tas_dispatch(&config_rest, &plan.rest, hint, "rest ");
  }
  if (EXIT_SUCCESS == result && NULL != config_panel.dgemm_blas) {
    /* one call of the full shape, threaded if the BLAS is */
    double duration;
    config_panel.dgemm_blas("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb,
      &beta, d, &ldc, 1, 1);
    t0 = libxs_timer_tick();
    for (r = 0; r < nrepeat; ++r) {
      config_panel.dgemm_blas("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb,
        &beta, d, &ldc, 1, 1);
    }
    t1 = libxs_timer_tick();
    duration = libxs_timer_duration(t0, t1) / nrepeat;
    if (0 < duration) {
      printf("BLAS call  : %.3f ms %.1f GB/s %.1f GFLOPS/s\n",
        1E3 * duration, 1E-9 * nbytes / duration, gflops / duration);
    }
  }
  if (EXIT_SUCCESS == result) {
    double duration;
    gemm_tas_exec(&plan, &config_panel, &config_rest, a, b, c, ldc, scratch, alpha, beta);
    t0 = libxs_timer_tick();
    for (r = 0; r < nrepeat; ++r) {
      gemm_tas_exec(&plan, &config_panel, &config_rest, a, b, c, ldc, scratch, alpha, beta);
    }
    t1 = libxs_timer_tick();
    duration = libxs_timer_duration(t0, t1) / nrepeat;
    if (0 < duration) {
      printf("TAS batch  : %.3f ms %.1f GB/s %.1f GFLOPS/s\n",
        1E3 * duration, 1E-9 * nbytes / duration, gflops / duration);
    }
  }
  if (EXIT_SUCCESS == result && 0 != check) {
    libxs_matdiff_t diff;
    if (EXIT_SUCCESS == libxs_matdiff(&diff, LIBXS_DATATYPE(double),
      m, n, NULL != config_panel.dgemm_blas ? d : NULL, c, &ldc, &ldc))
    {
      printf("diff: normf_rel=%.3e\n", diff.normf_rel);
      printf("checksum=%.6e\n", diff.l1_tst);
      if (NULL != config_panel.dgemm_blas && GEMM_TAS_TOLERANCE < diff.normf_rel) {
        fprintf(stderr, "FAILED: TAS batch differs from the BLAS call\n");
        result = EXIT_FAILURE;
      }
    }
  }
  free(a); free(b); free(c); free(d); free(scratch);
  libxs_finalize();
  return result;
}
