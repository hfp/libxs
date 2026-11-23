/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_source.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif
#define TEST_CHECK(CONDITION) do { \
    if (!(CONDITION)) { \
      FPRINTF(stderr, "ERROR line #%i: %s\n", __LINE__, #CONDITION); \
      result = EXIT_FAILURE; \
    } \
  } while(0)

#define MAXJIT 32
#define PAD 3


/* a generated kernel as a plain GEMM of the shape it was generated for */
typedef struct test_jit_t {
  int ta, tb, m, n, k, lda, ldb, ldc;
  double alpha, beta;
} test_jit_t;


static test_jit_t test_jits[MAXJIT];
static int test_njits, test_ncalls;


static void test_kernel(void* jitter, const double* a, const double* b, double* c)
{
  const test_jit_t *const jit = (const test_jit_t*)jitter;
  int i, j, p;
  for (j = 0; j < jit->n; ++j) {
    for (i = 0; i < jit->m; ++i) {
      double sum = 0;
      for (p = 0; p < jit->k; ++p) {
        sum += (0 == jit->ta ? a[i + p * jit->lda] : a[p + i * jit->lda])
             * (0 == jit->tb ? b[j * jit->ldb + p] : b[j + p * jit->ldb]);
      }
      c[i + j * jit->ldc] = jit->alpha * sum + (0 != jit->beta ? (jit->beta * c[i + j * jit->ldc]) : 0);
    }
  }
#if defined(_OPENMP)
# pragma omp atomic
#endif
  ++test_ncalls;
}


static int test_create(void** jitter, int layout, int transa, int transb, int m, int n, int k,
  double alpha, int lda, int ldb, double beta, int ldc)
{
  int result = 2; /* MKL_JIT_ERROR */
  LIBXS_UNUSED(layout);
  if (MAXJIT > test_njits) {
    test_jit_t *const jit = test_jits + test_njits++;
    jit->ta = (111 != transa); jit->tb = (111 != transb);
    jit->m = m; jit->n = n; jit->k = k;
    jit->lda = lda; jit->ldb = ldb; jit->ldc = ldc;
    jit->alpha = alpha; jit->beta = beta;
    *jitter = jit;
    result = 0;
  }
  return result;
}


static void* test_get(void* jitter)
{
  union { void (*fn)(void*, const double*, const double*, double*); void* ptr; } kernel;
  LIBXS_UNUSED(jitter);
  kernel.fn = test_kernel;
  return kernel.ptr;
}


static double value(int which, int i, int j)
{
  unsigned int h = (unsigned int)which * 7919U + (unsigned int)i * 104729U + (unsigned int)j * 1299709U + 1U;
  h ^= h >> 13;
  h *= 0x5bd1e995U;
  h ^= h >> 15;
  return (double)(h & 0xFFFF) / 32768.0 - 1.0;
}


static void put(void* x, libxs_data_t datatype, size_t i, double v)
{
  if (LIBXS_DATATYPE_F64 == datatype) ((double*)x)[i] = v;
  else ((float*)x)[i] = (float)v;
}


static double get(const void* x, libxs_data_t datatype, size_t i)
{
  return (LIBXS_DATATYPE_F64 == datatype ? ((const double*)x)[i] : (double)((const float*)x)[i]);
}


static double not_a_number(void)
{
  static volatile double zero = 0;
  return zero / zero;
}


static int run(const libxs_syrk_t* plan, double alpha, const void* a, const void* b,
  double beta, void* c, int ntasks)
{
  int result = EXIT_SUCCESS, i;
#if defined(_OPENMP)
# pragma omp parallel for schedule(dynamic) num_threads(4)
#endif
  for (i = 0; i < ntasks; ++i) {
    const int tid = ntasks - 1 - i; /* neither the order nor the thread matters */
    if (EXIT_SUCCESS != libxs_syrk_task(plan, alpha, a, b, beta, c, tid, ntasks)) {
#if defined(_OPENMP)
#     pragma omp critical
#endif
      result = EXIT_FAILURE;
    }
  }
  return result;
}


/* All variants of one size against the definition, and the same bits for any number of tasks. */
static int check(libxs_data_t datatype, char uplo, char trans, int twok, int n, int k)
{
  const double alphas[] = { 1.5, 1.5, 1.0, 0.0, 0.0, 0.0 };
  const double betas[] = { 0.0, 0.5, 1.0, 0.0, 0.5, 1.0 };
  const int upper = ('U' == uplo), notrans = ('N' == trans);
  const int rows = (0 != notrans ? n : k), cols = (0 != notrans ? k : n);
  const int lda = rows + PAD, ldb = rows + 2 * PAD, ldc = n + PAD;
  const size_t elemsize = LIBXS_TYPESIZE(datatype), nc = (size_t)ldc * n + PAD;
  const double tol = (LIBXS_DATATYPE_F64 == datatype ? 1E-12 : 2E-4) * (4.0 * k + 1);
  void *const a = malloc(((size_t)lda * cols + 1) * elemsize);
  void *const b = malloc(((size_t)ldb * cols + 1) * elemsize);
  void *const c = malloc(nc * elemsize), *const d = malloc(nc * elemsize);
  double *const sum = (double*)calloc((size_t)n * n + 1, sizeof(double));
  libxs_syrk_shape_t shape;
  libxs_syrk_t plan;
  int result = EXIT_SUCCESS, i, j, p, v;
  TEST_CHECK(NULL != a && NULL != b && NULL != c && NULL != d && NULL != sum);
  if (EXIT_SUCCESS == result) {
    for (j = 0; j < cols; ++j) {
      for (i = 0; i < lda; ++i) put(a, datatype, i + (size_t)j * lda, i < rows ? value(1, i, j) : 1E30);
      for (i = 0; i < ldb; ++i) put(b, datatype, i + (size_t)j * ldb, i < rows ? value(2, i, j) : 1E30);
    }
    for (p = 0; p < k; ++p) {
      for (j = 0; j < n; ++j) {
        const size_t ja = (0 != notrans ? (j + (size_t)p * lda) : (p + (size_t)j * lda));
        const size_t jb = (0 != notrans ? (j + (size_t)p * ldb) : (p + (size_t)j * ldb));
        const double aj = get(a, datatype, ja), bj = (0 != twok ? get(b, datatype, jb) : 0);
        const int end = (0 != upper ? (j + 1) : n);
        for (i = (0 != upper ? 0 : j); i < end; ++i) {
          const size_t ia = (0 != notrans ? (i + (size_t)p * lda) : (p + (size_t)i * lda));
          const size_t ib = (0 != notrans ? (i + (size_t)p * ldb) : (p + (size_t)i * ldb));
          sum[i + (size_t)j * n] += (0 != twok
            ? (get(a, datatype, ia) * bj + get(b, datatype, ib) * aj) : (get(a, datatype, ia) * aj));
        }
      }
    }
    shape.datatype = datatype;
    shape.uplo = uplo; shape.trans = trans;
    shape.n = n; shape.k = k;
    shape.lda = lda; shape.ldb = (0 != twok ? ldb : 0); shape.ldc = ldc;
    TEST_CHECK(0 != libxs_syrk_dispatch(&plan, &shape, NULL));
  }
  for (v = 0; v < (int)(sizeof(alphas) / sizeof(*alphas)) && EXIT_SUCCESS == result; ++v) {
    const double alpha = alphas[v], beta = betas[v];
    const int reads = (0 != alpha && 0 != k);
    const void *const pa = (0 != reads ? a : NULL), *const pb = ((0 != reads && 0 != twok) ? b : NULL);
    int tasks[6], run_i;
    size_t x;
    tasks[0] = 1; tasks[1] = 2; tasks[2] = 3; tasks[3] = 7;
    tasks[4] = LIBXS_MAX(plan.ntiles, 1); tasks[5] = plan.ntiles + 5;
    for (run_i = 0; run_i < 6 && EXIT_SUCCESS == result; ++run_i) {
      void *const out = (0 == run_i ? c : d);
      for (x = 0; x < nc; ++x) put(out, datatype, x, 0.375);
      if (0 == beta) { /* C is not read, and may hold anything */
        for (j = 0; j < n; ++j) {
          const int end = (0 != upper ? (j + 1) : n);
          for (i = (0 != upper ? 0 : j); i < end; ++i) put(out, datatype, i + (size_t)j * ldc, not_a_number());
        }
      }
      if (0 == run_i) TEST_CHECK(EXIT_SUCCESS == libxs_syrk(&plan, alpha, pa, pb, beta, out));
      else {
        TEST_CHECK(EXIT_SUCCESS == run(&plan, alpha, pa, pb, beta, out, tasks[run_i]));
        TEST_CHECK(0 == memcmp(c, d, nc * elemsize));
      }
    }
    for (x = 0; x < nc && EXIT_SUCCESS == result; ++x) {
      const int row = (int)(x % ldc), col = (int)(x / ldc);
      const int inside = (row < n && col < n && (0 != upper ? (row <= col) : (row >= col)));
      const double got = get(c, datatype, x);
      if (0 != inside) {
        const double ref = alpha * sum[row + (size_t)col * n] + (0 != beta ? (beta * 0.375) : 0);
        TEST_CHECK(LIBXS_DELTA(got, ref) <= tol);
      }
      else TEST_CHECK(0.375 == got);
    }
  }
  if (EXIT_SUCCESS != result) {
    FPRINTF(stderr, "  n=%i k=%i uplo=%c trans=%c twok=%i elemsize=%i\n", n, k, uplo, trans, twok, (int)elemsize);
  }
  free(a); free(b); free(c); free(d); free(sum);
  return result;
}


/* SYRK and GEMM of the same shape share a registry, either one first. */
static int check_registry(int syrk_first, int ldc)
{
  enum { N = 16, K = 32 };
  double a[N * K], c[64 * N + 1];
  libxs_gemm_backend_t backend;
  libxs_gemm_shape_t gshape;
  libxs_syrk_shape_t shape;
  libxs_gemm_config_t* gemm = NULL;
  libxs_syrk_t plan;
  libxs_registry_t *const registry = libxs_registry_create();
  int result = EXIT_SUCCESS, i, j, p, round;
  TEST_CHECK(NULL != registry);
  LIBXS_MEMZERO(&backend);
  LIBXS_MEMZERO(&plan);
  backend.jit_create_dgemm = test_create;
  backend.jit_get_dgemm = test_get;
  LIBXS_MEMZERO(&gshape);
  gshape.datatype = LIBXS_DATATYPE_F64;
  gshape.transa = 'N'; gshape.transb = 'T';
  gshape.m = gshape.n = gshape.lda = gshape.ldb = N;
  gshape.k = K; gshape.ldc = ldc;
  gshape.alpha = 1;
  shape.datatype = LIBXS_DATATYPE_F64;
  shape.uplo = 'L'; shape.trans = 'N';
  shape.n = shape.lda = N;
  shape.k = K; shape.ldb = 0; shape.ldc = ldc;
  for (round = 0; round < 2 && EXIT_SUCCESS == result; ++round) {
    for (i = 0; i < 64; ++i) { /* past the warm-up */
      if (round == (0 != syrk_first ? 0 : 1)) {
        TEST_CHECK(0 != libxs_syrk_dispatch_rt(&plan, &shape, NULL, &backend, registry));
      }
      else {
        gemm = libxs_gemm_dispatch_rt(&gshape, NULL, &backend, registry);
        TEST_CHECK(NULL != gemm);
      }
    }
  }
  if (EXIT_SUCCESS == result) {
    for (j = 0; j < K; ++j) {
      for (i = 0; i < N; ++i) a[i + j * N] = value(1, i, j);
    }
    for (round = 0; round < 2; ++round) {
      for (i = 0; i < 64 * N + 1; ++i) c[i] = 7;
      if (0 == round) libxs_gemm_call(gemm, a, a, c);
      else TEST_CHECK(EXIT_SUCCESS == libxs_syrk(&plan, 1.0, a, NULL, 0.0, c));
      for (j = 0; j < N; ++j) {
        for (i = 0; i < ldc; ++i) {
          double ref = 7;
          if (i < N && (0 == round || i >= j)) {
            ref = 0;
            for (p = 0; p < K; ++p) ref += a[i + p * N] * a[j + p * N];
          }
          TEST_CHECK(LIBXS_DELTA(c[i + j * ldc], ref) <= 1E-12);
        }
      }
      for (i = ldc * N; i < 64 * N + 1; ++i) TEST_CHECK(7 == c[i]);
    }
  }
  if (NULL != registry) libxs_registry_destroy(registry); /* the handles are the test's */
  return result;
}


/* What is invalid is refused rather than computed. */
static int check_arguments(void)
{
  double a[16], c[16];
  libxs_syrk_shape_t shape;
  libxs_syrk_t plan;
  int result = EXIT_SUCCESS;
  memset(a, 0, sizeof(a));
  memset(c, 0, sizeof(c));
  shape.datatype = LIBXS_DATATYPE_F64;
  shape.uplo = 'l'; shape.trans = 'c';
  shape.n = 4; shape.k = 4;
  shape.lda = 4; shape.ldb = 0; shape.ldc = 4;
  TEST_CHECK(0 != libxs_syrk_dispatch(&plan, &shape, NULL));
  TEST_CHECK('L' == plan.shape.uplo && 'T' == plan.shape.trans);
  TEST_CHECK(EXIT_SUCCESS == libxs_syrk(&plan, 1.0, a, NULL, 0.0, c));
  TEST_CHECK(EXIT_SUCCESS == libxs_syrk(&plan, 1.0, a, a, 0.0, c));
  TEST_CHECK(EXIT_SUCCESS != libxs_syrk(&plan, 1.0, a, c, 0.0, c)); /* B for a plan of SYRK */
  TEST_CHECK(EXIT_SUCCESS != libxs_syrk(&plan, 1.0, NULL, NULL, 0.0, c));
  TEST_CHECK(EXIT_SUCCESS != libxs_syrk(&plan, 1.0, a, NULL, 0.0, NULL));
  TEST_CHECK(EXIT_SUCCESS != libxs_syrk_task(&plan, 1.0, a, NULL, 0.0, c, 2, 2));
  TEST_CHECK(1 == libxs_syrk_ntasks(&plan, 8)); /* a single tile */
  shape.ldb = 4;
  TEST_CHECK(0 != libxs_syrk_dispatch(&plan, &shape, NULL));
  TEST_CHECK(EXIT_SUCCESS != libxs_syrk(&plan, 1.0, a, NULL, 0.0, c)); /* no B for a plan of SYR2K */
  shape.ldb = 3;
  TEST_CHECK(0 == libxs_syrk_dispatch(&plan, &shape, NULL));
  TEST_CHECK(EXIT_SUCCESS != libxs_syrk(&plan, 1.0, a, a, 0.0, c)); /* a failed dispatch */
  shape.ldb = 0; shape.ldc = 3;
  TEST_CHECK(0 == libxs_syrk_dispatch(&plan, &shape, NULL));
  shape.ldc = 4; shape.uplo = 'X';
  TEST_CHECK(0 == libxs_syrk_dispatch(&plan, &shape, NULL));
  shape.uplo = 'U'; shape.trans = 'X';
  TEST_CHECK(0 == libxs_syrk_dispatch(&plan, &shape, NULL));
  TEST_CHECK(0 == libxs_syrk_dispatch(&plan, NULL, NULL));
  libxs_syrk_release(&plan);
  libxs_syrk_release(&plan);
  libxs_syrk_release(NULL);
  return result;
}


/**
 * This test case is NOT an example of how to use LIBXS
 * since INTERNAL functions are tested which are not part
 * of the LIBXS API.
 */
int main(void)
{
  const int sizes[][2] = {
    { 0, 5 }, { 1, 1 }, { 4, 3 }, { 16, 32 }, { 32, 48 }, { 33, 49 }, { 64, 64 },
    { 100, 77 }, { 128, 300 }, { 129, 5 }, { 300, 0 }, { 300, 100 }, { 800, 20 }
  };
  const libxs_data_t types[] = { LIBXS_DATATYPE_F64, LIBXS_DATATYPE_F32 };
  int result = EXIT_SUCCESS, s, v;
  libxs_init();
#if defined(_OPENMP)
  omp_set_num_threads(1); /* bits are compared: no threaded BLAS in the serial calls */
#endif
  TEST_CHECK(EXIT_SUCCESS == check_arguments());
  TEST_CHECK(EXIT_SUCCESS == check_registry(1, 16));
  TEST_CHECK(EXIT_SUCCESS == check_registry(0, 16));
  TEST_CHECK(EXIT_SUCCESS == check_registry(1, 64));
  TEST_CHECK(EXIT_SUCCESS == check_registry(0, 64));
  if (0 < internal_libxs_gemm_jit_max && INTERNAL_GEMM_BACKEND_MKL_JIT >= internal_libxs_gemm_backend) {
    TEST_CHECK(0 < test_ncalls); /* otherwise the registry was not tested */
  }
  for (s = 0; s < (int)(sizeof(sizes) / sizeof(*sizes)) && EXIT_SUCCESS == result; ++s) {
    for (v = 0; v < 16 && EXIT_SUCCESS == result; ++v) {
      result = check(types[v >> 3], (char)(0 != (v & 1) ? 'U' : 'L'), (char)(0 != (v & 2) ? 'T' : 'N'),
        0 != (v & 4), sizes[s][0], sizes[s][1]);
    }
  }
  return result;
}
