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

/** Upper bound on LIBXS_GEMM_JIT_WARMUP (default: 8) taken by this test. */
#define TEST_MAXWARMUP 64

#define TEST_CHECK(EXPR) do { \
  if (!(EXPR)) { \
    fprintf(stderr, "FAIL: %s:%i (%s)\n", __FILE__, __LINE__, #EXPR); \
    return EXIT_FAILURE; \
  } \
} while(0)


static int jit_create_dgemm_calls;
static int jit_get_dgemm_calls;
static int jit_create_sgemm_calls;
static int jit_get_sgemm_calls;
static int jit_call_calls;
static int xgemm_call_calls;
static int jit_create_handle_calls;
static void* jit_call_jitter;
static int jit_handle;


static void test_dgemm_jit(void* jitter,
  const double* a, const double* b, double* c)
{
  LIBXS_UNUSED(a); LIBXS_UNUSED(b); LIBXS_UNUSED(c);
  jit_call_jitter = jitter;
  ++jit_call_calls;
}


static void test_sgemm_jit(void* jitter,
  const float* a, const float* b, float* c)
{
  LIBXS_UNUSED(a); LIBXS_UNUSED(b); LIBXS_UNUSED(c);
  jit_call_jitter = jitter;
  ++jit_call_calls;
}


static void test_xgemm_call(const void* param)
{
  LIBXS_UNUSED(param);
  ++xgemm_call_calls;
}


static void* test_vptr(void (*fn)(void))
{
  union { void (*fn)(void); void* vp; } u;
  u.fn = fn;
  return u.vp;
}


static int test_jit_create_dgemm(void** jitter,
  int layout, int transa, int transb, int m, int n, int k,
  double alpha, int lda, int ldb, double beta, int ldc)
{
  LIBXS_UNUSED(jitter); LIBXS_UNUSED(layout);
  LIBXS_UNUSED(transa); LIBXS_UNUSED(transb);
  LIBXS_UNUSED(m); LIBXS_UNUSED(n); LIBXS_UNUSED(k);
  LIBXS_UNUSED(alpha); LIBXS_UNUSED(lda); LIBXS_UNUSED(ldb);
  LIBXS_UNUSED(beta); LIBXS_UNUSED(ldc);
  ++jit_create_dgemm_calls;
  return 1; /* MKL_NO_JIT, handle left unpopulated */
}


static void* test_jit_get_dgemm(void* jitter)
{
  LIBXS_UNUSED(jitter);
  ++jit_get_dgemm_calls;
  return test_vptr((void (*)(void))test_dgemm_jit);
}


static int test_jit_create_sgemm(void** jitter,
  int layout, int transa, int transb, int m, int n, int k,
  float alpha, int lda, int ldb, float beta, int ldc)
{
  LIBXS_UNUSED(jitter); LIBXS_UNUSED(layout);
  LIBXS_UNUSED(transa); LIBXS_UNUSED(transb);
  LIBXS_UNUSED(m); LIBXS_UNUSED(n); LIBXS_UNUSED(k);
  LIBXS_UNUSED(alpha); LIBXS_UNUSED(lda); LIBXS_UNUSED(ldb);
  LIBXS_UNUSED(beta); LIBXS_UNUSED(ldc);
  ++jit_create_sgemm_calls;
  return 1; /* MKL_NO_JIT, handle left unpopulated */
}


static void* test_jit_get_sgemm(void* jitter)
{
  LIBXS_UNUSED(jitter);
  ++jit_get_sgemm_calls;
  return test_vptr((void (*)(void))test_sgemm_jit);
}


static int test_jit_create_handle(void** jitter,
  int layout, int transa, int transb, int m, int n, int k,
  double alpha, int lda, int ldb, double beta, int ldc)
{
  LIBXS_UNUSED(layout); LIBXS_UNUSED(transa); LIBXS_UNUSED(transb);
  LIBXS_UNUSED(m); LIBXS_UNUSED(n); LIBXS_UNUSED(k);
  LIBXS_UNUSED(alpha); LIBXS_UNUSED(lda); LIBXS_UNUSED(ldb);
  LIBXS_UNUSED(beta); LIBXS_UNUSED(ldc);
  *jitter = &jit_handle;
  ++jit_create_handle_calls;
  return 0; /* MKL_JIT_SUCCESS */
}


static void* test_jit_get_handle(void* jitter)
{
  LIBXS_UNUSED(jitter);
  return test_vptr((void (*)(void))test_dgemm_jit);
}


static int test_missing_jit_handle(void)
{
  libxs_gemm_backend_t backend;
  libxs_gemm_shape_t shape;
  libxs_gemm_config_t* config;
  libxs_registry_t* registry;
  int i;

  LIBXS_MEMZERO(&backend);
  backend.jit_create_dgemm = test_jit_create_dgemm;
  backend.jit_get_dgemm = test_jit_get_dgemm;
  backend.jit_create_sgemm = test_jit_create_sgemm;
  backend.jit_get_sgemm = test_jit_get_sgemm;

  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  LIBXS_MEMZERO(&shape);
  shape.datatype = LIBXS_DATATYPE_F64;
  shape.transa = 'N'; shape.transb = 'N';
  shape.m = 2; shape.n = 2; shape.k = 2;
  shape.lda = 2; shape.ldb = 2; shape.ldc = 2;
  shape.alpha = 1.0;
  config = NULL;
  for (i = 0; i < TEST_MAXWARMUP && 0 == jit_create_dgemm_calls; ++i) {
    config = libxs_gemm_dispatch_rt(&shape, NULL, &backend, registry);
  }
  TEST_CHECK(NULL != config);
  TEST_CHECK(1 == i); /* empty registry: no warm-up needed */
  TEST_CHECK(1 == jit_create_dgemm_calls);
  TEST_CHECK(0 == jit_get_dgemm_calls);
  TEST_CHECK(NULL == config->dgemm_jit);
  TEST_CHECK(NULL == config->jitter);
  TEST_CHECK(NULL != config->dgemm_blas);

  shape.datatype = LIBXS_DATATYPE_F32;
  config = NULL;
  for (i = 0; i < TEST_MAXWARMUP && 0 == jit_create_sgemm_calls; ++i) {
    config = libxs_gemm_dispatch_rt(&shape, NULL, &backend, registry);
  }
  TEST_CHECK(NULL != config);
  TEST_CHECK(1 == jit_create_sgemm_calls);
  TEST_CHECK(0 == jit_get_sgemm_calls);
  TEST_CHECK(NULL == config->sgemm_jit);
  TEST_CHECK(NULL == config->jitter);
  TEST_CHECK(NULL != config->sgemm_blas);

  libxs_gemm_release_registry(registry);
  return EXIT_SUCCESS;
}


static int test_incomplete_jit_config(void)
{
  libxs_gemm_config_t config;

  LIBXS_MEMZERO(&config);
  config.dgemm_jit = test_dgemm_jit;
  config.xgemm = test_xgemm_call;
  libxs_gemm_call(&config, NULL, NULL, NULL);
  TEST_CHECK(0 == jit_call_calls);
  TEST_CHECK(1 == xgemm_call_calls);

  LIBXS_MEMZERO(&config);
  config.sgemm_jit = test_sgemm_jit;
  config.xgemm = test_xgemm_call;
  libxs_gemm_call(&config, NULL, NULL, NULL);
  TEST_CHECK(0 == jit_call_calls);
  TEST_CHECK(2 == xgemm_call_calls);

  LIBXS_MEMZERO(&config);
  config.jitter = &jit_handle;
  config.xgemm = test_xgemm_call;
  libxs_gemm_call(&config, NULL, NULL, NULL);
  TEST_CHECK(0 == jit_call_calls);
  TEST_CHECK(3 == xgemm_call_calls);

  return EXIT_SUCCESS;
}


static int test_complete_jit_config(void)
{
  libxs_gemm_config_t config;

  LIBXS_MEMZERO(&config);
  config.dgemm_jit = test_dgemm_jit;
  config.jitter = &jit_handle;
  config.xgemm = test_xgemm_call;
  libxs_gemm_call(&config, NULL, NULL, NULL);
  TEST_CHECK(1 == jit_call_calls);
  TEST_CHECK(3 == xgemm_call_calls);
  TEST_CHECK(&jit_handle == jit_call_jitter);

  LIBXS_MEMZERO(&config);
  config.sgemm_jit = test_sgemm_jit;
  config.jitter = &jit_handle;
  config.xgemm = test_xgemm_call;
  jit_call_jitter = NULL;
  libxs_gemm_call(&config, NULL, NULL, NULL);
  TEST_CHECK(2 == jit_call_calls);
  TEST_CHECK(3 == xgemm_call_calls);
  TEST_CHECK(&jit_handle == jit_call_jitter);

  return EXIT_SUCCESS;
}


static int test_double_dispatch(void)
{
  libxs_gemm_backend_t backend;
  libxs_gemm_shape_t shape, kshape;
  libxs_gemm_config_t *config, *kernel;
  libxs_registry_t* registry;
  int i;

  LIBXS_MEMZERO(&backend);
  backend.jit_create_dgemm = test_jit_create_handle;
  backend.jit_get_dgemm = test_jit_get_handle;

  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  LIBXS_MEMZERO(&shape);
  shape.datatype = LIBXS_DATATYPE_F64;
  shape.transa = 'N'; shape.transb = 'T';
  shape.m = 8; shape.n = 8; shape.k = 8;
  shape.lda = 16; shape.ldb = 16; shape.ldc = 16;
  shape.alpha = 1.0; shape.beta = 0.0;
  kshape = shape;
  kshape.ldc = 8; kshape.beta = 1.0;
  config = NULL;
  for (i = 0; i < TEST_MAXWARMUP && 0 == jit_create_handle_calls; ++i) {
    config = libxs_gemm_dispatch_rt(&shape, &kshape, &backend, registry);
  }
  /* a kernel is generated even though warm-up registered the shape first */
  TEST_CHECK(1 == jit_create_handle_calls);
  TEST_CHECK(NULL != config);
  TEST_CHECK(NULL != config->dgemm_jit);
  TEST_CHECK(&jit_handle == config->jitter);
  kernel = (libxs_gemm_config_t*)libxs_registry_get(
    registry, &kshape, sizeof(kshape), NULL);
  TEST_CHECK(NULL != kernel);
  TEST_CHECK(config->jitter == kernel->jitter);
  /* exactly one config owns the handle, hence it is released once */
  TEST_CHECK(0 != (LIBXS_GEMM_FLAG_OWNJIT & kernel->flags));
  TEST_CHECK(0 == (LIBXS_GEMM_FLAG_OWNJIT & config->flags));
  libxs_gemm_release(config); /* alias: must not release the handle */
  TEST_CHECK(&jit_handle == kernel->jitter);
  TEST_CHECK(0 != (LIBXS_GEMM_FLAG_OWNJIT & kernel->flags));

  /* the mock handle must not reach the JIT-provider's destructor */
  kernel->jitter = NULL;
  kernel->flags = LIBXS_GEMM_FLAGS_DEFAULT;
  libxs_gemm_release_registry(registry);
  return EXIT_SUCCESS;
}


static int test_release_ownership(void)
{
  libxs_gemm_config_t config;

  LIBXS_MEMZERO(&config);
  config.dgemm_jit = test_dgemm_jit;
  config.jitter = &jit_handle;
  libxs_gemm_release(&config); /* not owned: nothing is released */
  TEST_CHECK(test_dgemm_jit == config.dgemm_jit);
  TEST_CHECK(&jit_handle == config.jitter);

  LIBXS_MEMZERO(&config);
  config.dgemm_jit = test_dgemm_jit;
  config.flags = LIBXS_GEMM_FLAG_OWNJIT; /* owned, but no handle */
  libxs_gemm_release(&config);
#if defined(mkl_jit_create_dgemm)
  TEST_CHECK(NULL == config.dgemm_jit);
  TEST_CHECK(0 == (LIBXS_GEMM_FLAG_OWNJIT & config.flags));
#endif

  return EXIT_SUCCESS;
}


static int test_config_cpy(void)
{
  libxs_gemm_config_t src, dst;
  const libxs_gemm_config_t* cfg;
  libxs_registry_t* registry;

  LIBXS_MEMZERO(&src);
  src.dgemm_jit = test_dgemm_jit;
  src.jitter = &jit_handle;
  src.flags = (libxs_gemm_flags_t)
    (LIBXS_GEMM_FLAG_OWNJIT | LIBXS_GEMM_FLAG_NOLOCK);
  src.shape.datatype = LIBXS_DATATYPE_F64;
  src.shape.m = 8; src.shape.n = 8; src.shape.k = 8;
  LIBXS_MEMZERO(&dst);
  TEST_CHECK(0 != libxs_gemm_config_cpy(&dst, &src));
  /* the copy shares the kernel but never owns the handle */
  TEST_CHECK(test_dgemm_jit == dst.dgemm_jit);
  TEST_CHECK(&jit_handle == dst.jitter);
  TEST_CHECK(0 == (LIBXS_GEMM_FLAG_OWNJIT & dst.flags));
  TEST_CHECK(0 != (LIBXS_GEMM_FLAG_NOLOCK & dst.flags));
  TEST_CHECK(LIBXS_DATATYPE_F64 == dst.shape.datatype);
  TEST_CHECK(8 == dst.shape.m && 8 == dst.shape.n && 8 == dst.shape.k);
  TEST_CHECK(0 != (LIBXS_GEMM_FLAG_OWNJIT & src.flags));
  libxs_gemm_release(&dst); /* not owned: must not release the handle */
  TEST_CHECK(&jit_handle == src.jitter);

  /* an unpopulated copy leaves the caller's config untouched */
  LIBXS_MEMZERO(&dst);
  dst.shape.m = 23;
  TEST_CHECK(0 == libxs_gemm_config_cpy(&dst, NULL));
  TEST_CHECK(23 == dst.shape.m);

  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  LIBXS_MEMZERO(&dst);
  TEST_CHECK(0 != libxs_syrk_dispatch_cpy(&dst, LIBXS_DATATYPE_F64,
    32, 8, 32, 32, registry));
  cfg = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, 32, 8, 32, 32, registry);
  TEST_CHECK(NULL != cfg);
  /* the copy carries the same shape as the registry-owned config */
  TEST_CHECK(cfg->shape.datatype == dst.shape.datatype);
  TEST_CHECK(cfg->shape.transa == dst.shape.transa);
  TEST_CHECK(cfg->shape.transb == dst.shape.transb);
  TEST_CHECK(cfg->shape.m == dst.shape.m);
  TEST_CHECK(cfg->shape.n == dst.shape.n);
  TEST_CHECK(cfg->shape.k == dst.shape.k);
  TEST_CHECK(cfg->shape.lda == dst.shape.lda);
  TEST_CHECK(cfg->shape.ldb == dst.shape.ldb);
  TEST_CHECK(cfg->shape.ldc == dst.shape.ldc);
  TEST_CHECK(cfg->dgemm_blas == dst.dgemm_blas);
  TEST_CHECK(0 == (LIBXS_GEMM_FLAG_OWNJIT & dst.flags));

  LIBXS_MEMZERO(&dst);
  TEST_CHECK(0 != libxs_gemm_dispatch_cpy(&dst, LIBXS_DATATYPE_F64,
    'N', 'N', 8, 8, 8, 8, 8, 8, NULL, NULL, registry));
  TEST_CHECK(LIBXS_DATATYPE_F64 == dst.shape.datatype);
  TEST_CHECK(8 == dst.shape.m && 8 == dst.shape.n && 8 == dst.shape.k);
  TEST_CHECK(1.0 == dst.shape.alpha && 0.0 == dst.shape.beta);
  TEST_CHECK(NULL != dst.dgemm_blas);
  TEST_CHECK(0 == (LIBXS_GEMM_FLAG_OWNJIT & dst.flags));

  libxs_gemm_release_registry(registry);
  return EXIT_SUCCESS;
}


static int test_warmup_sticky(void)
{
  libxs_gemm_backend_t backend;
  libxs_gemm_shape_t shape;
  libxs_gemm_config_t* config;
  libxs_registry_t* registry;
  int i, calls;

  LIBXS_MEMZERO(&backend);
  backend.jit_create_dgemm = test_jit_create_dgemm; /* MKL_NO_JIT */
  backend.jit_get_dgemm = test_jit_get_dgemm;

  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  LIBXS_MEMZERO(&shape);
  shape.datatype = LIBXS_DATATYPE_F64;
  shape.transa = 'N'; shape.transb = 'N';
  shape.m = 3; shape.n = 5; shape.k = 7;
  shape.lda = 3; shape.ldb = 7; shape.ldc = 3;
  shape.alpha = 1.0;
  /* occupy the registry, hence the shape below must prove reuse first */
  TEST_CHECK(NULL != libxs_gemm_dispatch_rt(&shape, NULL, &backend, registry));
  shape.m = 4; shape.n = 6; shape.k = 8;
  shape.lda = 4; shape.ldb = 8; shape.ldc = 4;
  jit_create_dgemm_calls = 0;
  config = NULL;
  for (i = 0; i < TEST_MAXWARMUP && 0 == jit_create_dgemm_calls; ++i) {
    config = libxs_gemm_dispatch_rt(&shape, NULL, &backend, registry);
    TEST_CHECK(NULL != config);
  }
  TEST_CHECK(1 < i); /* warm-up was required */
  TEST_CHECK(1 == jit_create_dgemm_calls);
  TEST_CHECK(NULL == config->dgemm_jit); /* MKL_NO_JIT yields no kernel */

  /* a shape whose JIT was attempted must not be attempted again */
  calls = jit_create_dgemm_calls;
  for (i = 0; i < TEST_MAXWARMUP; ++i) {
    TEST_CHECK(NULL != libxs_gemm_dispatch_rt(
      &shape, NULL, &backend, registry));
  }
  TEST_CHECK(calls == jit_create_dgemm_calls);

  libxs_gemm_release_registry(registry);
  return EXIT_SUCCESS;
}


static int test_registry_no_warmup_entries(void)
{
  libxs_gemm_backend_t backend;
  libxs_gemm_shape_t shape;
  libxs_gemm_config_t config;
  libxs_registry_t* registry;
  size_t n0;
  int i;

  LIBXS_MEMZERO(&backend);
  backend.jit_create_dgemm = test_jit_create_handle;
  backend.jit_get_dgemm = test_jit_get_handle;
  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  LIBXS_MEMZERO(&shape);
  shape.datatype = LIBXS_DATATYPE_F64;
  shape.transa = 'N'; shape.transb = 'N';
  shape.m = 5; shape.n = 5; shape.k = 5;
  shape.lda = 5; shape.ldb = 5; shape.ldc = 5;
  shape.alpha = 1.0;
  /* the first shape of a process skips warm-up, so spend that on a throwaway
     and keep these checks independent of the order main() runs them in */
  TEST_CHECK(NULL != libxs_gemm_dispatch_rt(
    &shape, NULL, &backend, registry));

  shape.m = 6; shape.n = 6; shape.k = 6;
  shape.lda = 6; shape.ldb = 6; shape.ldc = 6;
  n0 = libxs_registry_size(registry);
  jit_create_handle_calls = 0;
  for (i = 0; i < TEST_MAXWARMUP && 0 == jit_create_handle_calls; ++i) {
    LIBXS_MEMZERO(&config);
    TEST_CHECK(0 != libxs_gemm_dispatch_cpy_rt(
      &config, &shape, NULL, &backend, registry));
    if (0 == jit_create_handle_calls) { /* still warming up */
      TEST_CHECK(n0 == libxs_registry_size(registry));
      TEST_CHECK(NULL == config.dgemm_jit);
      TEST_CHECK(NULL != config.dgemm_blas); /* usable meanwhile */
    }
  }
  TEST_CHECK(1 < i); /* warm-up was required */
  TEST_CHECK(1 == jit_create_handle_calls);
  /* the kernel arrives and brings exactly one entry with it */
  TEST_CHECK(NULL != config.dgemm_jit);
  TEST_CHECK((n0 + 1) == libxs_registry_size(registry));
  /* a hot shape keeps returning the same kernel without growing further */
  for (i = 0; i < TEST_MAXWARMUP; ++i) {
    LIBXS_MEMZERO(&config);
    TEST_CHECK(0 != libxs_gemm_dispatch_cpy_rt(
      &config, &shape, NULL, &backend, registry));
    TEST_CHECK(NULL != config.dgemm_jit);
  }
  TEST_CHECK((n0 + 1) == libxs_registry_size(registry));
  TEST_CHECK(1 == jit_create_handle_calls);

  /* the mock handle must not reach the JIT-provider's destructor */
  { libxs_gemm_config_t *const e = (libxs_gemm_config_t*)libxs_registry_get(
      registry, &shape, sizeof(shape), NULL);
    TEST_CHECK(NULL != e);
    e->jitter = NULL;
    e->flags = LIBXS_GEMM_FLAGS_DEFAULT;
  }
  libxs_gemm_release_registry(registry);
  return EXIT_SUCCESS;
}


static int test_registry_no_blas_entries(void)
{
  libxs_gemm_backend_t backend;
  libxs_gemm_shape_t shape;
  libxs_gemm_config_t config;
  libxs_registry_t* registry;
  size_t n0;
  int i;

  LIBXS_MEMZERO(&backend);
  backend.jit_create_dgemm = test_jit_create_handle;
  backend.jit_get_dgemm = test_jit_get_handle;
  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  LIBXS_MEMZERO(&shape);
  shape.datatype = LIBXS_DATATYPE_F64;
  shape.transa = 'N'; shape.transb = 'N';
  shape.m = 7; shape.n = 7; shape.k = 7;
  shape.lda = 7; shape.ldb = 7; shape.ldc = 7;
  shape.alpha = 1.0;
  /* the first shape of a process skips warm-up, so spend that on a throwaway
     and keep these checks independent of the order main() runs them in */
  TEST_CHECK(NULL != libxs_gemm_dispatch_rt(
    &shape, NULL, &backend, registry));

  /* an arithmetic intensity of 2*128/24 is far above LIBXS_GEMM_JIT_MAX, so
     the gate refuses this shape whatever backend is offered */
  shape.m = 128; shape.n = 128; shape.k = 128;
  shape.lda = 128; shape.ldb = 128; shape.ldc = 128;
  n0 = libxs_registry_size(registry);
  jit_create_handle_calls = 0;
  for (i = 0; i < (4 * TEST_MAXWARMUP); ++i) {
    LIBXS_MEMZERO(&config);
    TEST_CHECK(0 != libxs_gemm_dispatch_cpy_rt(
      &config, &shape, NULL, &backend, registry));
    TEST_CHECK(NULL == config.dgemm_jit);
    TEST_CHECK(NULL != config.dgemm_blas);
    /* the registry must not grow by a single entry, ever */
    TEST_CHECK(n0 == libxs_registry_size(registry));
  }
  /* and the refusal is terminal: the gate is never re-evaluated by a JIT */
  TEST_CHECK(0 == jit_create_handle_calls);

  /* the pointer flavor still registers: a returned pointer must stay valid */
  TEST_CHECK(NULL != libxs_gemm_dispatch_rt(&shape, NULL, &backend, registry));
  TEST_CHECK((n0 + 1) == libxs_registry_size(registry));
  TEST_CHECK(NULL != libxs_gemm_dispatch_rt(&shape, NULL, &backend, registry));
  TEST_CHECK((n0 + 1) == libxs_registry_size(registry));

  libxs_gemm_release_registry(registry);
  return EXIT_SUCCESS;
}


/** The mock handle must not reach the JIT-provider's destructor. */
static void test_disown_jitters(libxs_registry_t* registry)
{
  const void* key = NULL;
  size_t cursor = 0;
  libxs_gemm_config_t* e = (libxs_gemm_config_t*)libxs_registry_begin(
    registry, &key, &cursor);
  while (NULL != e) {
    e->jitter = NULL;
    e->flags = LIBXS_GEMM_FLAGS_DEFAULT;
    e = (libxs_gemm_config_t*)libxs_registry_next(registry, &key, &cursor);
  }
}


static int test_registry_no_syrk_entries(void)
{
  libxs_gemm_backend_t backend;
  libxs_gemm_config_t config;
  libxs_registry_t* registry;
  int i;

  LIBXS_MEMZERO(&backend);
  backend.jit_create_dgemm = test_jit_create_handle;
  backend.jit_get_dgemm = test_jit_get_handle;
  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  /* spend the first-shape shortcut on a throwaway (see above) */
  TEST_CHECK(NULL != libxs_syrk_dispatch_rt(
    LIBXS_DATATYPE_F64, 9, 4, 9, 9, &backend, registry));
  libxs_gemm_release_registry(registry);

  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  jit_create_handle_calls = 0;
  /* a shape that fits one tile runs the kernel, hence it is dispatched: this
     double dispatch may enter neither the problem shape nor the tile while the
     shape is still proving reuse */
  for (i = 0; i < TEST_MAXWARMUP && 0 == jit_create_handle_calls; ++i) {
    LIBXS_MEMZERO(&config);
    TEST_CHECK(0 != libxs_syrk_dispatch_cpy_rt(
      &config, LIBXS_DATATYPE_F64, 16, 8, 16, 16, &backend, registry));
    if (0 == jit_create_handle_calls) {
      TEST_CHECK(0 == libxs_registry_size(registry));
      TEST_CHECK(NULL != config.dgemm_blas);
    }
  }
  TEST_CHECK(1 < i); /* warm-up was required */
  TEST_CHECK(1 == jit_create_handle_calls);
  /* exactly two entries once the kernel exists: the shape and its tile */
  TEST_CHECK(NULL != config.dgemm_jit);
  TEST_CHECK(2 == libxs_registry_size(registry));
  test_disown_jitters(registry);
  libxs_gemm_release_registry(registry);

  registry = libxs_registry_create();
  TEST_CHECK(NULL != registry);
  jit_create_handle_calls = 0;
  /* a shape wider than one tile runs the BLAS SYRK wherever that entry point
     is available, and a kernel nothing calls is then neither built nor entered;
     without the entry point the same shape tiles like the case above */
  for (i = 0; i < TEST_MAXWARMUP && 0 == jit_create_handle_calls; ++i) {
    LIBXS_MEMZERO(&config);
    TEST_CHECK(0 != libxs_syrk_dispatch_cpy_rt(
      &config, LIBXS_DATATYPE_F64, 890, 54, 890, 890, &backend, registry));
  }
  if (0 == jit_create_handle_calls) {
    TEST_CHECK(TEST_MAXWARMUP == i); /* no warm-up was ever due */
    TEST_CHECK(0 == libxs_registry_size(registry));
    TEST_CHECK(NULL == config.dgemm_jit);
    TEST_CHECK(NULL != config.dgemm_blas);
  }
  else {
    TEST_CHECK(1 == jit_create_handle_calls);
    TEST_CHECK(NULL != config.dgemm_jit);
    TEST_CHECK(2 == libxs_registry_size(registry));
  }
  test_disown_jitters(registry);
  libxs_gemm_release_registry(registry);
  return EXIT_SUCCESS;
}


int main(void)
{
  int result = test_missing_jit_handle();
  if (EXIT_SUCCESS == result) result = test_incomplete_jit_config();
  if (EXIT_SUCCESS == result) result = test_complete_jit_config();
  if (EXIT_SUCCESS == result) result = test_double_dispatch();
  if (EXIT_SUCCESS == result) result = test_release_ownership();
  if (EXIT_SUCCESS == result) result = test_config_cpy();
  if (EXIT_SUCCESS == result) result = test_warmup_sticky();
  if (EXIT_SUCCESS == result) result = test_registry_no_warmup_entries();
  if (EXIT_SUCCESS == result) result = test_registry_no_blas_entries();
  if (EXIT_SUCCESS == result) result = test_registry_no_syrk_entries();
  return result;
}
