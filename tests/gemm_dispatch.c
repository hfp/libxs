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

#include <stdio.h>


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
  return 1; /* MKL_NO_JIT */
}


static void* test_jit_get_dgemm(void* jitter)
{
  LIBXS_UNUSED(jitter);
  ++jit_get_dgemm_calls;
  return NULL;
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
  return 1; /* MKL_NO_JIT */
}


static void* test_jit_get_sgemm(void* jitter)
{
  LIBXS_UNUSED(jitter);
  ++jit_get_sgemm_calls;
  return NULL;
}


static void test_dgemm_jit(void* jitter,
  const double* a, const double* b, double* c)
{
  LIBXS_UNUSED(jitter); LIBXS_UNUSED(a);
  LIBXS_UNUSED(b); LIBXS_UNUSED(c);
  ++jit_call_calls;
}


static void test_sgemm_jit(void* jitter,
  const float* a, const float* b, float* c)
{
  LIBXS_UNUSED(jitter); LIBXS_UNUSED(a);
  LIBXS_UNUSED(b); LIBXS_UNUSED(c);
  ++jit_call_calls;
}


static void test_xgemm_call(const void* param)
{
  LIBXS_UNUSED(param);
  ++xgemm_call_calls;
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
  for (i = 0; i < 8; ++i) {
    config = libxs_gemm_dispatch_rt(&shape, NULL, &backend, registry);
  }
  TEST_CHECK(NULL != config);
  TEST_CHECK(1 == jit_create_dgemm_calls);
  TEST_CHECK(0 == jit_get_dgemm_calls);
  TEST_CHECK(NULL == config->dgemm_jit);
  TEST_CHECK(NULL == config->jitter);
  TEST_CHECK(NULL != config->dgemm_blas);

  shape.datatype = LIBXS_DATATYPE_F32;
  for (i = 0; i < 8; ++i) {
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

  return EXIT_SUCCESS;
}


int main(void)
{
  int result = test_missing_jit_handle();
  if (EXIT_SUCCESS == result) result = test_incomplete_jit_config();
  return result;
}
