/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_GEMM_INTERNAL_H
#define LIBXS_GEMM_INTERNAL_H

#include <libxs/libxs_gemm.h>
#include "libxs_main.h"

#if !defined(LIBXS_GEMM_PRINT) && 1
# define LIBXS_GEMM_PRINT
#endif

#define INTERNAL_GEMM_BACKEND_AUTO 0
#define INTERNAL_GEMM_BACKEND_MKL_JIT 1
#define INTERNAL_GEMM_BACKEND_LIBXSMM 2
#define INTERNAL_GEMM_BACKEND_BLAS 3
#define INTERNAL_GEMM_BACKEND_DEFAULT 4


typedef int (*internal_libxs_blas_query_t)(void);


/* GEMM settings and resolved entry points shared by SYRK and TAS (libxs_gemm.c) */
LIBXS_APIVAR_PRIVATE(int internal_libxs_gemm_bm);
LIBXS_APIVAR_PRIVATE(int internal_libxs_gemm_bn);
LIBXS_APIVAR_PRIVATE(int internal_libxs_gemm_bk);
LIBXS_APIVAR_PRIVATE(int internal_libxs_gemm_panel);
LIBXS_APIVAR_PRIVATE(int internal_libxs_gemm_jit_max);
LIBXS_APIVAR_PRIVATE(int internal_libxs_gemm_backend);
LIBXS_APIVAR_PRIVATE(libxs_gemm_dblas_t internal_libxs_dgemm_blas);
LIBXS_APIVAR_PRIVATE(libxs_gemm_sblas_t internal_libxs_sgemm_blas);
LIBXS_APIVAR_PRIVATE(internal_libxs_blas_query_t internal_libxs_mkl_nthreads);
LIBXS_APIVAR_PRIVATE(internal_libxs_blas_query_t internal_libxs_openblas_parallel);
LIBXS_APIVAR_PRIVATE(internal_libxs_blas_query_t internal_libxs_openblas_nthreads);
LIBXS_APIVAR_PRIVATE(libxs_blas_nthreads_t internal_libxs_blas_nthreads);


/** Built-in auto-vectorized GEMM, the last resort behind BLAS (libxs_gemm.c). */
LIBXS_API_INTERN void internal_libxs_dgemm_default(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc,
  size_t transa_len, size_t transb_len);
LIBXS_API_INTERN void internal_libxs_sgemm_default(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc,
  size_t transa_len, size_t transb_len);

/** Dispatch behind the public flavors; own is the caller's config (libxs_gemm.c). */
LIBXS_API_INTERN libxs_gemm_config_t* internal_libxs_gemm_dispatch(
  const libxs_gemm_shape_t* shape,
  const libxs_gemm_shape_t* kernel_shape,
  const libxs_gemm_backend_t* backend,
  void* registry, libxs_gemm_config_t* own);

/** BLAS call of an arbitrary shape with the transposes and entry points of config. */
LIBXS_API_INTERN void internal_libxs_gemm_blas(
  const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c,
  int m, int n, int k,
  int lda, int ldb, int ldc,
  double alpha, double beta);

/** Resolves the BLAS SYRK/SYR2K entry points, called by internal_libxs_gemm_init (libxs_syrk.c). */
LIBXS_API_INTERN void internal_libxs_syrk_init(void);

/**
 * Identifies the process behind a line, because ranks share one stream and
 * their lines interleave. The rank is preferred where it can be determined
 * (libxs_rid), and the value is stable for the lifetime of the process.
 */
LIBXS_API_INLINE unsigned int internal_libxs_gemm_origin(void)
{
  static unsigned int origin = 0xFFFFFFFF;
  if (0xFFFFFFFF == origin) origin = libxs_rid();
  return origin;
}

/* nonzero if the BLAS parallelizes by itself; a library that cannot tell counts as sequential */
LIBXS_API_INLINE int internal_libxs_blas_threaded(void)
{
  int result = 0;
  if (NULL != internal_libxs_mkl_nthreads) {
    result = (1 < internal_libxs_mkl_nthreads());
  }
  else if (NULL != internal_libxs_openblas_parallel
    && NULL != internal_libxs_openblas_nthreads)
  {
    result = (0 != internal_libxs_openblas_parallel()
      && 1 < internal_libxs_openblas_nthreads());
  }
  else if (NULL != internal_libxs_blas_nthreads) { /* backend of the caller (static link) */
    result = (1 < internal_libxs_blas_nthreads());
  }
  return result;
}

/** Assigns the BLAS entry points, either the backend's or the library's. */
LIBXS_API_INLINE void internal_libxs_gemm_blas_init(
  libxs_gemm_config_t* config,
  const libxs_gemm_backend_t* backend, int use_blas)
{
  if (NULL != backend && NULL != backend->blas_nthreads) {
    internal_libxs_blas_nthreads = backend->blas_nthreads;
  }
  if (0 != use_blas) {
    config->dgemm_blas = (NULL != backend && NULL != backend->dgemm_blas)
      ? backend->dgemm_blas : (NULL != internal_libxs_dgemm_blas)
      ? internal_libxs_dgemm_blas : internal_libxs_dgemm_default;
    config->sgemm_blas = (NULL != backend && NULL != backend->sgemm_blas)
      ? backend->sgemm_blas : (NULL != internal_libxs_sgemm_blas)
      ? internal_libxs_sgemm_blas : internal_libxs_sgemm_default;
  }
  else {
    config->dgemm_blas = internal_libxs_dgemm_default;
    config->sgemm_blas = internal_libxs_sgemm_default;
  }
}

#endif /*LIBXS_GEMM_INTERNAL_H*/
