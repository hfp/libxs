/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_SYRK_H
#define LIBXS_SYRK_H

#include "libxs_gemm.h"


/**
 * Dispatch a GEMM config suitable for libxs_syr2k (runtime, explicit
 * backend selection). The kernel shape is the SYRK tile, hence the
 * blocking (LIBXS_GEMM_BM/BN/BK) is applied inside. A shape wider than
 * one tile runs the BLAS SYRK where that entry point is available
 * (LIBXS_SYRK_BLAS) and the call is not split, which reads the shape of
 * the config but never its kernel. The caller-owned flavors then skip the kernel and the registry
 * alike, whereas this flavor keeps an entry, because a returned pointer
 * has to stay valid.
 * backend: optional backend pointers (NULL = built-in only).
 * Returns pointer to cached config (registry-owned), or NULL.
 */
LIBXS_API libxs_gemm_config_t* libxs_syr2k_dispatch_rt(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  const libxs_gemm_backend_t* LIBXS_ARGDEF(backend, NULL),
  void* LIBXS_ARGDEF(registry, NULL));

/**
 * Dispatch a GEMM config suitable for libxs_syrk (runtime, explicit
 * backend selection). Equivalent to libxs_syr2k_dispatch_rt with ldb=lda.
 */
LIBXS_API libxs_gemm_config_t* libxs_syrk_dispatch_rt(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  const libxs_gemm_backend_t* LIBXS_ARGDEF(backend, NULL),
  void* LIBXS_ARGDEF(registry, NULL));

/** libxs_syr2k_dispatch_rt populating a caller-owned config (see _cpy_rt). */
LIBXS_API int libxs_syr2k_dispatch_cpy_rt(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  const libxs_gemm_backend_t* LIBXS_ARGDEF(backend, NULL),
  void* LIBXS_ARGDEF(registry, NULL));

/** libxs_syrk_dispatch_rt populating a caller-owned config (see _cpy_rt). */
LIBXS_API int libxs_syrk_dispatch_cpy_rt(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  const libxs_gemm_backend_t* LIBXS_ARGDEF(backend, NULL),
  void* LIBXS_ARGDEF(registry, NULL));

/**
 * Dispatch a GEMM config suitable for libxs_syr2k, with the backend
 * detected at the caller's compile time (libxs_gemm_backend_init).
 * Returns pointer to cached config (registry-owned), or NULL.
 */
LIBXS_API_INLINE libxs_gemm_config_t* libxs_syr2k_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  void* LIBXS_ARGDEF(registry, NULL))
{
  libxs_gemm_backend_t be;
  libxs_gemm_backend_init(&be);
  return libxs_syr2k_dispatch_rt(datatype, n, k, lda, ldb, ldc,
    &be, registry);
}

/**
 * Dispatch a GEMM config suitable for libxs_syrk, with the backend
 * detected at the caller's compile time (libxs_gemm_backend_init).
 * Returns pointer to cached config (registry-owned), or NULL.
 */
LIBXS_API_INLINE libxs_gemm_config_t* libxs_syrk_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  void* LIBXS_ARGDEF(registry, NULL))
{
  return libxs_syr2k_dispatch(datatype, n, k, lda, lda, ldc, registry);
}

/**
 * Like libxs_syr2k_dispatch, but populates a caller-owned config.
 * Returns non-zero if config was populated, zero otherwise.
 */
LIBXS_API_INLINE int libxs_syr2k_dispatch_cpy(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  void* LIBXS_ARGDEF(registry, NULL))
{
  libxs_gemm_backend_t be;
  libxs_gemm_backend_init(&be);
  return libxs_syr2k_dispatch_cpy_rt(config,
    datatype, n, k, lda, ldb, ldc, &be, registry);
}

/**
 * Like libxs_syrk_dispatch, but populates a caller-owned config.
 * Returns non-zero if config was populated, zero otherwise.
 */
LIBXS_API_INLINE int libxs_syrk_dispatch_cpy(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  void* LIBXS_ARGDEF(registry, NULL))
{
  libxs_gemm_backend_t be;
  libxs_gemm_backend_init(&be);
  return libxs_syrk_dispatch_cpy_rt(config,
    datatype, n, k, lda, ldc, &be, registry);
}

/**
 * Symmetric rank-2k update: C := alpha*(A*B^T + B*A^T) + beta*C.
 * Only the triangle specified by uplo ('U' or 'L') is written.
 * Scratch is managed internally (TLS buffer, grown as needed).
 */
LIBXS_API void libxs_syr2k(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c);

/**
 * Per-thread form of libxs_syr2k. A BLAS SYR2K serves a single task, hence
 * it is taken for ntasks=1 only and any split runs the decomposition.
 */
LIBXS_API void libxs_syr2k_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c,
  int tid, int ntasks);

/**
 * Symmetric rank-k update: C := alpha*A*A^T + beta*C.
 * Only the triangle specified by uplo ('U' or 'L') is written.
 * Scratch is managed internally (TLS buffer, grown as needed).
 */
LIBXS_API void libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c);

/**
 * Per-thread form of libxs_syrk. A BLAS SYRK serves a single task, hence
 * it is taken for ntasks=1 only and any split runs the decomposition.
 */
LIBXS_API void libxs_syrk_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c,
  int tid, int ntasks);

/**
 * Number of tasks worth splitting libxs_syrk_task or libxs_syr2k_task into,
 * given nthreads available. Returns 1 where the unsplit call is expected to be
 * faster: a shape within one tile, a BLAS SYRK that is threaded by itself (MKL,
 * OpenBLAS), or fewer than LIBXS_SYRK_MINTASKS threads against a sequential BLAS
 * SYRK. Never exceeds the block columns there are to distribute. A threaded
 * BLAS is detected at runtime, or by the caller's backend (blas_nthreads). The Fortran module uses it to parallelize libxs_syrk/libxs_syr2k
 * when compiled with OpenMP.
 */
LIBXS_API int libxs_syrk_ntasks(const libxs_gemm_config_t* config, int nthreads);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_SYRK_H*/
