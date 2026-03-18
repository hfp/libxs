/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_GEMM_H
#define LIBXS_GEMM_H

#include "libxs.h"

/** Standard Fortran BLAS dgemm signature (e.g., dgemm_). */
typedef void (*libxs_dgemm_blas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc);

/** Standard Fortran BLAS sgemm signature (e.g., sgemm_). */
typedef void (*libxs_sgemm_blas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc);

/**
 * MKL JIT dgemm kernel signature (mkl_jit_get_dgemm_ptr).
 * Shape, alpha, beta, and transpose info are baked into the jitter handle.
 */
typedef void (*libxs_dgemm_jit_t)(void* jitter,
  const double* a, const double* b, double* c);

/**
 * MKL JIT sgemm kernel signature (mkl_jit_get_sgemm_ptr).
 * Shape, alpha, beta, and transpose info are baked into the jitter handle.
 */
typedef void (*libxs_sgemm_jit_t)(void* jitter,
  const float* a, const float* b, float* c);

/** Flags controlling GEMM batch synchronization (bitfield). */
typedef enum libxs_gemm_flags_t {
  LIBXS_GEMM_FLAGS_DEFAULT = 0,
  LIBXS_GEMM_FLAG_NOLOCK = 1
} libxs_gemm_flags_t;

/**
 * Configuration supplying GEMM kernels. Pass NULL to batch functions
 * to use the built-in default kernel (auto-vectorized, no BLAS dependency).
 * Users can supply their own kernels. Kernel selection priority:
 *   1. JIT kernel (dgemm_jit/sgemm_jit + jitter) if non-NULL,
 *   2. BLAS kernel (dgemm/sgemm) if non-NULL,
 *   3. built-in default kernel.
 * Only the function pointers matching the datatype need to be set.
 * By default (flags=0), _task variants synchronize C-matrix updates.
 * Set LIBXS_GEMM_FLAG_NOLOCK if no duplicate C pointers exist.
 */
typedef struct libxs_gemm_config_t {
  libxs_dgemm_blas_t dgemm_blas;
  libxs_sgemm_blas_t sgemm_blas;
  libxs_dgemm_jit_t dgemm_jit;
  libxs_sgemm_jit_t sgemm_jit;
  void* jitter;
  libxs_gemm_flags_t flags;
} libxs_gemm_config_t;

/**
 * Process a batch of GEMMs with strided access (constant offsets between matrices).
 * C_i := alpha * op(A_i) * op(B_i) + beta * C_i, for i in [0, batchsize).
 * Matrices are at: A + i*stride_a*elemsize, B + i*stride_b*elemsize, C + i*stride_c*elemsize.
 * Pass config=NULL to use the built-in kernel.
 */
LIBXS_API void libxs_gemm_strided(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, int stride_a,
                     const void* b, int ldb, int stride_b,
  const void* beta,        void* c, int ldc, int stride_c,
  int batchsize, const libxs_gemm_config_t* config);

/** Per-thread form of libxs_gemm_strided. */
LIBXS_API void libxs_gemm_strided_task(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, int stride_a,
                     const void* b, int ldb, int stride_b,
  const void* beta,        void* c, int ldc, int stride_c,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);

/**
 * Process a batch of GEMMs given arrays of pointers to matrices.
 * C_i := alpha * op(A_i) * op(B_i) + beta * C_i, for i in [0, batchsize).
 * Pass config=NULL to use the built-in kernel.
 */
LIBXS_API void libxs_gemm_batch(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a_array[], int lda,
                     const void* b_array[], int ldb,
  const void* beta,        void* c_array[], int ldc,
  int batchsize, const libxs_gemm_config_t* config);

/** Per-thread form of libxs_gemm_batch. */
LIBXS_API void libxs_gemm_batch_task(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a_array[], int lda,
                     const void* b_array[], int ldb,
  const void* beta,        void* c_array[], int ldc,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);

/**
 * Process groups of batched GEMMs with varying parameters.
 * Each group i has its own transa, transb, m, n, k, lda, ldb, ldc, and batchsize.
 * The a/b/c pointer arrays are concatenated across groups.
 * alpha and beta are arrays of ngroups scalars (each LIBXS_TYPESIZE Bytes).
 * Pass config=NULL to use the built-in kernel.
 */
LIBXS_API void libxs_gemm_groups(
  libxs_data_t datatype, const char transa_array[], const char transb_array[],
  const int m_array[], const int n_array[], const int k_array[],
  const void* alpha_array, const void* a_array[], const int lda_array[],
                           const void* b_array[], const int ldb_array[],
  const void* beta_array,        void* c_array[], const int ldc_array[],
  int ngroups, const int batchsize[],
  const libxs_gemm_config_t* config);

#endif /*LIBXS_GEMM_H*/
