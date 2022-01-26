/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_SPMDM_H
#define LIBXS_SPMDM_H

#include "libxs_typedefs.h"


typedef enum libxs_spmdm_datatype {
  LIBXS_SPMDM_DATATYPE_F32,
  LIBXS_SPMDM_DATATYPE_BFLOAT16
} libxs_spmdm_datatype;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_spmdm_handle {
  /* The following are the matrix multiply dimensions: A (sparse): m X k, B (dense): k X n, Output C (dense): m X n */
  int m;
  int n;
  int k;
  /* The block sizes for A, B and C. */
  /* Here we fix A to be divided into 128 X 128 blocks, B/C to be 128 X 48 for HSW/BDW and 128 X 96 for SKX */
  int bm;
  int bn;
  int bk;
  /* The number of blocks for the m, n and k dimensions */
  int mb;
  int nb;
  int kb;
  libxs_spmdm_datatype datatype;
  char* base_ptr_scratch_A;
  char* base_ptr_scratch_B_scratch_C;
  int memory_for_scratch_per_thread;
} libxs_spmdm_handle;

/**
 * This stores a single sparse splice (or block) of sparse matrix A using a CSR representation (rowidx, colidx, and values
 * Each splice corresponds to a bm X bk region of A, and stores local indexes
 */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_CSR_sparseslice {
  /* Since bm and bk are assumed to be <=256, a 16-bit integer is enough to store the local rowidx, colidx */
  uint16_t* rowidx;
  uint16_t* colidx;
  float*    values;
} libxs_CSR_sparseslice;


LIBXS_API void libxs_spmdm_init(
  int M, int N, int K,
  int max_threads,
  libxs_spmdm_handle* handle,
  libxs_CSR_sparseslice** libxs_output_csr);

LIBXS_API void libxs_spmdm_destroy(
  libxs_spmdm_handle* handle);

LIBXS_API int libxs_spmdm_get_num_createSparseSlice_blocks(
  const libxs_spmdm_handle* handle);

LIBXS_API int libxs_spmdm_get_num_compute_blocks(
  const libxs_spmdm_handle* handle);

/** This converts a dense representation of the sparse matrix to 2D array of sparse slices. */
LIBXS_API void libxs_spmdm_createSparseSlice_fp32_thread(
  const libxs_spmdm_handle* handle,
  char transa,
  const float* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads);

LIBXS_API void libxs_spmdm_createSparseSlice_bfloat16_thread(
  const libxs_spmdm_handle* handle,
  char transa,
  const libxs_bfloat16* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads);

/** NOTE: This code currently ignores alpha input to the matrix multiply */
LIBXS_API void libxs_spmdm_compute_fp32_thread(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const float* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const float* b,
  char transc,
  const float* beta,
  float* c,
  int block_id,
  int tid, int nthreads);

/** NOTE: This code currently ignores alpha input to the matrix multiply */
LIBXS_API void libxs_spmdm_compute_bfloat16_thread(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const libxs_bfloat16* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const libxs_bfloat16* b,
  char transc,
  const libxs_bfloat16* beta,
  float* c,
  int block_id,
  int tid, int nthreads);

#endif /*LIBXS_SPMDM_H*/
