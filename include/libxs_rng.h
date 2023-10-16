/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_UTILS_H
#define LIBXS_UTILS_H

/**
 * Any intrinsics interface (libxs_intrinsics_x86.h) shall be explicitly
 * included, i.e., separate from libxs_utils.h.
*/
#include "utils/libxs_lpflt_quant.h"
#include "utils/libxs_barrier.h"
#include "utils/libxs_timer.h"
#include "utils/libxs_math.h"
#include "utils/libxs_mhd.h"

#if defined(__BLAS) && (1 == __BLAS)
# if defined(__OPENBLAS)
    LIBXS_EXTERN void openblas_set_num_threads(int num_threads);
#   define LIBXS_BLAS_INIT openblas_set_num_threads(1);
# endif
#endif
#if !defined(LIBXS_BLAS_INIT)
# define LIBXS_BLAS_INIT
#endif

/** Call libxs_gemm_print using LIBXS's GEMM-flags. */
#define LIBXS_GEMM_PRINT(OSTREAM, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  LIBXS_GEMM_PRINT2(OSTREAM, PRECISION, PRECISION, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)
#define LIBXS_GEMM_PRINT2(OSTREAM, IPREC, OPREC, FLAGS, M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC) \
  libxs_gemm_dprint2(OSTREAM, (libxs_datatype)(IPREC), (libxs_datatype)(OPREC), \
    /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
    (char)(0 == (LIBXS_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'n' : 't'), \
    (char)(0 == (LIBXS_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'n' : 't'), \
    M, N, K, DALPHA, A, LDA, B, LDB, DBETA, C, LDC)

/**
 * Utility function, which either prints information about the GEMM call
 * or dumps (FILE/ostream=0) all input and output data into MHD files.
 * The Meta Image Format (MHD) is suitable for visual inspection using,
 * e.g., ITK-SNAP or ParaView.
 */
LIBXS_API void libxs_gemm_print(void* ostream,
  libxs_datatype precision, const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const void* alpha, const void* a, const libxs_blasint* lda,
  const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc);
LIBXS_API void libxs_gemm_print2(void* ostream,
  libxs_datatype iprec, libxs_datatype oprec, const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const void* alpha, const void* a, const libxs_blasint* lda,
  const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc);

#endif /*LIBXS_UTILS_H*/
