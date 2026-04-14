/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"
#include <libxs_malloc.h>

/**
 * Complex GEMM via block embedding into a single real GEMM:
 *   C = alpha * A * B + beta * C
 * where A, B, C are complex matrices stored in interleaved format
 * (pairs of GEMM_REAL_TYPE: real, imaginary).
 *
 * The block embedding constructs augmented real matrices:
 *
 *   A_hat = [ Ar  -Ai ]  (2M x 2K)    B_hat = [ Br ]  (2K x N)
 *           [ Ai   Ar ]                        [ Bi ]
 *
 * so that A_hat * B_hat = [ Re(A*B) ]  (2M x N).
 *                         [ Im(A*B) ]
 *
 * A single real GEMM of size (2M) x N x (2K) produces both real and
 * imaginary parts.  When the Ozaki wrapper intercepts this GEMM, the
 * block structure guarantees that Re and Im contributions from the same
 * complex row/column share a common Ozaki exponent base, eliminating
 * the catastrophic cancellation that plagued the 3M (Karatsuba) method.
 *
 * Complex alpha and beta scaling is applied in a finalize step.
 */

LIBXS_APIVAR_PRIVATE_DEF(zgemm_function_t zgemm_original);


/**
 * Construct block-augmented A_hat from interleaved complex A.
 *
 * For transa='N': A_hat = [Ar, -Ai; Ai, Ar]  (2*a_rows x 2*a_cols)
 * For transa='T': A_hat = [Ar, Ai; -Ai, Ar]  (2*a_rows x 2*a_cols)
 *
 * In both cases, op(A_hat) = [op(Ar), -op(Ai); op(Ai), op(Ar)]
 * so that op(A_hat) * [op(Br); op(Bi)] = [Re(C); Im(C)].
 */
LIBXS_API_INLINE void zgemm_block_construct_a(const GEMM_REAL_TYPE* LIBXS_RESTRICT a, GEMM_INT_TYPE lda,
  GEMM_REAL_TYPE* LIBXS_RESTRICT a_hat, GEMM_INT_TYPE a_rows, GEMM_INT_TYPE a_cols, int ta)
{
  const GEMM_INT_TYPE lda_hat = 2 * a_rows;
  GEMM_INT_TYPE i, j;
  for (j = 0; j < a_cols; ++j) {
    const GEMM_REAL_TYPE* aj = a + (size_t)j * lda * 2;
    GEMM_REAL_TYPE* left = a_hat + (size_t)j * lda_hat;
    GEMM_REAL_TYPE* right = a_hat + (size_t)(a_cols + j) * lda_hat;
    for (i = 0; i < a_rows; ++i) {
      const GEMM_REAL_TYPE re = aj[2 * i];
      const GEMM_REAL_TYPE im = aj[2 * i + 1];
      left[i] = re;                                /* Q1: Ar */
      left[a_rows + i] = ta ? -im : im;            /* Q3: ta ? -Ai : Ai */
      right[i] = ta ? im : -im;                    /* Q2: ta ? Ai : -Ai */
      right[a_rows + i] = re;                      /* Q4: Ar */
    }
  }
}


/**
 * Construct block-augmented B_hat from interleaved complex B.
 *
 * For transb='N': B_hat = [Br; Bi]  (2*b_rows x b_cols), stacked vertically
 * For transb='T': B_hat = [Br, Bi]  (b_rows x 2*b_cols), placed side by side
 *
 * In both cases, op(B_hat) = [op(Br); op(Bi)] (2K x N).
 */
LIBXS_API_INLINE void zgemm_block_construct_b(const GEMM_REAL_TYPE* LIBXS_RESTRICT b, GEMM_INT_TYPE ldb,
  GEMM_REAL_TYPE* LIBXS_RESTRICT b_hat, GEMM_INT_TYPE b_rows, GEMM_INT_TYPE b_cols, int tb)
{
  GEMM_INT_TYPE i, j;
  if (0 == tb) {
    /* transb='N': B_hat (2*b_rows x b_cols), ldb_hat = 2*b_rows */
    const GEMM_INT_TYPE ldb_hat = 2 * b_rows;
    for (j = 0; j < b_cols; ++j) {
      const GEMM_REAL_TYPE* bj = b + (size_t)j * ldb * 2;
      GEMM_REAL_TYPE* hj = b_hat + (size_t)j * ldb_hat;
      for (i = 0; i < b_rows; ++i) {
        hj[i] = bj[2 * i];               /* top: Br */
        hj[b_rows + i] = bj[2 * i + 1];  /* bottom: Bi */
      }
    }
  }
  else {
    /* transb='T': B_hat = [Br, Bi] (b_rows x 2*b_cols), ldb_hat = b_rows */
    const GEMM_INT_TYPE ldb_hat = b_rows;
    for (j = 0; j < b_cols; ++j) {
      const GEMM_REAL_TYPE* bj = b + (size_t)j * ldb * 2;
      GEMM_REAL_TYPE* left = b_hat + (size_t)j * ldb_hat;
      GEMM_REAL_TYPE* right = b_hat + (size_t)(b_cols + j) * ldb_hat;
      for (i = 0; i < b_rows; ++i) {
        left[i] = bj[2 * i];       /* Br (left half) */
        right[i] = bj[2 * i + 1];  /* Bi (right half) */
      }
    }
  }
}


/**
 * Extract Re/Im from block result C_hat (2M x N), apply complex alpha/beta,
 * write to interleaved complex output C.
 *
 * C_hat layout (2M x N, column-major, ldc_hat = 2*m):
 *   Rows 0..m-1   = Re(A*B)
 *   Rows m..2m-1  = Im(A*B)
 */
LIBXS_API_INLINE void zgemm_block_finalize(GEMM_REAL_TYPE* LIBXS_RESTRICT c, GEMM_INT_TYPE ldc,
  const GEMM_REAL_TYPE* LIBXS_RESTRICT c_hat, GEMM_INT_TYPE m, GEMM_INT_TYPE n,
  GEMM_REAL_TYPE ar, GEMM_REAL_TYPE ai, GEMM_REAL_TYPE br, GEMM_REAL_TYPE bi)
{
  const GEMM_INT_TYPE ldc_hat = 2 * m;
  GEMM_INT_TYPE i, j;
  for (j = 0; j < n; ++j) {
    GEMM_REAL_TYPE* cj = c + (size_t)j * ldc * 2;
    const GEMM_REAL_TYPE* hj = c_hat + (size_t)j * ldc_hat;
    if ((GEMM_REAL_TYPE)0 != br || (GEMM_REAL_TYPE)0 != bi) {
      for (i = 0; i < m; ++i) {
        const GEMM_REAL_TYPE re_ab = hj[i];
        const GEMM_REAL_TYPE im_ab = hj[m + i];
        const GEMM_REAL_TYPE c_re = cj[2 * i];
        const GEMM_REAL_TYPE c_im = cj[2 * i + 1];
        cj[2 * i] = ar * re_ab - ai * im_ab + br * c_re - bi * c_im;
        cj[2 * i + 1] = ar * im_ab + ai * re_ab + br * c_im + bi * c_re;
      }
    }
    else { /* beta == 0: do not read C (may contain NaN/Inf) */
      for (i = 0; i < m; ++i) {
        const GEMM_REAL_TYPE re_ab = hj[i];
        const GEMM_REAL_TYPE im_ab = hj[m + i];
        cj[2 * i] = ar * re_ab - ai * im_ab;
        cj[2 * i + 1] = ar * im_ab + ai * re_ab;
      }
    }
  }
}


/**
 * Complex GEMM via block embedding into a single real GEMM.
 *
 * All complex matrices are in standard BLAS interleaved format:
 * element (i,j) occupies a[2*(i + j*lda)] (real) and a[2*(i + j*lda)+1] (imag).
 * The alpha and beta arguments each point to 2 consecutive GEMM_REAL_TYPE values.
 *
 * GPU-aware: when OpenCL is available and block-embedding kernels are compiled,
 * this function delegates to the GPU-native path which keeps all intermediate
 * buffers on device, minimizing PCIe transfers.
 */
LIBXS_API_INTERN void zgemm3m(GEMM_ARGDECL)
{
#if defined(__LIBXSTREAM)
  /* GPU-native path: only when ozaki_3m >= 2 */
  if (2 <= ozaki_3m && NULL != ozaki_ocl_handle && ozaki_ocl_supports_zgemm3m(ozaki_ocl_handle)) {
    double alpha_d[2], beta_d[2];
    int result;
    alpha_d[0] = (double)alpha[0];
    alpha_d[1] = (double)alpha[1];
    beta_d[0] = (double)beta[0];
    beta_d[1] = (double)beta[1];
    result = ozaki_ocl_gemm3m(ozaki_ocl_handle, *transa, *transb, *m, *n, *k, alpha_d, a, *lda, b, *ldb, beta_d, c, *ldc);
    if (EXIT_SUCCESS == result) return;
    /* Fall through to CPU path on failure */
  }
#endif

  { /* CPU-based block-embedding path (ozaki_3m == 1, or GPU fallback) */
    const GEMM_INT_TYPE M = *m, N = *n, K = *k;
    const int ta = (*transa != 'N' && *transa != 'n');
    const int tb = (*transb != 'N' && *transb != 'n');

    /* Physical (stored) dimensions of A and B */
    const GEMM_INT_TYPE a_rows = ta ? K : M;
    const GEMM_INT_TYPE a_cols = ta ? M : K;
    const GEMM_INT_TYPE b_rows = tb ? N : K;
    const GEMM_INT_TYPE b_cols = tb ? K : N;

    /* Complex alpha = ar + i*ai, beta = br + i*bi */
    const GEMM_REAL_TYPE ar = alpha[0], ai = alpha[1];
    const GEMM_REAL_TYPE br = beta[0], bi = beta[1];

    /* Workspace: A_hat (2*a_rows x 2*a_cols), B_hat, C_hat (2*M x N) */
    const size_t sz_a_hat = (size_t)(2 * a_rows) * (2 * a_cols);
    const size_t sz_b_hat = tb
      ? (size_t)b_rows * (2 * b_cols)
      : (size_t)(2 * b_rows) * b_cols;
    const size_t sz_c_hat = (size_t)(2 * M) * N;
    GEMM_REAL_TYPE* workspace = (GEMM_REAL_TYPE*)libxs_malloc(
      gemm_pool, sizeof(GEMM_REAL_TYPE) * (sz_a_hat + sz_b_hat + sz_c_hat), 0 /*auto*/);

    if (NULL == workspace) {
      fprintf(stderr, "ERROR: " LIBXS_STRINGIFY(LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm_block))
        " allocation failed (m=%i, n=%i, k=%i), fallback to BLAS\n",
        (int)M, (int)N, (int)K);
      if (NULL != zgemm_original) {
        zgemm_original(GEMM_ARGPASS);
      }
      else {
        ZGEMM_REAL(GEMM_ARGPASS);
      }
    }
    else {
      GEMM_REAL_TYPE* const a_hat = workspace;
      GEMM_REAL_TYPE* const b_hat = a_hat + sz_a_hat;
      GEMM_REAL_TYPE* const c_hat = b_hat + sz_b_hat;
      const GEMM_REAL_TYPE one = 1, zero = 0;
      const GEMM_INT_TYPE m_hat = 2 * M;
      const GEMM_INT_TYPE k_hat = 2 * K;
      const GEMM_INT_TYPE lda_hat = 2 * a_rows;
      const GEMM_INT_TYPE ldb_hat = tb ? b_rows : 2 * b_rows;
      const GEMM_INT_TYPE ldc_hat = 2 * M;

      /* 1. Construct A_hat from interleaved complex A */
      zgemm_block_construct_a(a, *lda, a_hat, a_rows, a_cols, ta);

      /* 2. Construct B_hat from interleaved complex B */
      zgemm_block_construct_b(b, *ldb, b_hat, b_rows, b_cols, tb);

      /* 3. Single real GEMM: C_hat = op(A_hat) * op(B_hat) */
      GEMM(transa, transb, &m_hat, &N, &k_hat, &one, a_hat, &lda_hat,
        b_hat, &ldb_hat, &zero, c_hat, &ldc_hat);

      /* 4. Apply complex alpha/beta and write to interleaved C */
      zgemm_block_finalize(c, *ldc, c_hat, M, N, ar, ai, br, bi);
    }

    libxs_free(workspace);
  } /* end CPU path */
}


/**
 * Complex GEMM diff: save C, run block-embedding, reference ZGEMM, matdiff with C64/C32.
 */
LIBXS_API_INLINE void zgemm3m_diff(GEMM_ARGDECL,
  libxs_matdiff_t* diff)
{
  GEMM_REAL_TYPE* c_ref = NULL;
  size_t c_size = 0;
  /* Save C for reference comparison (before 3M modifies it) */
  if (NULL != diff) {
    c_size = 2 * (size_t)*ldc * (size_t)*n * sizeof(GEMM_REAL_TYPE);
    c_ref = (GEMM_REAL_TYPE*)libxs_malloc(gemm_pool, c_size, 0);
    if (NULL != c_ref) memcpy(c_ref, c, c_size);
  }
  /* Compute via 3M */
  gemm_dump_inhibit = 1;
  zgemm3m(GEMM_ARGPASS);
  if (2 == gemm_dump_inhibit) {
    const int result = gemm_dump_matrices(GEMM_ARGPASS, 2);
    if (0 != ozaki_exit) exit(EXIT_SUCCESS == result ? EXIT_FAILURE : result);
  }
  gemm_dump_inhibit = 0;
  /* Reference complex BLAS and diff.
   * Set gemm_nozaki so that any sgemm_ calls MKL's CGEMM makes
   * internally bypass Ozaki (--wrap redirects them to GEMM_WRAP). */
  if (NULL != c_ref) {
    const libxs_data_t dt = (GEMM_IS_DOUBLE ? LIBXS_DATATYPE_C64 : LIBXS_DATATYPE_C32);
    gemm_nozaki = 1;
    if (NULL != zgemm_original) {
      zgemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c_ref, ldc);
    }
    else {
      ZGEMM_REAL(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c_ref, ldc);
    }
    gemm_nozaki = 0;
    libxs_matdiff(diff, dt, *m, *n, c_ref, c, ldc, ldc);
    if (ozaki_diff_exceeds(diff)) memcpy(c, c_ref, c_size);
    libxs_free(c_ref);
  }
}


/**
 * Complex GEMM wrapper: dispatches based on OZAKI_3M env var.
 *   0 = pass through to original complex BLAS,
 *   1 = CPU-based block embedding (single real GEMM),
 *   2 = GPU-native block embedding (all on device, falls back to CPU).
 * Default: follows OZAKI (0 if OZAKI=0, else 2).
 */
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void ZGEMM_WRAP(GEMM_ARGDECL)
{
  gemm_init();
  if (0 != ozaki_3m) {
    OZAKI_GEMM_WRAPPER(zgemm3m_diff, ZGEMM_LABEL, 2)
  }
  else {
    /* Passthrough to original complex GEMM */
    if (NULL != zgemm_original) {
      zgemm_original(GEMM_ARGPASS);
    }
    else {
      ZGEMM_REAL(GEMM_ARGPASS);
    }
  }
}
