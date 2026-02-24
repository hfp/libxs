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
 * Complex GEMM via 3M (Karatsuba) method:
 *   C = alpha * A * B + beta * C
 * where A, B, C are complex matrices stored in interleaved format
 * (pairs of GEMM_REAL_TYPE: real, imaginary).
 *
 * Let alpha = ar + i*ai, A = Ar + i*Ai, B = Br + i*Bi, etc.
 * The 3M method computes:
 *   P1 = Ar * Br                         (real GEMM)
 *   P2 = Ai * Bi                         (real GEMM)
 *   P3 = (Ar + Ai) * (Br + Bi)           (real GEMM)
 * Then:
 *   Re(A*B) = P1 - P2
 *   Im(A*B) = P3 - P1 - P2
 * The complex alpha and beta scaling is applied afterwards.
 *
 * This wrapper intercepts ZGEMM (double) or CGEMM (float), deinterleaves
 * into split real/imaginary buffers, issues 3 real GEMM calls (which
 * themselves may be intercepted by the Ozaki wrapper), and recombines.
 */

LIBXS_APIVAR_PRIVATE_DEF(zgemm_function_t zgemm_original);


/** Deinterleave complex column-major matrix into separate real and imag parts. */
LIBXS_API_INLINE void zgemm3m_deinterleave(
  const GEMM_REAL_TYPE* LIBXS_RESTRICT z, GEMM_INT_TYPE ldz,
  GEMM_REAL_TYPE* LIBXS_RESTRICT re, GEMM_REAL_TYPE* LIBXS_RESTRICT im,
  GEMM_INT_TYPE ld_ri, GEMM_INT_TYPE rows, GEMM_INT_TYPE cols)
{
  GEMM_INT_TYPE i, j;
  for (j = 0; j < cols; ++j) {
    const GEMM_REAL_TYPE* zj = z + (size_t)j * ldz * 2; /* interleaved stride */
    GEMM_REAL_TYPE* rej = re + (size_t)j * ld_ri;
    GEMM_REAL_TYPE* imj = im + (size_t)j * ld_ri;
    for (i = 0; i < rows; ++i) {
      rej[i] = zj[2 * i];
      imj[i] = zj[2 * i + 1];
    }
  }
}


/** Interleave separate real and imag parts into complex column-major matrix. */
LIBXS_API_INLINE void zgemm3m_interleave(
  GEMM_REAL_TYPE* LIBXS_RESTRICT z, GEMM_INT_TYPE ldz,
  const GEMM_REAL_TYPE* LIBXS_RESTRICT re, const GEMM_REAL_TYPE* LIBXS_RESTRICT im,
  GEMM_INT_TYPE ld_ri, GEMM_INT_TYPE rows, GEMM_INT_TYPE cols)
{
  GEMM_INT_TYPE i, j;
  for (j = 0; j < cols; ++j) {
    GEMM_REAL_TYPE* zj = z + (size_t)j * ldz * 2;
    const GEMM_REAL_TYPE* rej = re + (size_t)j * ld_ri;
    const GEMM_REAL_TYPE* imj = im + (size_t)j * ld_ri;
    for (i = 0; i < rows; ++i) {
      zj[2 * i]     = rej[i];
      zj[2 * i + 1] = imj[i];
    }
  }
}


/** Elementwise matrix addition: dst[i,j] = a[i,j] + b[i,j]. */
LIBXS_API_INLINE void zgemm3m_matadd(
  GEMM_REAL_TYPE* LIBXS_RESTRICT dst, GEMM_INT_TYPE ld_dst,
  const GEMM_REAL_TYPE* LIBXS_RESTRICT a, GEMM_INT_TYPE ld_a,
  const GEMM_REAL_TYPE* LIBXS_RESTRICT b, GEMM_INT_TYPE ld_b,
  GEMM_INT_TYPE rows, GEMM_INT_TYPE cols)
{
  GEMM_INT_TYPE i, j;
  for (j = 0; j < cols; ++j) {
    for (i = 0; i < rows; ++i) {
      dst[i + (size_t)j * ld_dst] = a[i + (size_t)j * ld_a]
                                  + b[i + (size_t)j * ld_b];
    }
  }
}


/** Elementwise: dst[i,j] = a[i,j] - b[i,j]. */
LIBXS_API_INLINE void zgemm3m_matsub(
  GEMM_REAL_TYPE* LIBXS_RESTRICT dst, GEMM_INT_TYPE ld_dst,
  const GEMM_REAL_TYPE* LIBXS_RESTRICT a, GEMM_INT_TYPE ld_a,
  const GEMM_REAL_TYPE* LIBXS_RESTRICT b, GEMM_INT_TYPE ld_b,
  GEMM_INT_TYPE rows, GEMM_INT_TYPE cols)
{
  GEMM_INT_TYPE i, j;
  for (j = 0; j < cols; ++j) {
    for (i = 0; i < rows; ++i) {
      dst[i + (size_t)j * ld_dst] = a[i + (size_t)j * ld_a]
                                  - b[i + (size_t)j * ld_b];
    }
  }
}


/**
 * Apply complex alpha/beta to split Re/Im result and write back
 * to interleaved complex output C.
 *
 * For each element:
 *   C_new = (ar + i*ai) * (re_ab + i*im_ab) + (br + i*bi) * C_old
 * where re_ab = Re(A*B), im_ab = Im(A*B), C_old is read from c[].
 */
LIBXS_API_INLINE void zgemm3m_finalize(
  GEMM_REAL_TYPE* LIBXS_RESTRICT c, GEMM_INT_TYPE ldc,
  const GEMM_REAL_TYPE* LIBXS_RESTRICT re_ab, const GEMM_REAL_TYPE* LIBXS_RESTRICT im_ab,
  GEMM_INT_TYPE ld_ri, GEMM_INT_TYPE m, GEMM_INT_TYPE n,
  GEMM_REAL_TYPE ar, GEMM_REAL_TYPE ai, GEMM_REAL_TYPE br, GEMM_REAL_TYPE bi)
{
  GEMM_INT_TYPE i, j;
  for (j = 0; j < n; ++j) {
    GEMM_REAL_TYPE* cj = c + (size_t)j * ldc * 2;
    const GEMM_REAL_TYPE* rej = re_ab + (size_t)j * ld_ri;
    const GEMM_REAL_TYPE* imj = im_ab + (size_t)j * ld_ri;
    for (i = 0; i < m; ++i) {
      const GEMM_REAL_TYPE c_re = cj[2 * i];
      const GEMM_REAL_TYPE c_im = cj[2 * i + 1];
      cj[2 * i]     = ar * rej[i] - ai * imj[i] + br * c_re - bi * c_im;
      cj[2 * i + 1] = ar * imj[i] + ai * rej[i] + br * c_im + bi * c_re;
    }
  }
}


/**
 * Complex GEMM via 3 real GEMM calls (3M / Karatsuba method).
 *
 * All complex matrices are in standard BLAS interleaved format:
 * element (i,j) occupies a[2*(i + j*lda)] (real) and a[2*(i + j*lda)+1] (imag).
 * The alpha and beta arguments each point to 2 consecutive GEMM_REAL_TYPE values.
 */
LIBXS_API_INTERN void zgemm3m(GEMM_ARGDECL)
{
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
  const GEMM_REAL_TYPE br = beta[0],  bi = beta[1];

  /* Workspace: split Re/Im for A, B, and three products + temporaries */
  const size_t sz_a = (size_t)a_rows * a_cols;
  const size_t sz_b = (size_t)b_rows * b_cols;
  const size_t sz_c = (size_t)M * N;
  /* Ar, Ai, Br, Bi, Ta (=Ar+Ai), Tb (=Br+Bi), P1, P2, P3 */
  GEMM_REAL_TYPE* workspace = (GEMM_REAL_TYPE*)libxs_malloc(
    sizeof(GEMM_REAL_TYPE) * (3 * sz_a + 3 * sz_b + 3 * sz_c), 0/*auto*/);
  if (NULL == workspace) {
    fprintf(stderr, "zgemm3m: allocation failed (m=%i, n=%i, k=%i)\n",
      (int)M, (int)N, (int)K);
    return;
  }

  { GEMM_REAL_TYPE *const ar_buf = workspace;
    GEMM_REAL_TYPE *const ai_buf = ar_buf + sz_a;
    GEMM_REAL_TYPE *const br_buf = ai_buf + sz_a;
    GEMM_REAL_TYPE *const bi_buf = br_buf + sz_b;
    GEMM_REAL_TYPE *const ta_buf = bi_buf + sz_b; /* Ar + Ai */
    GEMM_REAL_TYPE *const tb_buf = ta_buf + sz_a; /* Br + Bi */
    GEMM_REAL_TYPE *const p1     = tb_buf + sz_b; /* Ar * Br */
    GEMM_REAL_TYPE *const p2     = p1 + sz_c;     /* Ai * Bi */
    GEMM_REAL_TYPE *const p3     = p2 + sz_c;     /* (Ar+Ai) * (Br+Bi) */
    const GEMM_REAL_TYPE one = 1, zero = 0;
    const GEMM_INT_TYPE lda_ri = a_rows;
    const GEMM_INT_TYPE ldb_ri = b_rows;
    const GEMM_INT_TYPE ldc_ri = M;

    /* 1. Deinterleave A and B into split real/imaginary */
    zgemm3m_deinterleave(a, *lda, ar_buf, ai_buf, lda_ri, a_rows, a_cols);
    zgemm3m_deinterleave(b, *ldb, br_buf, bi_buf, ldb_ri, b_rows, b_cols);

    /* 2. Form temporaries: Ta = Ar + Ai, Tb = Br + Bi */
    zgemm3m_matadd(ta_buf, lda_ri, ar_buf, lda_ri, ai_buf, lda_ri, a_rows, a_cols);
    zgemm3m_matadd(tb_buf, ldb_ri, br_buf, ldb_ri, bi_buf, ldb_ri, b_rows, b_cols);

    /* 3. Three real GEMM calls (Karatsuba)
     *    P1 = Ar * Br
     *    P2 = Ai * Bi
     *    P3 = (Ar + Ai) * (Br + Bi) */
    GEMM(transa, transb, &M, &N, &K, &one, ar_buf, &lda_ri,
                                           br_buf, &ldb_ri,
                                     &zero, p1,    &ldc_ri);
    GEMM(transa, transb, &M, &N, &K, &one, ai_buf, &lda_ri,
                                           bi_buf, &ldb_ri,
                                     &zero, p2,    &ldc_ri);
    GEMM(transa, transb, &M, &N, &K, &one, ta_buf, &lda_ri,
                                           tb_buf, &ldb_ri,
                                     &zero, p3,    &ldc_ri);

    /* 4. Recover complex product components
     *    Re(A*B) = P1 - P2           (stored in p1)
     *    Im(A*B) = P3 - P1 - P2     (stored in p3)
     *    First: p3 = p3 - p1 - p2 (Im part) */
    { GEMM_INT_TYPE i, j;
      for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
          const size_t idx = i + (size_t)j * ldc_ri;
          const GEMM_REAL_TYPE v1 = p1[idx];
          const GEMM_REAL_TYPE v2 = p2[idx];
          const GEMM_REAL_TYPE v3 = p3[idx];
          p1[idx] = v1 - v2;          /* Re(A*B) */
          p3[idx] = v3 - v1 - v2;     /* Im(A*B) */
        }
      }
    }

    /* 5. Apply complex alpha/beta and write back to interleaved C */
    zgemm3m_finalize(c, *ldc, p1, p3, ldc_ri, M, N, ar, ai, br, bi);
  }

  libxs_free(workspace);
}


/**
 * Complex GEMM wrapper: dispatches to zgemm3m (3M method) or falls back
 * to the original complex BLAS. Controlled by env var GEMM_OZAKI
 * (shared with the real GEMM Ozaki wrapper):
 *   0 = pass through to original complex BLAS (avoids 3M overhead),
 *   1 = use 3M method (default).
 */
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void ZGEMM_WRAP(GEMM_ARGDECL)
{
  if (0 != gemm_ozaki) {
    gemm_dump_inhibit = 1; /* suppress decomposed sub-GEMM dumps */
    zgemm3m(GEMM_ARGPASS);
    if (2 == gemm_dump_inhibit) {
      gemm_dump_matrices(GEMM_ARGPASS, 2);
      if (0 != gemm_exit) exit(EXIT_FAILURE);
    }
    gemm_dump_inhibit = 0;
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
