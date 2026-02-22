/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"

#if GEMM_IS_DOUBLE
# define OZ2_MAX_NPRIMES    16
# define OZ2_NPRIMES_DEFAULT 16
#else /* single-precision */
# define OZ2_MAX_NPRIMES    10
# define OZ2_NPRIMES_DEFAULT 7
#endif

/* IEEE-754 format parameters derived from GEMM_REAL_TYPE */
#if GEMM_IS_DOUBLE
# define OZ2_MANT_BITS  52
# define OZ2_EXP_BIAS   1023
#else /* single-precision */
# define OZ2_MANT_BITS  23
# define OZ2_EXP_BIAS   127
#endif
#define OZ2_BIAS_PLUS_MANT (OZ2_EXP_BIAS + OZ2_MANT_BITS)

#if !defined(BLOCK_M)
# define BLOCK_M 16
#endif
#if !defined(BLOCK_N)
# define BLOCK_N 16
#endif
#if !defined(BLOCK_K)
# define BLOCK_K 16
#endif

/* Chinese Remainder Theorem (CRT) primes: chosen < 256 so that residues
 * fit in uint8 and products (< 256^2) fit in uint32. The product
 * P = prod(p_i) must exceed 2 * BLOCK_K * (2^(MANT+1))^2 to represent
 * signed dot products without aliasing.
 * Double (53b mantissa, BLOCK_K=16): need P > 2^111 -> 16 primes (2^124).
 * Float  (24b mantissa, BLOCK_K=16): need P > 2^53  -> 7  primes (2^55). */
static const unsigned int oz2_primes[] = {
  251, 241, 239, 233, 229, 227, 223, 211,
  199, 197, 193, 191, 181, 179, 173, 167
};


/*=== Block-level helpers (duplicated from ozaki1.c for TU independence) =====*/

LIBXS_API_INLINE void oz2_scale_block_beta(GEMM_REAL_TYPE* mb, GEMM_INT_TYPE ldc,
  GEMM_INT_TYPE iblk, GEMM_INT_TYPE jblk, const GEMM_REAL_TYPE* beta,
  GEMM_REAL_TYPE* ref_blk, int capture_ref)
{
  GEMM_INT_TYPE mi, nj;
  for (mi = 0; mi < iblk; ++mi) {
    for (nj = 0; nj < jblk; ++nj) {
      if (0 != capture_ref) ref_blk[mi + nj * BLOCK_M] = mb[mi + nj * ldc];
      mb[mi + nj * ldc] *= (*beta);
    }
  }
}


LIBXS_API_INLINE void oz2_store_block_pair(GEMM_REAL_TYPE* ref_blk, GEMM_REAL_TYPE* recon_blk,
  GEMM_INT_TYPE ld, GEMM_INT_TYPE row, GEMM_INT_TYPE col,
  GEMM_REAL_TYPE ref_val, GEMM_REAL_TYPE recon_val)
{
  recon_blk[row + col * ld] = recon_val;
  ref_blk[row + col * ld] = ref_val;
}


LIBXS_API_INLINE void oz2_accumulate_block_diff(libxs_matdiff_info_t* acc,
  const GEMM_REAL_TYPE* ref_blk, const GEMM_REAL_TYPE* tst_blk,
  GEMM_INT_TYPE bm, GEMM_INT_TYPE bn, GEMM_INT_TYPE ld_ref, GEMM_INT_TYPE ld_tst)
{
  libxs_matdiff_info_t block_diff;
  const int ild_ref = (int)ld_ref, ild_tst = (int)ld_tst;
  if (EXIT_SUCCESS == libxs_matdiff(&block_diff, LIBXS_DATATYPE(GEMM_REAL_TYPE),
    bm, bn, ref_blk, tst_blk, &ild_ref, &ild_tst))
  {
    libxs_matdiff_reduce(acc, &block_diff);
  }
}


/*=== IEEE-754 decomposition =================================================*/

/** Decompose a floating-point value into sign (+1/-1), biased exponent,
 *  and full unsigned mantissa (with implicit leading 1-bit).
 *  Special values (zero, NaN, Inf, subnormal) yield mantissa=0, exp=0. */
LIBXS_API_INLINE void oz2_decompose(GEMM_REAL_TYPE value,
  int16_t* exp_biased, int8_t* sign_out, uint64_t* mantissa)
{
  const union { uint32_t raw; float value; } inf = { 0x7F800000U };

  LIBXS_ASSERT(NULL != exp_biased && NULL != sign_out && NULL != mantissa);

  if (value == (GEMM_REAL_TYPE)0 || LIBXS_ISNAN(value)
    || (float)value == inf.value || (float)value == -inf.value)
  {
    *exp_biased = 0; *sign_out = 1; *mantissa = 0;
    return;
  }

  *sign_out = (value < (GEMM_REAL_TYPE)0) ? -1 : 1;
  if (value < (GEMM_REAL_TYPE)0) value = -value;

#if GEMM_IS_DOUBLE
  { union { double d; uint64_t u; } cvt;
    cvt.d = value;
    { const uint64_t bits = cvt.u;
      const uint64_t frac = bits & ((1ULL << 52) - 1ULL);
      const uint16_t exp_raw = (uint16_t)((bits >> 52) & 0x7FFU);
      if (0 == exp_raw) { /* subnormal */
        *exp_biased = 0; *sign_out = 1; *mantissa = 0;
        return;
      }
      *exp_biased = (int16_t)exp_raw;
      *mantissa = (1ULL << 52) | frac; /* 53-bit mantissa */
    }
  }
#else /* single-precision */
  { union { float f; uint32_t u; } cvt;
    cvt.f = value;
    { const uint32_t bits = cvt.u;
      const uint32_t frac = bits & ((1U << 23) - 1U);
      const uint16_t exp_raw = (uint16_t)((bits >> 23) & 0xFFU);
      if (0 == exp_raw) { /* subnormal */
        *exp_biased = 0; *sign_out = 1; *mantissa = 0;
        return;
      }
      *exp_biased = (int16_t)exp_raw;
      *mantissa = (uint64_t)((1U << 23) | frac); /* 24-bit mantissa */
    }
  }
#endif
}


/** Reduce an aligned mantissa modulo all active primes.
 *  delta = max_exp - element_exp (>= 0); mantissa is right-shifted
 *  by delta bits for exponent alignment before reduction. */
LIBXS_API_INLINE void oz2_reduce(uint64_t mantissa, int delta,
  uint8_t residues[OZ2_MAX_NPRIMES], int nprimes)
{
  int i;
  if (delta > 0) {
    if (delta >= 64) mantissa = 0;
    else mantissa >>= delta;
  }
  for (i = 0; i < nprimes; ++i) {
    residues[i] = (uint8_t)(mantissa % oz2_primes[i]);
  }
}


/*=== Block preprocessing ====================================================*/

/** Preprocess rows of A for one (ib, kb) tile.
 *  Decomposes each element, finds per-row max exponent, aligns mantissas
 *  by right-shifting, and reduces mod each prime. */
LIBXS_API_INLINE void oz2_preprocess_rows(
  const GEMM_REAL_TYPE* a, GEMM_INT_TYPE lda, int ta,
  GEMM_INT_TYPE M, GEMM_INT_TYPE K,
  GEMM_INT_TYPE ib, GEMM_INT_TYPE kb,
  GEMM_INT_TYPE iblk, GEMM_INT_TYPE kblk, int nprimes,
  int16_t expa_row[BLOCK_M],
  int8_t a_sign[BLOCK_M][BLOCK_K],
  uint8_t a_res[BLOCK_M][BLOCK_K][OZ2_MAX_NPRIMES])
{
  int16_t elem_exp[BLOCK_M][BLOCK_K];
  uint64_t elem_mant[BLOCK_M][BLOCK_K];
  GEMM_INT_TYPE mi, kk;

  /* Pass 1: decompose and track per-row max exponent */
  for (mi = 0; mi < iblk; ++mi) {
    const GEMM_INT_TYPE row = ib + mi;
    int16_t row_max_exp = INT16_MIN;

    for (kk = 0; kk < kblk; ++kk) {
      const GEMM_INT_TYPE p = kb + kk;
      const GEMM_REAL_TYPE aval = ((row < M && p < K)
        ? a[LIBXS_INDEX(ta, lda, row, p)] : (GEMM_REAL_TYPE)0);
      oz2_decompose(aval, &elem_exp[mi][kk], &a_sign[mi][kk], &elem_mant[mi][kk]);
      row_max_exp = LIBXS_MAX(row_max_exp, elem_exp[mi][kk]);
    }

    expa_row[mi] = row_max_exp;

    /* Pass 2: align and reduce mod primes */
    for (kk = 0; kk < kblk; ++kk) {
      const int delta = (int)row_max_exp - (int)elem_exp[mi][kk];
      oz2_reduce(elem_mant[mi][kk], delta, a_res[mi][kk], nprimes);
    }
    /* Zero-pad remaining k-entries */
    for (kk = kblk; kk < BLOCK_K; ++kk) {
      a_sign[mi][kk] = 1;
      memset(a_res[mi][kk], 0, sizeof(uint8_t) * nprimes);
    }
  }
}


/** Preprocess columns of B for one (kb, jb) tile.
 *  Same as rows of A but with per-column max exponent. */
LIBXS_API_INLINE void oz2_preprocess_cols(
  const GEMM_REAL_TYPE* b, GEMM_INT_TYPE ldb, int tb,
  GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  GEMM_INT_TYPE jb, GEMM_INT_TYPE kb,
  GEMM_INT_TYPE jblk, GEMM_INT_TYPE kblk, int nprimes,
  int16_t expb_col[BLOCK_N],
  int8_t b_sign[BLOCK_K][BLOCK_N],
  uint8_t b_res[BLOCK_K][BLOCK_N][OZ2_MAX_NPRIMES])
{
  int16_t elem_exp[BLOCK_K][BLOCK_N];
  uint64_t elem_mant[BLOCK_K][BLOCK_N];
  GEMM_INT_TYPE nj, kk;

  for (nj = 0; nj < jblk; ++nj) expb_col[nj] = INT16_MIN;

  /* Pass 1: decompose and track per-column max exponent */
  for (kk = 0; kk < kblk; ++kk) {
    const GEMM_INT_TYPE p = kb + kk;
    for (nj = 0; nj < jblk; ++nj) {
      const GEMM_INT_TYPE col = jb + nj;
      const GEMM_REAL_TYPE bval = ((p < K && col < N)
        ? b[LIBXS_INDEX(tb, ldb, p, col)] : (GEMM_REAL_TYPE)0);
      oz2_decompose(bval, &elem_exp[kk][nj], &b_sign[kk][nj], &elem_mant[kk][nj]);
      expb_col[nj] = LIBXS_MAX(expb_col[nj], elem_exp[kk][nj]);
    }
  }

  /* Pass 2: align and reduce mod primes */
  for (kk = 0; kk < kblk; ++kk) {
    for (nj = 0; nj < jblk; ++nj) {
      const int delta = (int)expb_col[nj] - (int)elem_exp[kk][nj];
      oz2_reduce(elem_mant[kk][nj], delta, b_res[kk][nj], nprimes);
    }
  }
  /* Zero-pad remaining k-entries */
  for (kk = kblk; kk < BLOCK_K; ++kk) {
    for (nj = 0; nj < jblk; ++nj) {
      b_sign[kk][nj] = 1;
      memset(b_res[kk][nj], 0, sizeof(uint8_t) * nprimes);
    }
  }
}


/*=== CRT reconstruction (Garner + Horner) ===================================*/

/** Reconstruct a signed integer from its CRT residues.
 *
 *  Uses Garner's algorithm to compute mixed-radix digits, detects the
 *  sign from the most significant digit (centered representation), and
 *  evaluates via Horner's method into a double.
 *
 *  For negative values the digits are complemented before Horner
 *  evaluation so that the absolute value is computed directly, avoiding
 *  catastrophic cancellation from subtracting the large modulus product P.
 *
 *  Centered range: result in (-P/2, P/2], where P = prod(p_i).
 */
LIBXS_API_INLINE double oz2_reconstruct(
  const unsigned int residues[OZ2_MAX_NPRIMES],
  unsigned int garner_inv[OZ2_MAX_NPRIMES][OZ2_MAX_NPRIMES],
  int nprimes)
{
  unsigned int v[OZ2_MAX_NPRIMES]; /* mixed-radix digits */
  double result;
  int i, j, is_negative;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_MAX_NPRIMES);

  /* Garner's algorithm: compute mixed-radix digits v[i] in [0, p_i) */
  for (i = 0; i < nprimes; ++i) {
    unsigned int u = residues[i];
    for (j = 0; j < i; ++j) {
      const unsigned int pi = oz2_primes[i];
      /* v[j] was computed mod p_j which may exceed p_i (primes are
       * sorted descending), so reduce before subtraction. */
      const unsigned int vj = v[j] % pi;
      const unsigned int diff = (u >= vj) ? (u - vj) : (pi + u - vj);
      u = (diff * garner_inv[j][i]) % pi;
    }
    v[i] = u;
  }

  /* Sign detection */
  is_negative = (v[nprimes - 1] >= (oz2_primes[nprimes - 1] + 1) / 2)
    ? 1 : 0;

  /* Complement digits for negative values: P - 1 - V has digits
   * (p_i - 1 - v_i) in mixed-radix representation (no borrows needed).
   * Then |D| = (P - 1 - V) + 1 = P - V where V is the CRT result. */
  if (0 != is_negative) {
    for (i = 0; i < nprimes; ++i) {
      v[i] = oz2_primes[i] - 1 - v[i];
    }
  }

  /* Horner's method (MSB to LSB): numerically stable since each step is
   * just a multiply by a small prime and add of a small digit. */
  result = (double)v[nprimes - 1];
  for (i = nprimes - 2; i >= 0; --i) {
    result = result * (double)oz2_primes[i] + (double)v[i];
  }

  /* For negative values: result holds |D|-1 (the complement),
   * so the true value is -(result + 1.0). */
  if (0 != is_negative) {
    result = -(result + 1.0);
  }

  return result;
}


/** Reconstruct a single element's unsigned mantissa from its CRT residues.
 *  Uses Garner+Horner in uint64 arithmetic (exact for mantissa <= 2^53).
 *  Used for diff tracking of A/B matrices. */
LIBXS_API_INLINE uint64_t oz2_reconstruct_mantissa(
  const uint8_t residues[OZ2_MAX_NPRIMES],
  unsigned int garner_inv[OZ2_MAX_NPRIMES][OZ2_MAX_NPRIMES],
  int nprimes)
{
  unsigned int v[OZ2_MAX_NPRIMES];
  uint64_t result;
  int i, j;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_MAX_NPRIMES);

  for (i = 0; i < nprimes; ++i) {
    unsigned int u = (unsigned int)residues[i];
    for (j = 0; j < i; ++j) {
      const unsigned int pi = oz2_primes[i];
      const unsigned int vj = v[j] % pi;
      const unsigned int diff = (u >= vj) ? (u - vj) : (pi + u - vj);
      u = (diff * garner_inv[j][i]) % pi;
    }
    v[i] = u;
  }

  /* Horner in uint64 (exact for mantissa values up to 2^53) */
  result = (uint64_t)v[nprimes - 1];
  for (i = nprimes - 2; i >= 0; --i) {
    result = result * (uint64_t)oz2_primes[i] + (uint64_t)v[i];
  }
  return result;
}


/*=== Main CRT kernel ========================================================*/

LIBXS_API_INLINE void gemm_oz2_diff(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
  unsigned int diff_abc, libxs_matdiff_info_t* diff)
{
  unsigned int garner_inv[OZ2_MAX_NPRIMES][OZ2_MAX_NPRIMES];
  enum {
    BLOCK_MN = BLOCK_M * BLOCK_N,
    BLOCK_MK = BLOCK_M * BLOCK_K,
    BLOCK_KN = BLOCK_K * BLOCK_N,
    BLOCK_MNK = LIBXS_MAX(LIBXS_MAX(BLOCK_MN, BLOCK_MK), BLOCK_KN)
  };
  const int ta = (*transa != 'N' && *transa != 'n');
  const int tb = (*transb != 'N' && *transb != 'n');
  const GEMM_INT_TYPE M = *m, N = *n, K = *k;
  const GEMM_INT_TYPE ldcv = *ldc;
  const int nprimes = LIBXS_CLMP(gemm_ozn, 1, OZ2_MAX_NPRIMES);
  GEMM_INT_TYPE jb, ib;
  int i, j;
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE)
            || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));

  /* Precompute Garner modular inverse table:
   * garner_inv[i][j] = p_i^{-1} mod p_j  for i < j */
  memset(garner_inv, 0, sizeof(garner_inv));
  for (i = 0; i < nprimes; ++i) {
    for (j = i + 1; j < nprimes; ++j) {
      garner_inv[i][j] = libxs_mod_inverse_u32(
        oz2_primes[i] % oz2_primes[j], oz2_primes[j]);
    }
  }

#if defined(_OPENMP)
# pragma omp parallel if(NULL == diff)
#endif
  { uint8_t a_res[BLOCK_M][BLOCK_K][OZ2_MAX_NPRIMES];
    uint8_t b_res[BLOCK_K][BLOCK_N][OZ2_MAX_NPRIMES];
    int8_t  a_sign[BLOCK_M][BLOCK_K], b_sign[BLOCK_K][BLOCK_N];
    /* k-contiguous residues for dot product (transposed layout) */
    uint8_t ak[BLOCK_M][OZ2_MAX_NPRIMES][BLOCK_K];
    uint8_t bk[BLOCK_N][OZ2_MAX_NPRIMES][BLOCK_K];
    int8_t  ak_sign[BLOCK_M][BLOCK_K], bk_sign[BLOCK_N][BLOCK_K];
    int16_t expa_row[BLOCK_M], expb_col[BLOCK_N];
    GEMM_REAL_TYPE ref_blk[BLOCK_MNK], recon_blk[BLOCK_MNK];
    GEMM_INT_TYPE kb, mi, nj, kk;
    int pidx;

#if defined(_OPENMP)
#   pragma omp for LIBXS_OPENMP_COLLAPSE(2) schedule(dynamic)
#endif
    for (jb = 0; jb < N; jb += BLOCK_N) {
      for (ib = 0; ib < M; ib += BLOCK_M) {
        const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
        const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
        GEMM_REAL_TYPE *const mb = c + jb * ldcv + ib;

        oz2_scale_block_beta(mb, ldcv, iblk, jblk, beta, ref_blk,
          (NULL != diff && 0 == (diff_abc % 3)));

        for (kb = 0; kb < K; kb += BLOCK_K) {
          const GEMM_INT_TYPE kblk = LIBXS_MIN(BLOCK_K, K - kb);

          oz2_preprocess_rows(a, *lda, ta, M, K, ib, kb, iblk, kblk,
            nprimes, expa_row, a_sign, a_res);
          oz2_preprocess_cols(b, *ldb, tb, N, K, jb, kb, jblk, kblk,
            nprimes, expb_col, b_sign, b_res);

          /* Diff tracking for A decomposition (diff_abc == 1) */
          if (NULL != diff && 1 == (diff_abc % 3)) {
#if defined(_OPENMP)
#           pragma omp critical
#endif
            { for (mi = 0; mi < iblk; ++mi) {
                const GEMM_INT_TYPE row = ib + mi;
                for (kk = 0; kk < kblk; ++kk) {
                  const GEMM_INT_TYPE p = kb + kk;
                  const GEMM_REAL_TYPE aval = ((row < M && p < K)
                    ? a[LIBXS_INDEX(ta, *lda, row, p)] : (GEMM_REAL_TYPE)0);
                  const uint64_t mant_recon = oz2_reconstruct_mantissa(
                    a_res[mi][kk], garner_inv, nprimes);
                  const int sh = (int)expa_row[mi] - OZ2_BIAS_PLUS_MANT;
                  const double arecon = (double)a_sign[mi][kk]
                    * (double)mant_recon * libxs_pow2(sh);

                  oz2_store_block_pair(ref_blk, recon_blk, BLOCK_M, mi, kk,
                    aval, (GEMM_REAL_TYPE)arecon);
                }
              }
              oz2_accumulate_block_diff(diff, ref_blk, recon_blk,
                iblk, kblk, BLOCK_M, BLOCK_M);
            }
          }

          /* Diff tracking for B decomposition (diff_abc == 2) */
          if (NULL != diff && 2 == (diff_abc % 3)) {
#if defined(_OPENMP)
#           pragma omp critical
#endif
            { for (kk = 0; kk < kblk; ++kk) {
                const GEMM_INT_TYPE p = kb + kk;
                for (nj = 0; nj < jblk; ++nj) {
                  const GEMM_INT_TYPE col = jb + nj;
                  const GEMM_REAL_TYPE bval = ((p < K && col < N)
                    ? b[LIBXS_INDEX(tb, *ldb, p, col)] : (GEMM_REAL_TYPE)0);
                  const uint64_t mant_recon = oz2_reconstruct_mantissa(
                    b_res[kk][nj], garner_inv, nprimes);
                  const int sh = (int)expb_col[nj] - OZ2_BIAS_PLUS_MANT;
                  const double brecon = (double)b_sign[kk][nj]
                    * (double)mant_recon * libxs_pow2(sh);

                  oz2_store_block_pair(ref_blk, recon_blk, BLOCK_K, kk, nj,
                    bval, (GEMM_REAL_TYPE)brecon);
                }
              }
              oz2_accumulate_block_diff(diff, ref_blk, recon_blk,
                kblk, jblk, BLOCK_K, BLOCK_K);
            }
          }

          /* Transpose a_res/b_res into k-contiguous layout for dot products,
           * and gather signs into k-contiguous rows/columns. */
          for (mi = 0; mi < iblk; ++mi) {
            for (pidx = 0; pidx < nprimes; ++pidx) {
              for (kk = 0; kk < BLOCK_K; ++kk) {
                ak[mi][pidx][kk] = a_res[mi][kk][pidx];
              }
            }
            for (kk = 0; kk < BLOCK_K; ++kk) {
              ak_sign[mi][kk] = a_sign[mi][kk];
            }
          }
          for (nj = 0; nj < jblk; ++nj) {
            for (pidx = 0; pidx < nprimes; ++pidx) {
              for (kk = 0; kk < BLOCK_K; ++kk) {
                bk[nj][pidx][kk] = b_res[kk][nj][pidx];
              }
            }
            for (kk = 0; kk < BLOCK_K; ++kk) {
              bk_sign[nj][kk] = b_sign[kk][nj];
            }
          }

          /* CRT dot products: for each (i,j) pair, compute the dot product
           * modulo each prime, then reconstruct via CRT. This is *linear*
           * in nprimes (vs quadratic in nslices for Scheme 1). */
          for (mi = 0; mi < iblk; ++mi) {
            for (nj = 0; nj < jblk; ++nj) {
              unsigned int dot_residues[OZ2_MAX_NPRIMES];
              int8_t csign[BLOCK_K]; /* combined sign per k */
              double dot_value, contrib;
              int sh;

              /* Precompute combined sign = sign_a * sign_b per k element */
              for (kk = 0; kk < kblk; ++kk) {
                csign[kk] = (int8_t)(ak_sign[mi][kk] * bk_sign[nj][kk]);
              }

              /* For each prime: accumulate unsigned positive/negative partial
               * sums separately, then combine modularly (avoids signed % in
               * C89 where the result is implementation-defined). */
              LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_MAX_NPRIMES, OZ2_NPRIMES_DEFAULT)
              for (pidx = 0; pidx < nprimes; ++pidx) {
                const unsigned int p = oz2_primes[pidx];
                uint32_t pos_sum = 0, neg_sum = 0;
                for (kk = 0; kk < kblk; ++kk) {
                  const uint32_t prod = (uint32_t)ak[mi][pidx][kk]
                                      * (uint32_t)bk[nj][pidx][kk];
                  if (csign[kk] > 0) pos_sum += prod;
                  else neg_sum += prod;
                }
                dot_residues[pidx] = (unsigned int)(
                  (pos_sum % p + p - neg_sum % p) % p);
              }

              /* CRT reconstruction -> signed double */
              dot_value = oz2_reconstruct(dot_residues, garner_inv, nprimes);

              /* Apply exponent scale and alpha.
               * shift = exp_row + exp_col - 2*BIAS_PLUS_MANT covers the
               * full mantissa range (no per-slice offset like Scheme 1). */
              sh = (int)expa_row[mi] + (int)expb_col[nj]
                 - (2 * OZ2_BIAS_PLUS_MANT);
              contrib = (*alpha) * dot_value * libxs_pow2(sh);
              mb[mi + nj * ldcv] += (GEMM_REAL_TYPE)contrib;
            }
          }
        } /* end kb loop */

        /* Accumulate diff against reference GEMM (diff_abc == 0) */
        if (NULL != diff && 0 == (diff_abc % 3)) {
          const GEMM_INT_TYPE mref = BLOCK_M;
#if defined(_OPENMP)
#         pragma omp critical
#endif
          { if (NULL != gemm_original) {
              gemm_original(transa, transb, &iblk, &jblk, &K, alpha,
                a + LIBXS_INDEX(ta, *lda, ib, 0), lda,
                b + LIBXS_INDEX(tb, *ldb, 0, jb), ldb, beta, ref_blk, &mref);
            }
            else {
              GEMM_REAL(transa, transb, &iblk, &jblk, &K, alpha,
                a + LIBXS_INDEX(ta, *lda, ib, 0), lda,
                b + LIBXS_INDEX(tb, *ldb, 0, jb), ldb, beta, ref_blk, &mref);
            }
            oz2_accumulate_block_diff(diff, ref_blk, mb,
              iblk, jblk, mref, ldcv);
          }
        }
      }
    } /* end parallel for */
  } /* end parallel */
}


/*=== Public API =============================================================*/

LIBXS_API void gemm_oz2(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  if (0 == gemm_verbose) {
    gemm_oz2_diff(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      0 /*diff_abc*/, NULL /*diff*/);
  }
  else {
    double epsilon;
    libxs_matdiff_info_t diff;
    libxs_matdiff_clear(&diff);
    gemm_oz2_diff(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      LIBXS_ABS(gemm_diff_abc), &diff);

    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    libxs_matdiff_reduce(&gemm_diff, &diff); diff = gemm_diff;
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);

    epsilon = libxs_matdiff_epsilon(&diff);
    if (1 < gemm_verbose || 0 > gemm_verbose) {
      const int nth = (0 < gemm_verbose ? gemm_verbose : 1);
      if (0 == (diff.r % nth)) print_diff(stderr, &diff);
    }
    if (gemm_eps < epsilon || diff.rsq < gemm_rsq || 0 > gemm_verbose) {
      if (0 != gemm_dump_inhibit) {
        gemm_dump_inhibit = 2; /* signal pending composite dump */
      }
      else {
        gemm_dump_matrices(GEMM_ARGPASS, 1);
      }
    }
  }
}
