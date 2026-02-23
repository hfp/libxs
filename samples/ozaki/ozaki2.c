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
# define OZ2_MAX_NPRIMES     16
# define OZ2_NPRIMES_DEFAULT 15
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
 * Double (53b mantissa, BLOCK_K=16): need P > 2^111 -> 15 primes (2^116).
 * Float  (24b mantissa, BLOCK_K=16): need P > 2^53  -> 7  primes (2^55). */
static const unsigned int oz2_primes[] = {
  251, 241, 239, 233, 229, 227, 223, 211,
  199, 197, 193, 191, 181, 179, 173, 167
};

/* Barrett reciprocals: oz2_rcp[i] = floor(2^32 / oz2_primes[i]).
 * Used by oz2_mod() to replace expensive hardware integer division
 * (idiv ~25 cycles) with a multiply-shift (~5 cycles). */
static const uint32_t oz2_rcp[] = {
  (uint32_t)(0x100000000ULL / 251), (uint32_t)(0x100000000ULL / 241),
  (uint32_t)(0x100000000ULL / 239), (uint32_t)(0x100000000ULL / 233),
  (uint32_t)(0x100000000ULL / 229), (uint32_t)(0x100000000ULL / 227),
  (uint32_t)(0x100000000ULL / 223), (uint32_t)(0x100000000ULL / 211),
  (uint32_t)(0x100000000ULL / 199), (uint32_t)(0x100000000ULL / 197),
  (uint32_t)(0x100000000ULL / 193), (uint32_t)(0x100000000ULL / 191),
  (uint32_t)(0x100000000ULL / 181), (uint32_t)(0x100000000ULL / 179),
  (uint32_t)(0x100000000ULL / 173), (uint32_t)(0x100000000ULL / 167)
};

#if defined(LIBXS_INT128)
/* 64-bit Barrett reciprocals: oz2_rcp64[i] = floor(2^64 / oz2_primes[i]).
 * Used by oz2_mod64() to replace 64-bit hardware division (~40 cycles)
 * with a 128-bit multiply-shift (~5 cycles) in oz2_reduce(). */
# define OZ2_RCP64(P) ((uint64_t)((libxs_uint128_t)1 << 64) / (P))
static const uint64_t oz2_rcp64[] = {
  OZ2_RCP64(251), OZ2_RCP64(241), OZ2_RCP64(239), OZ2_RCP64(233),
  OZ2_RCP64(229), OZ2_RCP64(227), OZ2_RCP64(223), OZ2_RCP64(211),
  OZ2_RCP64(199), OZ2_RCP64(197), OZ2_RCP64(193), OZ2_RCP64(191),
  OZ2_RCP64(181), OZ2_RCP64(179), OZ2_RCP64(173), OZ2_RCP64(167)
};
# define OZ2_HAS_MOD64
#endif

/** Fast modular reduction: x mod oz2_primes[pidx].
 *  Barrett reduction: one 64-bit multiply + shift replaces hardware div.
 *  Valid for x < 2^32 (covers dot sums < BLOCK_K*255^2 and Garner products). */
LIBXS_API_INLINE unsigned int oz2_mod(uint32_t x, int pidx)
{
  const uint32_t p = (uint32_t)oz2_primes[pidx];
  const uint32_t q = (uint32_t)(((uint64_t)x * oz2_rcp[pidx]) >> 32);
  uint32_t r = x - q * p;
  if (r >= p) r -= p;
  return (unsigned int)r;
}

#if defined(OZ2_HAS_MOD64)
/** Fast 64-bit modular reduction: x mod oz2_primes[pidx].
 *  Barrett reduction via 128-bit multiply replaces 64-bit hardware div
 *  (~40 cycles) with a multiply-shift (~5 cycles). Used in oz2_reduce()
 *  for mantissa preprocessing (mantissa up to 2^53). */
LIBXS_API_INLINE unsigned int oz2_mod64(uint64_t x, int pidx)
{
  const uint32_t p = (uint32_t)oz2_primes[pidx];
  const uint64_t q = (uint64_t)(((libxs_uint128_t)x * oz2_rcp64[pidx]) >> 64);
  uint32_t r = (uint32_t)(x - q * p);
  if (r >= p) r -= p;
  return (unsigned int)r;
}
#endif

/* Number of (mi,nj) elements processed in one Garner reconstruction batch.
 * Batching exposes data-parallelism across independent elements so the
 * compiler can auto-vectorize the per-element Garner inner loop. */
#if !defined(OZ2_BATCH)
# define OZ2_BATCH 16
#endif


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
  nprimes = LIBXS_CLMP(nprimes, 0, OZ2_MAX_NPRIMES);
  if (delta > 0) {
    if (delta >= 64) mantissa = 0;
    else mantissa >>= delta;
  }
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_MAX_NPRIMES, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
#if defined(OZ2_HAS_MOD64)
    residues[i] = (uint8_t)oz2_mod64(mantissa, i);
#else
    residues[i] = (uint8_t)(mantissa % oz2_primes[i]);
#endif
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
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_MAX_NPRIMES, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    unsigned int u = residues[i];
    const unsigned int pi = oz2_primes[i];
    for (j = 0; j < i; ++j) {
      /* v[j] < p_j; since primes are descending and close (167..251),
       * at most one conditional subtract replaces v[j] % pi. */
      const unsigned int vj = (v[j] < pi) ? v[j] : (v[j] - pi);
      const unsigned int diff = (u >= vj) ? (u - vj) : (pi + u - vj);
      u = oz2_mod(diff * garner_inv[j][i], i);
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
    LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_MAX_NPRIMES, OZ2_NPRIMES_DEFAULT)
    for (i = 0; i < nprimes; ++i) {
      v[i] = oz2_primes[i] - 1 - v[i];
    }
  }

  /* Horner's method (MSB to LSB): numerically stable since each step is
   * just a multiply by a small prime and add of a small digit. */
  result = (double)v[nprimes - 1];
  LIBXS_PRAGMA_LOOP_COUNT(0, OZ2_MAX_NPRIMES - 1, OZ2_NPRIMES_DEFAULT - 1)
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

  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_MAX_NPRIMES, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    unsigned int u = (unsigned int)residues[i];
    const unsigned int pi = oz2_primes[i];
    for (j = 0; j < i; ++j) {
      const unsigned int vj = (v[j] < pi) ? v[j] : (v[j] - pi);
      const unsigned int diff = (u >= vj) ? (u - vj) : (pi + u - vj);
      u = oz2_mod(diff * garner_inv[j][i], i);
    }
    v[i] = u;
  }

  /* Horner in uint64 (exact for mantissa values up to 2^53) */
  result = (uint64_t)v[nprimes - 1];
  LIBXS_PRAGMA_LOOP_COUNT(0, OZ2_MAX_NPRIMES - 1, OZ2_NPRIMES_DEFAULT - 1)
  for (i = nprimes - 2; i >= 0; --i) {
    result = result * (uint64_t)oz2_primes[i] + (uint64_t)v[i];
  }
  return result;
}


/** Batched CRT reconstruction: process OZ2_BATCH elements in parallel
 *  through Garner's algorithm. The innermost loop over batch elements
 *  is data-parallel, enabling auto-vectorization. */
LIBXS_API_INLINE void oz2_reconstruct_batch(
  unsigned int batch_res[OZ2_BATCH][OZ2_MAX_NPRIMES],
  unsigned int garner_inv[OZ2_MAX_NPRIMES][OZ2_MAX_NPRIMES],
  int nprimes, int bsz, double result[OZ2_BATCH])
{
  unsigned int v[OZ2_BATCH][OZ2_MAX_NPRIMES];
  unsigned int u[OZ2_BATCH];
  int i, j, bi;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_MAX_NPRIMES);

  /* Garner's algorithm: vectorized across batch elements */
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_MAX_NPRIMES, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    const unsigned int pi = oz2_primes[i];
    for (bi = 0; bi < bsz; ++bi) u[bi] = batch_res[bi][i];

    for (j = 0; j < i; ++j) {
      const unsigned int inv_ji = garner_inv[j][i];
      for (bi = 0; bi < bsz; ++bi) {
        const unsigned int vj = (v[bi][j] < pi) ? v[bi][j] : (v[bi][j] - pi);
        const unsigned int diff = (u[bi] >= vj) ? (u[bi] - vj) : (pi + u[bi] - vj);
        u[bi] = oz2_mod(diff * inv_ji, i);
      }
    }

    for (bi = 0; bi < bsz; ++bi) v[bi][i] = u[bi];
  }

  /* Sign detection, complement, Horner — per element */
  for (bi = 0; bi < bsz; ++bi) {
    const int is_negative = (v[bi][nprimes - 1]
      >= (oz2_primes[nprimes - 1] + 1) / 2) ? 1 : 0;
    double r;
    if (0 != is_negative) {
      for (i = 0; i < nprimes; ++i) {
        v[bi][i] = oz2_primes[i] - 1 - v[bi][i];
      }
    }
    r = (double)v[bi][nprimes - 1];
    for (i = nprimes - 2; i >= 0; --i) {
      r = r * (double)oz2_primes[i] + (double)v[bi][i];
    }
    result[bi] = (0 != is_negative) ? -(r + 1.0) : r;
  }
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
  libxs_matdiff_info_t tdiff[256];
  GEMM_INT_TYPE jb, ib;
  int nthreads = 1;
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
# pragma omp parallel
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
    int tid = 0;
#if defined(_OPENMP)
    tid = omp_get_thread_num();
#   pragma omp single
    nthreads = omp_get_num_threads();
#endif
    if (NULL != diff) libxs_matdiff_clear(&tdiff[tid]);

#if defined(_OPENMP)
#   pragma omp for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
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
            for (mi = 0; mi < iblk; ++mi) {
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
            oz2_accumulate_block_diff(&tdiff[tid], ref_blk, recon_blk,
              iblk, kblk, BLOCK_M, BLOCK_M);
          }

          /* Diff tracking for B decomposition (diff_abc == 2) */
          if (NULL != diff && 2 == (diff_abc % 3)) {
            for (kk = 0; kk < kblk; ++kk) {
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
            oz2_accumulate_block_diff(&tdiff[tid], ref_blk, recon_blk,
              kblk, jblk, BLOCK_K, BLOCK_K);
          }

          /* Transpose a_res/b_res into k-contiguous layout for dot products,
           * and gather signs into k-contiguous rows/columns. */
          for (mi = 0; mi < iblk; ++mi) {
            LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_MAX_NPRIMES, OZ2_NPRIMES_DEFAULT)
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
            LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_MAX_NPRIMES, OZ2_NPRIMES_DEFAULT)
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
           * modulo each prime, then reconstruct via CRT. Batching across
           * nj exposes data-parallelism for auto-vectorized Garner. */
          for (mi = 0; mi < iblk; ++mi) {
            for (nj = 0; nj < jblk; nj += OZ2_BATCH) {
              const GEMM_INT_TYPE bsz = LIBXS_MIN(OZ2_BATCH,
                (int)(jblk - nj));
              unsigned int batch_res[OZ2_BATCH][OZ2_MAX_NPRIMES];
              double batch_val[OZ2_BATCH];
              int bi;

              for (bi = 0; bi < (int)bsz; ++bi) {
                const int col = (int)nj + bi;

                /* Branchless sign accumulation + Barrett reduction */
                LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_MAX_NPRIMES, OZ2_NPRIMES_DEFAULT)
                for (pidx = 0; pidx < nprimes; ++pidx) {
                  uint32_t pos_sum = 0, neg_sum = 0;
                  uint32_t pr, nr;
                  for (kk = 0; kk < kblk; ++kk) {
                    const uint32_t prod = (uint32_t)ak[mi][pidx][kk]
                                        * (uint32_t)bk[col][pidx][kk];
                    /* Branchless: sign bit of (sign_a * sign_b) gives mask */
                    const int32_t s = (int32_t)ak_sign[mi][kk]
                                    * (int32_t)bk_sign[col][kk];
                    const uint32_t neg = (uint32_t)(s >> 31);
                    pos_sum += prod & ~neg;
                    neg_sum += prod &  neg;
                  }
                  /* 2 Barrett mods + conditional subtract (was 3 divisions) */
                  pr = (uint32_t)oz2_mod(pos_sum, pidx);
                  nr = (uint32_t)oz2_mod(neg_sum, pidx);
                  batch_res[bi][pidx] = (pr >= nr)
                    ? (pr - nr) : ((uint32_t)oz2_primes[pidx] + pr - nr);
                }
              }

              /* Batched CRT reconstruction -> signed doubles */
              oz2_reconstruct_batch(batch_res, garner_inv, nprimes,
                (int)bsz, batch_val);

              /* Apply exponent scale and alpha */
              for (bi = 0; bi < (int)bsz; ++bi) {
                const int col = (int)nj + bi;
                const int sh = (int)expa_row[mi] + (int)expb_col[col]
                   - (2 * OZ2_BIAS_PLUS_MANT);
                const double contrib = (*alpha) * batch_val[bi]
                   * libxs_pow2(sh);
                mb[mi + col * ldcv] += (GEMM_REAL_TYPE)contrib;
              }
            }
          }
        } /* end kb loop */

        /* Accumulate diff against reference GEMM (diff_abc == 0) */
        if (NULL != diff && 0 == (diff_abc % 3)) {
          const GEMM_INT_TYPE mref = BLOCK_M;
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
            oz2_accumulate_block_diff(&tdiff[tid], ref_blk, mb,
              iblk, jblk, mref, ldcv);
          }
        }
      }
    } /* end parallel for */
  } /* end parallel */
  if (NULL != diff) {
    for (i = 0; i < nthreads; ++i) {
      libxs_matdiff_reduce(diff, &tdiff[i]);
    }
  }
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
      LIBXS_ABS(gemm_wrap), &diff);

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
