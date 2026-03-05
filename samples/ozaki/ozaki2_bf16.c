/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"

/* Number of (mi,nj) elements processed in one Garner reconstruction batch.
 * Batching exposes data-parallelism across independent elements so the
 * compiler can auto-vectorize the per-element Garner inner loop. */
#if !defined(OZ4_BATCH)
# define OZ4_BATCH 16
#endif

/* Maximum mixed-radix digits per uint64 Horner group.
 * 9 largest moduli product ~ 2^62.9, fits uint64.
 * Grouping eliminates FP64 from the inner Horner loop. */
#if !defined(OZ4_HORNER_GROUP)
# define OZ4_HORNER_GROUP 9
#endif


/* Chinese Remainder Theorem (CRT) moduli and precomputed Barrett tables.
 * Same moduli as Ozaki Scheme-2 (ozaki2_int8.c); the difference is that
 * residues are stored as BF16 values instead of int8, and dot products
 * use VDPBF16PS instead of VNNI.
 *
 * All residues (0..127 unsigned, -127..+127 signed) are exactly
 * representable in BF16 (8 bits of mantissa precision), and dot product
 * accumulation in FP32 is exact for BLOCK_K <= 64 since the maximum
 * absolute value BLOCK_K * 127^2 < 2^20 << 2^24 (FP32 mantissa).
 *
 * P = prod(m_i) must exceed 2 * BLOCK_K * (2^(MANT+1))^2 to represent
 * signed dot products without aliasing. */

/* 18 pairwise coprime moduli <= 128 (prime powers + primes).
 * Product of 17 ~ 2^112 > 2^111 (double).
 * Includes 128=2^7, 125=5^3, 121=11^2, 81=3^4 alongside primes.
 * Residues 0..127 fit exactly in BF16. */
static const uint16_t oz4_moduli[] = {
  128, 127, 125, 121, 113, 109, 107, 103,
  101,  97,  89,  83,  81,  79,  73,  71,
   67,  61
};

/* Barrett reciprocals: floor(2^32 / m_i) for single-word reduction. */
static const uint32_t oz4_rcp[] = {
  (uint32_t)(0x100000000ULL / 128), (uint32_t)(0x100000000ULL / 127),
  (uint32_t)(0x100000000ULL / 125), (uint32_t)(0x100000000ULL / 121),
  (uint32_t)(0x100000000ULL / 113), (uint32_t)(0x100000000ULL / 109),
  (uint32_t)(0x100000000ULL / 107), (uint32_t)(0x100000000ULL / 103),
  (uint32_t)(0x100000000ULL / 101), (uint32_t)(0x100000000ULL /  97),
  (uint32_t)(0x100000000ULL /  89), (uint32_t)(0x100000000ULL /  83),
  (uint32_t)(0x100000000ULL /  81), (uint32_t)(0x100000000ULL /  79),
  (uint32_t)(0x100000000ULL /  73), (uint32_t)(0x100000000ULL /  71),
  (uint32_t)(0x100000000ULL /  67), (uint32_t)(0x100000000ULL /  61)
};

/* 2^18 mod m_i and 2^36 mod m_i: used for 64-bit Barrett decomposition. */
static const uint32_t oz4_pow18[] = {
    0U /* 128 */,  16U /* 127 */,  19U /* 125 */,  58U /* 121 */,
   97U /* 113 */, 108U /* 109 */, 101U /* 107 */,   9U /* 103 */,
   49U /* 101 */,  50U /*  97 */,  39U /*  89 */,  30U /*  83 */,
   28U /*  81 */,  22U /*  79 */,   1U /*  73 */,  12U /*  71 */,
   40U /*  67 */,  27U /*  61 */
};
static const uint32_t oz4_pow36[] = {
    0U /* 128 */,   2U /* 127 */, 111U /* 125 */,  97U /* 121 */,
   30U /* 113 */,   1U /* 109 */,  36U /* 107 */,  81U /* 103 */,
   78U /* 101 */,  75U /*  97 */,   8U /*  89 */,  70U /*  83 */,
   55U /*  81 */,  10U /*  79 */,   1U /*  73 */,   2U /*  71 */,
   59U /*  67 */,  58U /*  61 */
};


/** Fast modular reduction: x mod oz4_moduli[pidx] (table-indexed wrapper). */
LIBXS_API_INLINE unsigned int oz4_mod(uint32_t x, int pidx)
{
  return libxs_mod_u32(x, oz4_moduli[pidx], oz4_rcp[pidx]);
}


/**
 *  Bounded modular reduction for mixed-radix digits: v mod oz4_moduli[pidx].
 *  v must be a Garner digit, i.e. v < max(moduli) = 128.
 *  Since min(moduli) = 61, floor(127/61) = 2, so at most two conditional
 *  subtracts suffice — avoiding the Barrett multiply in the O(nprimes^2)
 *  Garner inner loop.
 */
LIBXS_API_INLINE unsigned int oz4_mod_digit(unsigned int v, int pidx)
{
  const unsigned int p = oz4_moduli[pidx];
  if (v >= p) v -= p;
  if (v >= p) v -= p;
  return v;
}


/** Fast 64-bit modular reduction: x mod oz4_moduli[pidx] (table-indexed wrapper). */
LIBXS_API_INLINE unsigned int oz4_mod64(uint64_t x, int pidx)
{
  return libxs_mod_u64(x, oz4_moduli[pidx], oz4_rcp[pidx],
    oz4_pow18[pidx], oz4_pow36[pidx]);
}


/**
 *  Reduce an aligned mantissa modulo all active moduli.
 *  delta = max_exp - element_exp (>= 0); mantissa is right-shifted
 *  by delta bits for exponent alignment before reduction.
 */
LIBXS_API_INLINE void oz4_reduce(uint64_t mantissa, int delta,
  uint8_t residues[OZ2_NPRIMES_MAX], int nprimes)
{
  int i;
  nprimes = LIBXS_CLMP(nprimes, 0, OZ2_NPRIMES_MAX);
  if (delta > 0) {
    if (delta >= 64) mantissa = 0;
    else mantissa >>= delta;
  }
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    residues[i] = (uint8_t)oz4_mod64(mantissa, i);
  }
}


/* Forward declaration: needed by oz4_reconstruct_batch_avx512 (defined below). */
LIBXS_API_INLINE double oz4_horner_grouped(const unsigned int v[], int nprimes);

/* AVX-512 batched CRT reconstruction via Garner's algorithm.
 * Processes OZ4_BATCH (= 16) uint32 values in a single __m512i.
 * Uses libxs_mulhi_epu32 and libxs_mod_u32x16 from libxs_utils.h. */
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == OZ4_BATCH

/**
 *  AVX-512 batched CRT reconstruction via Garner's algorithm.
 *  Uses transposed internal layout vt[prime][batch] for contiguous
 *  SIMD access, bounded subtracts for digit reduction, and vectorized
 *  Barrett for the Garner product reduction.
 */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void oz4_reconstruct_batch_avx512(
  unsigned int batch_res[OZ4_BATCH][OZ2_NPRIMES_MAX],
  unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX],
  int nprimes, int bsz, double result[OZ4_BATCH])
{
  /* Transposed layout: vt[prime_idx][OZ4_BATCH] for contiguous SIMD loads */
  unsigned int vt[OZ2_NPRIMES_MAX][OZ4_BATCH];
  __m512i u_vec;
  int i, j, bi;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_NPRIMES_MAX);

  /* Garner's algorithm: SIMD across batch elements */
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    const unsigned int pi = oz4_moduli[i];
    const unsigned int rcp_i = oz4_rcp[i];
    const __m512i vpi = _mm512_set1_epi32((int)pi);

    { /* Load residues for prime i across batch elements (strided gather) */ unsigned int tmp[OZ4_BATCH];
      for (bi = 0; bi < bsz; ++bi) tmp[bi] = batch_res[bi][i];
      for (bi = bsz; bi < OZ4_BATCH; ++bi) tmp[bi] = 0;
      u_vec = _mm512_loadu_si512((__m512i*)tmp);
    }

    for (j = 0; j < i; ++j) {
      const unsigned int inv_ji = garner_inv[j][i];
      const __m512i vinv = _mm512_set1_epi32((int)inv_ji);

      /* Load vt[j] — contiguous */
      __m512i vj_vec = _mm512_loadu_si512((__m512i*)vt[j]);

      /* Bounded subtract: reduce vt[j][bi] mod pi.
       * Digits < max(moduli) = 128, min(moduli) = 61: two subtracts suffice. */
      { __mmask16 ge = _mm512_cmpge_epu32_mask(vj_vec, vpi);
        vj_vec = _mm512_mask_sub_epi32(vj_vec, ge, vj_vec, vpi);
        ge = _mm512_cmpge_epu32_mask(vj_vec, vpi);
        vj_vec = _mm512_mask_sub_epi32(vj_vec, ge, vj_vec, vpi);
      }

      { /* diff = (u >= vj) ? (u - vj) : (pi + u - vj) */ const __mmask16 ge = _mm512_cmpge_epu32_mask(u_vec, vj_vec);
        const __m512i d_pos = _mm512_sub_epi32(u_vec, vj_vec);
        const __m512i d_neg = _mm512_sub_epi32(
          _mm512_add_epi32(vpi, u_vec), vj_vec);
        __m512i diff_vec = _mm512_mask_blend_epi32(ge, d_neg, d_pos);

        /* u = (diff * inv_ji) mod pi via vectorized Barrett */
        diff_vec = _mm512_mullo_epi32(diff_vec, vinv);
        u_vec = libxs_mod_u32x16(diff_vec, pi, rcp_i);
      }
    }

    /* Store u into vt[i] */
    _mm512_storeu_si512((__m512i*)vt[i], u_vec);
  }

  { /* Sign detection, complement, Horner — per element (transpose back) */ unsigned int v_scalar[OZ4_BATCH][OZ2_NPRIMES_MAX];
    unsigned int tmp[OZ4_BATCH];
    for (i = 0; i < nprimes; ++i) {
      _mm512_storeu_si512((__m512i*)tmp,
        _mm512_loadu_si512((__m512i*)vt[i]));
      for (bi = 0; bi < bsz; ++bi) v_scalar[bi][i] = tmp[bi];
    }
    for (bi = 0; bi < bsz; ++bi) {
      const int is_negative = (v_scalar[bi][nprimes - 1]
        >= (unsigned int)(oz4_moduli[nprimes - 1] + 1) / 2) ? 1 : 0;
      double r;
      if (0 != is_negative) {
        for (i = 0; i < nprimes; ++i) {
          v_scalar[bi][i] = oz4_moduli[i] - 1 - v_scalar[bi][i];
        }
      }
      r = oz4_horner_grouped(v_scalar[bi], nprimes);
      result[bi] = (0 != is_negative) ? -(r + 1.0) : r;
    }
  }
}

#endif /* LIBXS_INTRINSICS_AVX512 && OZ4_BATCH == 16 */


/**
 *  Preprocess rows of A for one (ib, kb) tile.
 *  Decomposes each element, finds per-row max exponent, aligns mantissas
 *  by right-shifting, reduces mod each modulus, converts signed residues
 *  to BF16, and writes directly into the k-contiguous layout
 *  ak[M][P][K] used by BF16 dot products.
 */
LIBXS_API_INLINE void oz4_preprocess_rows(
  const GEMM_REAL_TYPE* a, GEMM_INT_TYPE lda, int ta,
  GEMM_INT_TYPE M, GEMM_INT_TYPE K,
  GEMM_INT_TYPE ib, GEMM_INT_TYPE kb,
  GEMM_INT_TYPE iblk, GEMM_INT_TYPE kblk, int nprimes,
  int16_t expa_row[BLOCK_M],
  libxs_bf16_t ak[BLOCK_M][OZ2_NPRIMES_MAX][BLOCK_K])
{
  int16_t elem_exp[BLOCK_M][BLOCK_K];
  uint64_t elem_mant[BLOCK_M][BLOCK_K];
  GEMM_INT_TYPE mi, kk;

  /* Pass 1: decompose and track per-row max exponent */
  for (mi = 0; mi < iblk; ++mi) {
    const GEMM_INT_TYPE row = ib + mi;
    int16_t row_max_exp = INT16_MIN;
    int8_t local_sign[BLOCK_K];
    int pidx;

    for (kk = 0; kk < kblk; ++kk) {
      const GEMM_INT_TYPE p = kb + kk;
      const GEMM_REAL_TYPE aval = ((row < M && p < K)
        ? a[LIBXS_INDEX(ta, lda, row, p)] : (GEMM_REAL_TYPE)0);
      local_sign[kk] = (int8_t)ozaki_extract_ieee(aval, &elem_exp[mi][kk], &elem_mant[mi][kk]);
      row_max_exp = LIBXS_MAX(row_max_exp, elem_exp[mi][kk]);
    }

    expa_row[mi] = row_max_exp;

    /* Pass 2: align, reduce mod moduli, convert to BF16 with sign,
     * scatter into ak[mi][pidx][kk] */
    for (kk = 0; kk < kblk; ++kk) {
      const int delta = (int)row_max_exp - (int)elem_exp[mi][kk];
      uint8_t tmp[OZ2_NPRIMES_MAX];
      oz4_reduce(elem_mant[mi][kk], delta, tmp, nprimes);
      LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
      for (pidx = 0; pidx < nprimes; ++pidx) {
        /* Convert signed integer residue to BF16.
         * Values are in -127..+127, exactly representable in BF16. */
        ak[mi][pidx][kk] = libxs_round_bf16(
          (double)(local_sign[kk] * (int)tmp[pidx]));
      }
    }
    /* Zero-pad remaining k-entries */
    LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
    for (pidx = 0; pidx < nprimes; ++pidx) {
      for (kk = kblk; kk < BLOCK_K; ++kk) ak[mi][pidx][kk] = 0;
    }
  }
}


/**
 *  Preprocess columns of B for one (kb, jb) tile.
 *  Same as rows of A but with per-column max exponent. Writes directly
 *  into bk[N][P][K] (k-contiguous layout) as BF16 values.
 */
LIBXS_API_INLINE void oz4_preprocess_cols(
  const GEMM_REAL_TYPE* b, GEMM_INT_TYPE ldb, int tb,
  GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  GEMM_INT_TYPE jb, GEMM_INT_TYPE kb,
  GEMM_INT_TYPE jblk, GEMM_INT_TYPE kblk, int nprimes,
  int16_t expb_col[BLOCK_N],
  libxs_bf16_t bk[BLOCK_N][OZ2_NPRIMES_MAX][BLOCK_K])
{
  int16_t elem_exp[BLOCK_K][BLOCK_N];
  uint64_t elem_mant[BLOCK_K][BLOCK_N];
  int8_t elem_sign[BLOCK_K][BLOCK_N];
  GEMM_INT_TYPE nj, kk;
  int pidx;

  for (nj = 0; nj < jblk; ++nj) expb_col[nj] = INT16_MIN;

  /* Pass 1: decompose and track per-column max exponent */
  for (kk = 0; kk < kblk; ++kk) {
    const GEMM_INT_TYPE p = kb + kk;
    for (nj = 0; nj < jblk; ++nj) {
      const GEMM_INT_TYPE col = jb + nj;
      const GEMM_REAL_TYPE bval = ((p < K && col < N)
        ? b[LIBXS_INDEX(tb, ldb, p, col)] : (GEMM_REAL_TYPE)0);
      elem_sign[kk][nj] = (int8_t)ozaki_extract_ieee(bval, &elem_exp[kk][nj], &elem_mant[kk][nj]);
      expb_col[nj] = LIBXS_MAX(expb_col[nj], elem_exp[kk][nj]);
    }
  }

  /* Pass 2: align, reduce mod moduli, convert to BF16 with sign,
   * scatter into bk[nj][pidx][kk] */
  for (kk = 0; kk < kblk; ++kk) {
    for (nj = 0; nj < jblk; ++nj) {
      const int delta = (int)expb_col[nj] - (int)elem_exp[kk][nj];
      uint8_t tmp[OZ2_NPRIMES_MAX];
      oz4_reduce(elem_mant[kk][nj], delta, tmp, nprimes);
      LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
      for (pidx = 0; pidx < nprimes; ++pidx) {
        bk[nj][pidx][kk] = libxs_round_bf16(
          (double)(elem_sign[kk][nj] * (int)tmp[pidx]));
      }
    }
  }
  /* Zero-pad remaining k-entries */
  for (nj = 0; nj < jblk; ++nj) {
    LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
    for (pidx = 0; pidx < nprimes; ++pidx) {
      for (kk = kblk; kk < BLOCK_K; ++kk) bk[nj][pidx][kk] = 0;
    }
  }
}


/**
 *  Evaluate mixed-radix digits v[0..nprimes-1] via grouped uint64 Horner.
 *  Partitions digits into groups of OZ4_HORNER_GROUP, evaluates each group
 *  exactly in uint64 (product of OZ4_HORNER_GROUP moduli < 2^63), then combines groups
 *  with Horner in FP64: ceil(nprimes/OZ4_HORNER_GROUP)-1 FP64 mul-adds.
 */
LIBXS_API_INLINE double oz4_horner_grouped(
  const unsigned int v[], int nprimes)
{
  const int ngroups = (nprimes + OZ4_HORNER_GROUP - 1) / OZ4_HORNER_GROUP;
  double result;
  int g;

  { /* MSB group: indices [(ngroups-1)*OZ4_HORNER_GROUP .. nprimes-1] */ const int lo = (ngroups - 1) * OZ4_HORNER_GROUP;
    uint64_t r = (uint64_t)v[nprimes - 1];
    int i;
    for (i = nprimes - 2; i >= lo; --i) {
      r = r * (uint64_t)oz4_moduli[i] + (uint64_t)v[i];
    }
    result = (double)r;
  }

  /* Combine remaining groups MSB to LSB */
  for (g = ngroups - 2; g >= 0; --g) {
    const int lo = g * OZ4_HORNER_GROUP;
    const int hi = lo + OZ4_HORNER_GROUP - 1;
    uint64_t gval, gprod = 1;
    int i;
    for (i = lo; i <= hi; ++i) gprod *= (uint64_t)oz4_moduli[i];
    gval = (uint64_t)v[hi];
    for (i = hi - 1; i >= lo; --i) {
      gval = gval * (uint64_t)oz4_moduli[i] + (uint64_t)v[i];
    }
    result = result * (double)gprod + (double)gval;
  }

  return result;
}


/**
 *  Reconstruct a signed integer from its CRT residues.
 *
 *  Uses Garner's algorithm to compute mixed-radix digits, detects the
 *  sign from the most significant digit (centered representation), and
 *  evaluates via Horner's method into a double.
 *
 *  For negative values the digits are complemented before Horner
 *  evaluation so that the absolute value is computed directly, avoiding
 *  catastrophic cancellation from subtracting the large modulus product P.
 *
 *  Centered range: result in (-P/2, P/2], where P = prod(m_i).
 */
LIBXS_API_INLINE double oz4_reconstruct(
  const unsigned int residues[OZ2_NPRIMES_MAX],
  unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX],
  int nprimes)
{
  unsigned int v[OZ2_NPRIMES_MAX]; /* mixed-radix digits */
  double result;
  int i, j, is_negative;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_NPRIMES_MAX);

  /* Garner's algorithm: compute mixed-radix digits v[i] in [0, m_i) */
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    unsigned int u = residues[i];
    const unsigned int pi = oz4_moduli[i];
    for (j = 0; j < i; ++j) {
      /* v[j] < m_j <= 128; reduce into [0, m_i) via bounded subtract
       * (avoids Barrett multiply; floor(127/61) = 2 so two subtracts suffice). */
      unsigned int vj = v[j];
      if (vj >= pi) vj -= pi;
      if (vj >= pi) vj -= pi;
      { const unsigned int diff = (u >= vj) ? (u - vj) : (pi + u - vj);
        u = oz4_mod(diff * garner_inv[j][i], i);
      }
    }
    v[i] = u;
  }

  /* Sign detection */
  is_negative = (v[nprimes - 1] >= (unsigned int)(oz4_moduli[nprimes - 1] + 1) / 2)
    ? 1 : 0;

  /* Complement digits for negative values: P - 1 - V has digits
   * (m_i - 1 - v_i) in mixed-radix representation (no borrows needed).
   * Then |D| = (P - 1 - V) + 1 = P - V where V is the CRT result. */
  if (0 != is_negative) {
    LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
    for (i = 0; i < nprimes; ++i) {
      v[i] = oz4_moduli[i] - 1 - v[i];
    }
  }

  /* Grouped Horner: uint64 within groups, FP64 only between groups */
  result = oz4_horner_grouped(v, nprimes);

  /* For negative values: result holds |D|-1 (the complement),
   * so the true value is -(result + 1.0). */
  if (0 != is_negative) {
    result = -(result + 1.0);
  }

  return result;
}


/**
 *  Reconstruct a single element's unsigned mantissa from its CRT residues.
 *  Uses Garner+Horner in uint64 arithmetic (exact for mantissa <= 2^53).
 *  Used for diff tracking of A/B matrices.
 */
LIBXS_API_INLINE uint64_t oz4_reconstruct_mantissa(
  const uint8_t residues[OZ2_NPRIMES_MAX],
  unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX],
  int nprimes)
{
  unsigned int v[OZ2_NPRIMES_MAX];
  uint64_t result;
  int i, j;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_NPRIMES_MAX);

  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    unsigned int u = (unsigned int)residues[i];
    const unsigned int pi = oz4_moduli[i];
    for (j = 0; j < i; ++j) {
      /* Bounded subtract: v[j] < m_j <= 128, two subtracts suffice. */
      unsigned int vj = v[j];
      if (vj >= pi) vj -= pi;
      if (vj >= pi) vj -= pi;
      { const unsigned int diff = (u >= vj) ? (u - vj) : (pi + u - vj);
        u = oz4_mod(diff * garner_inv[j][i], i);
      }
    }
    v[i] = u;
  }

  /* Horner in uint64 (exact for mantissa values up to 2^53) */
  result = (uint64_t)v[nprimes - 1];
  LIBXS_PRAGMA_LOOP_COUNT(0, OZ2_NPRIMES_MAX - 1, OZ2_NPRIMES_DEFAULT - 1)
  for (i = nprimes - 2; i >= 0; --i) {
    result = result * (uint64_t)oz4_moduli[i] + (uint64_t)v[i];
  }
  return result;
}


/**
 *  Batched CRT reconstruction: process OZ4_BATCH elements in parallel
 *  through Garner's algorithm. The innermost loop over batch elements
 *  is data-parallel, enabling auto-vectorization.
 */
LIBXS_API_INLINE void oz4_reconstruct_batch(
  unsigned int batch_res[OZ4_BATCH][OZ2_NPRIMES_MAX],
  unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX],
  int nprimes, int bsz, double result[OZ4_BATCH])
{
  unsigned int v[OZ4_BATCH][OZ2_NPRIMES_MAX];
  unsigned int u[OZ4_BATCH];
  int i, j, bi;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_NPRIMES_MAX);

  /* Garner's algorithm: vectorized across batch elements */
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    const unsigned int pi = oz4_moduli[i];
    for (bi = 0; bi < bsz; ++bi) u[bi] = batch_res[bi][i];

    for (j = 0; j < i; ++j) {
      const unsigned int inv_ji = garner_inv[j][i];
      for (bi = 0; bi < bsz; ++bi) {
        /* Bounded subtract: v[bi][j] < m_j <= 128, two subtracts suffice. */
        unsigned int vj = v[bi][j];
        if (vj >= pi) vj -= pi;
        if (vj >= pi) vj -= pi;
        { const unsigned int diff = (u[bi] >= vj) ? (u[bi] - vj) : (pi + u[bi] - vj);
          u[bi] = oz4_mod(diff * inv_ji, i);
        }
      }
    }

    for (bi = 0; bi < bsz; ++bi) v[bi][i] = u[bi];
  }

  /* Sign detection, complement, Horner — per element */
  for (bi = 0; bi < bsz; ++bi) {
    const int is_negative = (v[bi][nprimes - 1]
      >= (unsigned int)(oz4_moduli[nprimes - 1] + 1) / 2) ? 1 : 0;
    double r;
    if (0 != is_negative) {
      for (i = 0; i < nprimes; ++i) {
        v[bi][i] = oz4_moduli[i] - 1 - v[bi][i];
      }
    }
    r = oz4_horner_grouped(v[bi], nprimes);
    result[bi] = (0 != is_negative) ? -(r + 1.0) : r;
  }
}


LIBXS_API_INLINE void gemm_oz4_diff(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                                 const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
  unsigned int diff_abc, libxs_matdiff_info_t* diff)
{
  unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX];
  enum {
    BLOCK_MN = BLOCK_M * BLOCK_N,
    BLOCK_MK = BLOCK_M * BLOCK_K,
    BLOCK_KN = BLOCK_K * BLOCK_N,
    BLOCK_MNK = LIBXS_MAX(LIBXS_MAX(BLOCK_MN, BLOCK_MK), BLOCK_KN),
    NKB_MAX = BATCH_K /* sub-panels per batch */
  };
  const int ta = (*transa != 'N' && *transa != 'n');
  const int tb = (*transb != 'N' && *transb != 'n');
  const GEMM_INT_TYPE M = *m, N = *n, K = *k;
  const GEMM_INT_TYPE ldcv = *ldc;
  const int nprimes = LIBXS_CLMP(gemm_ozn, 1, OZ2_NPRIMES_MAX);
  const GEMM_INT_TYPE nblk_m = (M + BLOCK_M - 1) / BLOCK_M;
  const GEMM_INT_TYPE nblk_n = (N + BLOCK_N - 1) / BLOCK_N;
  /* Panel buffers: preprocessed A and B for one K-batch (shared).
   * BF16 panels instead of int8 — same CRT decomposition, BF16 storage. */
  libxs_bf16_t (*ak_panel)[BLOCK_M][OZ2_NPRIMES_MAX][BLOCK_K] = NULL;
  int16_t (*expa_panel)[BLOCK_M] = NULL;
  libxs_bf16_t (*bk_panel)[BLOCK_N][OZ2_NPRIMES_MAX][BLOCK_K] = NULL;
  int16_t (*expb_panel)[BLOCK_N] = NULL;
  GEMM_REAL_TYPE *ref_panel = NULL; /* diff mode 0 only */
  libxs_matdiff_info_t tdiff[256];
  int nthreads = 1;
  int i, j;
  const ozaki_dot_bf16_fn dot_bf16 = ozaki_dot_bf16_init();
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE)
            || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));
  LIBXS_ASSERT(1 <= BATCH_K);

  /* Precompute Garner modular inverse table:
   * garner_inv[i][j] = m_i^{-1} mod m_j  for i < j */
  memset(garner_inv, 0, sizeof(garner_inv));
  for (i = 0; i < nprimes; ++i) {
    for (j = i + 1; j < nprimes; ++j) {
      garner_inv[i][j] = libxs_mod_inverse_u32(
        oz4_moduli[i] % oz4_moduli[j], oz4_moduli[j]);
    }
  }

  ak_panel = (libxs_bf16_t (*)[BLOCK_M][OZ2_NPRIMES_MAX][BLOCK_K])libxs_malloc(
    gemm_pool, (size_t)NKB_MAX * nblk_m * sizeof(*ak_panel), 0);
  expa_panel = (int16_t (*)[BLOCK_M])libxs_malloc(
    gemm_pool, (size_t)NKB_MAX * nblk_m * sizeof(*expa_panel), 0);
  bk_panel = (libxs_bf16_t (*)[BLOCK_N][OZ2_NPRIMES_MAX][BLOCK_K])libxs_malloc(
    gemm_pool, (size_t)NKB_MAX * nblk_n * sizeof(*bk_panel), 0);
  expb_panel = (int16_t (*)[BLOCK_N])libxs_malloc(
    gemm_pool, (size_t)NKB_MAX * nblk_n * sizeof(*expb_panel), 0);
  if (NULL != diff && 0 == (diff_abc % 3)) {
    ref_panel = (GEMM_REAL_TYPE*)libxs_malloc(
      gemm_pool, (size_t)nblk_m * nblk_n * BLOCK_MN * sizeof(*ref_panel), 0);
  }

#if defined(_OPENMP)
# pragma omp parallel
#endif
  {
    GEMM_REAL_TYPE recon_blk[BLOCK_MNK];
    GEMM_INT_TYPE kb_batch, kb_sub, mi, nj, kk, jb, ib, ib_idx, jb_idx;
    int pidx;
    int tid = 0;
#if defined(_OPENMP)
    tid = omp_get_thread_num();
#   pragma omp single
    nthreads = omp_get_num_threads();
#endif
    if (NULL != diff) libxs_matdiff_clear(&tdiff[tid]);

    for (kb_batch = 0; kb_batch < K; kb_batch += BATCH_K * BLOCK_K) {
      const GEMM_INT_TYPE batch_end = LIBXS_MIN(kb_batch + BATCH_K * BLOCK_K, K);
      const GEMM_INT_TYPE nkb = (batch_end - kb_batch + BLOCK_K - 1) / BLOCK_K;

      /* Phase 1: preprocess all A row-blocks for all sub-panels in batch */
#if defined(_OPENMP)
#     pragma omp for schedule(dynamic) nowait
#endif
      for (ib_idx = 0; ib_idx < nblk_m * nkb; ++ib_idx) {
        const GEMM_INT_TYPE ibi = ib_idx / nkb;
        const GEMM_INT_TYPE ki = ib_idx % nkb;
        const GEMM_INT_TYPE kb = kb_batch + ki * BLOCK_K;
        const GEMM_INT_TYPE ib2 = ibi * BLOCK_M;
        const GEMM_INT_TYPE iblk2 = LIBXS_MIN(BLOCK_M, M - ib2);
        const GEMM_INT_TYPE kblk = LIBXS_MIN(BLOCK_K, K - kb);
        oz4_preprocess_rows(a, *lda, ta, M, K, ib2, kb, iblk2, kblk,
          nprimes, expa_panel[ki * nblk_m + ibi],
          ak_panel[ki * nblk_m + ibi]);
      }

      /* Phase 2: preprocess all B col-blocks for all sub-panels in batch */
#if defined(_OPENMP)
#     pragma omp for schedule(dynamic)
#endif
      for (jb_idx = 0; jb_idx < nblk_n * nkb; ++jb_idx) {
        const GEMM_INT_TYPE jbi = jb_idx / nkb;
        const GEMM_INT_TYPE ki = jb_idx % nkb;
        const GEMM_INT_TYPE kb = kb_batch + ki * BLOCK_K;
        const GEMM_INT_TYPE jb2 = jbi * BLOCK_N;
        const GEMM_INT_TYPE jblk2 = LIBXS_MIN(BLOCK_N, N - jb2);
        const GEMM_INT_TYPE kblk = LIBXS_MIN(BLOCK_K, K - kb);
        oz4_preprocess_cols(b, *ldb, tb, N, K, jb2, kb, jblk2, kblk,
          nprimes, expb_panel[ki * nblk_n + jbi],
          bk_panel[ki * nblk_n + jbi]);
      }
      /* implicit barrier ensures panels are ready */

      /* Phase 3: CRT dot products (BF16) + accumulate using panel data */
#if defined(_OPENMP)
#     pragma omp for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
#endif
      for (jb = 0; jb < N; jb += BLOCK_N) {
        for (ib = 0; ib < M; ib += BLOCK_M) {
          const GEMM_INT_TYPE ibi = ib / BLOCK_M;
          const GEMM_INT_TYPE jbi = jb / BLOCK_N;
          const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
          const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
          GEMM_REAL_TYPE *const cb = c + jb * ldcv + ib;

          /* Beta scaling at first K-batch only */
          if (0 == kb_batch) {
            if (NULL != ref_panel) {
              ozaki_scale_block_beta(cb, ldcv, iblk, jblk, beta,
                ref_panel + (jbi * nblk_m + ibi) * BLOCK_MN, 1);
            }
            else {
              ozaki_scale_block_beta(cb, ldcv, iblk, jblk, beta, NULL, 0);
            }
          }

          /* Loop over sub-panels within this batch */
          for (kb_sub = 0; kb_sub < nkb; ++kb_sub) {
            const GEMM_INT_TYPE kb = kb_batch + kb_sub * BLOCK_K;
            const GEMM_INT_TYPE kblk = LIBXS_MIN(BLOCK_K, K - kb);
            const GEMM_INT_TYPE a_idx = kb_sub * nblk_m + ibi;
            const GEMM_INT_TYPE b_idx = kb_sub * nblk_n + jbi;

            /* Diff tracking for A decomposition (diff_abc == 1) */
            if (NULL != diff && 1 == (diff_abc % 3)) {
              GEMM_REAL_TYPE ref_blk[BLOCK_MNK];
              for (mi = 0; mi < iblk; ++mi) {
                const GEMM_INT_TYPE row = ib + mi;
                for (kk = 0; kk < kblk; ++kk) {
                  const GEMM_INT_TYPE p = kb + kk;
                  const GEMM_REAL_TYPE aval = ((row < M && p < K)
                    ? a[LIBXS_INDEX(ta, *lda, row, p)] : (GEMM_REAL_TYPE)0);
                  uint8_t tmp[OZ2_NPRIMES_MAX];
                  uint64_t mant_recon;
                  int sh;
                  double arecon;
                  for (pidx = 0; pidx < nprimes; ++pidx) {
                    /* Convert BF16 back to signed integer, take absolute value */
                    const int rv = (int)libxs_bf16_to_f64(ak_panel[a_idx][mi][pidx][kk]);
                    tmp[pidx] = (uint8_t)(rv < 0 ? -rv : rv);
                  }
                  mant_recon = oz4_reconstruct_mantissa(tmp, garner_inv, nprimes);
                  sh = (int)expa_panel[a_idx][mi] - OZ_BIAS_PLUS_MANT;
                  { /* recover sign from any nonzero residue */
                    int ds = 1;
                    for (pidx = 0; pidx < nprimes; ++pidx) {
                      const int rv = (int)libxs_bf16_to_f64(ak_panel[a_idx][mi][pidx][kk]);
                      if (0 != rv) {
                        ds = (rv < 0) ? -1 : 1;
                        break;
                      }
                    }
                    arecon = (double)ds * (double)mant_recon * libxs_pow2(sh);
                  }
                  ozaki_store_block_pair(ref_blk, recon_blk, BLOCK_M, mi, kk,
                    aval, (GEMM_REAL_TYPE)arecon);
                }
              }
              ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, recon_blk,
                iblk, kblk, BLOCK_M, BLOCK_M);
            }

            /* Diff tracking for B decomposition (diff_abc == 2) */
            if (NULL != diff && 2 == (diff_abc % 3)) {
              GEMM_REAL_TYPE ref_blk[BLOCK_MNK];
              for (kk = 0; kk < kblk; ++kk) {
                const GEMM_INT_TYPE p = kb + kk;
                for (nj = 0; nj < jblk; ++nj) {
                  const GEMM_INT_TYPE col = jb + nj;
                  const GEMM_REAL_TYPE bval = ((p < K && col < N)
                    ? b[LIBXS_INDEX(tb, *ldb, p, col)] : (GEMM_REAL_TYPE)0);
                  uint8_t tmp[OZ2_NPRIMES_MAX];
                  uint64_t mant_recon;
                  int sh;
                  double brecon;
                  for (pidx = 0; pidx < nprimes; ++pidx) {
                    const int rv = (int)libxs_bf16_to_f64(bk_panel[b_idx][nj][pidx][kk]);
                    tmp[pidx] = (uint8_t)(rv < 0 ? -rv : rv);
                  }
                  mant_recon = oz4_reconstruct_mantissa(tmp, garner_inv, nprimes);
                  sh = (int)expb_panel[b_idx][nj] - OZ_BIAS_PLUS_MANT;
                  { int ds = 1;
                    for (pidx = 0; pidx < nprimes; ++pidx) {
                      const int rv = (int)libxs_bf16_to_f64(bk_panel[b_idx][nj][pidx][kk]);
                      if (0 != rv) {
                        ds = (rv < 0) ? -1 : 1;
                        break;
                      }
                    }
                    brecon = (double)ds * (double)mant_recon * libxs_pow2(sh);
                  }
                  ozaki_store_block_pair(ref_blk, recon_blk, BLOCK_K, kk, nj,
                    bval, (GEMM_REAL_TYPE)brecon);
                }
              }
              ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, recon_blk,
                kblk, jblk, BLOCK_K, BLOCK_K);
            }

            /* CRT dot products via BF16: for each (i,j) pair, compute
             * the dot product modulo each modulus using BF16 hardware
             * (VDPBF16PS) or scalar fallback, then reconstruct via CRT.
             * BF16 dot product of small-integer residues (-127..+127) is
             * exact in FP32 accumulation for BLOCK_K <= 64. */
            for (mi = 0; mi < iblk; ++mi) {
              for (nj = 0; nj < jblk; nj += OZ4_BATCH) {
                const GEMM_INT_TYPE bsz = LIBXS_MIN(OZ4_BATCH,
                  (int)(jblk - nj));
                unsigned int batch_res[OZ4_BATCH][OZ2_NPRIMES_MAX];
                double batch_val[OZ4_BATCH];
                int bi;

                for (bi = 0; bi < (int)bsz; ++bi) {
                  const int col = (int)nj + bi;

                  /* Dot products mod each modulus via BF16 */
                  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
                  for (pidx = 0; pidx < nprimes; ++pidx) {
                    /* BF16 dot product -> FP32 (exact for small integers) */
                    const float fdot = dot_bf16(
                      ak_panel[a_idx][mi][pidx],
                      bk_panel[b_idx][col][pidx]);
                    /* Convert exact FP32 integer to int32 for Barrett reduction */
                    const int32_t dot = (int32_t)fdot;
                    if (dot >= 0) {
                      batch_res[bi][pidx] = oz4_mod((uint32_t)dot, pidx);
                    } else {
                      const unsigned int r = oz4_mod((uint32_t)(-dot), pidx);
                      batch_res[bi][pidx] = (0 != r)
                        ? (oz4_moduli[pidx] - r) : 0;
                    }
                  }
                }

                /* Batched CRT reconstruction -> signed doubles */
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == OZ4_BATCH
# if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
                oz4_reconstruct_batch_avx512(batch_res, garner_inv,
                  nprimes, (int)bsz, batch_val);
# else
                if (LIBXS_X86_AVX512 <= ozaki_target_arch) {
                  oz4_reconstruct_batch_avx512(batch_res, garner_inv,
                    nprimes, (int)bsz, batch_val);
                } else
                {
                  oz4_reconstruct_batch(batch_res, garner_inv, nprimes,
                    (int)bsz, batch_val);
                }
# endif
#else
                oz4_reconstruct_batch(batch_res, garner_inv, nprimes,
                  (int)bsz, batch_val);
#endif

                for (bi = 0; bi < (int)bsz; ++bi) {
                  /* Apply exponent scale and alpha; guard against 0*Inf=NaN
                   * when large exponents make libxs_pow2(sh) overflow to Inf
                   * but the CRT dot product cancelled to exactly zero. */
                  if (0.0 != batch_val[bi] && (GEMM_REAL_TYPE)0 != *alpha) {
                    const int col = (int)nj + bi;
                    const int sh = (int)expa_panel[a_idx][mi] + (int)expb_panel[b_idx][col]
                       - (2 * OZ_BIAS_PLUS_MANT);
                    const double contrib = (*alpha) * batch_val[bi]
                       * libxs_pow2(sh);
                    cb[mi + col * ldcv] += (GEMM_REAL_TYPE)contrib;
                  }
                }
              }
            }
          } /* end kb_sub loop */
        }
      } /* implicit barrier before next kb_batch */
    } /* end kb_batch loop */

    /* Diff mode 0: reference GEMM comparison (after all kb) */
    if (NULL != diff && 0 == (diff_abc % 3)) {
#if defined(_OPENMP)
#     pragma omp for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
#endif
      for (jb = 0; jb < N; jb += BLOCK_N) {
        for (ib = 0; ib < M; ib += BLOCK_M) {
          const GEMM_INT_TYPE ibi = ib / BLOCK_M;
          const GEMM_INT_TYPE jbi = jb / BLOCK_N;
          const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
          const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
          GEMM_REAL_TYPE *const cb = c + jb * ldcv + ib;
          GEMM_REAL_TYPE *const ref_blk = ref_panel + (jbi * nblk_m + ibi) * BLOCK_MN;
          const GEMM_INT_TYPE mref = BLOCK_M;
          if (NULL != gemm_original) {
            gemm_original(transa, transb, &iblk, &jblk, &K, alpha,
              a + LIBXS_INDEX(ta, *lda, ib, 0), lda,
              b + LIBXS_INDEX(tb, *ldb, 0, jb), ldb, beta, ref_blk, &mref);
          }
          else {
            GEMM_REAL(transa, transb, &iblk, &jblk, &K, alpha,
              a + LIBXS_INDEX(ta, *lda, ib, 0), lda,
              b + LIBXS_INDEX(tb, *ldb, 0, jb), ldb, beta, ref_blk, &mref);
          }
          ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, cb, iblk, jblk, mref, ldcv);
        }
      }
    }
  } /* end parallel */

  if (NULL != diff) {
    for (i = 0; i < nthreads; ++i) {
      libxs_matdiff_reduce(diff, &tdiff[i]);
    }
  }
  libxs_free(ak_panel);
  libxs_free(expa_panel);
  libxs_free(bk_panel);
  libxs_free(expb_panel);
  libxs_free(ref_panel);
}


LIBXS_API void gemm_oz4(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                                 const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  OZAKI_GEMM_WRAPPER(gemm_oz4_diff)
}
