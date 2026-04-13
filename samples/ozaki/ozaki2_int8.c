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
#if !defined(OZ2_BATCH)
# define OZ2_BATCH 16
#endif

/* Maximum mixed-radix digits per uint64 Horner group.
 * u8: 8 largest moduli product ~ 2^63.2, fits uint64.
 * i8: 9 largest moduli product ~ 2^61.8, fits uint64.
 * Grouping eliminates FP64 from the inner Horner loop. */
#if !defined(OZ2_HORNER_GROUP)
# if defined(OZAKI_I8) && (OZAKI_I8)
#   define OZ2_HORNER_GROUP 9
# else
#   define OZ2_HORNER_GROUP 8
# endif
#endif


/* Chinese Remainder Theorem (CRT) moduli and precomputed Barrett tables.
 * Moduli must be pairwise coprime (not necessarily prime); using the
 * largest prime power of each prime that fits maximizes bits per channel.
 *
 * u8 (default): moduli <= 256, unsigned residues [0, p-1].
 *   Sign encoded via modular additive inverse (p - r ≡ -r mod p).
 *   Enables u8*u8 VNNI via VPDPBUSD with B-side bias correction.
 *   Larger moduli reduce prime count: fp64 16 (vs 19), fp32 9 (vs 10).
 *   Safe K without chunking: ~33K (255^2 * K per int32 accumulator).
 *
 * i8 (OZAKI_I8=1): moduli <= 128, signed residues [-127, +127].
 *   Sign folded by negation. Uses i8 VNNI via VPDPBUSD with A-side bias.
 *   Safe K without chunking: ~133K (127^2 * K per int32 accumulator).
 *
 * P = prod(m_i) must exceed 2 * BLOCK_K * (2^(MANT+1))^2 to represent
 * signed dot products without aliasing. */

#if defined(OZAKI_I8) && (OZAKI_I8)

/* 20 pairwise coprime moduli <= 128 (prime powers + primes).
 * 128=2^7, 125=5^3, 121=11^2, 119=7*17, 81=3^4 alongside primes. */
static const uint16_t oz2_moduli[] = {128, 127, 125, 121, 119, 113, 109, 107, 103, 101, 97, 89, 83, 81, 79, 73, 71, 67, 61, 59};
static const uint32_t oz2_rcp[] = {(uint32_t)(0x100000000ULL / 128), (uint32_t)(0x100000000ULL / 127),
  (uint32_t)(0x100000000ULL / 125), (uint32_t)(0x100000000ULL / 121), (uint32_t)(0x100000000ULL / 119),
  (uint32_t)(0x100000000ULL / 113), (uint32_t)(0x100000000ULL / 109), (uint32_t)(0x100000000ULL / 107),
  (uint32_t)(0x100000000ULL / 103), (uint32_t)(0x100000000ULL / 101), (uint32_t)(0x100000000ULL / 97),
  (uint32_t)(0x100000000ULL / 89), (uint32_t)(0x100000000ULL / 83), (uint32_t)(0x100000000ULL / 81),
  (uint32_t)(0x100000000ULL / 79), (uint32_t)(0x100000000ULL / 73), (uint32_t)(0x100000000ULL / 71),
  (uint32_t)(0x100000000ULL / 67), (uint32_t)(0x100000000ULL / 61), (uint32_t)(0x100000000ULL / 59)};
static const uint32_t oz2_pow18[] = {
  0U /* 128 */, 16U /* 127 */, 19U /* 125 */, 58U /* 121 */, 106U /* 119 */, 97U /* 113 */, 108U /* 109 */, 101U /* 107 */,
  9U /* 103 */, 49U /* 101 */, 50U /*  97 */, 39U /*  89 */, 30U /*  83 */, 28U /*  81 */, 22U /*  79 */, 1U /*  73 */,
  12U /*  71 */, 40U /*  67 */, 27U /*  61 */, 7U /*  59 */
};
static const uint32_t oz2_pow36[] = {
  0U /* 128 */, 2U /* 127 */, 111U /* 125 */, 97U /* 121 */, 50U /* 119 */, 30U /* 113 */, 1U /* 109 */, 36U /* 107 */,
  81U /* 103 */, 78U /* 101 */, 75U /*  97 */, 8U /*  89 */, 70U /*  83 */, 55U /*  81 */, 10U /*  79 */, 1U /*  73 */,
  2U /*  71 */, 59U /*  67 */, 58U /*  61 */, 49U /*  59 */
};
typedef int8_t oz2_res_t;

#else /* u8 default */

/* 20 pairwise coprime moduli <= 256 (prime powers + primes).
 * 256=2^8, 243=3^5, 169=13^2 alongside primes. */
static const uint16_t oz2_moduli[] = {
  256, 251, 243, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191, 181, 179, 173, 169, 167, 163};
static const uint32_t oz2_rcp[] = {(uint32_t)(0x100000000ULL / 256), (uint32_t)(0x100000000ULL / 251),
  (uint32_t)(0x100000000ULL / 243), (uint32_t)(0x100000000ULL / 241), (uint32_t)(0x100000000ULL / 239),
  (uint32_t)(0x100000000ULL / 233), (uint32_t)(0x100000000ULL / 229), (uint32_t)(0x100000000ULL / 227),
  (uint32_t)(0x100000000ULL / 223), (uint32_t)(0x100000000ULL / 211), (uint32_t)(0x100000000ULL / 199),
  (uint32_t)(0x100000000ULL / 197), (uint32_t)(0x100000000ULL / 193), (uint32_t)(0x100000000ULL / 191),
  (uint32_t)(0x100000000ULL / 181), (uint32_t)(0x100000000ULL / 179), (uint32_t)(0x100000000ULL / 173),
  (uint32_t)(0x100000000ULL / 169), (uint32_t)(0x100000000ULL / 167), (uint32_t)(0x100000000ULL / 163)};
static const uint32_t oz2_pow18[] = {
  0U /* 256 */, 100U /* 251 */, 190U /* 243 */, 177U /* 241 */, 200U /* 239 */, 19U /* 233 */, 168U /* 229 */, 186U /* 227 */,
  119U /* 223 */, 82U /* 211 */, 61U /* 199 */, 134U /* 197 */, 50U /* 193 */, 92U /* 191 */, 56U /* 181 */, 88U /* 179 */,
  49U /* 173 */, 25U /* 169 */, 121U /* 167 */, 40U /* 163 */
};
static const uint32_t oz2_pow36[] = {
  0U /* 256 */, 211U /* 251 */, 136U /* 243 */, 240U /* 241 */, 87U /* 239 */, 128U /* 233 */, 57U /* 229 */, 92U /* 227 */,
  112U /* 223 */, 183U /* 211 */, 139U /* 199 */, 29U /* 197 */, 184U /* 193 */, 60U /* 191 */, 59U /* 181 */, 47U /* 179 */,
  152U /* 173 */, 118U /* 169 */, 112U /* 167 */, 133U /* 163 */
};
typedef uint8_t oz2_res_t;

#endif /* OZAKI_I8 */

/* Sign folding for residue storage (used by host preprocessing wrappers).
 * u8: additive inverse (p - r) for negatives.
 * i8: sign negation (-r) for negatives. */
#if defined(OZAKI_I8) && (OZAKI_I8)
# define OZ2_SIGN_FOLD_(SIGN, RES, PIDX) ((oz2_res_t)((SIGN) * (int8_t)(RES)))
#else
# define OZ2_SIGN_FOLD_(SIGN, RES, PIDX) ((oz2_res_t)(((SIGN) < 0 && 0 != (RES)) ? (oz2_moduli[(PIDX)] - (RES)) : (RES)))
#endif

/** Fast modular reduction: x mod oz2_moduli[pidx] (table-indexed wrapper). */
LIBXS_API_INLINE unsigned int oz2_mod(uint32_t x, int pidx)
{
  return libxs_mod_u32(x, oz2_moduli[pidx], oz2_rcp[pidx]);
}


/**
 * Bounded modular reduction for mixed-radix digits: v mod oz2_moduli[pidx].
 * v must be a Garner digit, i.e. v < max(moduli).
 * u8: max(moduli)=256, min=163, floor(255/163)=1 -> one subtract suffices.
 * i8: max(moduli)=128, min= 59, floor(127/59) =2 -> two subtracts needed.
 */
LIBXS_API_INLINE unsigned int oz2_mod_digit(unsigned int v, int pidx)
{
  const unsigned int p = oz2_moduli[pidx];
  if (v >= p) v -= p;
#if defined(OZAKI_I8) && (OZAKI_I8)
  if (v >= p) v -= p;
#endif
  return v;
}


/** Fast 64-bit modular reduction: x mod oz2_moduli[pidx] (table-indexed wrapper). */
LIBXS_API_INLINE unsigned int oz2_mod64(uint64_t x, int pidx)
{
  return libxs_mod_u64(x, oz2_moduli[pidx], oz2_rcp[pidx], oz2_pow18[pidx], oz2_pow36[pidx]);
}


/**
 * Reduce an aligned mantissa modulo all active moduli.
 * delta = max_exp - element_exp (>= 0); mantissa is right-shifted
 * by delta bits for exponent alignment before reduction.
 */
LIBXS_API_INLINE void oz2_reduce(uint64_t mantissa, int delta, uint8_t residues[OZ2_NPRIMES_MAX], int nprimes)
{
  int i;
  nprimes = LIBXS_CLMP(nprimes, 0, OZ2_NPRIMES_MAX);
  if (delta > 0) {
    if (delta >= 64) mantissa = 0;
    else mantissa >>= delta;
  }
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    residues[i] = (uint8_t)oz2_mod64(mantissa, i);
  }
}


/* Forward declaration: needed by oz2_reconstruct_batch_avx512 (defined below). */
LIBXS_API_INLINE double oz2_horner_grouped(const unsigned int v[], int nprimes);

/* AVX-512 batched CRT reconstruction via Garner's algorithm.
 * Processes OZ2_BATCH (= 16) uint32 values in a single __m512i.
 * Uses libxs_mulhi_epu32 and libxs_mod_u32x16 from libxs_utils.h. */
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == OZ2_BATCH

/**
 * AVX-512 batched CRT reconstruction via Garner's algorithm.
 * Uses transposed internal layout vt[prime][batch] for contiguous
 * SIMD access, bounded subtracts for digit reduction, and vectorized
 * Barrett for the Garner product reduction.
 */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512) void oz2_reconstruct_batch_avx512(
  unsigned int batch_res[OZ2_BATCH][OZ2_NPRIMES_MAX], unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX], int nprimes,
  int bsz, double result[OZ2_BATCH])
{
  /* Transposed layout: vt[prime_idx][OZ2_BATCH] for contiguous SIMD loads */
  unsigned int vt[OZ2_NPRIMES_MAX][OZ2_BATCH];
  __m512i u_vec;
  int i, j, bi;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_NPRIMES_MAX);

  /* Garner's algorithm: SIMD across batch elements */
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    const unsigned int pi = oz2_moduli[i];
    const unsigned int rcp_i = oz2_rcp[i];
    const __m512i vpi = _mm512_set1_epi32((int)pi);

    { /* Load residues for prime i across batch elements (strided gather) */
      unsigned int tmp[OZ2_BATCH];
      for (bi = 0; bi < bsz; ++bi) tmp[bi] = batch_res[bi][i];
      for (bi = bsz; bi < OZ2_BATCH; ++bi) tmp[bi] = 0;
      u_vec = _mm512_loadu_si512((__m512i*)tmp);
    }

    for (j = 0; j < i; ++j) {
      const unsigned int inv_ji = garner_inv[j][i];
      const __m512i vinv = _mm512_set1_epi32((int)inv_ji);

      /* Load vt[j] — contiguous */
      __m512i vj_vec = _mm512_loadu_si512((__m512i*)vt[j]);

      /* Bounded subtract: reduce vt[j][bi] mod pi.
       * Digits < max(moduli) = 128, min(moduli) = 61: two subtracts suffice. */
      {
        __mmask16 ge = _mm512_cmpge_epu32_mask(vj_vec, vpi);
        vj_vec = _mm512_mask_sub_epi32(vj_vec, ge, vj_vec, vpi);
        ge = _mm512_cmpge_epu32_mask(vj_vec, vpi);
        vj_vec = _mm512_mask_sub_epi32(vj_vec, ge, vj_vec, vpi);
      }

      { /* diff = (u >= vj) ? (u - vj) : (pi + u - vj) */
        const __mmask16 ge = _mm512_cmpge_epu32_mask(u_vec, vj_vec);
        const __m512i d_pos = _mm512_sub_epi32(u_vec, vj_vec);
        const __m512i d_neg = _mm512_sub_epi32(_mm512_add_epi32(vpi, u_vec), vj_vec);
        __m512i diff_vec = _mm512_mask_blend_epi32(ge, d_neg, d_pos);

        /* u = (diff * inv_ji) mod pi via vectorized Barrett */
        diff_vec = _mm512_mullo_epi32(diff_vec, vinv);
        u_vec = libxs_mod_u32x16(diff_vec, pi, rcp_i);
      }
    }

    /* Store u into vt[i] */
    _mm512_storeu_si512((__m512i*)vt[i], u_vec);
  }

  { /* Sign detection, complement, Horner — per element (transpose back) */
    unsigned int v_scalar[OZ2_BATCH][OZ2_NPRIMES_MAX];
    unsigned int tmp[OZ2_BATCH];
    for (i = 0; i < nprimes; ++i) {
      _mm512_storeu_si512((__m512i*)tmp, _mm512_loadu_si512((__m512i*)vt[i]));
      for (bi = 0; bi < bsz; ++bi) v_scalar[bi][i] = tmp[bi];
    }
    for (bi = 0; bi < bsz; ++bi) {
      const int is_negative = (v_scalar[bi][nprimes - 1] >= (unsigned int)(oz2_moduli[nprimes - 1] + 1) / 2) ? 1 : 0;
      double r;
      if (0 != is_negative) {
        for (i = 0; i < nprimes; ++i) {
          v_scalar[bi][i] = oz2_moduli[i] - 1 - v_scalar[bi][i];
        }
      }
      r = oz2_horner_grouped(v_scalar[bi], nprimes);
      result[bi] = (0 != is_negative) ? -(r + 1.0) : r;
    }
  }
}

#endif /* LIBXS_INTRINSICS_AVX512 && OZ2_BATCH == 16 */


/**
 * Preprocess rows of A for one (ib, kb) tile.
 * Decomposes each element, finds per-row max exponent, aligns mantissas
 * by right-shifting, reduces mod each modulus, and writes directly into
 * the k-contiguous layout ak[M][P][K] used by dot products.
 * Signs are folded into int8 residues (-p..+p).
 * This avoids a separate transpose pass over an intermediate buffer.
 */
LIBXS_API_INLINE void oz2_preprocess_rows(const GEMM_REAL_TYPE* a, GEMM_INT_TYPE lda, int ta, GEMM_INT_TYPE M, GEMM_INT_TYPE K,
  GEMM_INT_TYPE ib, GEMM_INT_TYPE kb, GEMM_INT_TYPE iblk, GEMM_INT_TYPE kblk, int nprimes, int16_t expa_row[BLOCK_M],
  oz2_res_t ak[BLOCK_M][OZ2_NPRIMES_MAX][BLOCK_K])
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
      const GEMM_REAL_TYPE aval = ((row < M && p < K) ? a[LIBXS_INDEX(ta, lda, row, p)] : (GEMM_REAL_TYPE)0);
      local_sign[kk] = (int8_t)ozaki_extract_ieee(aval, &elem_exp[mi][kk], &elem_mant[mi][kk]);
      row_max_exp = LIBXS_MAX(row_max_exp, elem_exp[mi][kk]);
    }

    expa_row[mi] = row_max_exp;

    /* Pass 2: align, reduce mod moduli, scatter into ak[mi][pidx][kk].
     * u8: additive inverse (p - r) for negative elements.
     * i8: sign negation (-r) for negative elements. */
    for (kk = 0; kk < kblk; ++kk) {
      const int delta = (int)row_max_exp - (int)elem_exp[mi][kk];
      uint8_t tmp[OZ2_NPRIMES_MAX];
      oz2_reduce(elem_mant[mi][kk], delta, tmp, nprimes);
      LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
      for (pidx = 0; pidx < nprimes; ++pidx) {
#if defined(OZAKI_I8) && (OZAKI_I8)
        ak[mi][pidx][kk] = (int8_t)(local_sign[kk] * (int8_t)tmp[pidx]);
#else
        ak[mi][pidx][kk] = (uint8_t)((local_sign[kk] < 0 && 0 != tmp[pidx]) ? (oz2_moduli[pidx] - tmp[pidx]) : tmp[pidx]);
#endif
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
 * Preprocess columns of B for one (kb, jb) tile.
 * Same as rows of A but with per-column max exponent. Writes directly
 * into bk[N][P][K] (k-contiguous layout).
 * u8: additive inverse for sign.  i8: negation for sign.
 */
LIBXS_API_INLINE void oz2_preprocess_cols(const GEMM_REAL_TYPE* b, GEMM_INT_TYPE ldb, int tb, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  GEMM_INT_TYPE jb, GEMM_INT_TYPE kb, GEMM_INT_TYPE jblk, GEMM_INT_TYPE kblk, int nprimes, int16_t expb_col[BLOCK_N],
  oz2_res_t bk[BLOCK_N][OZ2_NPRIMES_MAX][BLOCK_K])
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
      const GEMM_REAL_TYPE bval = ((p < K && col < N) ? b[LIBXS_INDEX(tb, ldb, p, col)] : (GEMM_REAL_TYPE)0);
      elem_sign[kk][nj] = (int8_t)ozaki_extract_ieee(bval, &elem_exp[kk][nj], &elem_mant[kk][nj]);
      expb_col[nj] = LIBXS_MAX(expb_col[nj], elem_exp[kk][nj]);
    }
  }

  /* Pass 2: align, reduce mod moduli, scatter into bk[nj][pidx][kk] */
  for (kk = 0; kk < kblk; ++kk) {
    for (nj = 0; nj < jblk; ++nj) {
      const int delta = (int)expb_col[nj] - (int)elem_exp[kk][nj];
      uint8_t tmp[OZ2_NPRIMES_MAX];
      oz2_reduce(elem_mant[kk][nj], delta, tmp, nprimes);
      LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
      for (pidx = 0; pidx < nprimes; ++pidx) {
#if defined(OZAKI_I8) && (OZAKI_I8)
        bk[nj][pidx][kk] = (int8_t)(elem_sign[kk][nj] * (int8_t)tmp[pidx]);
#else
        bk[nj][pidx][kk] = (uint8_t)((elem_sign[kk][nj] < 0 && 0 != tmp[pidx]) ? (oz2_moduli[pidx] - tmp[pidx]) : tmp[pidx]);
#endif
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
 * Evaluate mixed-radix digits v[0..nprimes-1] via grouped uint64 Horner.
 * Partitions digits into groups of OZ2_HORNER_GROUP, evaluates each group
 * exactly in uint64 (product of OZ2_HORNER_GROUP moduli < 2^63), then combines groups
 * with Horner in FP64: ceil(nprimes/OZ2_HORNER_GROUP)-1 FP64 mul-adds.
 */
LIBXS_API_INLINE double oz2_horner_grouped(const unsigned int v[], int nprimes)
{
  const int ngroups = (nprimes + OZ2_HORNER_GROUP - 1) / OZ2_HORNER_GROUP;
  double result;
  int g;

  { /* MSB group: indices [(ngroups-1)*OZ2_HORNER_GROUP .. nprimes-1] */
    const int lo = (ngroups - 1) * OZ2_HORNER_GROUP;
    uint64_t r = (uint64_t)v[nprimes - 1];
    int i;
    for (i = nprimes - 2; i >= lo; --i) {
      r = r * (uint64_t)oz2_moduli[i] + (uint64_t)v[i];
    }
    result = (double)r;
  }

  /* Combine remaining groups MSB to LSB */
  for (g = ngroups - 2; g >= 0; --g) {
    const int lo = g * OZ2_HORNER_GROUP;
    const int hi = lo + OZ2_HORNER_GROUP - 1;
    uint64_t gval, gprod = 1;
    int i;
    for (i = lo; i <= hi; ++i) gprod *= (uint64_t)oz2_moduli[i];
    gval = (uint64_t)v[hi];
    for (i = hi - 1; i >= lo; --i) {
      gval = gval * (uint64_t)oz2_moduli[i] + (uint64_t)v[i];
    }
    result = result * (double)gprod + (double)gval;
  }

  return result;
}


/**
 * Reconstruct a signed integer from its CRT residues.
 *
 * Uses Garner's algorithm to compute mixed-radix digits, detects the
 * sign from the most significant digit (centered representation), and
 * evaluates via Horner's method into a double.
 *
 * For negative values the digits are complemented before Horner
 * evaluation so that the absolute value is computed directly, avoiding
 * catastrophic cancellation from subtracting the large modulus product P.
 *
 * Centered range: result in (-P/2, P/2], where P = prod(m_i).
 */
LIBXS_API_INLINE double oz2_reconstruct(
  const unsigned int residues[OZ2_NPRIMES_MAX], unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX], int nprimes)
{
  unsigned int v[OZ2_NPRIMES_MAX]; /* mixed-radix digits */
  double result;
  int i, j, is_negative;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_NPRIMES_MAX);

  /* Garner's algorithm: compute mixed-radix digits v[i] in [0, m_i) */
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    unsigned int u = residues[i];
    const unsigned int pi = oz2_moduli[i];
    for (j = 0; j < i; ++j) {
      /* v[j] < m_j <= 128; reduce into [0, m_i) via bounded subtract
       * (avoids Barrett multiply; floor(127/61) = 2 so two subtracts suffice). */
      unsigned int vj = v[j];
      if (vj >= pi) vj -= pi;
      if (vj >= pi) vj -= pi;
      {
        const unsigned int diff = (u >= vj) ? (u - vj) : (pi + u - vj);
        u = oz2_mod(diff * garner_inv[j][i], i);
      }
    }
    v[i] = u;
  }

  /* Sign detection */
  is_negative = (v[nprimes - 1] >= (unsigned int)(oz2_moduli[nprimes - 1] + 1) / 2) ? 1 : 0;

  /* Complement digits for negative values: P - 1 - V has digits
   * (m_i - 1 - v_i) in mixed-radix representation (no borrows needed).
   * Then |D| = (P - 1 - V) + 1 = P - V where V is the CRT result. */
  if (0 != is_negative) {
    LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
    for (i = 0; i < nprimes; ++i) {
      v[i] = oz2_moduli[i] - 1 - v[i];
    }
  }

  /* Grouped Horner: uint64 within groups, FP64 only between groups */
  result = oz2_horner_grouped(v, nprimes);

  /* For negative values: result holds |D|-1 (the complement),
   * so the true value is -(result + 1.0). */
  if (0 != is_negative) {
    result = -(result + 1.0);
  }

  return result;
}


/**
 * Reconstruct a single element's unsigned mantissa from its CRT residues.
 * Uses Garner+Horner in uint64 arithmetic (exact for mantissa <= 2^53).
 * Used for diff tracking of A/B matrices.
 */
LIBXS_API_INLINE uint64_t oz2_reconstruct_mantissa(
  const uint8_t residues[OZ2_NPRIMES_MAX], unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX], int nprimes)
{
  unsigned int v[OZ2_NPRIMES_MAX];
  uint64_t result;
  int i, j;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_NPRIMES_MAX);

  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    unsigned int u = (unsigned int)residues[i];
    const unsigned int pi = oz2_moduli[i];
    for (j = 0; j < i; ++j) {
      /* Bounded subtract: v[j] < m_j <= 128, two subtracts suffice. */
      unsigned int vj = v[j];
      if (vj >= pi) vj -= pi;
      if (vj >= pi) vj -= pi;
      {
        const unsigned int diff = (u >= vj) ? (u - vj) : (pi + u - vj);
        u = oz2_mod(diff * garner_inv[j][i], i);
      }
    }
    v[i] = u;
  }

  /* Horner in uint64 (exact for mantissa values up to 2^53) */
  result = (uint64_t)v[nprimes - 1];
  LIBXS_PRAGMA_LOOP_COUNT(0, OZ2_NPRIMES_MAX - 1, OZ2_NPRIMES_DEFAULT - 1)
  for (i = nprimes - 2; i >= 0; --i) {
    result = result * (uint64_t)oz2_moduli[i] + (uint64_t)v[i];
  }
  return result;
}


/**
 * Batched CRT reconstruction: process OZ2_BATCH elements in parallel
 * through Garner's algorithm. The innermost loop over batch elements
 * is data-parallel, enabling auto-vectorization.
 */
LIBXS_API_INLINE void oz2_reconstruct_batch(unsigned int batch_res[OZ2_BATCH][OZ2_NPRIMES_MAX],
  unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX], int nprimes, int bsz, double result[OZ2_BATCH])
{
  unsigned int v[OZ2_BATCH][OZ2_NPRIMES_MAX];
  unsigned int u[OZ2_BATCH];
  int i, j, bi;
  nprimes = LIBXS_CLMP(nprimes, 1, OZ2_NPRIMES_MAX);

  /* Garner's algorithm: vectorized across batch elements */
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
  for (i = 0; i < nprimes; ++i) {
    const unsigned int pi = oz2_moduli[i];
    for (bi = 0; bi < bsz; ++bi) u[bi] = batch_res[bi][i];

    for (j = 0; j < i; ++j) {
      const unsigned int inv_ji = garner_inv[j][i];
      for (bi = 0; bi < bsz; ++bi) {
        /* Bounded subtract: v[bi][j] < m_j <= 128, two subtracts suffice. */
        unsigned int vj = v[bi][j];
        if (vj >= pi) vj -= pi;
        if (vj >= pi) vj -= pi;
        {
          const unsigned int diff = (u[bi] >= vj) ? (u[bi] - vj) : (pi + u[bi] - vj);
          u[bi] = oz2_mod(diff * inv_ji, i);
        }
      }
    }

    for (bi = 0; bi < bsz; ++bi) v[bi][i] = u[bi];
  }

  /* Sign detection, complement, Horner — per element */
  for (bi = 0; bi < bsz; ++bi) {
    const int is_negative = (v[bi][nprimes - 1] >= (unsigned int)(oz2_moduli[nprimes - 1] + 1) / 2) ? 1 : 0;
    double r;
    if (0 != is_negative) {
      for (i = 0; i < nprimes; ++i) {
        v[bi][i] = oz2_moduli[i] - 1 - v[bi][i];
      }
    }
    r = oz2_horner_grouped(v[bi], nprimes);
    result[bi] = (0 != is_negative) ? -(r + 1.0) : r;
  }
}


LIBXS_API_INLINE void gemm_oz2_diff(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc, unsigned int diff_stat,
  libxs_matdiff_t* diff)
{
  unsigned int garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX];
  /* Max K per int32 accumulation pass: K_CHUNK * max_residue^2 < 2^31.
   * u8 (max 255): 255^2 * 32768 ~ 2.13e9 < 2^31. K_CHUNK = 32768.
   * i8 (max 127): 127^2 * 131072 ~ 2.11e9 < 2^31. K_CHUNK = 131072. */
#if defined(OZAKI_I8) && (OZAKI_I8)
  enum { K_CHUNK = 131072 };
#else
  enum { K_CHUNK = 32768 };
#endif
  const int ta = (*transa != 'N' && *transa != 'n');
  const int tb = (*transb != 'N' && *transb != 'n');
  const GEMM_INT_TYPE M = *m, N = *n, K = *k;
  const GEMM_INT_TYPE ldcv = *ldc;
  const int nprimes = LIBXS_CLMP(ozaki_n, 1, OZ2_NPRIMES_MAX);
  const GEMM_INT_TYPE K_grp_size = (0 < ozaki_maxk ? (GEMM_INT_TYPE)ozaki_maxk : K);
  const GEMM_INT_TYPE K_grp_max = LIBXS_MIN(K_grp_size, K);
  const GEMM_INT_TYPE K_grp_pad = ((K_grp_max + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
  const GEMM_INT_TYPE nblk_m = (M + BLOCK_M - 1) / BLOCK_M;
  const GEMM_INT_TYPE nblk_n = (N + BLOCK_N - 1) / BLOCK_N;
  oz2_res_t* a_res = NULL;
  oz2_res_t* b_res = NULL;
  int16_t* expa_raw = NULL;
  int16_t* expb_raw = NULL;
  double* expa_fp = NULL;
  double* expb_fp = NULL;
  GEMM_REAL_TYPE* ref_panel = NULL;
  libxs_matdiff_t tdiff[256];
  ozaki_rsq_acc_t trsq[256];
  GEMM_PROFILE_DECL;
  int nthreads = 1;
  int i, j;
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE) || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));

  /* Precompute Garner modular inverse table */
  memset(garner_inv, 0, sizeof(garner_inv));
  for (i = 0; i < nprimes; ++i) {
    for (j = i + 1; j < nprimes; ++j) {
      garner_inv[i][j] = libxs_mod_inverse_u32(oz2_moduli[i] % oz2_moduli[j], oz2_moduli[j]);
    }
  }

  a_res = (oz2_res_t*)libxs_malloc(gemm_pool, (size_t)nprimes * M * K_grp_pad, 0);
  b_res = (oz2_res_t*)libxs_malloc(gemm_pool, (size_t)nprimes * N * K_grp_pad, 0);
  expa_raw = (int16_t*)libxs_malloc(gemm_pool, (size_t)M * sizeof(int16_t), 0);
  expb_raw = (int16_t*)libxs_malloc(gemm_pool, (size_t)N * sizeof(int16_t), 0);
  expa_fp = (double*)libxs_malloc(gemm_pool, (size_t)M * sizeof(double), 0);
  expb_fp = (double*)libxs_malloc(gemm_pool, (size_t)N * sizeof(double), 0);
  if (NULL != diff && 0 == (diff_stat % 3)) {
    ref_panel = (GEMM_REAL_TYPE*)libxs_malloc(gemm_pool, (size_t)nblk_m * nblk_n * BLOCK_M * BLOCK_N * sizeof(*ref_panel), 0);
  }

#if defined(_OPENMP)
# pragma omp parallel
#endif
  {
    GEMM_INT_TYPE row, col, ib, jb, mi, nj, kb, kb_grp;
    int pidx, tid;
    tid = 0;
#if defined(_OPENMP)
    tid = omp_get_thread_num();
# pragma omp single
    nthreads = omp_get_num_threads();
#endif
    if (NULL != diff) {
      libxs_matdiff_clear(&tdiff[tid]);
      trsq[tid].ss_res = trsq[tid].ss_tot = 0;
    }
    GEMM_PROFILE_TICK(t_start, tid);

    /* Phase 3: scale C by beta (once, before K-group loop) */
#if defined(_OPENMP)
# pragma omp for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
#endif
    for (jb = 0; jb < N; jb += BLOCK_N) {
      for (ib = 0; ib < M; ib += BLOCK_M) {
        const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
        const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
        GEMM_REAL_TYPE* const cb = c + jb * ldcv + ib;
        if (NULL != ref_panel) {
          const GEMM_INT_TYPE ibi = ib / BLOCK_M;
          const GEMM_INT_TYPE jbi = jb / BLOCK_N;
          ozaki_scale_block_beta(cb, ldcv, iblk, jblk, beta, ref_panel + (jbi * nblk_m + ibi) * BLOCK_M * BLOCK_N, 1);
        }
        else {
          ozaki_scale_block_beta(cb, ldcv, iblk, jblk, beta, NULL, 0);
        }
      }
    }

    GEMM_PROFILE_TICK(t_preprocess, tid);
    GEMM_PROFILE_TICK(t_kernel, tid);

    /* K-group loop: process K in chunks of ozaki_maxk */
    for (kb_grp = 0; kb_grp < K; kb_grp += K_grp_size) {
      const GEMM_INT_TYPE K_len = LIBXS_MIN(K_grp_size, K - kb_grp);

      /* Phase 1: preprocess rows of A for this K-group */
#if defined(_OPENMP)
# pragma omp for schedule(static) nowait
#endif
      for (row = 0; row < M; ++row) {
        int16_t row_max_exp = 0;
        GEMM_INT_TYPE kk;
        /* Zero this row's residue buffers */
        for (pidx = 0; pidx < nprimes; ++pidx) {
          memset(a_res + (long)pidx * M * K_grp_pad + (long)row * K_grp_pad, 0, (size_t)K_grp_pad);
        }
        for (kk = kb_grp; kk < kb_grp + K_len; ++kk) {
          int16_t e;
          uint64_t mt;
          ozaki_extract_ieee(a[LIBXS_INDEX(ta, *lda, row, kk)], &e, &mt);
          if (e > row_max_exp) row_max_exp = e;
        }
        expa_raw[row] = row_max_exp;
        for (kk = kb_grp; kk < kb_grp + K_len; ++kk) {
          int16_t e;
          uint64_t mt;
          int sign;
          sign = ozaki_extract_ieee(a[LIBXS_INDEX(ta, *lda, row, kk)], &e, &mt);
          if (0 != mt) {
            const int delta = (int)row_max_exp - (int)e;
            uint8_t tmp[OZ2_NPRIMES_MAX];
            oz2_reduce(mt, delta, tmp, nprimes);
            LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
            for (pidx = 0; pidx < nprimes; ++pidx) {
#if defined(OZAKI_I8) && (OZAKI_I8)
              a_res[(long)pidx * M * K_grp_pad + (long)row * K_grp_pad + (kk - kb_grp)] = (oz2_res_t)(sign * (int8_t)tmp[pidx]);
#else
              a_res[(long)pidx * M * K_grp_pad + (long)row * K_grp_pad + (kk - kb_grp)] =
                (oz2_res_t)((sign < 0 && 0 != tmp[pidx]) ? (oz2_moduli[pidx] - tmp[pidx]) : tmp[pidx]);
#endif
            }
          }
        }
      }

      /* Phase 2: preprocess columns of B for this K-group */
#if defined(_OPENMP)
# pragma omp for schedule(static)
#endif
      for (col = 0; col < N; ++col) {
        int16_t col_max_exp = 0;
        GEMM_INT_TYPE kk;
        /* Zero this column's residue buffers */
        for (pidx = 0; pidx < nprimes; ++pidx) {
          memset(b_res + (long)pidx * N * K_grp_pad + (long)col * K_grp_pad, 0, (size_t)K_grp_pad);
        }
        for (kk = kb_grp; kk < kb_grp + K_len; ++kk) {
          int16_t e;
          uint64_t mt;
          ozaki_extract_ieee(b[LIBXS_INDEX(tb, *ldb, kk, col)], &e, &mt);
          if (e > col_max_exp) col_max_exp = e;
        }
        expb_raw[col] = col_max_exp;
        for (kk = kb_grp; kk < kb_grp + K_len; ++kk) {
          int16_t e;
          uint64_t mt;
          int sign;
          sign = ozaki_extract_ieee(b[LIBXS_INDEX(tb, *ldb, kk, col)], &e, &mt);
          if (0 != mt) {
            const int delta = (int)col_max_exp - (int)e;
            uint8_t tmp[OZ2_NPRIMES_MAX];
            oz2_reduce(mt, delta, tmp, nprimes);
            LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
            for (pidx = 0; pidx < nprimes; ++pidx) {
#if defined(OZAKI_I8) && (OZAKI_I8)
              b_res[(long)pidx * N * K_grp_pad + (long)col * K_grp_pad + (kk - kb_grp)] = (oz2_res_t)(sign * (int8_t)tmp[pidx]);
#else
              b_res[(long)pidx * N * K_grp_pad + (long)col * K_grp_pad + (kk - kb_grp)] =
                (oz2_res_t)((sign < 0 && 0 != tmp[pidx]) ? (oz2_moduli[pidx] - tmp[pidx]) : tmp[pidx]);
#endif
            }
          }
        }
      }
      /* implicit barrier: preprocessing done */

      /* Phase 2b: compute FP exponent scale factors */
#if defined(_OPENMP)
# pragma omp for schedule(static) nowait
#endif
      for (row = 0; row < M; ++row) {
        expa_fp[row] = libxs_pow2((int)expa_raw[row] - OZ_BIAS_PLUS_MANT);
      }
#if defined(_OPENMP)
# pragma omp for schedule(static)
#endif
      for (col = 0; col < N; ++col) {
        expb_fp[col] = libxs_pow2((int)expb_raw[col] - OZ_BIAS_PLUS_MANT);
      }

      /* Phase 2c: diff tracking for A decomposition (mode 1) */
      if (NULL != diff && 1 == (diff_stat % 3)) {
#if defined(_OPENMP)
# pragma omp for schedule(static)
#endif
        for (row = 0; row < M; ++row) {
          GEMM_REAL_TYPE ref_blk[BLOCK_K];
          GEMM_REAL_TYPE recon_blk[BLOCK_K];
          GEMM_INT_TYPE kk;
          const int sh = (int)expa_raw[row] - OZ_BIAS_PLUS_MANT;
          for (kk = kb_grp; kk < kb_grp + K_len; ++kk) {
            const GEMM_INT_TYPE kk_local = kk - kb_grp;
            uint8_t tmp[OZ2_NPRIMES_MAX];
            uint64_t mant_recon;
            int8_t ds;
            double arecon;
            for (pidx = 0; pidx < nprimes; ++pidx) {
              const oz2_res_t rv = a_res[(long)pidx * M * K_grp_pad + (long)row * K_grp_pad + kk_local];
              tmp[pidx] = (uint8_t)((int8_t)rv < 0 ? -rv : rv);
            }
            mant_recon = oz2_reconstruct_mantissa(tmp, garner_inv, nprimes);
            ds = 1;
            for (pidx = 0; pidx < nprimes; ++pidx) {
              const oz2_res_t rv = a_res[(long)pidx * M * K_grp_pad + (long)row * K_grp_pad + kk_local];
              if (0 != rv) {
                ds = ((int8_t)rv < 0) ? -1 : 1;
                break;
              }
            }
            arecon = (double)ds * (double)mant_recon * libxs_pow2(sh);
            ref_blk[kk_local % BLOCK_K] = a[LIBXS_INDEX(ta, *lda, row, kk)];
            recon_blk[kk_local % BLOCK_K] = (GEMM_REAL_TYPE)arecon;
            if (BLOCK_K - 1 == (kk_local % BLOCK_K) || kk == kb_grp + K_len - 1) {
              const GEMM_INT_TYPE bsize = (kk_local % BLOCK_K) + 1;
              libxs_matdiff_t bd;
              const int ild = 1, itd = 1;
              if (EXIT_SUCCESS == libxs_matdiff(&bd, LIBXS_DATATYPE(GEMM_REAL_TYPE), bsize, 1, ref_blk, recon_blk, &ild, &itd)) {
                libxs_matdiff_reduce(&tdiff[tid], &bd);
              }
            }
          }
        }
      }

      /* Phase 2d: diff tracking for B decomposition (mode 2) */
      if (NULL != diff && 2 == (diff_stat % 3)) {
#if defined(_OPENMP)
# pragma omp for schedule(static)
#endif
        for (col = 0; col < N; ++col) {
          GEMM_REAL_TYPE ref_blk[BLOCK_K];
          GEMM_REAL_TYPE recon_blk[BLOCK_K];
          GEMM_INT_TYPE kk;
          const int sh = (int)expb_raw[col] - OZ_BIAS_PLUS_MANT;
          for (kk = kb_grp; kk < kb_grp + K_len; ++kk) {
            const GEMM_INT_TYPE kk_local = kk - kb_grp;
            uint8_t tmp[OZ2_NPRIMES_MAX];
            uint64_t mant_recon;
            int8_t ds;
            double brecon;
            for (pidx = 0; pidx < nprimes; ++pidx) {
              const oz2_res_t rv = b_res[(long)pidx * N * K_grp_pad + (long)col * K_grp_pad + kk_local];
              tmp[pidx] = (uint8_t)((int8_t)rv < 0 ? -rv : rv);
            }
            mant_recon = oz2_reconstruct_mantissa(tmp, garner_inv, nprimes);
            ds = 1;
            for (pidx = 0; pidx < nprimes; ++pidx) {
              const oz2_res_t rv = b_res[(long)pidx * N * K_grp_pad + (long)col * K_grp_pad + kk_local];
              if (0 != rv) {
                ds = ((int8_t)rv < 0) ? -1 : 1;
                break;
              }
            }
            brecon = (double)ds * (double)mant_recon * libxs_pow2(sh);
            ref_blk[kk_local % BLOCK_K] = b[LIBXS_INDEX(tb, *ldb, kk, col)];
            recon_blk[kk_local % BLOCK_K] = (GEMM_REAL_TYPE)brecon;
            if (BLOCK_K - 1 == (kk_local % BLOCK_K) || kk == kb_grp + K_len - 1) {
              const GEMM_INT_TYPE bsize = (kk_local % BLOCK_K) + 1;
              libxs_matdiff_t bd;
              const int ild = 1, itd = 1;
              if (EXIT_SUCCESS == libxs_matdiff(&bd, LIBXS_DATATYPE(GEMM_REAL_TYPE), bsize, 1, ref_blk, recon_blk, &ild, &itd)) {
                libxs_matdiff_reduce(&tdiff[tid], &bd);
              }
            }
          }
        }
      }

      /* Phase 4: CRT dot products + accumulate for this K-group.
       * K_CHUNK loop retained for int32 safety when K_GRP > K_CHUNK. */
#if defined(_OPENMP)
# pragma omp for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
#endif
      for (jb = 0; jb < N; jb += BLOCK_N) {
        for (ib = 0; ib < M; ib += BLOCK_M) {
          const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
          const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
          GEMM_REAL_TYPE* const cb = c + jb * ldcv + ib;
          unsigned int tile_res[BLOCK_M * BLOCK_N][OZ2_NPRIMES_MAX];
          memset(tile_res, 0, sizeof(tile_res));

          /* Accumulate per-prime residues via GEMM + mod-reduce */
          for (kb = 0; kb < K_grp_pad; kb += K_CHUNK) {
            const GEMM_INT_TYPE chunk_k = ((GEMM_INT_TYPE)K_CHUNK < K_grp_pad - kb) ? (GEMM_INT_TYPE)K_CHUNK : (K_grp_pad - kb);
            LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
            for (pidx = 0; pidx < nprimes; ++pidx) {
              int32_t partial[BLOCK_M * BLOCK_N];
#if defined(OZAKI_I8) && (OZAKI_I8)
              ozaki_gemm_s8s8s32('N', 'T', iblk, jblk, chunk_k,
                (const int8_t*)(a_res + (long)pidx * M * K_grp_pad + (long)ib * K_grp_pad + kb), K_grp_pad,
                (const int8_t*)(b_res + (long)pidx * N * K_grp_pad + (long)jb * K_grp_pad + kb), K_grp_pad, 0, partial, jblk);
#else
              ozaki_gemm_u8u8s32('N', 'T', iblk, jblk, chunk_k,
                (const uint8_t*)(a_res + (long)pidx * M * K_grp_pad + (long)ib * K_grp_pad + kb), K_grp_pad,
                (const uint8_t*)(b_res + (long)pidx * N * K_grp_pad + (long)jb * K_grp_pad + kb), K_grp_pad, 0, partial, jblk);
#endif
              for (mi = 0; mi < iblk; ++mi) {
                for (nj = 0; nj < jblk; ++nj) {
                  const int32_t dot = partial[mi * jblk + nj];
                  unsigned int r;
#if defined(OZAKI_I8) && (OZAKI_I8)
                  if (dot >= 0) {
                    r = oz2_mod((uint32_t)dot, pidx);
                  }
                  else {
                    r = oz2_mod((uint32_t)(-dot), pidx);
                    r = (0 != r) ? (oz2_moduli[pidx] - r) : 0;
                  }
#else
                  r = oz2_mod((uint32_t)dot, pidx);
#endif
                  r += tile_res[mi * jblk + nj][pidx];
                  if (r >= oz2_moduli[pidx]) r -= oz2_moduli[pidx];
                  tile_res[mi * jblk + nj][pidx] = r;
                }
              }
            }
          }

          /* CRT reconstruct, scale, and accumulate to C */
          for (mi = 0; mi < iblk; ++mi) {
            for (nj = 0; nj < jblk; nj += OZ2_BATCH) {
              const GEMM_INT_TYPE bsz = LIBXS_MIN(OZ2_BATCH, (int)(jblk - nj));
              unsigned int batch_res[OZ2_BATCH][OZ2_NPRIMES_MAX];
              double batch_val[OZ2_BATCH];
              int bi;
              for (bi = 0; bi < (int)bsz; ++bi) {
                LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
                for (pidx = 0; pidx < nprimes; ++pidx) {
                  batch_res[bi][pidx] = tile_res[mi * jblk + nj + bi][pidx];
                }
              }

#if defined(LIBXS_INTRINSICS_AVX512) && 16 == OZ2_BATCH
# if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
              oz2_reconstruct_batch_avx512(batch_res, garner_inv, nprimes, (int)bsz, batch_val);
# else
              if (LIBXS_X86_AVX512 <= ozaki_target_arch) {
                oz2_reconstruct_batch_avx512(batch_res, garner_inv, nprimes, (int)bsz, batch_val);
              }
              else {
                oz2_reconstruct_batch(batch_res, garner_inv, nprimes, (int)bsz, batch_val);
              }
# endif
#else
              oz2_reconstruct_batch(batch_res, garner_inv, nprimes, (int)bsz, batch_val);
#endif

              for (bi = 0; bi < (int)bsz; ++bi) {
                if (0.0 != batch_val[bi] && (GEMM_REAL_TYPE)0 != *alpha) {
                  const GEMM_INT_TYPE jcol = nj + bi;
                  const double contrib = (*alpha) * batch_val[bi] * expa_fp[ib + mi] * expb_fp[jb + jcol];
                  cb[mi + jcol * ldcv] += (GEMM_REAL_TYPE)contrib;
                }
              }
            }
          }
        }
      }
#if defined(_OPENMP)
# pragma omp barrier
#endif
    } /* end K-group loop */

    GEMM_PROFILE_END(tid, M, N, K);

    /* Phase 5 (diff mode 0): reference GEMM comparison */
    if (NULL != diff && 0 == (diff_stat % 3)) {
#if defined(_OPENMP)
# pragma omp for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
#endif
      for (jb = 0; jb < N; jb += BLOCK_N) {
        for (ib = 0; ib < M; ib += BLOCK_M) {
          const GEMM_INT_TYPE ibi = ib / BLOCK_M;
          const GEMM_INT_TYPE jbi = jb / BLOCK_N;
          const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
          const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
          GEMM_REAL_TYPE* const cb = c + jb * ldcv + ib;
          GEMM_REAL_TYPE* const ref_blk = ref_panel + (jbi * nblk_m + ibi) * BLOCK_M * BLOCK_N;
          const GEMM_INT_TYPE mref = BLOCK_M;
          if (NULL != gemm_original) {
            gemm_original(transa, transb, &iblk, &jblk, &K, alpha, a + LIBXS_INDEX(ta, *lda, ib, 0), lda,
              b + LIBXS_INDEX(tb, *ldb, 0, jb), ldb, beta, ref_blk, &mref);
          }
          else {
            GEMM_REAL(transa, transb, &iblk, &jblk, &K, alpha, a + LIBXS_INDEX(ta, *lda, ib, 0), lda,
              b + LIBXS_INDEX(tb, *ldb, 0, jb), ldb, beta, ref_blk, &mref);
          }
          ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, cb, iblk, jblk, mref, ldcv, &trsq[tid]);
        }
      }
    }
  } /* end parallel */

  if (NULL != diff) {
    double total_ss_res = 0, total_ss_tot = 0;
    for (i = 0; i < nthreads; ++i) {
      libxs_matdiff_reduce(diff, &tdiff[i]);
      total_ss_res += trsq[i].ss_res;
      total_ss_tot += trsq[i].ss_tot;
    }
    /* global rsq from accumulated SS_res/SS_tot (per-block min-reduction is meaningless
     * when individual blocks have near-zero variance) */
    if (0 < total_ss_tot) {
      diff->rsq = LIBXS_MAX(0.0, 1.0 - total_ss_res / total_ss_tot);
    }
    if (NULL != ref_panel && ozaki_diff_exceeds(diff)) {
      ozaki_repair_from_ref_panel(c, ref_panel, M, N, ldcv, nblk_m);
    }
  }
  libxs_free(a_res);
  libxs_free(b_res);
  libxs_free(expa_raw);
  libxs_free(expb_raw);
  libxs_free(expa_fp);
  libxs_free(expb_fp);
  libxs_free(ref_panel);
}


LIBXS_API void gemm_oz2(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  OZAKI_GEMM_WRAPPER(gemm_oz2_diff)
}


#if defined(__LIBXSTREAM)
/**
 * Host preprocessing wrapper for A (rows) — scheme 2 (CRT int8).
 * Handles K-grouping: the per-row max exponent spans KGROUP panels.
 * Layout: slices[panel][BM][NPRIMES][BK], exp[group][BM].
 *   panel = ki * nblk + ib_idx,
 *   group = ki_group * nblk + ib_idx.
 */
void oz2_host_preprocess_a(const void* matrix, int ld, int trans, int dim, int K, int kb_batch, int nkb, int nblk, int brc, int bk,
  int nslices_p, int kgroup_p, int use_xmx_p, void* slices, void* exp)
{
  const GEMM_REAL_TYPE* a = (const GEMM_REAL_TYPE*)matrix;
  const int nkb_groups = (nkb + kgroup_p - 1) / kgroup_p;
  int ki_group, ib_idx;
  (void)use_xmx_p;
  LIBXS_ASSERT(brc == BLOCK_M && bk == BLOCK_K);
# if defined(_OPENMP)
#   pragma omp parallel for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
# endif
  for (ki_group = 0; ki_group < nkb_groups; ++ki_group) {
    for (ib_idx = 0; ib_idx < nblk; ++ib_idx) {
      const int group_idx = ki_group * nblk + ib_idx;
      const int ib = ib_idx * brc;
      const GEMM_INT_TYPE iblk = (GEMM_INT_TYPE)LIBXS_MIN(brc, dim - ib);
      int16_t group_max_exp[BLOCK_M];
      int mi, s;
      /* Phase 1: find per-row max exponent across all K-blocks in group */
      for (mi = 0; mi < BLOCK_M; ++mi) group_max_exp[mi] = 0;
      for (s = 0; s < kgroup_p; ++s) {
        const int ki = ki_group * kgroup_p + s;
        const int kb = kb_batch + ki * bk;
        if (ki >= nkb) break;
        for (mi = 0; mi < iblk; ++mi) {
          const GEMM_INT_TYPE row = (GEMM_INT_TYPE)(ib + mi);
          const GEMM_INT_TYPE kblk = (GEMM_INT_TYPE)LIBXS_MIN(bk, K - kb);
          int kk;
          for (kk = 0; kk < kblk; ++kk) {
            int16_t elem_exp;
            uint64_t elem_mant;
            const GEMM_INT_TYPE p = (GEMM_INT_TYPE)(kb + kk);
            const GEMM_REAL_TYPE aval = ((row < dim && p < K) ? a[LIBXS_INDEX(trans, (GEMM_INT_TYPE)ld, row, p)]
                                                              : (GEMM_REAL_TYPE)0);
            ozaki_extract_ieee(aval, &elem_exp, &elem_mant);
            if (elem_exp > group_max_exp[mi]) group_max_exp[mi] = elem_exp;
          }
        }
      }
      /* Write group exponent */
      memcpy((short*)exp + (long)group_idx * brc, group_max_exp, (size_t)brc * sizeof(short));
      /* Phase 2: compute residues per panel using group max exponent */
      for (s = 0; s < kgroup_p; ++s) {
        const int ki = ki_group * kgroup_p + s;
        const int panel = ki * nblk + ib_idx;
        const int kb = kb_batch + ki * bk;
        const GEMM_INT_TYPE kblk = (GEMM_INT_TYPE)LIBXS_MIN(bk, K - kb);
        if (ki >= nkb) break;
        for (mi = 0; mi < brc; ++mi) {
          const GEMM_INT_TYPE row = (GEMM_INT_TYPE)(ib + mi);
          int kk;
          for (kk = 0; kk < kblk; ++kk) {
            const GEMM_INT_TYPE p = (GEMM_INT_TYPE)(kb + kk);
            int16_t elem_exp;
            uint64_t elem_mant;
            int elem_sign;
            int pidx;
            uint8_t tmp[OZ2_NPRIMES_MAX];
            const GEMM_REAL_TYPE aval = ((row < dim && p < K) ? a[LIBXS_INDEX(trans, (GEMM_INT_TYPE)ld, row, p)]
                                                              : (GEMM_REAL_TYPE)0);
            elem_sign = ozaki_extract_ieee(aval, &elem_exp, &elem_mant);
            {
              const int delta = (int)group_max_exp[mi] - (int)elem_exp;
              oz2_reduce(elem_mant, delta, tmp, nslices_p);
            }
            for (pidx = 0; pidx < nslices_p; ++pidx) {
              ((char*)slices)[(((long)panel * brc + mi) * nslices_p + pidx) * bk + kk] = (char)OZ2_SIGN_FOLD_(
                elem_sign, tmp[pidx], pidx);
            }
          }
          /* Zero-pad remaining k-entries */
          {
            int kk2;
            for (kk2 = kblk; kk2 < bk; ++kk2) {
              int pidx2;
              for (pidx2 = 0; pidx2 < nslices_p; ++pidx2) {
                ((char*)slices)[(((long)panel * brc + mi) * nslices_p + pidx2) * bk + kk2] = 0;
              }
            }
          }
        }
      }
    }
  }
}


/**
 * Host preprocessing wrapper for B (columns) — scheme 2 (CRT int8).
 * Layout (non-XMX): slices[panel][BN][NPRIMES][BK], exp[group][BN].
 * Layout (XMX):     slices[panel][NPRIMES][BK][bn_pad], exp[group][BN].
 *   panel = ki * nblk + jb_idx.
 */
void oz2_host_preprocess_b(const void* matrix, int ld, int trans, int dim, int K, int kb_batch, int nkb, int nblk, int brc, int bk,
  int nslices_p, int kgroup_p, int use_xmx_p, void* slices, void* exp)
{
  const GEMM_REAL_TYPE* b = (const GEMM_REAL_TYPE*)matrix;
  const int bn_pad = (0 != use_xmx_p) ? LIBXS_MAX(brc, 64) : brc;
  const int nkb_groups = (nkb + kgroup_p - 1) / kgroup_p;
  int ki_group, jb_idx;
  LIBXS_ASSERT(brc == BLOCK_N && bk == BLOCK_K);
# if defined(_OPENMP)
#   pragma omp parallel for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
# endif
  for (ki_group = 0; ki_group < nkb_groups; ++ki_group) {
    for (jb_idx = 0; jb_idx < nblk; ++jb_idx) {
      const int group_idx = ki_group * nblk + jb_idx;
      const int jb = jb_idx * brc;
      const GEMM_INT_TYPE jblk = (GEMM_INT_TYPE)LIBXS_MIN(brc, dim - jb);
      int16_t group_max_exp[BLOCK_N];
      int nj, s;
      /* Phase 1: find per-column max exponent across group */
      for (nj = 0; nj < BLOCK_N; ++nj) group_max_exp[nj] = 0;
      for (s = 0; s < kgroup_p; ++s) {
        const int ki = ki_group * kgroup_p + s;
        const int kb = kb_batch + ki * bk;
        const GEMM_INT_TYPE kblk = (GEMM_INT_TYPE)LIBXS_MIN(bk, K - kb);
        int kk;
        if (ki >= nkb) break;
        for (kk = 0; kk < kblk; ++kk) {
          const GEMM_INT_TYPE p = (GEMM_INT_TYPE)(kb + kk);
          for (nj = 0; nj < jblk; ++nj) {
            int16_t elem_exp;
            uint64_t elem_mant;
            const GEMM_INT_TYPE col = (GEMM_INT_TYPE)(jb + nj);
            const GEMM_REAL_TYPE bval = ((p < K && col < dim) ? b[LIBXS_INDEX(trans, (GEMM_INT_TYPE)ld, p, col)]
                                                              : (GEMM_REAL_TYPE)0);
            ozaki_extract_ieee(bval, &elem_exp, &elem_mant);
            if (elem_exp > group_max_exp[nj]) group_max_exp[nj] = elem_exp;
          }
        }
      }
      memcpy((short*)exp + (long)group_idx * brc, group_max_exp, (size_t)brc * sizeof(short));
      /* Phase 2: compute residues per panel using group max exponent */
      for (s = 0; s < kgroup_p; ++s) {
        const int ki = ki_group * kgroup_p + s;
        const int panel = ki * nblk + jb_idx;
        const int kb = kb_batch + ki * bk;
        const GEMM_INT_TYPE kblk = (GEMM_INT_TYPE)LIBXS_MIN(bk, K - kb);
        int kk;
        if (ki >= nkb) break;
        for (kk = 0; kk < kblk; ++kk) {
          const GEMM_INT_TYPE p = (GEMM_INT_TYPE)(kb + kk);
          for (nj = 0; nj < brc; ++nj) {
            const GEMM_INT_TYPE col = (GEMM_INT_TYPE)(jb + nj);
            int16_t elem_exp;
            uint64_t elem_mant;
            int elem_sign;
            int pidx;
            uint8_t tmp[OZ2_NPRIMES_MAX];
            const GEMM_REAL_TYPE bval = ((p < K && col < dim) ? b[LIBXS_INDEX(trans, (GEMM_INT_TYPE)ld, p, col)]
                                                              : (GEMM_REAL_TYPE)0);
            elem_sign = ozaki_extract_ieee(bval, &elem_exp, &elem_mant);
            {
              const int delta = (int)group_max_exp[nj] - (int)elem_exp;
              oz2_reduce(elem_mant, delta, tmp, nslices_p);
            }
            if (0 == use_xmx_p) {
              for (pidx = 0; pidx < nslices_p; ++pidx) {
                ((char*)slices)[(((long)panel * brc + nj) * nslices_p + pidx) * bk + kk] = (char)OZ2_SIGN_FOLD_(
                  elem_sign, tmp[pidx], pidx);
              }
            }
            else {
              for (pidx = 0; pidx < nslices_p; ++pidx) {
                ((char*)slices)[(((long)panel * nslices_p + pidx) * bk + kk) * bn_pad + nj] = (char)OZ2_SIGN_FOLD_(
                  elem_sign, tmp[pidx], pidx);
              }
            }
          }
        }
        /* Zero-pad remaining k-entries */
        {
          int kk2;
          for (kk2 = kblk; kk2 < bk; ++kk2) {
            for (nj = 0; nj < brc; ++nj) {
              int pidx2;
              if (0 == use_xmx_p) {
                for (pidx2 = 0; pidx2 < nslices_p; ++pidx2) {
                  ((char*)slices)[(((long)panel * brc + nj) * nslices_p + pidx2) * bk + kk2] = 0;
                }
              }
              else {
                for (pidx2 = 0; pidx2 < nslices_p; ++pidx2) {
                  ((char*)slices)[(((long)panel * nslices_p + pidx2) * bk + kk2) * bn_pad + nj] = 0;
                }
              }
            }
          }
        }
      }
    }
  }
}
#endif /* __LIBXSTREAM */
