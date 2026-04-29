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

static const uint32_t oz2_hier_gprod[] = {245872000u, 156832361u, 89809099u, 38771541u, 17120443u};
static const uint64_t oz2_hier_l2_barrett[] = {75025802343ull, 117620776452ull, 205399500486ull, 475780523495ull, 1077468852512ull};

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

static const uint32_t oz2_hier_gprod[] = {3763024128u, 2894777321u, 1844618759u, 1194324337u, 795860377u};
static const uint64_t oz2_hier_l2_barrett[] = {4902106243ull, 6372422479ull, 10000301679ull, 15445338843ull, 23178367219ull};

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


/* Hierarchical CRT: two-level Garner reconstruction.
 * Level 1: HIER_GS primes per group (small Garner, 32-bit).
 * Level 2: Garner over HIER_NGROUPS group-moduli (32-bit, 64-bit Barrett). */
#define HIER_GS 4
#define HIER_NGROUPS_MAX ((OZ2_NPRIMES_MAX + HIER_GS - 1) / HIER_GS)
#define HIER_L2_HORNER_GROUP 2

LIBXS_API_INLINE unsigned int oz2_mod_l2(uint64_t x, int gidx)
{
  const uint32_t m = oz2_hier_gprod[gidx];
#if defined(__SIZEOF_INT128__)
  const uint64_t q = (uint64_t)(((__uint128_t)x * oz2_hier_l2_barrett[gidx]) >> 64);
#else
  const uint64_t x_lo = (uint32_t)x, x_hi = x >> 32;
  const uint64_t b_lo = (uint32_t)oz2_hier_l2_barrett[gidx], b_hi = oz2_hier_l2_barrett[gidx] >> 32;
  const uint64_t q = (x_hi * b_hi) + ((x_hi * b_lo + x_lo * b_hi + ((x_lo * b_lo) >> 32)) >> 32);
#endif
  { uint32_t r = (uint32_t)(x - q * (uint64_t)m);
    return (r >= m) ? (r - m) : r;
  }
}

LIBXS_API_INLINE unsigned int oz2_hier_l1_garner(const unsigned int group_residues[], int g,
  uint8_t garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX], int nprimes)
{
  const int lo = g * HIER_GS;
  const int hi = (lo + HIER_GS <= nprimes) ? (lo + HIER_GS) : nprimes;
  const int gsz = hi - lo;
  unsigned int v[HIER_GS];
  uint64_t hval;
  int li, lj;

  for (li = 0; li < gsz; ++li) {
    unsigned int u = group_residues[li];
    const unsigned int pi = oz2_moduli[lo + li];
    for (lj = 0; lj < li; ++lj) {
      unsigned int vj = v[lj];
      if (vj >= pi) vj -= pi;
      if (vj >= pi) vj -= pi;
      { const unsigned int diff = (u >= vj) ? (u - vj) : (pi + u - vj);
        u = oz2_mod(diff * (unsigned int)garner_inv[lo + lj][lo + li], lo + li);
      }
    }
    v[li] = u;
  }

  hval = (uint64_t)v[gsz - 1];
  for (li = gsz - 2; li >= 0; --li) {
    hval = hval * (uint64_t)oz2_moduli[lo + li] + (uint64_t)v[li];
  }
  return (uint32_t)hval;
}

LIBXS_API_INLINE int oz2_hier_l2_garner(const unsigned int gval[], unsigned int d[],
  const uint32_t* l2_garner_inv, int ngroups)
{
  int i, j, is_negative;

  for (i = 0; i < ngroups; ++i) {
    unsigned int u = gval[i];
    const unsigned int mi = oz2_hier_gprod[i];
    for (j = 0; j < i; ++j) {
      unsigned int dj = d[j];
      if (dj >= mi) dj = oz2_mod_l2((uint64_t)dj, i);
      { const unsigned int diff = (u >= dj) ? (u - dj) : (mi + u - dj);
        u = oz2_mod_l2((uint64_t)diff * (uint64_t)l2_garner_inv[j * HIER_NGROUPS_MAX + i], i);
      }
    }
    d[i] = u;
  }

  is_negative = (d[ngroups - 1] >= (oz2_hier_gprod[ngroups - 1] + 1) / 2) ? 1 : 0;
  if (0 != is_negative) {
    for (i = 0; i < ngroups; ++i) {
      d[i] = oz2_hier_gprod[i] - 1 - d[i];
    }
  }
  return is_negative;
}

LIBXS_API_INLINE double oz2_hier_horner(const unsigned int d[], int ngroups)
{
  const int nsuper = (ngroups + HIER_L2_HORNER_GROUP - 1) / HIER_L2_HORNER_GROUP;
  double result;
  int sg, i;

  { const int lo = (nsuper - 1) * HIER_L2_HORNER_GROUP;
    uint64_t r = (uint64_t)d[ngroups - 1];
    for (i = ngroups - 2; i >= lo; --i) {
      r = r * (uint64_t)oz2_hier_gprod[i] + (uint64_t)d[i];
    }
    result = (double)r;
  }

  for (sg = nsuper - 2; sg >= 0; --sg) {
    const int lo = sg * HIER_L2_HORNER_GROUP;
    const int hi = lo + HIER_L2_HORNER_GROUP - 1;
    uint64_t sgval, sgprod = 1;
    for (i = lo; i <= hi; ++i) sgprod *= (uint64_t)oz2_hier_gprod[i];
    sgval = (uint64_t)d[hi];
    for (i = hi - 1; i >= lo; --i) {
      sgval = sgval * (uint64_t)oz2_hier_gprod[i] + (uint64_t)d[i];
    }
    result = result * (double)sgprod + (double)sgval;
  }

  return result;
}

LIBXS_API_INLINE void oz2_reconstruct_batch(unsigned int batch_res[OZ2_BATCH][OZ2_NPRIMES_MAX],
  uint8_t garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX],
  uint32_t (*l2_garner_inv)[HIER_NGROUPS_MAX],
  int nprimes, int bsz, double result[OZ2_BATCH])
{
  const int ngroups = (nprimes + HIER_GS - 1) / HIER_GS;
  int bi, g;

  for (bi = 0; bi < bsz; ++bi) {
    unsigned int gval[HIER_NGROUPS_MAX];
    unsigned int d[HIER_NGROUPS_MAX];
    int is_negative;
    double r;

    for (g = 0; g < ngroups; ++g) {
      const int lo = g * HIER_GS;
      const int hi = (lo + HIER_GS <= nprimes) ? (lo + HIER_GS) : nprimes;
      unsigned int gr[HIER_GS];
      int li;
      for (li = 0; li < hi - lo; ++li) gr[li] = batch_res[bi][lo + li];
      gval[g] = oz2_hier_l1_garner(gr, g, garner_inv, nprimes);
    }

    is_negative = oz2_hier_l2_garner(gval, d, l2_garner_inv, ngroups);
    r = oz2_hier_horner(d, ngroups);
    result[bi] = (0 != is_negative) ? -(r + 1.0) : r;
  }
}

/* AVX-512 batched CRT reconstruction via Garner's algorithm.
 * Processes OZ2_BATCH (= 16) uint32 values in a single __m512i.
 * Uses libxs_mulhi_epu32 and libxs_mod_u32x16 from libxs_utils.h. */
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == OZ2_BATCH

/**
 * AVX-512 hierarchical batched CRT reconstruction.
 * Level 1: vectorized Garner across 16 batch elements per group.
 * Level 2: scalar Garner + Horner per element (only HIER_NGROUPS_MAX steps).
 */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512) void oz2_reconstruct_batch_avx512(
  unsigned int batch_res[OZ2_BATCH][OZ2_NPRIMES_MAX],
  uint8_t garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX],
  uint32_t (*l2_garner_inv)[HIER_NGROUPS_MAX],
  int nprimes, int bsz, double result[OZ2_BATCH])
{
  const int ngroups = (nprimes + HIER_GS - 1) / HIER_GS;
  unsigned int gval_all[OZ2_BATCH][HIER_NGROUPS_MAX];
  int g, bi;

  for (g = 0; g < ngroups; ++g) {
    const int lo = g * HIER_GS;
    const int hi = (lo + HIER_GS <= nprimes) ? (lo + HIER_GS) : nprimes;
    const int gsz = hi - lo;
    unsigned int vt[HIER_GS][OZ2_BATCH];
    __m512i u_vec;
    int li, lj;

    { unsigned int tmp[OZ2_BATCH];
      for (bi = 0; bi < bsz; ++bi) tmp[bi] = batch_res[bi][lo];
      for (bi = bsz; bi < OZ2_BATCH; ++bi) tmp[bi] = 0;
      u_vec = _mm512_loadu_si512((__m512i*)tmp);
      _mm512_storeu_si512((__m512i*)vt[0], u_vec);
    }

    for (li = 1; li < gsz; ++li) {
      const unsigned int pi = oz2_moduli[lo + li];
      const unsigned int rcp_i = oz2_rcp[lo + li];
      const __m512i vpi = _mm512_set1_epi32((int)pi);
      { unsigned int tmp[OZ2_BATCH];
        for (bi = 0; bi < bsz; ++bi) tmp[bi] = batch_res[bi][lo + li];
        for (bi = bsz; bi < OZ2_BATCH; ++bi) tmp[bi] = 0;
        u_vec = _mm512_loadu_si512((__m512i*)tmp);
      }

      for (lj = 0; lj < li; ++lj) {
        const unsigned int inv_ji = garner_inv[lo + lj][lo + li];
        const __m512i vinv = _mm512_set1_epi32((int)inv_ji);
        __m512i vj_vec = _mm512_loadu_si512((__m512i*)vt[lj]);
        { __mmask16 ge = _mm512_cmpge_epu32_mask(vj_vec, vpi);
          vj_vec = _mm512_mask_sub_epi32(vj_vec, ge, vj_vec, vpi);
          ge = _mm512_cmpge_epu32_mask(vj_vec, vpi);
          vj_vec = _mm512_mask_sub_epi32(vj_vec, ge, vj_vec, vpi);
        }
        { const __mmask16 ge = _mm512_cmpge_epu32_mask(u_vec, vj_vec);
          const __m512i d_pos = _mm512_sub_epi32(u_vec, vj_vec);
          const __m512i d_neg = _mm512_sub_epi32(_mm512_add_epi32(vpi, u_vec), vj_vec);
          __m512i diff_vec = _mm512_mask_blend_epi32(ge, d_neg, d_pos);
          diff_vec = _mm512_mullo_epi32(diff_vec, vinv);
          u_vec = libxs_mod_u32x16(diff_vec, pi, rcp_i);
        }
      }
      _mm512_storeu_si512((__m512i*)vt[li], u_vec);
    }

    { unsigned int hval_arr[OZ2_BATCH];
      unsigned int vtmp[OZ2_BATCH];
      int k;
      _mm512_storeu_si512((__m512i*)hval_arr, _mm512_loadu_si512((__m512i*)vt[gsz - 1]));
      for (k = gsz - 2; k >= 0; --k) {
        _mm512_storeu_si512((__m512i*)vtmp, _mm512_loadu_si512((__m512i*)vt[k]));
        for (bi = 0; bi < bsz; ++bi) {
          hval_arr[bi] = (uint32_t)((uint64_t)hval_arr[bi] * (uint64_t)oz2_moduli[lo + k] + (uint64_t)vtmp[bi]);
        }
      }
      for (bi = 0; bi < bsz; ++bi) gval_all[bi][g] = hval_arr[bi];
    }
  }

  for (bi = 0; bi < bsz; ++bi) {
    unsigned int d[HIER_NGROUPS_MAX];
    const int is_negative = oz2_hier_l2_garner(gval_all[bi], d, l2_garner_inv, ngroups);
    const double r = oz2_hier_horner(d, ngroups);
    result[bi] = (0 != is_negative) ? -(r + 1.0) : r;
  }
}

#endif /* LIBXS_INTRINSICS_AVX512 && OZ2_BATCH == 16 */


LIBXS_API_INLINE void gemm_oz2_diff(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc, libxs_matdiff_t* diff)
{
  uint8_t garner_inv[OZ2_NPRIMES_MAX][OZ2_NPRIMES_MAX];
  uint32_t l2_garner_inv[HIER_NGROUPS_MAX * HIER_NGROUPS_MAX];
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
  int nprimes = LIBXS_CLMP(ozaki_n, 1, OZ2_NPRIMES_MAX);
  int oztrim_bits = 0;
  const GEMM_INT_TYPE K_grp_size = (0 < ozaki_maxk ? (GEMM_INT_TYPE)ozaki_maxk : K);
  const GEMM_INT_TYPE K_grp_max = LIBXS_MIN(K_grp_size, K);
  const GEMM_INT_TYPE K_grp_pad = ((K_grp_max + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
  oz2_res_t* a_res = NULL;
  oz2_res_t* b_res = NULL;
  int16_t* expa_raw = NULL;
  int16_t* expb_raw = NULL;
  double* expa_fp = NULL;
  double* expb_fp = NULL;
  GEMM_REAL_TYPE* c_ref = NULL;
  const size_t c_size = (size_t)ldcv * (size_t)N * sizeof(GEMM_REAL_TYPE);
  GEMM_PROFILE_DECL;
  int i, j;
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE) || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));

  /* Trim: truncate mantissa bits to reduce nprimes (mirroring GPU Scheme 2).
   * Each trim level drops 2 input bits (4 product bits). Auto-reduce nprimes
   * when cumulative CRT product bits exceed the required representation. */
  if (0 < ozaki_trim) {
    const int mant = GEMM_IS_DOUBLE ? 52 : 23;
    const int max_levels = mant / 2;
#if defined(OZAKI_I8) && (OZAKI_I8)
    static const int cumbits[20] = {7, 13, 20, 27, 34, 41, 48, 55, 61, 68, 75, 81, 87, 94, 100, 106, 112, 118, 124, 130};
#else
    static const int cumbits[20] = {8, 15, 23, 31, 39, 47, 55, 63, 71, 78, 86, 94, 101, 109, 116, 124, 131, 139, 146, 153};
#endif
    oztrim_bits = LIBXS_MIN(ozaki_trim, max_levels) * 2;
    {
      const int req = 2 * (mant - oztrim_bits) + 23;
      int np;
      for (np = 0; np < OZ2_NPRIMES_MAX && cumbits[np] < req; ++np);
      nprimes = LIBXS_CLMP((np < OZ2_NPRIMES_MAX) ? np + 1 : OZ2_NPRIMES_MAX, 1, nprimes);
    }
  }

  /* Precompute Garner modular inverse table */
  memset(garner_inv, 0, sizeof(garner_inv));
  for (i = 0; i < nprimes; ++i) {
    for (j = i + 1; j < nprimes; ++j) {
      garner_inv[i][j] = (uint8_t)libxs_mod_inverse_u32(oz2_moduli[i] % oz2_moduli[j], oz2_moduli[j]);
    }
  }
  { const int ngroups = (nprimes + HIER_GS - 1) / HIER_GS;
    memset(l2_garner_inv, 0, sizeof(l2_garner_inv));
    for (i = 0; i < ngroups; ++i) {
      for (j = i + 1; j < ngroups; ++j) {
        l2_garner_inv[i * HIER_NGROUPS_MAX + j] = libxs_mod_inverse_u32(oz2_hier_gprod[i] % oz2_hier_gprod[j], oz2_hier_gprod[j]);
      }
    }
  }

  a_res = (oz2_res_t*)libxs_malloc(gemm_pool, (size_t)nprimes * M * K_grp_pad, 0);
  b_res = (oz2_res_t*)libxs_malloc(gemm_pool, (size_t)nprimes * N * K_grp_pad, 0);
  expa_raw = (int16_t*)libxs_malloc(gemm_pool, (size_t)M * sizeof(int16_t), 0);
  expb_raw = (int16_t*)libxs_malloc(gemm_pool, (size_t)N * sizeof(int16_t), 0);
  expa_fp = (double*)libxs_malloc(gemm_pool, (size_t)M * sizeof(double), 0);
  expb_fp = (double*)libxs_malloc(gemm_pool, (size_t)N * sizeof(double), 0);
  /* Save C for reference comparison (before Ozaki modifies it) */
  if (NULL != diff) {
    c_ref = (GEMM_REAL_TYPE*)libxs_malloc(gemm_pool, c_size, 0);
    if (NULL != c_ref) memcpy(c_ref, c, c_size);
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
#endif
    GEMM_PROFILE_START(tid);
    GEMM_PROFILE_PAIRS(nprimes);

    /* Phase 3: scale C by beta (once, before K-group loop).
     * Per BLAS spec, beta=0 must zero C unconditionally (NaN/Inf safe). */
#if defined(_OPENMP)
# pragma omp for OZAKI_OMP_SCHEDULE
#endif
    for (jb = 0; jb < N; ++jb) {
      GEMM_REAL_TYPE* const cj = c + jb * ldcv;
      if ((GEMM_REAL_TYPE)0 != *beta) {
        for (ib = 0; ib < M; ++ib) cj[ib] *= *beta;
      }
      else {
        for (ib = 0; ib < M; ++ib) cj[ib] = (GEMM_REAL_TYPE)0;
      }
    }

    /* K-group loop: process K in chunks of ozaki_maxk */
    for (kb_grp = 0; kb_grp < K; kb_grp += K_grp_size) {
      const GEMM_INT_TYPE K_len = LIBXS_MIN(K_grp_size, K - kb_grp);

      /* Phase 1: preprocess rows of A for this K-group */
#if defined(_OPENMP)
# pragma omp for OZAKI_OMP_SCHEDULE nowait
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
            const int delta = (int)row_max_exp - (int)e + oztrim_bits;
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
# pragma omp for OZAKI_OMP_SCHEDULE
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
            const int delta = (int)col_max_exp - (int)e + oztrim_bits;
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
# pragma omp for OZAKI_OMP_SCHEDULE nowait
#endif
      for (row = 0; row < M; ++row) {
        expa_fp[row] = libxs_pow2((int)expa_raw[row] - OZ_BIAS_PLUS_MANT + oztrim_bits);
      }
#if defined(_OPENMP)
# pragma omp for OZAKI_OMP_SCHEDULE
#endif
      for (col = 0; col < N; ++col) {
        expb_fp[col] = libxs_pow2((int)expb_raw[col] - OZ_BIAS_PLUS_MANT + oztrim_bits);
      }

      /* Phase 4: CRT dot products + accumulate for this K-group.
       * K_CHUNK loop retained for int32 safety when K_GRP > K_CHUNK. */
#if defined(_OPENMP)
# pragma omp for LIBXS_OPENMP_COLLAPSE(2) OZAKI_OMP_SCHEDULE
#endif
      for (jb = 0; jb < N; jb += BLOCK_N) {
        for (ib = 0; ib < M; ib += BLOCK_M) {
          const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
          const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
          GEMM_REAL_TYPE* const cb = c + jb * ldcv + ib;
          uint8_t tile_res[BLOCK_M * BLOCK_N][OZ2_NPRIMES_MAX];
          memset(tile_res, 0, sizeof(tile_res));

          /* Fused GEMM + mod-reduce: inline VNNI panel per prime, Barrett-
           * reduce accumulators in-register, accumulate into tile_res.
           * Eliminates partial[] buffer and per-prime function call overhead. */
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == BLOCK_N && \
  (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH || LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
          if (BLOCK_N == jblk && LIBXS_X86_AVX512 <= ozaki_target_arch) {
            LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
            for (pidx = 0; pidx < nprimes; ++pidx) {
              const unsigned int pi = oz2_moduli[pidx];
              const unsigned int rcp_i = oz2_rcp[pidx];
              const __m512i vpi = _mm512_set1_epi32((int)pi);
              const oz2_res_t* const a_prime = a_res + (long)pidx * M * K_grp_pad + (long)ib * K_grp_pad;
              const oz2_res_t* const b_prime = b_res + (long)pidx * N * K_grp_pad + (long)jb * K_grp_pad;
              for (kb = 0; kb < K_grp_pad; kb += K_CHUNK) {
                const GEMM_INT_TYPE chunk_k = ((GEMM_INT_TYPE)K_CHUNK < K_grp_pad - kb) ? (GEMM_INT_TYPE)K_CHUNK : (K_grp_pad - kb);
                __m512i acc[BLOCK_M];
                GEMM_INT_TYPE kk;
                for (mi = 0; mi < iblk; ++mi) acc[mi] = _mm512_setzero_si512();
# if defined(OZAKI_I8) && (OZAKI_I8)
                {
                  const __m512i bias = _mm512_set1_epi32((int32_t)0x80808080);
                  const __m512i ones = _mm512_set1_epi32(0x01010101);
                  __m512i bsum = _mm512_setzero_si512();
                  {
                    const __m512i vidx = OZAKI_GATHER_VIDX(K_grp_pad);
                    for (kk = kb; kk - kb < chunk_k; kk += BLOCK_K) {
                      LIBXS_ALIGNED(int32_t bv[(BLOCK_K / 4) * BLOCK_N], LIBXS_ALIGNMENT);
                      int bk;
                      OZAKI_REFORMAT_B_IMPL(vidx, b_prime, kk, BLOCK_N, bv, BLOCK_K);
                      for (bk = 0; bk < BLOCK_K; bk += 4) {
                        const __m512i vb = _mm512_load_si512((__m512i*)(bv + (bk >> 2) * BLOCK_N));
                        bsum = _mm512_dpbusd_epi32(bsum, ones, vb);
                        LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M)
                        for (mi = 0; mi < iblk; ++mi) {
                          const __m512i va = _mm512_xor_si512(
                            _mm512_set1_epi32(*(const int32_t*)(a_prime + (long)mi * K_grp_pad + kk + bk)), bias);
                          acc[mi] = _mm512_dpbusd_epi32(acc[mi], va, vb);
                        }
                      }
                    }
                  }
                  {
                    const __m512i correction = _mm512_mullo_epi32(_mm512_set1_epi32(128), bsum);
                    for (mi = 0; mi < iblk; ++mi) acc[mi] = _mm512_sub_epi32(acc[mi], correction);
                  }
                }
                for (mi = 0; mi < iblk; ++mi) {
                  const __mmask16 neg = _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), acc[mi]);
                  __m512i vr = libxs_mod_u32x16(_mm512_abs_epi32(acc[mi]), pi, rcp_i);
                  {
                    const __mmask16 nz = _mm512_cmpgt_epu32_mask(vr, _mm512_setzero_si512());
                    vr = _mm512_mask_sub_epi32(vr, (__mmask16)(neg & nz), vpi, vr);
                  }
                  {
                    LIBXS_ALIGNED(unsigned int tmp[BLOCK_N], LIBXS_ALIGNMENT);
                    __m512i vacc;
                    int nj2;
                    for (nj2 = 0; nj2 < BLOCK_N; ++nj2) tmp[nj2] = tile_res[mi * BLOCK_N + nj2][pidx];
                    vacc = _mm512_add_epi32(_mm512_loadu_si512((__m512i*)tmp), vr);
                    {
                      const __mmask16 ge = _mm512_cmpge_epu32_mask(vacc, vpi);
                      vacc = _mm512_mask_sub_epi32(vacc, ge, vacc, vpi);
                    }
                    _mm512_storeu_si512((__m512i*)tmp, vacc);
                    for (nj2 = 0; nj2 < BLOCK_N; ++nj2) tile_res[mi * BLOCK_N + nj2][pidx] = (uint8_t)tmp[nj2];
                  }
                }
# else /* u8 */
                {
                  const __m512i vidx = OZAKI_GATHER_VIDX(K_grp_pad);
                  for (kk = kb; kk - kb < chunk_k; kk += BLOCK_K) {
                    LIBXS_ALIGNED(int32_t bv[(BLOCK_K / 4) * BLOCK_N], LIBXS_ALIGNMENT);
                    int bk;
                    OZAKI_REFORMAT_B_XOR_IMPL(vidx, b_prime, kk, BLOCK_N, bv, BLOCK_K);
                    for (bk = 0; bk < BLOCK_K; bk += 4) {
                      const __m512i vb = _mm512_load_si512((__m512i*)(bv + (bk >> 2) * BLOCK_N));
                      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M)
                      for (mi = 0; mi < iblk; ++mi) {
                        const __m512i va = _mm512_set1_epi32(*(const int32_t*)(a_prime + (long)mi * K_grp_pad + kk + bk));
                        acc[mi] = _mm512_dpbusd_epi32(acc[mi], va, vb);
                      }
                    }
                  }
                }
                for (mi = 0; mi < iblk; ++mi) {
                  int32_t asum = 0;
                  for (kk = kb; kk - kb < chunk_k; ++kk) {
                    asum += (int32_t)a_prime[mi * K_grp_pad + kk];
                  }
                  {
                    const __m512i vr = libxs_mod_u32x16(_mm512_add_epi32(acc[mi], _mm512_set1_epi32(128 * asum)), pi, rcp_i);
                    LIBXS_ALIGNED(unsigned int tmp[BLOCK_N], LIBXS_ALIGNMENT);
                    __m512i vacc;
                    int nj2;
                    for (nj2 = 0; nj2 < BLOCK_N; ++nj2) tmp[nj2] = tile_res[mi * BLOCK_N + nj2][pidx];
                    vacc = _mm512_add_epi32(_mm512_loadu_si512((__m512i*)tmp), vr);
                    {
                      const __mmask16 ge = _mm512_cmpge_epu32_mask(vacc, vpi);
                      vacc = _mm512_mask_sub_epi32(vacc, ge, vacc, vpi);
                    }
                    _mm512_storeu_si512((__m512i*)tmp, vacc);
                    for (nj2 = 0; nj2 < BLOCK_N; ++nj2) tile_res[mi * BLOCK_N + nj2][pidx] = tmp[nj2];
                  }
                }
# endif
              }
            }
          }
          else
#endif
          { /* Scalar fallback */
            for (kb = 0; kb < K_grp_pad; kb += K_CHUNK) {
              const GEMM_INT_TYPE chunk_k = ((GEMM_INT_TYPE)K_CHUNK < K_grp_pad - kb) ? (GEMM_INT_TYPE)K_CHUNK : (K_grp_pad - kb);
              LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NPRIMES_MAX, OZ2_NPRIMES_DEFAULT)
              for (pidx = 0; pidx < nprimes; ++pidx) {
                LIBXS_ALIGNED(int32_t partial[BLOCK_M * BLOCK_N], LIBXS_ALIGNMENT);
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
                    tile_res[mi * jblk + nj][pidx] = (uint8_t)r;
                  }
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
              oz2_reconstruct_batch_avx512(batch_res, garner_inv, l2_garner_inv, nprimes, (int)bsz, batch_val);
# else
              if (LIBXS_X86_AVX512 <= ozaki_target_arch) {
                oz2_reconstruct_batch_avx512(batch_res, garner_inv, l2_garner_inv, nprimes, (int)bsz, batch_val);
              }
              else {
                oz2_reconstruct_batch(batch_res, garner_inv, l2_garner_inv, nprimes, (int)bsz, batch_val);
              }
# endif
#else
              oz2_reconstruct_batch(batch_res, garner_inv, l2_garner_inv, nprimes, (int)bsz, batch_val);
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
    } /* end K-group loop */

    GEMM_PROFILE_END(tid, M, N, K);
  } /* end parallel */

  /* Reference BLAS and diff comparison (whole-matrix, consistent with GPU path) */
  if (NULL != c_ref) {
    ozaki_diff_reference(GEMM_ARGPASS, c_ref, c_size, diff);
  }
  libxs_free(a_res);
  libxs_free(b_res);
  libxs_free(expa_raw);
  libxs_free(expb_raw);
  libxs_free(expa_fp);
  libxs_free(expb_fp);
  libxs_free(c_ref);
}


OZAKI_API void gemm_oz2(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  OZAKI_GEMM_WRAPPER(gemm_oz2_diff, GEMM_LABEL, 1)
}
