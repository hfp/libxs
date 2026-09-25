/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"

/**
 * Number of (mi,nj) elements processed in one Garner reconstruction batch.
 * Batching exposes data-parallelism across independent elements so the
 * compiler can auto-vectorize the per-element Garner inner loop.
 */
#if !defined(OZ2_BATCH)
# define OZ2_BATCH 16
#endif

#define OZ2_AMX_NACC 6
/* Below this problem size (cube root of M*N*K) AMX loses to VNNI by up to 1.6x, above it wins up to 4x */
#define OZ2_AMX_MIN 640

/* VPDPBUUD can be compiled (it runs if the CPU has AVX512_INT8) */
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K) && \
  (LIBXS_X86_AVX512_INT8 <= LIBXS_MAX_STATIC_TARGET_ARCH)
# define OZ2_BUUD
#endif

/** Sign folding for residue storage: additive inverse (p - r) for negatives. */
#define OZ2_SIGN_FOLD(SIGN, RES, PIDX) ((oz2_res_t)(((SIGN) < 0 && 0 != (RES)) ? (oz2_moduli[(PIDX)] - (RES)) : (RES)))

typedef uint8_t oz2_res_t;

/**
 * Chinese Remainder Theorem (CRT) moduli and precomputed Barrett tables:
 * 20 pairwise coprime moduli <= 256, not necessarily prime (256=2^8,
 * 243=3^5, 169=13^2); the largest prime power of each prime that fits
 * maximizes bits per channel.
 *
 * Unsigned residues [0, p-1] encode the sign via the modular additive
 * inverse (p - r == -r mod p), which enables u8*u8 VNNI via VPDPBUSD with
 * B-side bias correction. Signed moduli <= 128 would need fp64 19 (vs 16)
 * and fp32 10 (vs 9). Safe K without chunking: ~33K (255^2 * K per int32
 * accumulator).
 *
 * P = prod(m_i) must exceed 2 * BLOCK_K * (2^(MANT+1))^2 to represent
 * signed dot products without aliasing.
 */
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

/** Fast modular reduction: x mod oz2_moduli[pidx] (table-indexed wrapper). */
LIBXS_INLINE unsigned int oz2_mod(uint32_t x, int pidx)
{
  return libxs_mod_u32(x, oz2_moduli[pidx], oz2_rcp[pidx]);
}


/** Fast 64-bit modular reduction: x mod oz2_moduli[pidx] (table-indexed wrapper). */
LIBXS_INLINE unsigned int oz2_mod64(uint64_t x, int pidx)
{
  return libxs_mod_u64(x, oz2_moduli[pidx], oz2_rcp[pidx], oz2_pow18[pidx], oz2_pow36[pidx]);
}


#if defined(LIBXS_INTRINSICS_AVX512)
LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512) void oz2_reduce_avx512(uint64_t mantissa,
  uint8_t residues[OZ2_NMODULI_MAX], int nmoduli)
{
  const __m512i vmoduli = _mm512_setr_epi32((int)oz2_moduli[0], (int)oz2_moduli[1],
    (int)oz2_moduli[2], (int)oz2_moduli[3], (int)oz2_moduli[4], (int)oz2_moduli[5],
    (int)oz2_moduli[6], (int)oz2_moduli[7], (int)oz2_moduli[8], (int)oz2_moduli[9],
    (int)oz2_moduli[10], (int)oz2_moduli[11], (int)oz2_moduli[12], (int)oz2_moduli[13],
    (int)oz2_moduli[14], (int)oz2_moduli[15]);
  const __m512i vrcp = _mm512_loadu_si512((const __m512i*)oz2_rcp);
  const __m512i vpow18 = _mm512_loadu_si512((const __m512i*)oz2_pow18);
  const __m512i vpow36 = _mm512_loadu_si512((const __m512i*)oz2_pow36);
  const __m512i va0 = _mm512_set1_epi32((int)(mantissa & 0x3FFFFU));
  const __m512i va1 = _mm512_set1_epi32((int)((mantissa >> 18) & 0x3FFFFU));
  const __m512i va2 = _mm512_set1_epi32((int)(mantissa >> 36));
  LIBXS_ALIGNED(uint32_t tmp[16], LIBXS_ALIGNMENT);
  __m512i value, quotient, remainder;
  int i;

  value = _mm512_add_epi32(va0, _mm512_add_epi32(_mm512_mullo_epi32(va1, vpow18),
    _mm512_mullo_epi32(va2, vpow36)));
  quotient = libxs_mulhi_epu32(value, vrcp);
  remainder = _mm512_sub_epi32(value, _mm512_mullo_epi32(quotient, vmoduli));
  {
    const __mmask16 ge = _mm512_cmpge_epu32_mask(remainder, vmoduli);
    remainder = _mm512_mask_sub_epi32(remainder, ge, remainder, vmoduli);
  }
  _mm512_store_si512((__m512i*)tmp, remainder);
  for (i = 0; i < nmoduli && i < 16; ++i) residues[i] = (uint8_t)tmp[i];
}
#endif


/**
 * Reduce an aligned mantissa modulo all active moduli.
 * delta = max_exp - element_exp (>= 0); mantissa is right-shifted
 * by delta bits for exponent alignment before reduction.
 */
LIBXS_INLINE void oz2_reduce(uint64_t mantissa, int delta, uint8_t residues[OZ2_NMODULI_MAX], int nmoduli)
{
  int i = 0;
  nmoduli = LIBXS_CLMP(nmoduli, 0, OZ2_NMODULI_MAX);
  if (delta > 0) {
    if (delta >= 64) mantissa = 0;
    else mantissa >>= delta;
  }
#if defined(LIBXS_INTRINSICS_AVX512)
  if (16 <= nmoduli && LIBXS_X86_AVX512 <= ozaki_target_arch) {
    oz2_reduce_avx512(mantissa, residues, nmoduli);
    i = 16;
  }
#endif
  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NMODULI_MAX, OZ2_NMODULI_DEFAULT)
  for (; i < nmoduli; ++i) {
    residues[i] = (uint8_t)oz2_mod64(mantissa, i);
  }
}


/**
 * Hierarchical CRT: two-level Garner reconstruction.
 * Level 1: HIER_GS moduli per group (small Garner, 32-bit).
 * Level 2: Garner over HIER_NGROUPS group-moduli (32-bit, 64-bit Barrett).
 */
#define HIER_GS 4
#define HIER_NGROUPS_MAX ((OZ2_NMODULI_MAX + HIER_GS - 1) / HIER_GS)
#define HIER_L2_HORNER_GROUP 2

LIBXS_INLINE unsigned int oz2_mod_l2(uint64_t x, uint32_t m, uint64_t barrett)
{
#if defined(__SIZEOF_INT128__)
  const uint64_t q = (uint64_t)(((__uint128_t)x * barrett) >> 64);
#else
  const uint64_t x_lo = (uint32_t)x, x_hi = x >> 32;
  const uint64_t b_lo = (uint32_t)barrett, b_hi = barrett >> 32;
  const uint64_t q = (x_hi * b_hi) + ((x_hi * b_lo + x_lo * b_hi + ((x_lo * b_lo) >> 32)) >> 32);
#endif
  { uint32_t r = (uint32_t)(x - q * (uint64_t)m);
    return (r >= m) ? (r - m) : r;
  }
}

LIBXS_INLINE unsigned int oz2_hier_l1_garner(const unsigned int group_residues[], int g,
  uint8_t garner_inv[OZ2_NMODULI_MAX][OZ2_NMODULI_MAX], int nmoduli)
{
  const int lo = g * HIER_GS;
  const int hi = (lo + HIER_GS <= nmoduli) ? (lo + HIER_GS) : nmoduli;
  const int gsz = hi - lo;
  unsigned int v[HIER_GS];
  uint64_t hval = 0;
  int li, lj;

  if (gsz >= 1) {
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
  }
  return (uint32_t)hval;
}

LIBXS_INLINE int oz2_hier_l2_garner(const unsigned int gval[], unsigned int d[],
  const uint32_t* l2_garner_inv, const uint32_t* gprod, const uint64_t* l2_barrett, int ngroups)
{
  int i, j, is_negative;

  for (i = 0; i < ngroups; ++i) {
    unsigned int u = gval[i];
    const unsigned int mi = gprod[i];
    for (j = 0; j < i; ++j) {
      unsigned int dj = d[j];
      if (dj >= mi) dj = oz2_mod_l2((uint64_t)dj, mi, l2_barrett[i]);
      { const unsigned int diff = (u >= dj) ? (u - dj) : (mi + u - dj);
        u = oz2_mod_l2((uint64_t)diff * (uint64_t)l2_garner_inv[j * HIER_NGROUPS_MAX + i], mi, l2_barrett[i]);
      }
    }
    d[i] = u;
  }

  is_negative = (d[ngroups - 1] >= (gprod[ngroups - 1] + 1) / 2) ? 1 : 0;
  /**
   * Two's-complement the level-2 digits and complete the negation with a +1
   * carry propagated in integer space, so the Horner evaluation already yields
   * the magnitude. Adding the 1 to the reconstructed value in floating point
   * instead would be inexact once the magnitude exceeds 2^53.
   */
  if (0 != is_negative) {
    for (i = 0; i < ngroups; ++i) {
      d[i] = gprod[i] - 1 - d[i];
    }
    for (i = 0; i < ngroups; ++i) {
      if (d[i] + 1 < gprod[i]) {
        ++d[i];
        break;
      }
      d[i] = 0;
    }
  }
  return is_negative;
}

LIBXS_INLINE double oz2_hier_horner(const unsigned int d[], const uint32_t* gprod, int ngroups)
{
  const int nsuper = LIBXS_UPDIV(ngroups, HIER_L2_HORNER_GROUP);
  double result;
  int sg, i;

  { const int lo = (nsuper - 1) * HIER_L2_HORNER_GROUP;
    uint64_t r = (uint64_t)d[ngroups - 1];
    for (i = ngroups - 2; i >= lo; --i) {
      r = r * (uint64_t)gprod[i] + (uint64_t)d[i];
    }
    result = (double)r;
  }

  for (sg = nsuper - 2; sg >= 0; --sg) {
    const int lo = sg * HIER_L2_HORNER_GROUP;
    const int hi = lo + HIER_L2_HORNER_GROUP - 1;
    uint64_t sgval, sgprod = 1;
    for (i = lo; i <= hi; ++i) sgprod *= (uint64_t)gprod[i];
    sgval = (uint64_t)d[hi];
    for (i = hi - 1; i >= lo; --i) {
      sgval = sgval * (uint64_t)gprod[i] + (uint64_t)d[i];
    }
    result = result * (double)sgprod + (double)sgval;
  }

  return result;
}

LIBXS_INLINE void oz2_reconstruct_batch(unsigned int batch_res[OZ2_BATCH][OZ2_NMODULI_MAX],
  uint8_t garner_inv[OZ2_NMODULI_MAX][OZ2_NMODULI_MAX],
  const uint32_t* l2_garner_inv, const uint32_t* gprod, const uint64_t* l2_barrett,
  int nmoduli, int bsz, double result[OZ2_BATCH])
{
  const int ngroups = LIBXS_UPDIV(nmoduli, HIER_GS);
  int bi, g;

  for (bi = 0; bi < bsz; ++bi) {
    unsigned int gval[HIER_NGROUPS_MAX];
    unsigned int d[HIER_NGROUPS_MAX];
    int is_negative;
    double r;

    for (g = 0; g < ngroups; ++g) {
      const int lo = g * HIER_GS;
      const int hi = (lo + HIER_GS <= nmoduli) ? (lo + HIER_GS) : nmoduli;
      unsigned int gr[HIER_GS];
      int li;
      for (li = 0; li < hi - lo; ++li) gr[li] = batch_res[bi][lo + li];
      gval[g] = oz2_hier_l1_garner(gr, g, garner_inv, nmoduli);
    }

    is_negative = oz2_hier_l2_garner(gval, d, l2_garner_inv, gprod, l2_barrett, ngroups);
    r = oz2_hier_horner(d, gprod, ngroups);
    result[bi] = (0 != is_negative) ? -r : r;
  }
}

/**
 * AVX-512 batched CRT reconstruction via Garner's algorithm.
 * Processes OZ2_BATCH (= 16) uint32 values in a single __m512i.
 * Uses libxs_mulhi_epu32 and libxs_mod_u32x16 from libxs_utils.h.
 */
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == OZ2_BATCH

/**
 * AVX-512 hierarchical batched CRT reconstruction.
 * Level 1: vectorized Garner across 16 batch elements per group.
 * Level 2: scalar Garner + Horner per element (only HIER_NGROUPS_MAX steps).
 */
LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512) void oz2_reconstruct_batch_avx512(
  unsigned int batch_res[OZ2_BATCH][OZ2_NMODULI_MAX],
  uint8_t garner_inv[OZ2_NMODULI_MAX][OZ2_NMODULI_MAX],
  const uint32_t* l2_garner_inv, const uint32_t* gprod, const uint64_t* l2_barrett,
  int nmoduli, int bsz, double result[OZ2_BATCH])
{
  const int ngroups = LIBXS_UPDIV(nmoduli, HIER_GS);
  unsigned int gval_all[OZ2_BATCH][HIER_NGROUPS_MAX];
  int g, bi;

  for (g = 0; g < ngroups; ++g) {
    const int lo = g * HIER_GS;
    const int hi = (lo + HIER_GS <= nmoduli) ? (lo + HIER_GS) : nmoduli;
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
    const int is_negative = oz2_hier_l2_garner(gval_all[bi], d, l2_garner_inv, gprod, l2_barrett, ngroups);
    const double r = oz2_hier_horner(d, gprod, ngroups);
    result[bi] = (0 != is_negative) ? -r : r;
  }
}

#endif /* LIBXS_INTRINSICS_AVX512 && OZ2_BATCH == 16 */


#if defined(OZ2_BUUD)
/** acc[M] += A[M,K] * B'[16,K] via VPDPBUUD: u8*u8 natively, so B needs no bias and A no row-sum. */
LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_INT8) void oz2_gemm_buud(GEMM_INT_TYPE M, GEMM_INT_TYPE K,
  const oz2_res_t* a, GEMM_INT_TYPE lda, const oz2_res_t* b, GEMM_INT_TYPE ldb, __m512i acc[BLOCK_M])
{
  const __m512i vidx = OZAKI_GATHER_VIDX(ldb);
  GEMM_INT_TYPE kk, mi;
  int bk;
  for (kk = 0; kk < K; kk += BLOCK_K) {
    LIBXS_ALIGNED(int32_t bv[(BLOCK_K / 4) * BLOCK_N], LIBXS_ALIGNMENT);
    OZAKI_REFORMAT_B_IMPL(vidx, b, kk, BLOCK_N, bv, BLOCK_K);
    for (bk = 0; bk < BLOCK_K; bk += 4) {
      const __m512i vb = _mm512_load_si512((const __m512i*)(bv + (bk >> 2) * BLOCK_N));
      LIBXS_PRAGMA_LOOP_COUNT(1, BLOCK_M, BLOCK_M)
      for (mi = 0; mi < M; ++mi) {
        const __m512i va = _mm512_set1_epi32(*(const int32_t*)(a + (long)mi * lda + kk + bk));
        acc[mi] = _mm512_dpbuud_epi32(acc[mi], va, vb);
      }
    }
  }
}
#endif


#if defined(LIBXS_INTRINSICS_AMX) && defined(LIBXS_INTRINSICS_AVX512) && \
  16 == BLOCK_M && 16 == BLOCK_N
LIBXS_INLINE void oz2_amx_tilecfg_init(ozaki_amx_tilecfg_t* cfg)
{
  int tile;
  memset(cfg, 0, sizeof(*cfg));
  cfg->data[0] = 1;
  for (tile = 0; tile < OZ2_AMX_NACC; ++tile) {
    *(uint16_t*)(cfg->data + 16 + tile * 2) = 64;
    cfg->data[48 + tile] = 16;
  }
  *(uint16_t*)(cfg->data + 16 + 6 * 2) = 64;
  cfg->data[48 + 6] = 16;
  *(uint16_t*)(cfg->data + 16 + 7 * 2) = 64;
  cfg->data[48 + 7] = 16;
}


LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_AMX) void oz2_amx_configure(void)
{
  ozaki_amx_tilecfg_t cfg;
  oz2_amx_tilecfg_init(&cfg);
  _tile_loadconfig(&cfg);
}


LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_AMX) void oz2_amx_release(void)
{
  _tile_release();
}


/** AMX for npanel (1..OZ2_AMX_NACC) adjacent column blocks sharing each A tile. */
LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_AMX) void oz2_amx_gemm(
  GEMM_INT_TYPE K, int pidx, int npanel, const oz2_res_t* a, GEMM_INT_TYPE lda,
  const int32_t* b, GEMM_INT_TYPE ldb, size_t bpanel_stride,
  uint8_t tile_res[OZ2_AMX_NACC][BLOCK_M * BLOCK_N][OZ2_NMODULI_MAX])
{
  LIBXS_ALIGNED(int32_t partial[OZ2_AMX_NACC][BLOCK_M * BLOCK_N], LIBXS_ALIGNMENT);
  GEMM_INT_TYPE kb;
  const unsigned int pi = oz2_moduli[pidx];
  const unsigned int rcp_i = oz2_rcp[pidx];
  const __m512i vpi = _mm512_set1_epi32((int)pi);
  int mi, panel;

  LIBXS_ASSERT(0 == (K % 64) && 0 < npanel && OZ2_AMX_NACC >= npanel);
  for (panel = 0; panel < npanel; ++panel) { /* tile numbers must be immediates */
    switch (panel) {
      case 0: _tile_zero(0); break;
      case 1: _tile_zero(1); break;
      case 2: _tile_zero(2); break;
      case 3: _tile_zero(3); break;
      case 4: _tile_zero(4); break;
      default: _tile_zero(5); break;
    }
  }
  for (kb = 0; kb < K; kb += 64) {
    _tile_loadd(6, a + kb, (int)lda);
    for (panel = 0; panel < npanel; ++panel) {
      _tile_loadd(7, b + (size_t)panel * bpanel_stride + (kb / 4) * ldb,
        (int)(ldb * sizeof(int32_t)));
      switch (panel) {
        case 0: _tile_dpbuud(0, 6, 7); break;
        case 1: _tile_dpbuud(1, 6, 7); break;
        case 2: _tile_dpbuud(2, 6, 7); break;
        case 3: _tile_dpbuud(3, 6, 7); break;
        case 4: _tile_dpbuud(4, 6, 7); break;
        default: _tile_dpbuud(5, 6, 7); break;
      }
    }
  }
  for (panel = 0; panel < npanel; ++panel) {
    switch (panel) {
      case 0: _tile_stored(0, partial[0], BLOCK_N * (int)sizeof(int32_t)); break;
      case 1: _tile_stored(1, partial[1], BLOCK_N * (int)sizeof(int32_t)); break;
      case 2: _tile_stored(2, partial[2], BLOCK_N * (int)sizeof(int32_t)); break;
      case 3: _tile_stored(3, partial[3], BLOCK_N * (int)sizeof(int32_t)); break;
      case 4: _tile_stored(4, partial[4], BLOCK_N * (int)sizeof(int32_t)); break;
      default: _tile_stored(5, partial[5], BLOCK_N * (int)sizeof(int32_t)); break;
    }
  }

  for (panel = 0; panel < npanel; ++panel) {
    for (mi = 0; mi < BLOCK_M; ++mi) {
      __m512i vr;
      LIBXS_ALIGNED(unsigned int tmp[BLOCK_N], LIBXS_ALIGNMENT);
      __m512i vacc;
      int nj;
      vr = libxs_mod_u32x16(_mm512_load_si512((const __m512i*)(partial[panel] + mi * BLOCK_N)), pi, rcp_i);
      for (nj = 0; nj < BLOCK_N; ++nj) tmp[nj] = tile_res[panel][mi * BLOCK_N + nj][pidx];
      vacc = _mm512_add_epi32(_mm512_load_si512((const __m512i*)tmp), vr);
      {
        const __mmask16 ge = _mm512_cmpge_epu32_mask(vacc, vpi);
        vacc = _mm512_mask_sub_epi32(vacc, ge, vacc, vpi);
      }
      _mm512_store_si512((__m512i*)tmp, vacc);
      for (nj = 0; nj < BLOCK_N; ++nj) tile_res[panel][mi * BLOCK_N + nj][pidx] = (uint8_t)tmp[nj];
    }
  }
}
#endif


LIBXS_INLINE void gemm_oz2_diff(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc, libxs_matdiff_t* diff)
{
  uint8_t garner_inv[OZ2_NMODULI_MAX][OZ2_NMODULI_MAX];
  uint32_t hier_gprod[HIER_NGROUPS_MAX];
  uint64_t hier_l2_barrett[HIER_NGROUPS_MAX];
  uint32_t l2_garner_inv[HIER_NGROUPS_MAX * HIER_NGROUPS_MAX];
  /**
   * Max K per int32 accumulation pass: K_CHUNK * max_residue^2 < 2^31.
   * u8 (max 255): 255^2 * 32768 ~ 2.13e9 < 2^31. K_CHUNK = 32768.
   */
  enum { K_CHUNK = 32768 };
  const int ta = (*transa != 'N' && *transa != 'n');
  const int tb = (*transb != 'N' && *transb != 'n');
  const GEMM_INT_TYPE M = *m, N = *n, K = *k;
  const GEMM_INT_TYPE ldcv = *ldc;
  const GEMM_INT_TYPE K_grp_size = (0 < ozaki_maxk ? (GEMM_INT_TYPE)ozaki_maxk : K);
  const GEMM_INT_TYPE K_grp_max = LIBXS_MIN(K_grp_size, K);
  const size_t c_size = (size_t)ldcv * (size_t)N * sizeof(GEMM_REAL_TYPE);
  GEMM_INT_TYPE K_grp_pad, nchunk, jstep;
  int32_t* a_rowsum = NULL;
  oz2_res_t* a_res = NULL;
  oz2_res_t* b_res = NULL;
  int32_t* b_packed = NULL;
  int16_t* expa_raw = NULL;
  int16_t* expb_raw = NULL;
  double* expa_fp = NULL;
  double* expb_fp = NULL;
  GEMM_REAL_TYPE* c_ref = NULL;
#if defined(__LIBXSMM)
  ozaki_xsmm_t xsmm[2]; /* whole K_CHUNK, chunk tail */
  int use_xsmm = 0;
#endif
  int nmoduli = LIBXS_CLMP(ozaki_n, 1, OZ2_NMODULI_MAX);
  int oztrim_bits = 0, use_amx = 0;
  int i, j;
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE) || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));
#if defined(LIBXS_INTRINSICS_AMX) && defined(LIBXS_INTRINSICS_AVX512) && \
  16 == BLOCK_M && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
  use_amx = (BLOCK_M <= M && BLOCK_N <= N && LIBXS_X86_AVX512_AMX <= ozaki_target_arch
             && (0 < ozaki_amx || (double)OZ2_AMX_MIN * OZ2_AMX_MIN * OZ2_AMX_MIN <= (double)M * N * K));
#endif
#if defined(__LIBXSMM)
  if (0 != ozaki_xsmm) { /* LIBXSMM ahead of the built-in kernels */
    const GEMM_INT_TYPE kpad = LIBXS_UP(K_grp_max, BLOCK_K), ktail = (K_CHUNK < kpad ? kpad % K_CHUNK : 0);
    use_xsmm = (EXIT_SUCCESS == ozaki_xsmm_init(&xsmm[0], LIBXSMM_DATATYPE_U8, M, LIBXS_MIN(kpad, K_CHUNK), kpad, 1)
      && (0 == ktail || EXIT_SUCCESS == ozaki_xsmm_init(&xsmm[1], LIBXSMM_DATATYPE_U8, M, ktail, kpad, 1)));
    if (0 != use_xsmm) use_amx = 0;
  }
#endif
  K_grp_pad = LIBXS_UP(K_grp_max, 0 != use_amx ? 64 : BLOCK_K);
  nchunk = LIBXS_UPDIV(K_grp_pad, K_CHUNK);

  /**
   * Trim counts moduli, the unit of work, and the truncation follows from what those
   * moduli carry: a product of two b-bit significands summed over K terms needs
   * 2*b + ceil(log2(K)) + 1 bits, so asking for fewer bits than the moduli hold
   * loses accuracy at no saving. Same accounting as GPU Scheme 2 (ozaki_crt_bits in
   * LIBXSTREAM), with the pass's own K rather than a declared bound, which is
   * tighter. A negative trim adds moduli and buys the precision back.
   */
  {
    const int sig = GEMM_IS_DOUBLE ? 53 : 24;
    static const int cumbits[20] = {8, 15, 23, 31, 39, 47, 55, 63, 71, 78, 86, 94, 101, 109, 116, 124, 131, 139, 146, 153};
    uint64_t kk = (uint64_t)K_grp_max - 1;
    int lgk = 1, avail;
    while (0 < kk) { ++lgk; kk >>= 1; }
    nmoduli = LIBXS_CLMP(nmoduli - ozaki_trim, 2, OZ2_NMODULI_MAX);
    avail = cumbits[nmoduli-1] - lgk;
    oztrim_bits = LIBXS_CLMP(sig - (0 < avail ? avail / 2 : 0), 0, sig - 1);
  }

  /* Precompute Garner modular inverse table */
  memset(garner_inv, 0, sizeof(garner_inv));
  for (i = 0; i < nmoduli; ++i) {
    for (j = i + 1; j < nmoduli; ++j) {
      garner_inv[i][j] = (uint8_t)libxs_mod_inverse_u32(oz2_moduli[i] % oz2_moduli[j], oz2_moduli[j]);
    }
  }
  { const int ngroups = LIBXS_UPDIV(nmoduli, HIER_GS);
    for (i = 0; i < ngroups; ++i) {
      const int lo = i * HIER_GS;
      const int hi = (lo + HIER_GS <= nmoduli) ? (lo + HIER_GS) : nmoduli;
      uint32_t p = 1;
      for (j = lo; j < hi; ++j) p *= (uint32_t)oz2_moduli[j];
      hier_gprod[i] = p;
      hier_l2_barrett[i] = (uint64_t)(-1) / (uint64_t)p; /* floor(2^64-1 / p) ~ floor(2^64/p) */
    }
    memset(l2_garner_inv, 0, sizeof(l2_garner_inv));
    for (i = 0; i < ngroups; ++i) {
      for (j = i + 1; j < ngroups; ++j) {
        l2_garner_inv[i * HIER_NGROUPS_MAX + j] = libxs_mod_inverse_u32(hier_gprod[i] % hier_gprod[j], hier_gprod[j]);
      }
    }
  }

  a_res = (oz2_res_t*)libxs_malloc(gemm_pool, (size_t)nmoduli * M * K_grp_pad, 0);
  b_res = (oz2_res_t*)libxs_malloc(gemm_pool, (size_t)nmoduli * N * K_grp_pad, 0);
#if defined(LIBXS_INTRINSICS_AMX) && defined(LIBXS_INTRINSICS_AVX512) && \
  16 == BLOCK_M && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
  if (0 != use_amx) {
    const GEMM_INT_TYPE N_blocks = LIBXS_UPDIV(N, BLOCK_N);
    b_packed = (int32_t*)libxs_malloc(gemm_pool,
      (size_t)nmoduli * N_blocks * (K_grp_pad / 4) * BLOCK_N * sizeof(int32_t), 0);
    if (NULL == b_packed) use_amx = 0;
  }
#endif
  /* AMX takes a unit of OZ2_AMX_NACC column blocks per work item, the other paths one block */
  jstep = (0 != use_amx ? OZ2_AMX_NACC * BLOCK_N : BLOCK_N);
#if defined(__LIBXSMM)
  if (0 != use_xsmm) {
    b_packed = (int32_t*)libxs_malloc(gemm_pool, (size_t)nmoduli * LIBXS_UPDIV(N, BLOCK_N) * K_grp_pad * BLOCK_N, 0);
    if (NULL == b_packed) use_xsmm = 0;
  }
#endif
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == BLOCK_N && \
  (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH || LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
  { /* VPDPBUSD is corrected by +128*row_sum(A), which depends on row, modulus, and K chunk but not on the tile */
    int biased = (LIBXS_X86_AVX512 <= ozaki_target_arch);
#if defined(__LIBXSMM)
    biased = (biased && 0 == use_xsmm);
#endif
# if defined(OZ2_BUUD)
    if (LIBXS_X86_AVX512_INT8 <= ozaki_target_arch) biased = 0;
# endif
    if (0 != biased) {
      a_rowsum = (int32_t*)libxs_malloc(gemm_pool, (size_t)nmoduli * M * nchunk * sizeof(int32_t), 0);
    }
  }
#endif
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
    GEMM_INT_TYPE row, col, ib, jb, jw, mi, nj, kb, kb_grp;
    int pidx;
    /**
     * Phase 3: scale C by beta (once, before K-group loop).
     * Per BLAS spec, beta=0 must zero C unconditionally (NaN/Inf safe).
     */
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
        for (pidx = 0; pidx < nmoduli; ++pidx) {
          memset(a_res + (long)pidx * M * K_grp_pad + (long)row * K_grp_pad, 0, (size_t)K_grp_pad);
          if (NULL != a_rowsum) memset(a_rowsum + ((size_t)pidx * M + row) * nchunk, 0, (size_t)nchunk * sizeof(int32_t));
        }
        for (kk = kb_grp; kk < kb_grp + K_len; ++kk) {
          int16_t e;
          uint64_t mt;
          ozaki_extract_ieee(a[LIBXS_INDEX(ta, *lda, row, kk)], &e, &mt);
          if (e > row_max_exp) row_max_exp = e;
        }
        expa_raw[row] = row_max_exp;
        for (kk = kb_grp; kk < kb_grp + K_len; kk += K_CHUNK) { /* local row sums: rows of two threads share lines */
          const GEMM_INT_TYPE kend = LIBXS_MIN(kk + (GEMM_INT_TYPE)K_CHUNK, kb_grp + K_len);
          int32_t rsum[OZ2_NMODULI_MAX];
          GEMM_INT_TYPE kc;
          memset(rsum, 0, sizeof(rsum));
          for (kc = kk; kc < kend; ++kc) {
            int16_t e;
            uint64_t mt;
            int sign;
            sign = ozaki_extract_ieee(a[LIBXS_INDEX(ta, *lda, row, kc)], &e, &mt);
            if (0 != mt) {
              const int delta = (int)row_max_exp - (int)e + oztrim_bits;
              uint8_t tmp[OZ2_NMODULI_MAX];
              oz2_reduce(mt, delta, tmp, nmoduli);
              LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NMODULI_MAX, OZ2_NMODULI_DEFAULT)
              for (pidx = 0; pidx < nmoduli; ++pidx) {
                const oz2_res_t r = OZ2_SIGN_FOLD(sign, tmp[pidx], pidx);
                a_res[(long)pidx * M * K_grp_pad + (long)row * K_grp_pad + (kc - kb_grp)] = r;
                rsum[pidx] += r;
              }
            }
          }
          if (NULL != a_rowsum) {
            for (pidx = 0; pidx < nmoduli; ++pidx) {
              a_rowsum[((size_t)pidx * M + row) * nchunk + (kk - kb_grp) / K_CHUNK] = rsum[pidx];
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
        for (pidx = 0; pidx < nmoduli; ++pidx) {
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
            uint8_t tmp[OZ2_NMODULI_MAX];
            oz2_reduce(mt, delta, tmp, nmoduli);
            LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NMODULI_MAX, OZ2_NMODULI_DEFAULT)
            for (pidx = 0; pidx < nmoduli; ++pidx) {
              b_res[(long)pidx * N * K_grp_pad + (long)col * K_grp_pad + (kk - kb_grp)] = OZ2_SIGN_FOLD(sign, tmp[pidx], pidx);
            }
          }
        }
      } /* implicit barrier: preprocessing done */

#if defined(LIBXS_INTRINSICS_AMX) && defined(LIBXS_INTRINSICS_AVX512) && \
  16 == BLOCK_M && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
      if (0 != use_amx) {
        const GEMM_INT_TYPE N_blocks = LIBXS_UPDIV(N, BLOCK_N);
        const GEMM_INT_TYPE bp_stride = (K_grp_pad / 4) * BLOCK_N;
#if defined(_OPENMP)
#       pragma omp for LIBXS_OPENMP_COLLAPSE(2) OZAKI_OMP_SCHEDULE
#endif
        for (jb = 0; jb < N; jb += BLOCK_N) {
          for (pidx = 0; pidx < nmoduli; ++pidx) {
            const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
            if (BLOCK_N == jblk) {
              const __m512i vidx = OZAKI_GATHER_VIDX(K_grp_pad);
              int32_t* const dst = b_packed + (long)pidx * N_blocks * bp_stride + (long)(jb / BLOCK_N) * bp_stride;
              for (kb = 0; kb < K_grp_pad; kb += BLOCK_K) {
                OZAKI_REFORMAT_B_IMPL(vidx,
                  b_res + (long)pidx * N * K_grp_pad + (long)jb * K_grp_pad,
                  kb, BLOCK_N, dst + (kb / 4) * BLOCK_N, BLOCK_K);
              }
            }
          }
        }
      }
#endif
#if defined(__LIBXSMM)
      if (0 != use_xsmm) {
        const GEMM_INT_TYPE N_blocks = LIBXS_UPDIV(N, BLOCK_N);
        const size_t bp_bytes = (size_t)K_grp_pad * BLOCK_N;
# if defined(_OPENMP)
#       pragma omp for LIBXS_OPENMP_COLLAPSE(2) OZAKI_OMP_SCHEDULE
# endif
        for (jb = 0; jb < N; jb += BLOCK_N) {
          for (pidx = 0; pidx < nmoduli; ++pidx) {
            ozaki_xsmm_pack(xsmm[0].pf, (const char*)(b_res + (long)pidx * N * K_grp_pad + (long)jb * K_grp_pad),
              K_grp_pad, LIBXS_MIN(BLOCK_N, N - jb), K_grp_pad,
              (char*)b_packed + ((size_t)pidx * N_blocks + (size_t)(jb / BLOCK_N)) * bp_bytes);
          }
        }
      }
#endif

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

      /**
       * Phase 4: CRT dot products + accumulate for this K-group.
       * K_CHUNK loop retained for int32 safety when K_GRP > K_CHUNK.
       */
#if defined(LIBXS_INTRINSICS_AMX) && defined(LIBXS_INTRINSICS_AVX512) && \
  16 == BLOCK_M && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
      if (0 != use_amx) oz2_amx_configure();
#endif
#if defined(_OPENMP)
# pragma omp for LIBXS_OPENMP_COLLAPSE(2) OZAKI_OMP_SCHEDULE
#endif
      for (jw = 0; jw < N; jw += jstep) {
        for (ib = 0; ib < M; ib += BLOCK_M) {
          const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
          const GEMM_INT_TYPE jend = LIBXS_MIN(jw + jstep, N);
          GEMM_INT_TYPE jamx = jw; /* columns [jw, jamx) are done by AMX */
#if defined(LIBXS_INTRINSICS_AMX) && defined(LIBXS_INTRINSICS_AVX512) && \
  16 == BLOCK_M && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
          if (0 != use_amx && BLOCK_M == iblk) { /* edge rows stay on the per-tile path: a_res has no row padding */
            const int npanel = (int)((jend - jw) / BLOCK_N);
            if (0 < npanel) {
              const GEMM_INT_TYPE N_blocks = LIBXS_UPDIV(N, BLOCK_N);
              const GEMM_INT_TYPE bp_stride = (K_grp_pad / 4) * BLOCK_N;
              const size_t bmod_stride = (size_t)N_blocks * bp_stride;
              uint8_t amx_res[OZ2_AMX_NACC][BLOCK_M * BLOCK_N][OZ2_NMODULI_MAX];
              int panel;
              memset(amx_res, 0, sizeof(amx_res));
              for (pidx = 0; pidx < nmoduli; ++pidx) {
                for (kb = 0; kb < K_grp_pad; kb += K_CHUNK) {
                  const GEMM_INT_TYPE chunk_k = ((GEMM_INT_TYPE)K_CHUNK < K_grp_pad - kb) ? (GEMM_INT_TYPE)K_CHUNK : (K_grp_pad - kb);
                  oz2_amx_gemm(chunk_k, pidx, npanel,
                    a_res + (long)pidx * M * K_grp_pad + (long)ib * K_grp_pad + kb, K_grp_pad,
                    b_packed + (long)pidx * bmod_stride + (long)(jw / BLOCK_N) * bp_stride + (long)(kb / 4) * BLOCK_N,
                    BLOCK_N, bp_stride, amx_res);
                }
              }
              for (panel = 0; panel < npanel; ++panel) {
                GEMM_REAL_TYPE* const amx_cb = c + (jw + panel * BLOCK_N) * ldcv + ib;
                for (mi = 0; mi < BLOCK_M; ++mi) {
                  unsigned int batch_res[OZ2_BATCH][OZ2_NMODULI_MAX];
                  double batch_val[OZ2_BATCH];
                  int bi;
                  for (bi = 0; bi < OZ2_BATCH; ++bi) {
                    LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NMODULI_MAX, OZ2_NMODULI_DEFAULT)
                    for (pidx = 0; pidx < nmoduli; ++pidx) {
                      batch_res[bi][pidx] = amx_res[panel][mi * BLOCK_N + bi][pidx];
                    }
                  }
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == OZ2_BATCH
# if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
                  oz2_reconstruct_batch_avx512(batch_res, garner_inv, l2_garner_inv, hier_gprod, hier_l2_barrett,
                    nmoduli, OZ2_BATCH, batch_val);
# else
                  if (LIBXS_X86_AVX512 <= ozaki_target_arch) {
                    oz2_reconstruct_batch_avx512(batch_res, garner_inv, l2_garner_inv, hier_gprod, hier_l2_barrett,
                      nmoduli, OZ2_BATCH, batch_val);
                  }
                  else {
                    oz2_reconstruct_batch(batch_res, garner_inv, l2_garner_inv, hier_gprod, hier_l2_barrett,
                      nmoduli, OZ2_BATCH, batch_val);
                  }
# endif
#else
                  oz2_reconstruct_batch(batch_res, garner_inv, l2_garner_inv, hier_gprod, hier_l2_barrett,
                    nmoduli, OZ2_BATCH, batch_val);
#endif
                  for (bi = 0; bi < OZ2_BATCH; ++bi) {
                    if (0.0 != batch_val[bi] && (GEMM_REAL_TYPE)0 != *alpha) {
                      const double contrib = (*alpha) * batch_val[bi] * expa_fp[ib + mi] * expb_fp[jw + panel * BLOCK_N + bi];
                      amx_cb[mi + bi * ldcv] += (GEMM_REAL_TYPE)contrib;
                    }
                  }
                }
              }
            }
            jamx = jw + npanel * BLOCK_N;
          }
#endif
          for (jb = jamx; jb < jend; jb += BLOCK_N) {
            const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
            GEMM_REAL_TYPE* const cb = c + jb * ldcv + ib;
            uint8_t tile_res[BLOCK_M * BLOCK_N][OZ2_NMODULI_MAX];
            memset(tile_res, 0, sizeof(tile_res));

#if defined(__LIBXSMM)
            if (0 != use_xsmm) {
              const size_t bp_bytes = (size_t)K_grp_pad * BLOCK_N;
              const char* const bp_jb = (const char*)b_packed + (size_t)(jb / BLOCK_N) * bp_bytes;
              const size_t bmod_bytes = (size_t)LIBXS_UPDIV(N, BLOCK_N) * bp_bytes;
              LIBXS_ALIGNED(int32_t partial[BLOCK_M * BLOCK_N], LIBXS_ALIGNMENT);
              LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NMODULI_MAX, OZ2_NMODULI_DEFAULT)
              for (pidx = 0; pidx < nmoduli; ++pidx) {
                for (kb = 0; kb < K_grp_pad; kb += K_CHUNK) {
                  const ozaki_xsmm_t* const x = &xsmm[(0 == kb || K_CHUNK <= K_grp_pad - kb) ? 0 : 1];
                  ozaki_xsmm_call(BLOCK_M == iblk ? x->full[0] : x->edge[0],
                    bp_jb + (size_t)pidx * bmod_bytes + (size_t)kb * BLOCK_N,
                    a_res + (long)pidx * M * K_grp_pad + (long)ib * K_grp_pad + kb, partial);
                  for (mi = 0; mi < iblk; ++mi) {
                    for (nj = 0; nj < jblk; ++nj) {
                      unsigned int r = oz2_mod((uint32_t)partial[mi * BLOCK_N + nj], pidx);
                      r += tile_res[mi * jblk + nj][pidx];
                      if (r >= oz2_moduli[pidx]) r -= oz2_moduli[pidx];
                      tile_res[mi * jblk + nj][pidx] = (uint8_t)r;
                    }
                  }
                }
              }
            }
            else
#endif
            /**
             * Fused GEMM + mod-reduce: inline VNNI panel per modulus, Barrett-
             * reduce accumulators in-register, accumulate into tile_res.
             * Eliminates partial[] buffer and per-modulus function call overhead.
             */
#if defined(LIBXS_INTRINSICS_AVX512) && 16 == BLOCK_N && \
    (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH || LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
            if (BLOCK_N == jblk && LIBXS_X86_AVX512 <= ozaki_target_arch) {
              LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NMODULI_MAX, OZ2_NMODULI_DEFAULT)
              for (pidx = 0; pidx < nmoduli; ++pidx) {
                const unsigned int pi = oz2_moduli[pidx];
                const unsigned int rcp_i = oz2_rcp[pidx];
                const __m512i vpi = _mm512_set1_epi32((int)pi);
                const oz2_res_t* const a_prime = a_res + (long)pidx * M * K_grp_pad + (long)ib * K_grp_pad;
                const oz2_res_t* const b_prime = b_res + (long)pidx * N * K_grp_pad + (long)jb * K_grp_pad;
                for (kb = 0; kb < K_grp_pad; kb += K_CHUNK) {
                  const GEMM_INT_TYPE chunk_k = ((GEMM_INT_TYPE)K_CHUNK < K_grp_pad - kb) ? (GEMM_INT_TYPE)K_CHUNK : (K_grp_pad - kb);
                  __m512i acc[BLOCK_M];
                  GEMM_INT_TYPE kk;
                  int biased = 1; /* VPDPBUSD reads B signed: B is XOR-biased, then corrected by +128*row_sum(A) */
                  for (mi = 0; mi < iblk; ++mi) acc[mi] = _mm512_setzero_si512();
#if defined(OZ2_BUUD)
                  if (LIBXS_X86_AVX512_INT8 <= ozaki_target_arch) {
                    oz2_gemm_buud(iblk, chunk_k, a_prime + kb, K_grp_pad, b_prime + kb, K_grp_pad, acc);
                    biased = 0;
                  }
                  else
#endif
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
                    if (0 != biased) {
                      if (NULL != a_rowsum) asum = a_rowsum[((size_t)pidx * M + ib + mi) * nchunk + kb / K_CHUNK];
                      else {
                        for (kk = kb; kk - kb < chunk_k; ++kk) {
                          asum += (int32_t)a_prime[mi * K_grp_pad + kk];
                        }
                      }
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
                }
              }
            }
            else
#endif
            { /* Scalar fallback */
              for (kb = 0; kb < K_grp_pad; kb += K_CHUNK) {
                const GEMM_INT_TYPE chunk_k = ((GEMM_INT_TYPE)K_CHUNK < K_grp_pad - kb) ? (GEMM_INT_TYPE)K_CHUNK : (K_grp_pad - kb);
                LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NMODULI_MAX, OZ2_NMODULI_DEFAULT)
                for (pidx = 0; pidx < nmoduli; ++pidx) {
                  LIBXS_ALIGNED(int32_t partial[BLOCK_M * BLOCK_N], LIBXS_ALIGNMENT);
                  ozaki_gemm_u8u8s32('N', 'T', iblk, jblk, chunk_k,
                    (const uint8_t*)(a_res + (long)pidx * M * K_grp_pad + (long)ib * K_grp_pad + kb), K_grp_pad,
                    (const uint8_t*)(b_res + (long)pidx * N * K_grp_pad + (long)jb * K_grp_pad + kb), K_grp_pad, 0, partial, jblk);
                  for (mi = 0; mi < iblk; ++mi) {
                    for (nj = 0; nj < jblk; ++nj) {
                      const int32_t dot = partial[mi * jblk + nj];
                      unsigned int r = oz2_mod((uint32_t)dot, pidx);
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
                unsigned int batch_res[OZ2_BATCH][OZ2_NMODULI_MAX];
                double batch_val[OZ2_BATCH];
                int bi;
                for (bi = 0; bi < (int)bsz; ++bi) {
                  LIBXS_PRAGMA_LOOP_COUNT(1, OZ2_NMODULI_MAX, OZ2_NMODULI_DEFAULT)
                  for (pidx = 0; pidx < nmoduli; ++pidx) {
                    batch_res[bi][pidx] = tile_res[mi * jblk + nj + bi][pidx];
                  }
                }

#if defined(LIBXS_INTRINSICS_AVX512) && 16 == OZ2_BATCH
# if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
                oz2_reconstruct_batch_avx512(batch_res, garner_inv, l2_garner_inv, hier_gprod, hier_l2_barrett, nmoduli, (int)bsz, batch_val);
# else
                if (LIBXS_X86_AVX512 <= ozaki_target_arch) {
                  oz2_reconstruct_batch_avx512(batch_res, garner_inv, l2_garner_inv, hier_gprod, hier_l2_barrett, nmoduli, (int)bsz, batch_val);
                }
                else {
                  oz2_reconstruct_batch(batch_res, garner_inv, l2_garner_inv, hier_gprod, hier_l2_barrett, nmoduli, (int)bsz, batch_val);
                }
# endif
#else
                oz2_reconstruct_batch(batch_res, garner_inv, l2_garner_inv, hier_gprod, hier_l2_barrett, nmoduli, (int)bsz, batch_val);
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
      }
#if defined(LIBXS_INTRINSICS_AMX) && defined(LIBXS_INTRINSICS_AVX512) && \
  16 == BLOCK_M && 16 == BLOCK_N && (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
      if (0 != use_amx) oz2_amx_release();
#endif
    } /* end K-group loop */

  } /* end parallel */

  /* Reference BLAS and diff comparison (whole-matrix, consistent with GPU path) */
  if (NULL != c_ref) {
    ozaki_diff_reference(GEMM_ARGPASS, c_ref, c_size, diff);
  }
  libxs_free(a_res);
  libxs_free(b_res);
  libxs_free(b_packed);
  libxs_free(a_rowsum);
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
