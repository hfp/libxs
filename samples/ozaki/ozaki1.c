/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"



LIBXS_API_INLINE double reconstruct_from_digits(const int8_t digits[MAX_NSLICES],
  int exp_base, const int8_t slice_low_bit[MAX_NSLICES])
{
  double recon = 0.0;
  int slice = 0;

  for (; slice < gemm_ozn; ++slice) {
    const int16_t digit = (int16_t)digits[slice];
    if (0 != digit) {
      const int sh = exp_base + slice_low_bit[slice];
      recon += (double)digit * libxs_pow2(sh);
    }
  }

  return recon;
}


/** Preprocess rows of A: decompose, align, split into digits, and write
 *  directly into the k-contiguous layout ak[M][S][K] used by dot products.
 *  This avoids a separate transpose pass over an intermediate buffer. */
LIBXS_API_INLINE void preprocess_rows(const GEMM_REAL_TYPE* a, GEMM_INT_TYPE lda, int ta,
  GEMM_INT_TYPE M, GEMM_INT_TYPE K, GEMM_INT_TYPE ib, GEMM_INT_TYPE kb, GEMM_INT_TYPE iblk,
  GEMM_INT_TYPE kblk, int16_t expa_row[BLOCK_M], int8_t ak[BLOCK_M][MAX_NSLICES][BLOCK_K])
{
  int16_t elem_exp[BLOCK_M][BLOCK_K];
  uint64_t elem_mant[BLOCK_M][BLOCK_K];
  int elem_sign[BLOCK_M][BLOCK_K];
  GEMM_INT_TYPE mi, kk;

  /* Pass 1: extract sign, exponent, mantissa and track per-row max exponent */
  for (mi = 0; mi < iblk; ++mi) {
    const GEMM_INT_TYPE row = ib + mi;
    int16_t row_max_exp = INT16_MIN;

    for (kk = 0; kk < kblk; ++kk) {
      const GEMM_INT_TYPE p = kb + kk;
      const GEMM_REAL_TYPE aval = ((row < M && p < K)
        ? a[LIBXS_INDEX(ta, lda, row, p)] : (GEMM_REAL_TYPE)0);
      elem_sign[mi][kk] = ozaki_extract_ieee(aval, &elem_exp[mi][kk], &elem_mant[mi][kk]);
      row_max_exp = LIBXS_MAX(row_max_exp, elem_exp[mi][kk]);
    }

    expa_row[mi] = row_max_exp;

    /* Pass 2: align mantissa, split into digits, scatter into ak[mi][s][kk] */
    for (kk = 0; kk < kblk; ++kk) {
      const int delta = (int)row_max_exp - (int)elem_exp[mi][kk];
      uint64_t aligned = elem_mant[mi][kk];
      int8_t tmp[MAX_NSLICES];
      int s;
      if (delta > 0) {
        aligned = (delta < 64) ? (aligned >> delta) : 0;
      }
      ozaki_split_digits(aligned, elem_sign[mi][kk], tmp);
      LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
      for (s = 0; s < gemm_ozn; ++s) ak[mi][s][kk] = tmp[s];
    }
    /* Zero-pad remaining k-entries for fixed-length dot products */
    { int s;
      LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
      for (s = 0; s < gemm_ozn; ++s) {
        for (kk = kblk; kk < BLOCK_K; ++kk) ak[mi][s][kk] = 0;
      }
    }
  }
}


/** Preprocess columns of B: decompose, align, split into digits, and write
 *  directly into the k-contiguous layout bk[N][S][K] used by dot products.
 *  This avoids a separate transpose pass over an intermediate buffer. */
LIBXS_API_INLINE void preprocess_cols(const GEMM_REAL_TYPE* b, GEMM_INT_TYPE ldb, int tb,
  GEMM_INT_TYPE N, GEMM_INT_TYPE K, GEMM_INT_TYPE jb, GEMM_INT_TYPE kb, GEMM_INT_TYPE jblk,
  GEMM_INT_TYPE kblk, int16_t expb_col[BLOCK_N], int8_t bk[BLOCK_N][MAX_NSLICES][BLOCK_K])
{
  int16_t elem_exp[BLOCK_K][BLOCK_N];
  uint64_t elem_mant[BLOCK_K][BLOCK_N];
  int elem_sign[BLOCK_K][BLOCK_N];
  GEMM_INT_TYPE nj, kk;

  for (nj = 0; nj < jblk; ++nj) {
    expb_col[nj] = INT16_MIN;
  }

  /* Pass 1: extract sign, exponent, mantissa and track per-column max exponent */
  for (kk = 0; kk < kblk; ++kk) {
    const GEMM_INT_TYPE p = kb + kk;
    for (nj = 0; nj < jblk; ++nj) {
      const GEMM_INT_TYPE col = jb + nj;
      const GEMM_REAL_TYPE bval = ((p < K && col < N)
        ? b[LIBXS_INDEX(tb, ldb, p, col)] : (GEMM_REAL_TYPE)0);
      elem_sign[kk][nj] = ozaki_extract_ieee(bval, &elem_exp[kk][nj], &elem_mant[kk][nj]);
      expb_col[nj] = LIBXS_MAX(expb_col[nj], elem_exp[kk][nj]);
    }
  }

  /* Pass 2: align mantissa, split into digits, scatter into bk[nj][s][kk] */
  for (kk = 0; kk < kblk; ++kk) {
    for (nj = 0; nj < jblk; ++nj) {
      const int delta = (int)expb_col[nj] - (int)elem_exp[kk][nj];
      uint64_t aligned = elem_mant[kk][nj];
      int8_t tmp[MAX_NSLICES];
      int s;
      if (delta > 0) {
        aligned = (delta < 64) ? (aligned >> delta) : 0;
      }
      ozaki_split_digits(aligned, elem_sign[kk][nj], tmp);
      LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
      for (s = 0; s < gemm_ozn; ++s) bk[nj][s][kk] = tmp[s];
    }
  }
  /* Zero-pad remaining k-entries for fixed-length dot products */
  { int s;
    for (nj = 0; nj < jblk; ++nj) {
      LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
      for (s = 0; s < gemm_ozn; ++s) {
        for (kk = kblk; kk < BLOCK_K; ++kk) bk[nj][s][kk] = 0;
      }
    }
  }
}


/* VPDPBUSD (AVX-512 VNNI) accelerated int8 dot product.
 * The instruction is unsigned*signed, so we convert a[] from signed to
 * unsigned by XOR with 0x80 (+128), then subtract 128*sum(b[]) to
 * compensate.  AVX-512 VNNI exposes VPDPBUSD at all three register
 * widths (XMM/128, YMM/256, ZMM/512); we select the narrowest that
 * covers BLOCK_K bytes, falling back to scalar for other sizes. */
#if defined(LIBXS_INTRINSICS_AVX512)

# if 16 == BLOCK_K /* 128-bit: one __m128i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m128i bias = _mm_set1_epi8((char)0x80);
  const __m128i va = _mm_xor_si128(_mm_loadu_si128((const __m128i*)a), bias);
  const __m128i vb = _mm_loadu_si128((const __m128i*)b);
  const __m128i ones = _mm_set1_epi8(1);
  __m128i dp = _mm_dpbusd_epi32(_mm_setzero_si128(), va, vb);
  __m128i sb = _mm_dpbusd_epi32(_mm_setzero_si128(), ones, vb);
  dp = _mm_hadd_epi32(dp, sb);
  dp = _mm_hadd_epi32(dp, dp);
  return _mm_extract_epi32(dp, 0) - 128 * _mm_extract_epi32(dp, 1);
}

# elif 32 == BLOCK_K /* 256-bit: one __m256i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m256i bias = _mm256_set1_epi8((char)0x80);
  const __m256i va = _mm256_xor_si256(
    _mm256_loadu_si256((const __m256i*)a), bias);
  const __m256i vb = _mm256_loadu_si256((const __m256i*)b);
  const __m256i ones = _mm256_set1_epi8(1);
  __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), va, vb);
  __m256i sb = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, vb);
  /* reduce 8 dwords → 4 → 2 → 1 */
  { const __m128i hi_dp = _mm256_extracti128_si256(dp, 1);
    const __m128i hi_sb = _mm256_extracti128_si256(sb, 1);
    __m128i lo_dp = _mm256_castsi256_si128(dp);
    __m128i lo_sb = _mm256_castsi256_si128(sb);
    lo_dp = _mm_add_epi32(lo_dp, hi_dp);
    lo_sb = _mm_add_epi32(lo_sb, hi_sb);
    lo_dp = _mm_hadd_epi32(lo_dp, lo_sb);
    lo_dp = _mm_hadd_epi32(lo_dp, lo_dp);
    return _mm_extract_epi32(lo_dp, 0) - 128 * _mm_extract_epi32(lo_dp, 1);
  }
}

# elif 64 == BLOCK_K /* 512-bit: one __m512i */

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m512i bias = _mm512_set1_epi8((char)0x80);
  const __m512i va = _mm512_xor_si512(
    _mm512_loadu_si512((const __m512i*)a), bias);
  const __m512i vb = _mm512_loadu_si512((const __m512i*)b);
  const __m512i ones = _mm512_set1_epi8(1);
  __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), va, vb);
  __m512i sb = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, vb);
  /* reduce 16 dwords → scalar via _mm512_reduce_add_epi32 */
  return _mm512_reduce_add_epi32(dp)
       - 128 * _mm512_reduce_add_epi32(sb);
}

# endif /* BLOCK_K width selection */
#endif /* LIBXS_INTRINSICS_AVX512 */


LIBXS_API_INLINE int32_t ozaki_dot_i8_sw(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  int32_t dot = 0;
  int kk;
  for (kk = 0; kk < BLOCK_K; ++kk) {
    dot += (int32_t)a[kk] * (int32_t)b[kk];
  }
  return dot;
}


/* Runtime dispatch: use VNNI when AVX-512 is detected and
 * BLOCK_K matches a supported register width, else scalar. */
#if defined(LIBXS_INTRINSICS_AVX512) && \
    (16 == BLOCK_K || 32 == BLOCK_K || 64 == BLOCK_K)
# define ozaki_dot_i8(A, B) \
    ((LIBXS_X86_AVX512 <= ozaki_target_arch) ? ozaki_dot_i8_vnni(A, B) : ozaki_dot_i8_sw(A, B))
#else
# define ozaki_dot_i8(A, B) ozaki_dot_i8_sw(A, B)
#endif


LIBXS_API_INLINE void gemm_oz1_diff(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
  unsigned int diff_abc, libxs_matdiff_info_t* diff)
{
  int8_t slice_low_bit[MAX_NSLICES];
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
  const int nslices = LIBXS_CLMP(gemm_ozn, 1, MAX_NSLICES);
  libxs_matdiff_info_t tdiff[256];
  int nthreads = 1;
  int s;
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE)
            || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));

  LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
  for (s = 0; s < nslices; ++s) {
    const int high = OZ_MANT_BITS - (7 * s);
    const int low = (high >= 0) ? (high - 6) : 0;
    slice_low_bit[s] = (low > 0 ? low : 0);
  }

#if defined(_OPENMP)
# pragma omp parallel
#endif
  { int8_t ak[BLOCK_M][MAX_NSLICES][BLOCK_K]; /* k-contiguous for dot product */
    int8_t bk[BLOCK_N][MAX_NSLICES][BLOCK_K]; /* k-contiguous for dot product */
    int16_t expa_row[BLOCK_M], expb_col[BLOCK_N];
    GEMM_REAL_TYPE ref_blk[BLOCK_MNK], recon_blk[BLOCK_MNK];
    GEMM_INT_TYPE kb, mi, nj, kk, jb, ib;
    int slice_a, slice_b;
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
        GEMM_REAL_TYPE *const cb = c + jb * ldcv + ib;

        ozaki_scale_block_beta(cb, ldcv, iblk, jblk, beta, ref_blk,
          (NULL != diff && 0 == (diff_abc % 3)));

        for (kb = 0; kb < K; kb += BLOCK_K) {
          const GEMM_INT_TYPE kblk = LIBXS_MIN(BLOCK_K, K - kb);

          preprocess_rows(a, *lda, ta, M, K, ib, kb, iblk, kblk, expa_row, ak);
          preprocess_cols(b, *ldb, tb, N, K, jb, kb, jblk, kblk, expb_col, bk);

          /* Track differences between original A block and reconstructed digits */
          if (NULL != diff && 1 == (diff_abc % 3)) {
            for (mi = 0; mi < iblk; ++mi) {
              const GEMM_INT_TYPE row = ib + mi;
              for (kk = 0; kk < kblk; ++kk) {
                const GEMM_INT_TYPE p = kb + kk;
                const GEMM_REAL_TYPE aval = ((row < M && p < K) ?
                  a[LIBXS_INDEX(ta, *lda, row, p)] : (GEMM_REAL_TYPE)0);
                const int exp_base = (int)expa_row[mi] - OZ_BIAS_PLUS_MANT;
                int8_t tmp[MAX_NSLICES];
                int si;
                for (si = 0; si < gemm_ozn; ++si) tmp[si] = ak[mi][si][kk];
                { const double arecon = reconstruct_from_digits(tmp,
                    exp_base, slice_low_bit);
                  ozaki_store_block_pair(ref_blk, recon_blk, BLOCK_M, mi, kk,
                    (GEMM_REAL_TYPE)aval, (GEMM_REAL_TYPE)arecon);
                }
              }
            }
            ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, recon_blk, iblk, kblk, BLOCK_M, BLOCK_M);
          }

          /* Track differences between original B block and reconstructed digits */
          if (NULL != diff && 2 == (diff_abc % 3)) {
            for (kk = 0; kk < kblk; ++kk) {
              const GEMM_INT_TYPE p = kb + kk;
              for (nj = 0; nj < jblk; ++nj) {
                const GEMM_INT_TYPE col = jb + nj;
                const GEMM_REAL_TYPE bval = ((p < K && col < N)
                  ? b[LIBXS_INDEX(tb, *ldb, p, col)] : (GEMM_REAL_TYPE)0);
                const int exp_base = (int)expb_col[nj] - OZ_BIAS_PLUS_MANT;
                int8_t tmp[MAX_NSLICES];
                int si;
                for (si = 0; si < gemm_ozn; ++si) tmp[si] = bk[nj][si][kk];
                { const double brecon = reconstruct_from_digits(tmp,
                    exp_base, slice_low_bit);
                  ozaki_store_block_pair(ref_blk, recon_blk, BLOCK_K, kk, nj,
                    (GEMM_REAL_TYPE)bval, (GEMM_REAL_TYPE)brecon);
                }
              }
            }
            ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, recon_blk, kblk, jblk, BLOCK_K, BLOCK_K);
          }

          { /* Diagonal-trim loop: iterate pairs (sa,sb) with sa+sb <= cutoff.
             * trim=0 means all pairs (exact); larger values drop the least
             * significant diagonals (~7 bits each). */
            const int cutoff = LIBXS_MAX(0, 2 * (nslices - 1) - gemm_oztrim);
            LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
            for (slice_a = 0; slice_a < nslices && slice_a <= cutoff; ++slice_a) {
              const int sb_start = (0 != (gemm_ozflags & OZ1_TRIANGULAR))
                ? slice_a : 0;
              const int sb_end = LIBXS_MIN(nslices, cutoff + 1 - slice_a);
              for (slice_b = sb_start; slice_b < sb_end; ++slice_b) {
                const int low_bit_sum = (int)slice_low_bit[slice_a]
                  + slice_low_bit[slice_b];
                /* When SYMMETRIZE is active and TRIANGULAR drops the mirror
                 * pair (sb,sa), compute both D(sa,sb) and D(sb,sa) in the
                 * same iteration. Both share the same power-of-two shift
                 * (low_bit_sum is commutative). */
                const int do_mirror = (0 != (gemm_ozflags & OZ1_SYMMETRIZE))
                  && (slice_a != slice_b);
                for (mi = 0; mi < iblk; ++mi) {
                  for (nj = 0; nj < jblk; ++nj) {
                    int32_t dot = ozaki_dot_i8(ak[mi][slice_a], bk[nj][slice_b]);
                    if (do_mirror) {
                      dot += ozaki_dot_i8(ak[mi][slice_b], bk[nj][slice_a]);
                    }
                    if (0 != dot && (GEMM_REAL_TYPE)0 != *alpha) {
                      const int sh = (int)expa_row[mi] + (int)expb_col[nj]
                        - (2 * OZ_BIAS_PLUS_MANT) + low_bit_sum;
                      const double contrib = (*alpha) * (double)dot * libxs_pow2(sh);
                      cb[mi + nj * ldcv] += (GEMM_REAL_TYPE)contrib;
                    }
                  }
                }
              }
            }
          }
        }

        /* compute reference GEMM on the saved block and accumulate diff */
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
            ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, cb, iblk, jblk, mref, ldcv);
          }
        }
      }
    } /* end parallel for */
  } /* end parallel */
  if (NULL != diff) {
    for (s = 0; s < nthreads; ++s) {
      libxs_matdiff_reduce(diff, &tdiff[s]);
    }
  }
}


LIBXS_API void gemm_oz1(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  OZAKI_GEMM_WRAPPER(gemm_oz1_diff)
}
