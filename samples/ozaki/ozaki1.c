/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"


LIBXS_API_INLINE void ozaki_decompose(GEMM_REAL_TYPE value, int16_t* exp_biased, int8_t digits[MAX_NSLICES])
{
  const union { uint32_t raw; float value; } inf = { 0x7F800000U };
  int sign = 1;

  LIBXS_ASSERT(NULL != exp_biased);
  if (value == (GEMM_REAL_TYPE)0 || LIBXS_ISNAN(value)
    || (float)value == inf.value || (float)value == -inf.value)
  {
    if (NULL != digits) memset(digits, 0, sizeof(int8_t) * gemm_ozn);
    *exp_biased = 0;
    return;
  }

  if (value < (GEMM_REAL_TYPE)0) {
    value = -value;
    sign = -1;
  }

#if GEMM_IS_DOUBLE
  { union { double d; uint64_t u; } cvt;
    cvt.d = value;
    { const uint64_t bits = cvt.u;
      const uint64_t frac = bits & ((1ULL << 52) - 1ULL);
      const uint16_t exp_raw = (uint16_t)((bits >> 52) & 0x7FFU);

      if (0 == exp_raw) { /* subnormal treated as zero here */
        *exp_biased = 0;
        if (NULL != digits) memset(digits, 0, sizeof(int8_t) * gemm_ozn);
        return;
      }

      { const uint64_t mant_full = (1ULL << 52) | frac; /* 53 bits */
        int s = 0;
        *exp_biased = (int16_t)exp_raw;

        if (NULL != digits) {
          LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
          for (; s < gemm_ozn; ++s) {
            const int high = 52 - (7 * s);
            if (high < 0) {
              digits[s] = 0;
              continue;
            }
            { const int low = high - 6;
              uint64_t chunk;
              if (low >= 0) {
                chunk = (mant_full >> low) & 0x7FULL;
              }
              else {
                const int width = high + 1; /* 0..6 */
                chunk = mant_full & ((1ULL << width) - 1ULL);
              }
              digits[s] = (int8_t)(sign * (int64_t)chunk);
            }
          }
        }
      }
    }
  }
#else /* single-precision */
  { union { float f; uint32_t u; } cvt;
    cvt.f = value;
    { const uint32_t bits = cvt.u;
      const uint32_t frac = bits & ((1U << 23) - 1U);
      const uint16_t exp_raw = (uint16_t)((bits >> 23) & 0xFFU);

      if (0 == exp_raw) { /* subnormal treated as zero here */
        *exp_biased = 0;
        if (NULL != digits) memset(digits, 0, sizeof(int8_t) * gemm_ozn);
        return;
      }

      { const uint32_t mant_full = (1U << 23) | frac; /* 24 bits */
        int s = 0;
        *exp_biased = (int16_t)exp_raw;

        if (NULL != digits) {
          LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
          for (; s < gemm_ozn; ++s) {
            const int high = 23 - (7 * s);
            if (high < 0) {
              digits[s] = 0;
              continue;
            }
            { const int low = high - 6;
              uint32_t chunk;
              if (low >= 0) {
                chunk = (mant_full >> low) & 0x7FU;
              }
              else {
                const int width = high + 1; /* 0..6 */
                chunk = mant_full & ((1U << width) - 1U);
              }
              digits[s] = (int8_t)(sign * (int32_t)chunk);
            }
          }
        }
      }
    }
  }
#endif
}


LIBXS_API_INLINE void rescale_digits(int8_t dst[MAX_NSLICES], const int8_t src[MAX_NSLICES], int delta)
{
  int i;

  if (delta == 0) {
    if (dst != src) memcpy(dst, src, sizeof(int8_t) * gemm_ozn);
    return;
  }

  if (delta >= 7) {
    LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
    for (i = 0; i < gemm_ozn; ++i) dst[i] = (src[i] >= 0) ? INT8_MAX : INT8_MIN;
    return;
  }

  if (delta <= -15) {
    memset(dst, 0, sizeof(int8_t) * gemm_ozn);
    return;
  }

  if (delta > 0) {
    const int sh = delta;
    LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
    for (i = 0; i < gemm_ozn; ++i) {
      const int32_t v = ((int32_t)src[i]) << sh;
      dst[i] = (int8_t)LIBXS_CLMP(v, INT8_MIN, INT8_MAX);
    }
  }
  else {
    const int sh = -delta;
    LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
    for (i = 0; i < gemm_ozn; ++i) {
      const int32_t v = ((int32_t)src[i]) >> sh;
      dst[i] = (int8_t)LIBXS_CLMP(v, INT8_MIN, INT8_MAX);
    }
  }
}


LIBXS_API_INLINE double reconstruct_from_digits(const int8_t digits[MAX_NSLICES],
  int exp_base, const int8_t slice_low_bit[MAX_NSLICES])
{
  double recon = 0.0;
  int slice = 0;

  for (; slice < gemm_ozn; ++slice) {
    const int16_t digit = (int16_t)digits[slice];
    if (0 != digit) {
      int sh = exp_base + slice_low_bit[slice];
      sh = LIBXS_CLMP(sh, -60, 60);
      if (sh >= 0) {
        recon += (double)digit * (double)(1ULL << sh);
      }
      else {
        recon += (double)digit / (double)(1ULL << (-sh));
      }
    }
  }

  return recon;
}


LIBXS_API_INLINE void preprocess_rows(const GEMM_REAL_TYPE* a, GEMM_INT_TYPE lda, int ta,
  GEMM_INT_TYPE M, GEMM_INT_TYPE K, GEMM_INT_TYPE ib, GEMM_INT_TYPE kb, GEMM_INT_TYPE iblk,
  GEMM_INT_TYPE kblk, int16_t expa_row[BLOCK_M], int8_t am[BLOCK_M][BLOCK_K][MAX_NSLICES])
{
  int16_t elem_exp[BLOCK_M][BLOCK_K];
  GEMM_INT_TYPE mi, kk;

  /* Single pass: decompose and track per-row max exponent */
  for (mi = 0; mi < iblk; ++mi) {
    const GEMM_INT_TYPE row = ib + mi;
    int16_t row_max_exp = INT16_MIN;

    for (kk = 0; kk < kblk; ++kk) {
      const GEMM_INT_TYPE p = kb + kk;
      const GEMM_REAL_TYPE aval = ((row < M && p < K)
        ? a[LIBXS_INDEX(ta, lda, row, p)] : (GEMM_REAL_TYPE)0);
      ozaki_decompose(aval, &elem_exp[mi][kk], am[mi][kk]);
      row_max_exp = LIBXS_MAX(row_max_exp, elem_exp[mi][kk]);
    }

    expa_row[mi] = row_max_exp;

    /* Rescale digits to align with row's max exponent */
    for (kk = 0; kk < kblk; ++kk) {
      const int delta = (int)elem_exp[mi][kk] - (int)row_max_exp;
      if (0 != delta) rescale_digits(am[mi][kk], am[mi][kk], delta);
    }
    /* Zero-pad remaining k-entries for fixed-length dot products */
    for (kk = kblk; kk < BLOCK_K; ++kk) {
      memset(am[mi][kk], 0, sizeof(int8_t) * gemm_ozn);
    }
  }
}


LIBXS_API_INLINE void preprocess_cols(const GEMM_REAL_TYPE* b, GEMM_INT_TYPE ldb, int tb,
  GEMM_INT_TYPE N, GEMM_INT_TYPE K, GEMM_INT_TYPE jb, GEMM_INT_TYPE kb, GEMM_INT_TYPE jblk,
  GEMM_INT_TYPE kblk, int16_t expb_col[BLOCK_N], int8_t bm[BLOCK_K][BLOCK_N][MAX_NSLICES])
{
  int16_t elem_exp[BLOCK_K][BLOCK_N];
  GEMM_INT_TYPE nj, kk;

  for (nj = 0; nj < jblk; ++nj) {
    expb_col[nj] = INT16_MIN;
  }

  /* Single pass: decompose and track per-column max exponent */
  for (kk = 0; kk < kblk; ++kk) {
    const GEMM_INT_TYPE p = kb + kk;
    for (nj = 0; nj < jblk; ++nj) {
      const GEMM_INT_TYPE col = jb + nj;
      const GEMM_REAL_TYPE bval = ((p < K && col < N)
        ? b[LIBXS_INDEX(tb, ldb, p, col)] : (GEMM_REAL_TYPE)0);
      ozaki_decompose(bval, &elem_exp[kk][nj], bm[kk][nj]);
      expb_col[nj] = LIBXS_MAX(expb_col[nj], elem_exp[kk][nj]);
    }
  }

  /* Rescale digits to align with column's max exponent */
  for (kk = 0; kk < kblk; ++kk) {
    for (nj = 0; nj < jblk; ++nj) {
      const int delta = (int)elem_exp[kk][nj] - (int)expb_col[nj];
      if (0 != delta) rescale_digits(bm[kk][nj], bm[kk][nj], delta);
    }
  }
  /* Zero-pad remaining k-entries for fixed-length dot products */
  for (kk = kblk; kk < BLOCK_K; ++kk) {
    for (nj = 0; nj < jblk; ++nj) {
      memset(bm[kk][nj], 0, sizeof(int8_t) * gemm_ozn);
    }
  }
}


/* AVX-512 VNNI accelerated dot product (BLOCK_K=16 fits one __m128i).
 * VPDPBUSD is unsigned*signed, so we convert a[] from signed to unsigned
 * by XOR with 0x80 (+128), then subtract 128*sum(b[]) to compensate. */
#if defined(LIBXS_INTRINSICS_AVX512)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int32_t ozaki_dot_i8_vnni(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  const __m128i bias = _mm_set1_epi8((char)0x80);
  const __m128i va = _mm_xor_si128(_mm_loadu_si128((const __m128i*)a), bias);
  const __m128i vb = _mm_loadu_si128((const __m128i*)b);
  const __m128i ones = _mm_set1_epi8(1);
  /* dot = sum(((uint8_t)a[k]+128) * b[k]) for k in groups of 4 */
  __m128i dp = _mm_dpbusd_epi32(_mm_setzero_si128(), va, vb);
  /* sum_b = sum(b[k]) via dpbusd with all-ones as the unsigned operand */
  __m128i sb = _mm_dpbusd_epi32(_mm_setzero_si128(), ones, vb);
  /* horizontal sum of 4 dwords */
  dp = _mm_hadd_epi32(dp, sb);
  dp = _mm_hadd_epi32(dp, dp);
  return _mm_extract_epi32(dp, 0) - 128 * _mm_extract_epi32(dp, 1);
}
#endif


LIBXS_API_INLINE int32_t ozaki_dot_i8_sw(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  int32_t dot = 0;
  int kk;
  for (kk = 0; kk < BLOCK_K; ++kk) {
    dot += (int32_t)a[kk] * (int32_t)b[kk];
  }
  return dot;
}


/* Runtime dispatch: use VNNI when AVX-512 is detected, else scalar. */
#if defined(LIBXS_INTRINSICS_AVX512)
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
  { int8_t am[BLOCK_M][BLOCK_K][MAX_NSLICES], bm[BLOCK_K][BLOCK_N][MAX_NSLICES];
    int8_t ak[BLOCK_M][MAX_NSLICES][BLOCK_K]; /* k-contiguous for dot product */
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
        GEMM_REAL_TYPE *const mb = c + jb * ldcv + ib;

        ozaki_scale_block_beta(mb, ldcv, iblk, jblk, beta, ref_blk,
          (NULL != diff && 0 == (diff_abc % 3)));

        for (kb = 0; kb < K; kb += BLOCK_K) {
          const GEMM_INT_TYPE kblk = LIBXS_MIN(BLOCK_K, K - kb);

          preprocess_rows(a, *lda, ta, M, K, ib, kb, iblk, kblk, expa_row, am);
          preprocess_cols(b, *ldb, tb, N, K, jb, kb, jblk, kblk, expb_col, bm);

          /* Track differences between original A block and reconstructed digits */
          if (NULL != diff && 1 == (diff_abc % 3)) {
            for (mi = 0; mi < iblk; ++mi) {
              const GEMM_INT_TYPE row = ib + mi;
              for (kk = 0; kk < kblk; ++kk) {
                const GEMM_INT_TYPE p = kb + kk;
                const GEMM_REAL_TYPE aval = ((row < M && p < K) ?
                  a[LIBXS_INDEX(ta, *lda, row, p)] : (GEMM_REAL_TYPE)0);
                const int exp_base = (int)expa_row[mi] - OZ_BIAS_PLUS_MANT;
                const double arecon = reconstruct_from_digits(am[mi][kk],
                  exp_base, slice_low_bit);

                ozaki_store_block_pair(ref_blk, recon_blk, BLOCK_M, mi, kk,
                  (GEMM_REAL_TYPE)aval, (GEMM_REAL_TYPE)arecon);
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
                const double brecon = reconstruct_from_digits(bm[kk][nj],
                  exp_base, slice_low_bit);

                ozaki_store_block_pair(ref_blk, recon_blk, BLOCK_K, kk, nj,
                  (GEMM_REAL_TYPE)bval, (GEMM_REAL_TYPE)brecon);
              }
            }
            ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, recon_blk, kblk, jblk, BLOCK_K, BLOCK_K);
          }

          /* Transpose am/bm into k-contiguous layout for dot product */
          for (mi = 0; mi < iblk; ++mi) {
            LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
            for (slice_a = 0; slice_a < nslices; ++slice_a) {
              for (kk = 0; kk < BLOCK_K; ++kk) {
                ak[mi][slice_a][kk] = am[mi][kk][slice_a];
              }
            }
          }
          for (nj = 0; nj < jblk; ++nj) {
            LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
            for (slice_b = 0; slice_b < nslices; ++slice_b) {
              for (kk = 0; kk < BLOCK_K; ++kk) {
                bk[nj][slice_b][kk] = bm[kk][nj][slice_b];
              }
            }
          }

          LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
          for (slice_a = 0; slice_a < ((0 != (gemm_ozflags & OZ1_TRIM_FORWARD)) ? nslices / 2 : nslices); ++slice_a) {
            slice_b = (0 != (gemm_ozflags & OZ1_TRIANGULAR)) ? slice_a : 0;
            for (; slice_b < nslices; ++slice_b) {
              const int low_bit_sum = (int)slice_low_bit[slice_a] + slice_low_bit[slice_b];
              /* Double off-diagonal terms whose mirror (sb,sa) is not explicitly
               * computed. When REVERSE_PASS is also active, the mirror IS
               * recovered when: sb >= S/2 && sa <= S-1-sb, so skip doubling. */
              const double sym_alpha = (0 != (gemm_ozflags & OZ1_SYMMETRIZE))
                ? (*alpha) * ((slice_a != slice_b
                    && !(0 != (gemm_ozflags & OZ1_REVERSE_PASS)
                      && slice_b >= nslices / 2 && slice_a <= nslices - 1 - slice_b)
                    ) ? 2.0 : 1.0)
                : (*alpha);
              for (mi = 0; mi < iblk; ++mi) {
                for (nj = 0; nj < jblk; ++nj) {
                  const int32_t dot = ozaki_dot_i8(ak[mi][slice_a], bk[nj][slice_b]);
                  if (0 != dot) {
                    int sh = (int)expa_row[mi] + (int)expb_col[nj] - (2 * OZ_BIAS_PLUS_MANT) + low_bit_sum;
                    double contrib = sym_alpha * (double)dot;
                    sh = LIBXS_CLMP(sh, -60, 60);
                    if (sh >= 0) {
                      contrib *= (double)(1ULL << sh);
                    }
                    else {
                      contrib /= (double)(1ULL << (-sh));
                    }
                    mb[mi + nj * ldcv] += (GEMM_REAL_TYPE)contrib;
                  }
                }
              }
            }
          }
          if (0 != (gemm_ozflags & OZ1_REVERSE_PASS)) {
          /* Reverse pass: explicitly recover the most significant lower-triangle
           * terms (slice_a >= S/2, slice_b from S-1-slice_a downward). These are
           * the dropped terms with the largest exponents (small slice_b index
           * paired with large slice_a index). */
          LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
          for (slice_a = nslices / 2; slice_a < nslices; ++slice_a) {
            for (slice_b = nslices - 1 - slice_a; slice_b >= 0; --slice_b) {
              const int low_bit_sum = (int)slice_low_bit[slice_a] + slice_low_bit[slice_b];
              for (mi = 0; mi < iblk; ++mi) {
                for (nj = 0; nj < jblk; ++nj) {
                  const int32_t dot = ozaki_dot_i8(ak[mi][slice_a], bk[nj][slice_b]);
                  if (0 != dot) {
                    int sh = (int)expa_row[mi] + (int)expb_col[nj] - (2 * OZ_BIAS_PLUS_MANT) + low_bit_sum;
                    double contrib = (*alpha) * (double)dot;
                    sh = LIBXS_CLMP(sh, -60, 60);
                    if (sh >= 0) {
                      contrib *= (double)(1ULL << sh);
                    }
                    else {
                      contrib /= (double)(1ULL << (-sh));
                    }
                    mb[mi + nj * ldcv] += (GEMM_REAL_TYPE)contrib;
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
            ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, mb, iblk, jblk, mref, ldcv);
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
