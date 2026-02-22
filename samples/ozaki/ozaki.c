/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"
#include <libxs_sync.h>
#include <libxs_mhd.h>

/* Runtime flag-set controlling the Ozaki scheme (GEMM_OZ1 env var).
 * Bit 0 (1): TRIANGULAR  - drop symmetric contributions (speed for accuracy)
 * Bit 1 (2): SYMMETRIZE  - double off-diagonal upper-triangle terms
 * Bit 2 (4): REVERSE_PASS - recover most significant lower-triangle terms
 * Bit 3 (8): TRIM_FORWARD - limit forward pass to slice_a < S/2
 * Default 15 = all enabled. */
#define OZ1_TRIANGULAR   1
#define OZ1_SYMMETRIZE   2
#define OZ1_REVERSE_PASS 4
#define OZ1_TRIM_FORWARD 8
#define OZ1_DEFAULT (OZ1_TRIANGULAR | OZ1_SYMMETRIZE | OZ1_REVERSE_PASS | OZ1_TRIM_FORWARD)
/* IEEE-754 format parameters derived from GEMM_REAL_TYPE */
#if GEMM_IS_DOUBLE
# define OZ1_MANT_BITS  52
# define OZ1_EXP_BIAS   1023
#else /* single-precision */
# define OZ1_MANT_BITS  23
# define OZ1_EXP_BIAS   127
#endif
#define OZ1_BIAS_PLUS_MANT (OZ1_EXP_BIAS + OZ1_MANT_BITS)
#if !defined(BLOCK_M)
# define BLOCK_M 16
#endif
#if !defined(BLOCK_N)
# define BLOCK_N 16
#endif
#if !defined(BLOCK_K)
# define BLOCK_K 16
#endif
#if !defined(MAX_NSLICES)
# if GEMM_IS_DOUBLE
#   define MAX_NSLICES 16
# else
#   define MAX_NSLICES 8
# endif
#endif
#if !defined(NSLICES_DEFAULT)
# if GEMM_IS_DOUBLE
#   define NSLICES_DEFAULT 8
# else
#   define NSLICES_DEFAULT 4
# endif
#endif


LIBXS_APIVAR_PUBLIC_DEF(libxs_matdiff_info_t gemm_diff);
LIBXS_APIVAR_PUBLIC_DEF(int gemm_verbose);

LIBXS_APIVAR_PRIVATE_DEF(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
LIBXS_APIVAR_PRIVATE_DEF(gemm_function_t gemm_original);
LIBXS_APIVAR_PRIVATE_DEF(int gemm_oz1_nslices);
LIBXS_APIVAR_PRIVATE_DEF(int gemm_oz1_flags);
LIBXS_APIVAR_PRIVATE_DEF(int gemm_diff_abc);
LIBXS_APIVAR_PRIVATE_DEF(double gemm_eps);
LIBXS_APIVAR_PRIVATE_DEF(double gemm_rsq);


LIBXS_API_INLINE void ozaki_decompose(GEMM_REAL_TYPE value, int16_t* exp_biased, int8_t digits[MAX_NSLICES])
{
  const union { uint32_t raw; float value; } inf = { 0x7F800000U };
  int sign = 1;

  LIBXS_ASSERT(NULL != exp_biased);
  if (value == (GEMM_REAL_TYPE)0 || LIBXS_ISNAN(value)
    || (float)value == inf.value || (float)value == -inf.value)
  {
    if (NULL != digits) memset(digits, 0, sizeof(int8_t) * gemm_oz1_nslices);
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
        if (NULL != digits) memset(digits, 0, sizeof(int8_t) * gemm_oz1_nslices);
        return;
      }

      { const uint64_t mant_full = (1ULL << 52) | frac; /* 53 bits */
        int s = 0;
        *exp_biased = (int16_t)exp_raw;

        if (NULL != digits) {
          for (; s < gemm_oz1_nslices; ++s) {
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
        if (NULL != digits) memset(digits, 0, sizeof(int8_t) * gemm_oz1_nslices);
        return;
      }

      { const uint32_t mant_full = (1U << 23) | frac; /* 24 bits */
        int s = 0;
        *exp_biased = (int16_t)exp_raw;

        if (NULL != digits) {
          for (; s < gemm_oz1_nslices; ++s) {
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
    if (dst != src) memcpy(dst, src, sizeof(int8_t) * gemm_oz1_nslices);
    return;
  }

  if (delta >= 7) {
    for (i = 0; i < gemm_oz1_nslices; ++i) dst[i] = (src[i] >= 0) ? INT8_MAX : INT8_MIN;
    return;
  }

  if (delta <= -15) {
    memset(dst, 0, sizeof(int8_t) * gemm_oz1_nslices);
    return;
  }

  if (delta > 0) {
    const int sh = delta;
    for (i = 0; i < gemm_oz1_nslices; ++i) {
      const int32_t v = ((int32_t)src[i]) << sh;
      dst[i] = (int8_t)LIBXS_CLMP(v, INT8_MIN, INT8_MAX);
    }
  }
  else {
    const int sh = -delta;
    for (i = 0; i < gemm_oz1_nslices; ++i) {
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

  for (; slice < gemm_oz1_nslices; ++slice) {
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


LIBXS_API_INLINE void store_block_pair(GEMM_REAL_TYPE* ref_blk, GEMM_REAL_TYPE* recon_blk,
  GEMM_INT_TYPE ld, GEMM_INT_TYPE row, GEMM_INT_TYPE col, GEMM_REAL_TYPE ref_val, GEMM_REAL_TYPE recon_val)
{
  recon_blk[row + col * ld] = recon_val;
  ref_blk[row + col * ld] = ref_val;
}


LIBXS_API_INLINE void accumulate_block_diff(libxs_matdiff_info_t* acc, const GEMM_REAL_TYPE* ref_blk,
  const GEMM_REAL_TYPE* tst_blk, GEMM_INT_TYPE m, GEMM_INT_TYPE n, GEMM_INT_TYPE ld_ref, GEMM_INT_TYPE ld_tst)
{
  libxs_matdiff_info_t block_diff;
  const int ild_ref = (int)ld_ref, ild_tst = (int)ld_tst;
  if (EXIT_SUCCESS == libxs_matdiff(&block_diff, LIBXS_DATATYPE(GEMM_REAL_TYPE),
    m, n, ref_blk /*ref*/, tst_blk /*tst*/, &ild_ref, &ild_tst))
  {
    libxs_matdiff_reduce(acc, &block_diff);
  }
}


LIBXS_API_INLINE void scale_block_beta(GEMM_REAL_TYPE* mb, GEMM_INT_TYPE ldc,
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
      memset(am[mi][kk], 0, sizeof(int8_t) * gemm_oz1_nslices);
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
      memset(bm[kk][nj], 0, sizeof(int8_t) * gemm_oz1_nslices);
    }
  }
}


LIBXS_API_INLINE int32_t ozaki_dot_i8(const int8_t a[BLOCK_K], const int8_t b[BLOCK_K])
{
  int32_t dot = 0;
  int kk;
  for (kk = 0; kk < BLOCK_K; ++kk) {
    dot += (int32_t)a[kk] * (int32_t)b[kk];
  }
  return dot;
}


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
  const int nslices = gemm_oz1_nslices;
  GEMM_INT_TYPE jb, ib;
  int s;
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE)
            || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));

  for (s = 0; s < nslices; ++s) {
    const int high = OZ1_MANT_BITS - (7 * s);
    const int low = (high >= 0) ? (high - 6) : 0;
    slice_low_bit[s] = (low > 0 ? low : 0);
  }

#if defined(_OPENMP)
# pragma omp parallel if(NULL == diff)
#endif
  { int8_t am[BLOCK_M][BLOCK_K][MAX_NSLICES], bm[BLOCK_K][BLOCK_N][MAX_NSLICES];
    int8_t ak[BLOCK_M][MAX_NSLICES][BLOCK_K]; /* k-contiguous for dot product */
    int8_t bk[BLOCK_N][MAX_NSLICES][BLOCK_K]; /* k-contiguous for dot product */
    int16_t expa_row[BLOCK_M], expb_col[BLOCK_N];
    GEMM_REAL_TYPE ref_blk[BLOCK_MNK], recon_blk[BLOCK_MNK];
    GEMM_INT_TYPE kb, mi, nj, kk;
    int slice_a, slice_b;

#if defined(_OPENMP)
#   pragma omp for LIBXS_OPENMP_COLLAPSE(2) schedule(dynamic)
#endif
    for (jb = 0; jb < N; jb += BLOCK_N) {
      for (ib = 0; ib < M; ib += BLOCK_M) {
        const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
        const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
        GEMM_REAL_TYPE *const mb = c + jb * ldcv + ib;

        scale_block_beta(mb, ldcv, iblk, jblk, beta, ref_blk,
          (NULL != diff && 0 == (diff_abc % 3)));

        for (kb = 0; kb < K; kb += BLOCK_K) {
          const GEMM_INT_TYPE kblk = LIBXS_MIN(BLOCK_K, K - kb);

          preprocess_rows(a, *lda, ta, M, K, ib, kb, iblk, kblk, expa_row, am);
          preprocess_cols(b, *ldb, tb, N, K, jb, kb, jblk, kblk, expb_col, bm);

          /* Track differences between original A block and reconstructed digits */
          if (NULL != diff && 1 == (diff_abc % 3)) {
#if defined(_OPENMP)
#           pragma omp critical
#endif
            { for (mi = 0; mi < iblk; ++mi) {
                const GEMM_INT_TYPE row = ib + mi;
                for (kk = 0; kk < kblk; ++kk) {
                  const GEMM_INT_TYPE p = kb + kk;
                  const GEMM_REAL_TYPE aval = ((row < M && p < K) ?
                    a[LIBXS_INDEX(ta, *lda, row, p)] : (GEMM_REAL_TYPE)0);
                  const int exp_base = (int)expa_row[mi] - OZ1_BIAS_PLUS_MANT;
                  const double arecon = reconstruct_from_digits(am[mi][kk],
                    exp_base, slice_low_bit);

                  store_block_pair(ref_blk, recon_blk, BLOCK_M, mi, kk,
                    (GEMM_REAL_TYPE)aval, (GEMM_REAL_TYPE)arecon);
                }
              }
              accumulate_block_diff(diff, ref_blk, recon_blk, iblk, kblk, BLOCK_M, BLOCK_M);
            }
          }

          /* Track differences between original B block and reconstructed digits */
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
                  const int exp_base = (int)expb_col[nj] - OZ1_BIAS_PLUS_MANT;
                  const double brecon = reconstruct_from_digits(bm[kk][nj],
                    exp_base, slice_low_bit);

                  store_block_pair(ref_blk, recon_blk, BLOCK_K, kk, nj,
                    (GEMM_REAL_TYPE)bval, (GEMM_REAL_TYPE)brecon);
                }
              }
              accumulate_block_diff(diff, ref_blk, recon_blk, kblk, jblk, BLOCK_K, BLOCK_K);
            }
          }

          /* Transpose am/bm into k-contiguous layout for dot product */
          for (mi = 0; mi < iblk; ++mi) {
            for (slice_a = 0; slice_a < nslices; ++slice_a) {
              for (kk = 0; kk < BLOCK_K; ++kk) {
                ak[mi][slice_a][kk] = am[mi][kk][slice_a];
              }
            }
          }
          for (nj = 0; nj < jblk; ++nj) {
            for (slice_b = 0; slice_b < nslices; ++slice_b) {
              for (kk = 0; kk < BLOCK_K; ++kk) {
                bk[nj][slice_b][kk] = bm[kk][nj][slice_b];
              }
            }
          }

          for (slice_a = 0; slice_a < ((0 != (gemm_oz1_flags & OZ1_TRIM_FORWARD)) ? nslices / 2 : nslices); ++slice_a) {
            slice_b = (0 != (gemm_oz1_flags & OZ1_TRIANGULAR)) ? slice_a : 0;
            for (; slice_b < nslices; ++slice_b) {
              const int low_bit_sum = (int)slice_low_bit[slice_a] + slice_low_bit[slice_b];
              /* Double off-diagonal terms whose mirror (sb,sa) is not explicitly
               * computed. When REVERSE_PASS is also active, the mirror IS
               * recovered when: sb >= S/2 && sa <= S-1-sb, so skip doubling. */
              const double sym_alpha = (0 != (gemm_oz1_flags & OZ1_SYMMETRIZE))
                ? (*alpha) * ((slice_a != slice_b
                    && !(0 != (gemm_oz1_flags & OZ1_REVERSE_PASS)
                      && slice_b >= nslices / 2 && slice_a <= nslices - 1 - slice_b)
                    ) ? 2.0 : 1.0)
                : (*alpha);
              for (mi = 0; mi < iblk; ++mi) {
                for (nj = 0; nj < jblk; ++nj) {
                  const int32_t dot = ozaki_dot_i8(ak[mi][slice_a], bk[nj][slice_b]);
                  if (0 != dot) {
                    int sh = (int)expa_row[mi] + (int)expb_col[nj] - (2 * OZ1_BIAS_PLUS_MANT) + low_bit_sum;
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
          if (0 != (gemm_oz1_flags & OZ1_REVERSE_PASS)) {
          /* Reverse pass: explicitly recover the most significant lower-triangle
           * terms (slice_a >= S/2, slice_b from S-1-slice_a downward). These are
           * the dropped terms with the largest exponents (small slice_b index
           * paired with large slice_a index). */
          for (slice_a = nslices / 2; slice_a < nslices; ++slice_a) {
            for (slice_b = nslices - 1 - slice_a; slice_b >= 0; --slice_b) {
              const int low_bit_sum = (int)slice_low_bit[slice_a] + slice_low_bit[slice_b];
              for (mi = 0; mi < iblk; ++mi) {
                for (nj = 0; nj < jblk; ++nj) {
                  const int32_t dot = ozaki_dot_i8(ak[mi][slice_a], bk[nj][slice_b]);
                  if (0 != dot) {
                    int sh = (int)expa_row[mi] + (int)expb_col[nj] - (2 * OZ1_BIAS_PLUS_MANT) + low_bit_sum;
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
            accumulate_block_diff(diff, ref_blk, mb, iblk, jblk, mref, ldcv);
          }
        }
      }
    } /* end parallel for */
  } /* end parallel */
}


LIBXS_API void gemm_oz1(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  if (0 == gemm_verbose) {
    gemm_oz1_diff(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      0 /*gemm_diff_abc*/, NULL /*diff*/);
  }
  else {
    double epsilon;
    libxs_matdiff_info_t diff;
    libxs_matdiff_clear(&diff);
    gemm_oz1_diff(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
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
      char extension[
        sizeof(char) /*trans*/ +
        sizeof(GEMM_INT_TYPE) /*ld*/+
        sizeof(GEMM_REAL_TYPE) /* alpha/beta */
      ];
      libxs_mhd_info_t mhd_info = { 2, 1, LIBXS_DATATYPE(GEMM_REAL_TYPE), 0 };
      char fname[64];
      size_t size[2], ld[2];
      FILE *file = NULL;
      int result = EXIT_SUCCESS;
      LIBXS_SNPRINTF(fname, sizeof(fname), "ozaki-%i-a.mhd", diff.r);
      file = fopen(fname, "rb");
      if (NULL == file) { /* Never overwrite an existing file */
        size[0] = *m; size[1] = *k; ld[0] = *lda; ld[1] = *k;
        *(char*)extension = *transa;
        memcpy(extension + sizeof(char), lda, sizeof(GEMM_INT_TYPE));
        memcpy(extension + sizeof(char) + sizeof(GEMM_INT_TYPE), alpha, sizeof(GEMM_REAL_TYPE));
        result |= libxs_mhd_write(fname, NULL/*offset*/, size, ld,
          &mhd_info, a, NULL/*handler_info*/, NULL/*handler*/,
          NULL/*extension_header*/, extension, sizeof(extension));
      }
      else fclose(file);
      LIBXS_SNPRINTF(fname, sizeof(fname), "ozaki-%i-b.mhd", diff.r);
      file = fopen(fname, "rb");
      if (NULL == file) { /* Never overwrite an existing file */
        size[0] = *k; size[1] = *n; ld[0] = *ldb; ld[1] = *n;
        *(char*)extension = *transb;
        memcpy(extension + sizeof(char), ldb, sizeof(GEMM_INT_TYPE));
        memcpy(extension + sizeof(char) + sizeof(GEMM_INT_TYPE), beta, sizeof(GEMM_REAL_TYPE));
        result |= libxs_mhd_write(fname, NULL/*offset*/, size, ld,
          &mhd_info, b, NULL/*handler_info*/, NULL/*handler*/,
          NULL/*extension_header*/, extension, sizeof(extension));
      }
      else fclose(file);
      if (EXIT_SUCCESS == result) {
        print_gemm(stdout, transa, transb, m, n, k,
          alpha, a, lda, b, ldb, beta, c, ldc);
      }
      /* avoid repeated dumps */
      gemm_rsq = diff.rsq;
      gemm_eps = epsilon;
    }
  }
}


LIBXS_API_INTERN void print_diff_atexit(void);
LIBXS_API_INTERN void print_diff_atexit(void)
{
  if (0 != gemm_verbose && 0 < gemm_diff.r) print_diff(stderr, &gemm_diff);
}


/** Function gemm_oz1 is called here with the original GEMM as fallback and for comparison. */
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void GEMM_WRAP(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  static int gemm_initialized = 0, gemm_ozaki = 1;
  LIBXS_ASSERT(NULL != lda && NULL != ldb && NULL != ldc);
  LIBXS_ASSERT(NULL != a && NULL != b && NULL != c);
  LIBXS_ASSERT(NULL != m && NULL != n && NULL != k);
  LIBXS_ASSERT(NULL != transa && NULL != transb);

  if (0 == gemm_initialized) {
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    if (0 == gemm_initialized) {
      const union { uint32_t raw; float value; } inf = { 0x7F800000U };
      const char *const gemm_diff_abc_env = getenv("GEMM_DIFF");
      const char *const gemm_verbose_env = getenv("GEMM_VERBOSE");
      const char *const gemm_ozaki_env = getenv("GEMM_OZAKI");
      const char *const gemm_oz1n_env = getenv("GEMM_OZ1N");
      const char *const gemm_oz1_env = getenv("GEMM_OZ1");
      const char *const gemm_eps_env = getenv("GEMM_EPS");
      const char *const gemm_rsq_env = getenv("GEMM_RSQ");
      libxs_matdiff_clear(&gemm_diff);
      gemm_diff_abc = (NULL == gemm_diff_abc_env ? 0 : atoi(gemm_diff_abc_env));
      gemm_verbose = (NULL == gemm_verbose_env ? 0 : atoi(gemm_verbose_env));
      gemm_ozaki = (NULL == gemm_ozaki_env ? 1 : atoi(gemm_ozaki_env));
      gemm_oz1_nslices = LIBXS_CLMP((NULL == gemm_oz1n_env
        ? NSLICES_DEFAULT : atoi(gemm_oz1n_env)), 1, MAX_NSLICES);
      gemm_oz1_flags = (NULL == gemm_oz1_env
        ? OZ1_DEFAULT : atoi(gemm_oz1_env));
      if (NULL == gemm_eps_env) gemm_eps = inf.value;
      else {
        if (0 == gemm_verbose) gemm_verbose = 1;
        gemm_eps = atof(gemm_eps_env);
      }
      if (NULL == gemm_rsq_env) gemm_rsq = 0;
      else {
        if (0 == gemm_verbose) gemm_verbose = 1;
        gemm_rsq = atof(gemm_rsq_env);
      }
      LIBXS_EXPECT(EXIT_SUCCESS == atexit(print_diff_atexit));
      gemm_initialized = 1;
    }
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
  }
  LIBXS_ASSERT(0 != gemm_initialized);

  if (0 == gemm_ozaki) { /* only run original GEMM right away */
    if (NULL != gemm_original) {
      gemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    else {
      GEMM_REAL(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    if (0 != gemm_verbose) {
      LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
      ++gemm_diff.r;
      LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
    }
  }
  else { /* run LP-GEMM */
    gemm_oz1(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}
