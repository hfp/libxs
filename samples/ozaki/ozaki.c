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

#if !defined(BLOCK_M)
# define BLOCK_M 16
#endif
#if !defined(BLOCK_N)
# define BLOCK_N 16
#endif
#if !defined(BLOCK_K)
# define BLOCK_K 16
#endif
#if !defined(NSLICES)
# define NSLICES 8
#endif


LIBXS_APIVAR_PUBLIC_DEF(libxs_matdiff_info_t gemm_diff);
LIBXS_APIVAR_PUBLIC_DEF(int gemm_verbose);

LIBXS_APIVAR_PRIVATE_DEF(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
LIBXS_APIVAR_PRIVATE_DEF(gemm_function_t gemm_original);
LIBXS_APIVAR_PRIVATE_DEF(int gemm_diff_abc);


LIBXS_API_INLINE void ozaki_decompose(double value, int16_t* exp_biased, int8_t digits[NSLICES])
{
  const union { int raw; float value; } inf = { 0x7F800000 };
  union { double d; uint64_t u; } cvt;
  int sign = 1;

  LIBXS_ASSERT(NULL != exp_biased);
  if (value == 0.0 || LIBXS_ISNAN(value) || inf.value == value) {
    if (NULL != digits) memset(digits, 0, sizeof(int8_t) * NSLICES);
    *exp_biased = 0;
    return;
  }

  if (value < 0.0) {
    value = -value;
    sign = -1;
  }

  cvt.d = value;
  { const uint64_t bits = cvt.u;
    const uint64_t frac = bits & ((1ULL << 52) - 1ULL);
    const uint16_t exp_raw = (uint16_t)((bits >> 52) & 0x7FFU);

    if (0 == exp_raw) { /* subnormal treated as zero here */
      *exp_biased = 0;
      if (NULL != digits) memset(digits, 0, sizeof(int8_t) * NSLICES);
      return;
    }

    { const uint64_t mant_full = (1ULL << 52) | frac; /* 53 bits */
      int s = 0;
      *exp_biased = (int16_t)exp_raw;

      if (NULL != digits) {
        for (; s < NSLICES; ++s) {
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


LIBXS_API_INLINE void rescale_digits(int8_t dst[NSLICES], const int8_t src[NSLICES], const int delta)
{
  int i;

  if (delta == 0) {
    memcpy(dst, src, sizeof(int8_t) * NSLICES);
    return;
  }

  if (delta >= 7) {
    for (i = 0; i < NSLICES; ++i) dst[i] = (src[i] >= 0) ? INT8_MAX : INT8_MIN;
    return;
  }

  if (delta <= -15) {
    memset(dst, 0, sizeof(int8_t) * NSLICES);
    return;
  }

  if (delta > 0) {
    const int sh = delta;
    for (i = 0; i < NSLICES; ++i) {
      const int32_t v = ((int32_t)src[i]) << sh;
      if (v > INT8_MAX) dst[i] = INT8_MAX; else if (v < INT8_MIN) dst[i] = INT8_MIN; else dst[i] = (int8_t)v;
    }
  }
  else {
    const int sh = -delta;
    for (i = 0; i < NSLICES; ++i) {
      const int32_t v = ((int32_t)src[i]) >> sh;
      if (v > INT8_MAX) dst[i] = INT8_MAX; else if (v < INT8_MIN) dst[i] = INT8_MIN; else dst[i] = (int8_t)v;
    }
  }
}


LIBXS_API_INLINE void preprocess_rows(const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE lda, const int ta,
  const GEMM_INT_TYPE M, const GEMM_INT_TYPE K,
  const GEMM_INT_TYPE ib, const GEMM_INT_TYPE kb,
  const GEMM_INT_TYPE iblk, const GEMM_INT_TYPE kblk,
  int16_t expa_row[BLOCK_M], int8_t am[BLOCK_M][BLOCK_K][NSLICES])
{
  GEMM_INT_TYPE mi, kk;

  for (mi = 0; mi < BLOCK_M; ++mi) {
    const GEMM_INT_TYPE row = ib + mi;
    int16_t row_max_exp = INT16_MIN;

    for (kk = 0; kk < BLOCK_K; ++kk) {
      const GEMM_INT_TYPE p = kb + kk;
      const double aval = (row < M && p < K) ? a[LIBXS_INDEX(ta, lda, row, p)] : 0.0;
      int16_t e;
      ozaki_decompose(aval, &e, NULL /*digits*/);
      row_max_exp = LIBXS_MAX(row_max_exp, e);
    }

    expa_row[mi] = row_max_exp;

    for (kk = 0; kk < BLOCK_K; ++kk) {
      const GEMM_INT_TYPE p = kb + kk;
      const double aval = (row < M && p < K) ? a[LIBXS_INDEX(ta, lda, row, p)] : 0.0;
      int16_t e;
      int8_t digits[NSLICES];
      ozaki_decompose(aval, &e, digits);
      rescale_digits(am[mi][kk], digits, (int)e - (int)row_max_exp);
    }
  }
}


LIBXS_API_INLINE void preprocess_cols(const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE ldb, const int tb,
  const GEMM_INT_TYPE N, const GEMM_INT_TYPE K,
  const GEMM_INT_TYPE jb, const GEMM_INT_TYPE kb,
  const GEMM_INT_TYPE jblk, const GEMM_INT_TYPE kblk,
  int16_t expb_col[BLOCK_N], int8_t bm[BLOCK_K][BLOCK_N][NSLICES])
{
  GEMM_INT_TYPE nj, kk;
  int16_t col_max_exp[BLOCK_N];

  for (nj = 0; nj < BLOCK_N; ++nj) {
    col_max_exp[nj] = INT16_MIN;
  }

  for (kk = 0; kk < BLOCK_K; ++kk) {
    const GEMM_INT_TYPE p = kb + kk;
    for (nj = 0; nj < BLOCK_N; ++nj) {
      const GEMM_INT_TYPE col = jb + nj;
      const double bval = (p < K && col < N) ? b[LIBXS_INDEX(tb, ldb, p, col)] : 0.0;
      int16_t e;
      ozaki_decompose(bval, &e, NULL /*digits*/);
      col_max_exp[nj] = LIBXS_MAX(col_max_exp[nj], e);
    }
  }

  for (nj = 0; nj < BLOCK_N; ++nj) {
    expb_col[nj] = col_max_exp[nj];
  }

  for (kk = 0; kk < BLOCK_K; ++kk) {
    const GEMM_INT_TYPE p = kb + kk;
    for (nj = 0; nj < BLOCK_N; ++nj) {
      const GEMM_INT_TYPE col = jb + nj;
      const double bval = (p < K && col < N) ? b[LIBXS_INDEX(tb, ldb, p, col)] : 0.0;
      int16_t e;
      int8_t digits[NSLICES];
      ozaki_decompose(bval, &e, digits);
      rescale_digits(bm[kk][nj], digits, (int)e - (int)col_max_exp[nj]);
    }
  }
}


LIBXS_API_INLINE void gemm_oz1_diff(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
  int diff_abc, libxs_matdiff_info_t* diff)
{
  const int ta = (*transa != 'N' && *transa != 'n');
  const int tb = (*transb != 'N' && *transb != 'n');
  int8_t am[BLOCK_M][BLOCK_K][NSLICES];
  int8_t bm[BLOCK_K][BLOCK_N][NSLICES];
  int16_t expa_row[BLOCK_M];
  int16_t expb_col[BLOCK_N];
  int8_t slice_low_bit[NSLICES];
  int32_t acc[BLOCK_M][BLOCK_N];
  const GEMM_INT_TYPE M = *m, N = *n, K = *k;
  GEMM_INT_TYPE jb, ib, kb, mi, nj, kk;
  int slice_a, slice_b;
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE));

  for (slice_a = 0; slice_a < NSLICES; ++slice_a) {
    const int high = 52 - (7 * slice_a);
    const int low = (high >= 0) ? (high - 6) : 0;
    slice_low_bit[slice_a] = (low > 0 ? low : 0);
  }

  for (jb = 0; jb < N; jb += BLOCK_N) {
    const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);

    for (ib = 0; ib < M; ib += BLOCK_M) {
      const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
      memset(acc, 0, sizeof(acc));

      for (kb = 0; kb < K; kb += BLOCK_K) {
        const GEMM_INT_TYPE kblk = LIBXS_MIN(BLOCK_K, K - kb);

        preprocess_rows(a, *lda, ta, M, K, ib, kb, iblk, kblk, expa_row, am);
        preprocess_cols(b, *ldb, tb, N, K, jb, kb, jblk, kblk, expb_col, bm);

        for (slice_a = 0; slice_a < NSLICES; ++slice_a) {
          for (slice_b = slice_a; slice_b < NSLICES; ++slice_b) {
            for (mi = 0; mi < iblk; ++mi) {
              for (nj = 0; nj < jblk; ++nj) {
                for (kk = 0; kk < kblk; ++kk) {
                  const int16_t a_digit = (int16_t)am[mi][kk][slice_a];
                  const int16_t b_digit = (int16_t)bm[kk][nj][slice_b];

                  if (a_digit | b_digit) {
                    const int exp_term = (int)expa_row[mi] + (int)expb_col[nj] - 2150 + slice_low_bit[slice_a] + slice_low_bit[slice_b];
                    int64_t contrib = (int64_t)a_digit * (int64_t)b_digit;

                    if (exp_term >= 0) {
                      if (exp_term >= 31) {
                        contrib = (contrib >= 0) ? INT32_MAX : INT32_MIN;
                      }
                      else {
                        contrib = contrib << exp_term;
                      }
                    }
                    else {
                      const int sh = -exp_term;
                      if (sh >= 31) {
                        contrib = 0; /* shifted out */
                      }
                      else {
                        contrib = contrib >> sh;
                      }
                    }

                    {
                      const int64_t sum = (int64_t)acc[mi][nj] + contrib;
                      if (sum > INT32_MAX) {
                        acc[mi][nj] = INT32_MAX;
                      }
                      else if (sum < INT32_MIN) {
                        acc[mi][nj] = INT32_MIN;
                      }
                      else {
                        acc[mi][nj] = (int32_t)sum;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      for (mi = 0; mi < iblk; ++mi) {
        const GEMM_INT_TYPE row = ib + mi;
        for (nj = 0; nj < jblk; ++nj) {
          const GEMM_INT_TYPE col = jb + nj;
          const GEMM_INT_TYPE idx = row + col * (*ldc);
          c[idx] = (*beta) * c[idx] + (*alpha) * (GEMM_REAL_TYPE)acc[mi][nj];
        }
      }
    }
  }
}


LIBXS_API void print_diff(FILE* stream, const libxs_matdiff_info_t* diff)
{
  LIBXS_ASSERT(NULL != diff);
  fprintf(stream, "OZAKI GEMM: ncalls=%i linf_abs=%f linf_rel=%f l2_abs=%f l2_rel=%f rsq=%f\n",
    diff->r, diff->linf_abs, diff->linf_rel, diff->l2_abs, diff->l2_rel, diff->rsq);
}


LIBXS_API void gemm_oz1(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  if (0 != gemm_verbose) {
    gemm_oz1_diff(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      0 /*gemm_diff_abc*/, NULL /*diff*/);
  }
  else {
    libxs_matdiff_info_t diff;
    gemm_oz1_diff(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      gemm_diff_abc, &diff);

    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
    libxs_matdiff_reduce(&gemm_diff, &diff);
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_SEQ_CST);

    if (1 < gemm_verbose || 0 > gemm_verbose) {
      const int nth = (0 < gemm_verbose ? gemm_verbose : 1);
      if (0 == (gemm_diff.r % nth)) print_diff(stderr, &gemm_diff);
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
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
    if (0 == gemm_initialized) {
      const char *const gemm_diff_abc_env = getenv("GEMM_DIFF");
      const char *const gemm_verbose_env = getenv("GEMM_VERBOSE");
      const char *const gemm_ozaki_env = getenv("GEMM_OZAKI");
      libxs_matdiff_clear(&gemm_diff);
      gemm_diff_abc = (NULL == gemm_diff_abc_env ? 0 : atoi(gemm_diff_abc_env));
      gemm_verbose = (NULL == gemm_verbose_env ? 0 : atoi(gemm_verbose_env));
      gemm_ozaki = (NULL == gemm_ozaki_env ? 1 : atoi(gemm_ozaki_env));
      LIBXS_EXPECT(EXIT_SUCCESS == atexit(print_diff_atexit));
      gemm_initialized = 1;
    }
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_SEQ_CST);
  }
  LIBXS_ASSERT(0 != gemm_initialized);

  if (0 == gemm_ozaki) { /* only run original GEMM right away */
    if (NULL != gemm_original) {
      gemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    else {
      GEMM_REAL(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
  }
  else if (0 == gemm_verbose) { /* only run LP-GEMM; no statistics */
    gemm_oz1_diff(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      0 /*gemm_diff_abc*/, NULL /*diff*/);
  }
  else { /* run LP-GEMM and GEMM; build statistics of differences */
    gemm_oz1(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}
