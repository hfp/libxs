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
    BLOCK_MNK = LIBXS_MAX(LIBXS_MAX(BLOCK_MN, BLOCK_MK), BLOCK_KN),
    NKB_MAX = BATCH_K /* sub-panels per batch */
  };
  const int ta = (*transa != 'N' && *transa != 'n');
  const int tb = (*transb != 'N' && *transb != 'n');
  const GEMM_INT_TYPE M = *m, N = *n, K = *k;
  const GEMM_INT_TYPE ldcv = *ldc;
  const int nslices = LIBXS_CLMP(gemm_ozn, 1, MAX_NSLICES);
  const GEMM_INT_TYPE nblk_m = (M + BLOCK_M - 1) / BLOCK_M;
  const GEMM_INT_TYPE nblk_n = (N + BLOCK_N - 1) / BLOCK_N;
  /* Panel buffers: preprocessed A and B for one K-batch (shared).
   * First index is the sub-panel (BLOCK_K slice within BATCH_K*BLOCK_K). */
  int8_t (*ak_panel)[BLOCK_M][MAX_NSLICES][BLOCK_K] = NULL;
  int16_t (*expa_panel)[BLOCK_M] = NULL;
  int8_t (*bk_panel)[BLOCK_N][MAX_NSLICES][BLOCK_K] = NULL;
  int16_t (*expb_panel)[BLOCK_N] = NULL;
  GEMM_REAL_TYPE *ref_panel = NULL; /* diff mode 0 only */
  libxs_matdiff_info_t tdiff[256];
  int nthreads = 1;
  int s;
  const ozaki_dot_i8_fn dot_i8 = ozaki_dot_i8_init();
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE)
            || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));
  LIBXS_ASSERT(1 <= BATCH_K);

  LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
  for (s = 0; s < nslices; ++s) {
    const int high = OZ_MANT_BITS - (7 * s);
    const int low = (high >= 0) ? (high - 6) : 0;
    slice_low_bit[s] = (low > 0 ? low : 0);
  }

  ak_panel = (int8_t (*)[BLOCK_M][MAX_NSLICES][BLOCK_K])libxs_malloc(
    (size_t)NKB_MAX * nblk_m * sizeof(*ak_panel), 0);
  expa_panel = (int16_t (*)[BLOCK_M])libxs_malloc(
    (size_t)NKB_MAX * nblk_m * sizeof(*expa_panel), 0);
  bk_panel = (int8_t (*)[BLOCK_N][MAX_NSLICES][BLOCK_K])libxs_malloc(
    (size_t)NKB_MAX * nblk_n * sizeof(*bk_panel), 0);
  expb_panel = (int16_t (*)[BLOCK_N])libxs_malloc(
    (size_t)NKB_MAX * nblk_n * sizeof(*expb_panel), 0);
  if (NULL != diff && 0 == (diff_abc % 3)) {
    ref_panel = (GEMM_REAL_TYPE*)libxs_malloc(
      (size_t)nblk_m * nblk_n * BLOCK_MN * sizeof(*ref_panel), 0);
  }

#if defined(_OPENMP)
# pragma omp parallel
#endif
  { GEMM_REAL_TYPE recon_blk[BLOCK_MNK];
    GEMM_INT_TYPE kb_batch, kb_sub, mi, nj, kk, jb, ib, ib_idx, jb_idx;
    int slice_a, slice_b;
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
        preprocess_rows(a, *lda, ta, M, K, ib2, kb, iblk2, kblk,
          expa_panel[ki * nblk_m + ibi], ak_panel[ki * nblk_m + ibi]);
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
        preprocess_cols(b, *ldb, tb, N, K, jb2, kb, jblk2, kblk,
          expb_panel[ki * nblk_n + jbi], bk_panel[ki * nblk_n + jbi]);
      }
      /* implicit barrier ensures panels are ready */

      /* Phase 3: dot products + accumulate using panel data */
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

            /* Track differences between original A block and reconstructed digits */
            if (NULL != diff && 1 == (diff_abc % 3)) {
              GEMM_REAL_TYPE ref_blk[BLOCK_MNK];
              for (mi = 0; mi < iblk; ++mi) {
                const GEMM_INT_TYPE row = ib + mi;
                for (kk = 0; kk < kblk; ++kk) {
                  const GEMM_INT_TYPE p = kb + kk;
                  const GEMM_REAL_TYPE aval = ((row < M && p < K) ?
                    a[LIBXS_INDEX(ta, *lda, row, p)] : (GEMM_REAL_TYPE)0);
                  const int exp_base = (int)expa_panel[a_idx][mi] - OZ_BIAS_PLUS_MANT;
                  int8_t tmp[MAX_NSLICES];
                  int si;
                  for (si = 0; si < gemm_ozn; ++si) tmp[si] = ak_panel[a_idx][mi][si][kk];
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
              GEMM_REAL_TYPE ref_blk[BLOCK_MNK];
              for (kk = 0; kk < kblk; ++kk) {
                const GEMM_INT_TYPE p = kb + kk;
                for (nj = 0; nj < jblk; ++nj) {
                  const GEMM_INT_TYPE col = jb + nj;
                  const GEMM_REAL_TYPE bval = ((p < K && col < N)
                    ? b[LIBXS_INDEX(tb, *ldb, p, col)] : (GEMM_REAL_TYPE)0);
                  const int exp_base = (int)expb_panel[b_idx][nj] - OZ_BIAS_PLUS_MANT;
                  int8_t tmp[MAX_NSLICES];
                  int si;
                  for (si = 0; si < gemm_ozn; ++si) tmp[si] = bk_panel[b_idx][nj][si][kk];
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
                      int32_t dot = dot_i8(ak_panel[a_idx][mi][slice_a], bk_panel[b_idx][nj][slice_b]);
                      if (do_mirror) {
                        dot += dot_i8(ak_panel[a_idx][mi][slice_b], bk_panel[b_idx][nj][slice_a]);
                      }
                      if (0 != dot && (GEMM_REAL_TYPE)0 != *alpha) {
                        const int sh = (int)expa_panel[a_idx][mi] + (int)expb_panel[b_idx][nj]
                          - (2 * OZ_BIAS_PLUS_MANT) + low_bit_sum;
                        const double contrib = (*alpha) * (double)dot * libxs_pow2(sh);
                        cb[mi + nj * ldcv] += (GEMM_REAL_TYPE)contrib;
                      }
                    }
                  }
                }
              }
            }
          } /* end kb_sub loop */
        }
      } /* implicit barrier before next kb_batch */
    } /* end kb_batch loop */

    /* Diff mode 0: compute reference GEMM and compare (after all kb) */
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
    for (s = 0; s < nthreads; ++s) {
      libxs_matdiff_reduce(diff, &tdiff[s]);
    }
  }
  libxs_free(ak_panel); libxs_free(expa_panel);
  libxs_free(bk_panel); libxs_free(expb_panel);
  libxs_free(ref_panel);
}


LIBXS_API void gemm_oz1(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  OZAKI_GEMM_WRAPPER(gemm_oz1_diff)
}
