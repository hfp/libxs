/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"


/**
 * Split a (pre-aligned) mantissa into signed 7-bit digits.
 * The mantissa is expected to be in the same format as produced
 * by ozaki_extract_ieee (implicit bit at position OZ_MANT_BITS),
 * but may have been right-shifted for exponent alignment.
 */
static void split_digits(uint64_t mantissa, int sign, int8_t digits[MAX_NSLICES])
{
  int s;
  if (0 == mantissa) {
    memset(digits, 0, sizeof(int8_t) * ozaki_n);
  }
  else {
    LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
    for (s = 0; s < ozaki_n; ++s) {
      const int high = OZ_MANT_BITS - (7 * s);
      if (high < 0) {
        digits[s] = 0;
        continue;
      }
      {
        const int low = high - 6;
        uint64_t chunk;
        if (low >= 0) {
          chunk = (mantissa >> low) & 0x7FULL;
        }
        else {
          const int width = high + 1;
          chunk = mantissa & ((1ULL << width) - 1ULL);
        }
        digits[s] = (int8_t)(sign * (int64_t)chunk);
      }
    }
  }
}


/**
 * Reconstruct a floating-point value from its signed 7-bit digit
 * representation.  Used by diff tracking modes 1 and 2.
 */
static double reconstruct_from_digits(const int8_t digits[MAX_NSLICES], int exp_base, const int8_t slice_low_bit[MAX_NSLICES])
{
  double recon = 0.0;
  int slice = 0;

  for (; slice < ozaki_n; ++slice) {
    const int16_t digit = (int16_t)digits[slice];
    if (0 != digit) {
      const int sh = exp_base + slice_low_bit[slice];
      recon += (double)digit * libxs_pow2(sh);
    }
  }

  return recon;
}


/**
 * Preprocess rows of A: decompose, align, split into digits, and write
 * directly into the k-contiguous layout ak[M][S][K] used by dot products.
 * This avoids a separate transpose pass over an intermediate buffer.
 */
LIBXS_API_INLINE void preprocess_rows(const GEMM_REAL_TYPE* a, GEMM_INT_TYPE lda, int ta, GEMM_INT_TYPE M, GEMM_INT_TYPE K,
  GEMM_INT_TYPE ib, GEMM_INT_TYPE kb, GEMM_INT_TYPE iblk, GEMM_INT_TYPE kblk, int16_t expa_row[BLOCK_M],
  int8_t ak[BLOCK_M][MAX_NSLICES][BLOCK_K])
{
  int16_t elem_exp[BLOCK_M][BLOCK_K];
  uint64_t elem_mant[BLOCK_M][BLOCK_K];
  int elem_sign[BLOCK_M][BLOCK_K];
  GEMM_INT_TYPE mi, kk;

  /* Pass 1: extract sign, exponent, mantissa and track per-row max exponent */
  for (mi = 0; mi < iblk; ++mi) {
    const GEMM_INT_TYPE row = ib + mi;
    int16_t row_max_exp = INT16_MIN;
    int s;

    for (kk = 0; kk < kblk; ++kk) {
      const GEMM_INT_TYPE p = kb + kk;
      const GEMM_REAL_TYPE aval = ((row < M && p < K) ? a[LIBXS_INDEX(ta, lda, row, p)] : (GEMM_REAL_TYPE)0);
      elem_sign[mi][kk] = ozaki_extract_ieee(aval, &elem_exp[mi][kk], &elem_mant[mi][kk]);
      row_max_exp = LIBXS_MAX(row_max_exp, elem_exp[mi][kk]);
    }

    expa_row[mi] = row_max_exp;

    /* Pass 2: align mantissa, split into digits, scatter into ak[mi][s][kk] */
    for (kk = 0; kk < kblk; ++kk) {
      const int delta = (int)row_max_exp - (int)elem_exp[mi][kk];
      uint64_t aligned = elem_mant[mi][kk];
      int8_t tmp[MAX_NSLICES];
      if (delta > 0) {
        aligned = (delta < 64) ? (aligned >> delta) : 0;
      }
      split_digits(aligned, elem_sign[mi][kk], tmp);
      LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
      for (s = 0; s < ozaki_n; ++s) ak[mi][s][kk] = tmp[s];
    }
    /* Zero-pad remaining k-entries for fixed-length dot products */
    LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
    for (s = 0; s < ozaki_n; ++s) {
      for (kk = kblk; kk < BLOCK_K; ++kk) ak[mi][s][kk] = 0;
    }
  }
}


/**
 * Preprocess columns of B: decompose, align, split into digits, and write
 * directly into the k-contiguous layout bk[N][S][K] used by dot products.
 * This avoids a separate transpose pass over an intermediate buffer.
 */
LIBXS_API_INLINE void preprocess_cols(const GEMM_REAL_TYPE* b, GEMM_INT_TYPE ldb, int tb, GEMM_INT_TYPE N, GEMM_INT_TYPE K,
  GEMM_INT_TYPE jb, GEMM_INT_TYPE kb, GEMM_INT_TYPE jblk, GEMM_INT_TYPE kblk, int16_t expb_col[BLOCK_N],
  int8_t bk[BLOCK_N][MAX_NSLICES][BLOCK_K])
{
  int16_t elem_exp[BLOCK_K][BLOCK_N];
  uint64_t elem_mant[BLOCK_K][BLOCK_N];
  int elem_sign[BLOCK_K][BLOCK_N];
  GEMM_INT_TYPE nj, kk;
  int s;

  for (nj = 0; nj < jblk; ++nj) {
    expb_col[nj] = INT16_MIN;
  }

  /* Pass 1: extract sign, exponent, mantissa and track per-column max exponent */
  for (kk = 0; kk < kblk; ++kk) {
    const GEMM_INT_TYPE p = kb + kk;
    for (nj = 0; nj < jblk; ++nj) {
      const GEMM_INT_TYPE col = jb + nj;
      const GEMM_REAL_TYPE bval = ((p < K && col < N) ? b[LIBXS_INDEX(tb, ldb, p, col)] : (GEMM_REAL_TYPE)0);
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
      if (delta > 0) {
        aligned = (delta < 64) ? (aligned >> delta) : 0;
      }
      split_digits(aligned, elem_sign[kk][nj], tmp);
      LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
      for (s = 0; s < ozaki_n; ++s) bk[nj][s][kk] = tmp[s];
    }
  }
  /* Zero-pad remaining k-entries for fixed-length dot products */
  for (nj = 0; nj < jblk; ++nj) {
    LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
    for (s = 0; s < ozaki_n; ++s) {
      for (kk = kblk; kk < BLOCK_K; ++kk) bk[nj][s][kk] = 0;
    }
  }
}


LIBXS_API_INLINE void gemm_oz1_diff(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc, unsigned int diff_stat,
  libxs_matdiff_t* diff)
{
  int8_t slice_low_bit[MAX_NSLICES];
  double pow2_low[MAX_NSLICES];
  const int ta = (*transa != 'N' && *transa != 'n');
  const int tb = (*transb != 'N' && *transb != 'n');
  const GEMM_INT_TYPE M = *m, N = *n, K = *k;
  const GEMM_INT_TYPE ldcv = *ldc;
  const int nslices = LIBXS_CLMP(ozaki_n, 1, MAX_NSLICES);
  const int trim = LIBXS_MIN(ozaki_trim, 2 * (nslices - 1));
  const int cutoff = 2 * (nslices - 1) - trim;
  const GEMM_INT_TYPE K_grp_size = (0 < ozaki_maxk ? (GEMM_INT_TYPE)ozaki_maxk : K);
  const GEMM_INT_TYPE K_grp_max = LIBXS_MIN(K_grp_size, K);
  const GEMM_INT_TYPE K_grp_pad = ((K_grp_max + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;
  const GEMM_INT_TYPE nblk_m = (M + BLOCK_M - 1) / BLOCK_M;
  const GEMM_INT_TYPE nblk_n = (N + BLOCK_N - 1) / BLOCK_N;
  int8_t* a_slices = NULL;
  int8_t* b_slices = NULL;
  int16_t* expa_raw = NULL;
  int16_t* expb_raw = NULL;
  double* expa_fp = NULL;
  double* expb_fp = NULL;
  GEMM_REAL_TYPE* ref_panel = NULL;
  libxs_matdiff_t tdiff[256];
  ozaki_rsq_acc_t trsq[256];
  GEMM_PROFILE_DECL;
  int nthreads = 1;
  int s;
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE) || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));

  LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
  for (s = 0; s < nslices; ++s) {
    const int high = OZ_MANT_BITS - (7 * s);
    const int low = (high >= 0) ? (high - 6) : 0;
    slice_low_bit[s] = (low > 0 ? low : 0);
  }
  for (s = 0; s < nslices; ++s) {
    pow2_low[s] = libxs_pow2((int)slice_low_bit[s]);
  }

  a_slices = (int8_t*)libxs_malloc(gemm_pool, (size_t)nslices * M * K_grp_pad, 0);
  b_slices = (int8_t*)libxs_malloc(gemm_pool, (size_t)nslices * N * K_grp_pad, 0);
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
    GEMM_INT_TYPE row, col, ib, jb, mi, nj, kb_grp;
    int slice_a, slice_b, tid;
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
        int ss;
        /* Zero this row's slice buffers */
        for (ss = 0; ss < nslices; ++ss) {
          memset(a_slices + (long)ss * M * K_grp_pad + (long)row * K_grp_pad, 0, (size_t)K_grp_pad);
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
            int delta = (int)row_max_exp - (int)e;
            uint64_t aligned = (delta < 64) ? (mt >> delta) : 0;
            int8_t tmp[MAX_NSLICES];
            split_digits(aligned, sign, tmp);
            LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
            for (ss = 0; ss < nslices; ++ss) {
              a_slices[(long)ss * M * K_grp_pad + (long)row * K_grp_pad + (kk - kb_grp)] = tmp[ss];
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
        int ss;
        /* Zero this column's slice buffers */
        for (ss = 0; ss < nslices; ++ss) {
          memset(b_slices + (long)ss * N * K_grp_pad + (long)col * K_grp_pad, 0, (size_t)K_grp_pad);
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
            int delta = (int)col_max_exp - (int)e;
            uint64_t aligned = (delta < 64) ? (mt >> delta) : 0;
            int8_t tmp[MAX_NSLICES];
            split_digits(aligned, sign, tmp);
            LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
            for (ss = 0; ss < nslices; ++ss) {
              b_slices[(long)ss * N * K_grp_pad + (long)col * K_grp_pad + (kk - kb_grp)] = tmp[ss];
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

      /* Phase 2c: diff tracking for A/B decomposition (modes 1 and 2) */
      if (NULL != diff && 1 == (diff_stat % 3)) {
#if defined(_OPENMP)
# pragma omp for schedule(static)
#endif
        for (row = 0; row < M; ++row) {
          GEMM_REAL_TYPE ref_blk[BLOCK_K];
          GEMM_REAL_TYPE recon_blk[BLOCK_K];
          GEMM_INT_TYPE kk;
          const int exp_base = (int)expa_raw[row] - OZ_BIAS_PLUS_MANT;
          for (kk = kb_grp; kk < kb_grp + K_len; ++kk) {
            const GEMM_INT_TYPE kk_local = kk - kb_grp;
            int8_t tmp[MAX_NSLICES];
            int si;
            double arecon;
            LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
            for (si = 0; si < nslices; ++si) {
              tmp[si] = a_slices[(long)si * M * K_grp_pad + (long)row * K_grp_pad + kk_local];
            }
            arecon = reconstruct_from_digits(tmp, exp_base, slice_low_bit);
            ref_blk[kk_local % BLOCK_K] = a[LIBXS_INDEX(ta, *lda, row, kk)];
            recon_blk[kk_local % BLOCK_K] = (GEMM_REAL_TYPE)arecon;
            if (BLOCK_K - 1 == (kk_local % BLOCK_K) || kk == kb_grp + K_len - 1) {
              GEMM_INT_TYPE bsize = (kk_local % BLOCK_K) + 1;
              libxs_matdiff_t bd;
              const int ild = 1, itd = 1;
              if (EXIT_SUCCESS == libxs_matdiff(&bd, LIBXS_DATATYPE(GEMM_REAL_TYPE), bsize, 1, ref_blk, recon_blk, &ild, &itd)) {
                libxs_matdiff_reduce(&tdiff[tid], &bd);
              }
            }
          }
        }
      }
      if (NULL != diff && 2 == (diff_stat % 3)) {
#if defined(_OPENMP)
# pragma omp for schedule(static)
#endif
        for (col = 0; col < N; ++col) {
          GEMM_REAL_TYPE ref_blk[BLOCK_K];
          GEMM_REAL_TYPE recon_blk[BLOCK_K];
          GEMM_INT_TYPE kk;
          const int exp_base = (int)expb_raw[col] - OZ_BIAS_PLUS_MANT;
          for (kk = kb_grp; kk < kb_grp + K_len; ++kk) {
            const GEMM_INT_TYPE kk_local = kk - kb_grp;
            int8_t tmp[MAX_NSLICES];
            int si;
            double brecon;
            LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
            for (si = 0; si < nslices; ++si) {
              tmp[si] = b_slices[(long)si * N * K_grp_pad + (long)col * K_grp_pad + kk_local];
            }
            brecon = reconstruct_from_digits(tmp, exp_base, slice_low_bit);
            ref_blk[kk_local % BLOCK_K] = b[LIBXS_INDEX(tb, *ldb, kk, col)];
            recon_blk[kk_local % BLOCK_K] = (GEMM_REAL_TYPE)brecon;
            if (BLOCK_K - 1 == (kk_local % BLOCK_K) || kk == kb_grp + K_len - 1) {
              GEMM_INT_TYPE bsize = (kk_local % BLOCK_K) + 1;
              libxs_matdiff_t bd;
              const int ild = 1, itd = 1;
              if (EXIT_SUCCESS == libxs_matdiff(&bd, LIBXS_DATATYPE(GEMM_REAL_TYPE), bsize, 1, ref_blk, recon_blk, &ild, &itd)) {
                libxs_matdiff_reduce(&tdiff[tid], &bd);
              }
            }
          }
        }
      }

      /* Phase 4: tile-first GEMM + accumulate for this K-group.
       * Tiles outermost (single omp for): C stays in a local buffer
       * across all slice pairs -- one load + one store per tile instead
       * of a read-modify-write per pair.  Also eliminates per-pair
       * omp-for barriers (implicit barrier at end of tile loop suffices). */
#if defined(_OPENMP)
# pragma omp for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
#endif
      for (jb = 0; jb < N; jb += BLOCK_N) {
        for (ib = 0; ib < M; ib += BLOCK_M) {
          const GEMM_INT_TYPE iblk = LIBXS_MIN(BLOCK_M, M - ib);
          const GEMM_INT_TYPE jblk = LIBXS_MIN(BLOCK_N, N - jb);
          GEMM_REAL_TYPE* const cb = c + jb * ldcv + ib;
          GEMM_REAL_TYPE c_local[BLOCK_M * BLOCK_N];
          int32_t c_acc[BLOCK_M * BLOCK_N];

          /* Load current C tile into contiguous local buffer */
          for (nj = 0; nj < jblk; ++nj) {
            for (mi = 0; mi < iblk; ++mi) {
              c_local[mi * jblk + nj] = cb[mi + nj * ldcv];
            }
          }

          /* Accumulate all slice pairs into c_local */
          LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
          for (slice_a = 0; slice_a < nslices && slice_a <= cutoff; ++slice_a) {
            const int sb_start = (0 != (ozaki_flags & OZ1_TRIANGULAR)) ? slice_a : 0;
            const int sb_end = LIBXS_MIN(nslices, cutoff + 1 - slice_a);
            for (slice_b = sb_start; slice_b < sb_end; ++slice_b) {
              const double pair_scale = (*alpha) * pow2_low[slice_a] * pow2_low[slice_b];
              const int do_mirror = (0 != (ozaki_flags & OZ1_SYMMETRIZE)) && (slice_a != slice_b);

              ozaki_gemm_s8s8s32('N', 'T', iblk, jblk, K_grp_pad,
                a_slices + (long)slice_a * M * K_grp_pad + (long)ib * K_grp_pad, K_grp_pad,
                b_slices + (long)slice_b * N * K_grp_pad + (long)jb * K_grp_pad, K_grp_pad, 0, c_acc, jblk);
              if (do_mirror) {
                ozaki_gemm_s8s8s32('N', 'T', iblk, jblk, K_grp_pad,
                  a_slices + (long)slice_b * M * K_grp_pad + (long)ib * K_grp_pad, K_grp_pad,
                  b_slices + (long)slice_a * N * K_grp_pad + (long)jb * K_grp_pad, K_grp_pad, 1, c_acc, jblk);
              }

              for (mi = 0; mi < iblk; ++mi) {
                const double ea = pair_scale * expa_fp[ib + mi];
                for (nj = 0; nj < jblk; ++nj) {
                  if (0 != c_acc[mi * jblk + nj]) {
                    c_local[mi * jblk + nj] += (GEMM_REAL_TYPE)(ea * expb_fp[jb + nj] * (double)c_acc[mi * jblk + nj]);
                  }
                }
              }
            }
          }

          /* Store c_local back to C */
          for (nj = 0; nj < jblk; ++nj) {
            for (mi = 0; mi < iblk; ++mi) {
              cb[mi + nj * ldcv] = c_local[mi * jblk + nj];
            }
          }
        }
      }
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
    for (s = 0; s < nthreads; ++s) {
      libxs_matdiff_reduce(diff, &tdiff[s]);
      total_ss_res += trsq[s].ss_res;
      total_ss_tot += trsq[s].ss_tot;
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
  libxs_free(a_slices);
  libxs_free(b_slices);
  libxs_free(expa_raw);
  libxs_free(expb_raw);
  libxs_free(expa_fp);
  libxs_free(expb_fp);
  libxs_free(ref_panel);
}


LIBXS_API void gemm_oz1(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  OZAKI_GEMM_WRAPPER(gemm_oz1_diff)
}


#if defined(__LIBXSTREAM)
/**
 * Host preprocessing wrapper for A (rows) — scheme 1 (int8).
 * Fills the flat slice buffer and exponent buffer in GPU-compatible layout.
 * Layout: slices[panel][BM][nslices][BK], exp[panel][BM].
 *   panel = ki * nblk + ib_idx.
 */
void oz1_host_preprocess_a(const void* matrix, int ld, int trans, int dim, int K, int kb_batch, int nkb, int nblk, int brc, int bk,
  int nslices_p, int kgroup_p, int use_xmx_p, void* slices, void* exp)
{
  const GEMM_REAL_TYPE* a = (const GEMM_REAL_TYPE*)matrix;
  int ki, ib_idx;
  (void)kgroup_p;
  (void)use_xmx_p;
  LIBXS_ASSERT(brc == BLOCK_M && bk == BLOCK_K);
# if defined(_OPENMP)
#   pragma omp parallel for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
# endif
  for (ki = 0; ki < nkb; ++ki) {
    for (ib_idx = 0; ib_idx < nblk; ++ib_idx) {
      const int panel = ki * nblk + ib_idx;
      const int ib = ib_idx * brc;
      const int kb = kb_batch + ki * bk;
      const GEMM_INT_TYPE iblk = (GEMM_INT_TYPE)LIBXS_MIN(brc, dim - ib);
      const GEMM_INT_TYPE kblk = (GEMM_INT_TYPE)LIBXS_MIN(bk, K - kb);
      int16_t expa_row[BLOCK_M];
      int8_t ak_tile[BLOCK_M][MAX_NSLICES][BLOCK_K];
      int mi;
      preprocess_rows(a, (GEMM_INT_TYPE)ld, trans, (GEMM_INT_TYPE)dim, (GEMM_INT_TYPE)K, (GEMM_INT_TYPE)ib, (GEMM_INT_TYPE)kb, iblk,
        kblk, expa_row, ak_tile);
      for (mi = 0; mi < brc; ++mi) {
        memcpy((char*)slices + ((long)panel * brc + mi) * nslices_p * bk, &ak_tile[mi][0][0], (size_t)nslices_p * bk);
      }
      memcpy((short*)exp + (long)panel * brc, expa_row, (size_t)brc * sizeof(short));
    }
  }
}


/**
 * Host preprocessing wrapper for B (columns) — scheme 1 (int8).
 * Layout (non-XMX): slices[panel][BN][nslices][BK], exp[panel][BN].
 * Layout (XMX):     slices[panel][nslices][BK][bn_pad], exp[panel][BN].
 *   panel = ki * nblk + jb_idx.
 */
void oz1_host_preprocess_b(const void* matrix, int ld, int trans, int dim, int K, int kb_batch, int nkb, int nblk, int brc, int bk,
  int nslices_p, int kgroup_p, int use_xmx_p, void* slices, void* exp)
{
  const GEMM_REAL_TYPE* b = (const GEMM_REAL_TYPE*)matrix;
  const int bn_pad = (0 != use_xmx_p) ? LIBXS_MAX(brc, 64) : brc;
  int ki, jb_idx;
  (void)kgroup_p;
  LIBXS_ASSERT(brc == BLOCK_N && bk == BLOCK_K);
# if defined(_OPENMP)
#   pragma omp parallel for LIBXS_OPENMP_COLLAPSE(2) schedule(static)
# endif
  for (ki = 0; ki < nkb; ++ki) {
    for (jb_idx = 0; jb_idx < nblk; ++jb_idx) {
      const int panel = ki * nblk + jb_idx;
      const int jb = jb_idx * brc;
      const int kb = kb_batch + ki * bk;
      const GEMM_INT_TYPE jblk = (GEMM_INT_TYPE)LIBXS_MIN(brc, dim - jb);
      const GEMM_INT_TYPE kblk = (GEMM_INT_TYPE)LIBXS_MIN(bk, K - kb);
      int16_t expb_col[BLOCK_N];
      int8_t bk_tile[BLOCK_N][MAX_NSLICES][BLOCK_K];
      int nj, s, kk;
      preprocess_cols(b, (GEMM_INT_TYPE)ld, trans, (GEMM_INT_TYPE)dim, (GEMM_INT_TYPE)K, (GEMM_INT_TYPE)jb, (GEMM_INT_TYPE)kb, jblk,
        kblk, expb_col, bk_tile);
      if (0 == use_xmx_p) {
        for (nj = 0; nj < brc; ++nj) {
          memcpy((char*)slices + ((long)panel * brc + nj) * nslices_p * bk, &bk_tile[nj][0][0], (size_t)nslices_p * bk);
        }
      }
      else {
        /* XMX layout: slices[panel][nslices][BK][bn_pad] (K-major) */
        for (s = 0; s < nslices_p; ++s) {
          for (kk = 0; kk < bk; ++kk) {
            long base = (((long)panel * nslices_p + s) * bk + kk) * bn_pad;
            for (nj = 0; nj < brc; ++nj) {
              ((char*)slices)[base + nj] = bk_tile[nj][s][kk];
            }
            for (nj = brc; nj < bn_pad; ++nj) {
              ((char*)slices)[base + nj] = 0;
            }
          }
        }
      }
      memcpy((short*)exp + (long)panel * brc, expb_col, (size_t)brc * sizeof(short));
    }
  }
}
#endif /* __LIBXSTREAM */
