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


LIBXS_API_INLINE void gemm_oz1_diff(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
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
  int8_t* a_slices = NULL;
  int8_t* b_slices = NULL;
  int16_t* expa_raw = NULL;
  int16_t* expb_raw = NULL;
  double* expa_fp = NULL;
  double* expb_fp = NULL;
  GEMM_REAL_TYPE* c_ref = NULL;
  const size_t c_size = (size_t)ldcv * (size_t)N * sizeof(GEMM_REAL_TYPE);
  GEMM_PROFILE_DECL;
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
  /* Save C for reference comparison (before Ozaki modifies it) */
  if (NULL != diff) {
    c_ref = (GEMM_REAL_TYPE*)libxs_malloc(gemm_pool, c_size, 0);
    if (NULL != c_ref) memcpy(c_ref, c, c_size);
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
#endif
    GEMM_PROFILE_START(tid);

    /* Phase 3: scale C by beta (once, before K-group loop).
     * Per BLAS spec, beta=0 must zero C unconditionally (NaN/Inf safe). */
#if defined(_OPENMP)
# pragma omp for schedule(static)
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
  } /* end parallel */

  /* Reference BLAS and diff comparison (whole-matrix, consistent with GPU path) */
  if (NULL != c_ref) {
    ozaki_diff_reference(GEMM_ARGPASS, c_ref, c_size, diff);
  }
  libxs_free(a_slices);
  libxs_free(b_slices);
  libxs_free(expa_raw);
  libxs_free(expb_raw);
  libxs_free(expa_fp);
  libxs_free(expb_fp);
  libxs_free(c_ref);
}


LIBXS_API void gemm_oz1(const char* transa, const char* transb, const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n,
  const GEMM_INT_TYPE* k, const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, const GEMM_REAL_TYPE* b,
  const GEMM_INT_TYPE* ldb, const GEMM_REAL_TYPE* beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  OZAKI_GEMM_WRAPPER(gemm_oz1_diff, GEMM_LABEL, 1)
}
