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
 *  Preprocess rows of A: Dekker-split each element into BF16 slices
 *  and write directly into the k-contiguous layout ak[M][S][K].
 *  No shared per-row exponent — each BF16 value carries its own.
 */
LIBXS_API_INLINE void oz3_preprocess_rows(const GEMM_REAL_TYPE* a, GEMM_INT_TYPE lda, int ta,
  GEMM_INT_TYPE M, GEMM_INT_TYPE K, GEMM_INT_TYPE ib, GEMM_INT_TYPE kb, GEMM_INT_TYPE iblk,
  GEMM_INT_TYPE kblk, libxs_bf16_t ak[BLOCK_M][MAX_NSLICES][BLOCK_K])
{
  GEMM_INT_TYPE mi, kk;
  for (mi = 0; mi < iblk; ++mi) {
    const GEMM_INT_TYPE row = ib + mi;
    int s;
    for (kk = 0; kk < kblk; ++kk) {
      const GEMM_INT_TYPE p = kb + kk;
      const GEMM_REAL_TYPE aval = ((row < M && p < K)
        ? a[LIBXS_INDEX(ta, lda, row, p)] : (GEMM_REAL_TYPE)0);
      libxs_bf16_t tmp[MAX_NSLICES];
      ozaki_split_to_bf16(aval, tmp);
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
 *  Preprocess columns of B: Dekker-split each element into BF16 slices
 *  and write directly into the k-contiguous layout bk[N][S][K].
 *  No shared per-column exponent — each BF16 value carries its own.
 */
LIBXS_API_INLINE void oz3_preprocess_cols(const GEMM_REAL_TYPE* b, GEMM_INT_TYPE ldb, int tb,
  GEMM_INT_TYPE N, GEMM_INT_TYPE K, GEMM_INT_TYPE jb, GEMM_INT_TYPE kb, GEMM_INT_TYPE jblk,
  GEMM_INT_TYPE kblk, libxs_bf16_t bk[BLOCK_N][MAX_NSLICES][BLOCK_K])
{
  GEMM_INT_TYPE nj, kk;
  int s;
  for (kk = 0; kk < kblk; ++kk) {
    const GEMM_INT_TYPE p = kb + kk;
    for (nj = 0; nj < jblk; ++nj) {
      const GEMM_INT_TYPE col = jb + nj;
      const GEMM_REAL_TYPE bval = ((p < K && col < N)
        ? b[LIBXS_INDEX(tb, ldb, p, col)] : (GEMM_REAL_TYPE)0);
      libxs_bf16_t tmp[MAX_NSLICES];
      ozaki_split_to_bf16(bval, tmp);
      LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
      for (s = 0; s < ozaki_n; ++s) bk[nj][s][kk] = tmp[s];
    }
  }
  /* Zero-pad remaining k-entries */
  for (nj = 0; nj < jblk; ++nj) {
    LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
    for (s = 0; s < ozaki_n; ++s) {
      for (kk = kblk; kk < BLOCK_K; ++kk) bk[nj][s][kk] = 0;
    }
  }
}


LIBXS_API_INLINE void gemm_oz3_diff(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc,
  unsigned int diff_abc, libxs_matdiff_info_t* diff)
{
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
  const int nslices = LIBXS_CLMP(ozaki_n, 1, MAX_NSLICES);
  const GEMM_INT_TYPE nblk_m = (M + BLOCK_M - 1) / BLOCK_M;
  const GEMM_INT_TYPE nblk_n = (N + BLOCK_N - 1) / BLOCK_N;
  /* Panel buffers: preprocessed A and B for one K-batch.
   * No per-row/per-column exponent panels needed — BF16 slices are
   * self-describing.  This halves the metadata versus oz1/oz2. */
  libxs_bf16_t (*ak_panel)[BLOCK_M][MAX_NSLICES][BLOCK_K] = NULL;
  libxs_bf16_t (*bk_panel)[BLOCK_N][MAX_NSLICES][BLOCK_K] = NULL;
  GEMM_REAL_TYPE *ref_panel = NULL; /* diff mode 0 only */
  libxs_matdiff_info_t tdiff[256];
  int nthreads = 1;
  int s;
  const ozaki_dot_bf16_fn dot_bf16 = ozaki_dot_bf16_init();
  LIBXS_ASSERT(LIBXS_DATATYPE_F64 == LIBXS_DATATYPE(GEMM_REAL_TYPE)
            || LIBXS_DATATYPE_F32 == LIBXS_DATATYPE(GEMM_REAL_TYPE));
  LIBXS_ASSERT(1 <= BATCH_K);

  ak_panel = (libxs_bf16_t (*)[BLOCK_M][MAX_NSLICES][BLOCK_K])libxs_malloc(
    gemm_pool, (size_t)NKB_MAX * nblk_m * sizeof(*ak_panel), 0);
  bk_panel = (libxs_bf16_t (*)[BLOCK_N][MAX_NSLICES][BLOCK_K])libxs_malloc(
    gemm_pool, (size_t)NKB_MAX * nblk_n * sizeof(*bk_panel), 0);
  if (NULL != diff && 0 == (diff_abc % 3)) {
    ref_panel = (GEMM_REAL_TYPE*)libxs_malloc(
      gemm_pool, (size_t)nblk_m * nblk_n * BLOCK_MN * sizeof(*ref_panel), 0);
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

      /* Phase 1: preprocess all A row-blocks — Dekker split only, no exponents */
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
        oz3_preprocess_rows(a, *lda, ta, M, K, ib2, kb, iblk2, kblk,
          ak_panel[ki * nblk_m + ibi]);
      }

      /* Phase 2: preprocess all B col-blocks — Dekker split only, no exponents */
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
        oz3_preprocess_cols(b, *ldb, tb, N, K, jb2, kb, jblk2, kblk,
          bk_panel[ki * nblk_n + jbi]);
      }
      /* implicit barrier ensures panels are ready */

      /* Phase 3: BF16 dot products + accumulate.
       * No exponent reconstruction needed — the dot product result from
       * VDPBF16PS (or the scalar fallback) is already a properly scaled
       * FP32 value because each BF16 slice carries its own exponent. */
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

            /* Diff mode 1: track A decomposition accuracy */
            if (NULL != diff && 1 == (diff_abc % 3)) {
              GEMM_REAL_TYPE ref_blk[BLOCK_MNK];
              for (mi = 0; mi < iblk; ++mi) {
                const GEMM_INT_TYPE row = ib + mi;
                for (kk = 0; kk < kblk; ++kk) {
                  const GEMM_INT_TYPE p = kb + kk;
                  const GEMM_REAL_TYPE aval = ((row < M && p < K) ?
                    a[LIBXS_INDEX(ta, *lda, row, p)] : (GEMM_REAL_TYPE)0);
                  double arecon = 0.0;
                  int si;
                  for (si = 0; si < ozaki_n; ++si) {
                    arecon += libxs_bf16_to_f64(ak_panel[a_idx][mi][si][kk]);
                  }
                  ozaki_store_block_pair(ref_blk, recon_blk, BLOCK_M, mi, kk,
                    (GEMM_REAL_TYPE)aval, (GEMM_REAL_TYPE)arecon);
                }
              }
              ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, recon_blk, iblk, kblk, BLOCK_M, BLOCK_M);
            }

            /* Diff mode 2: track B decomposition accuracy */
            if (NULL != diff && 2 == (diff_abc % 3)) {
              GEMM_REAL_TYPE ref_blk[BLOCK_MNK];
              for (kk = 0; kk < kblk; ++kk) {
                const GEMM_INT_TYPE p = kb + kk;
                for (nj = 0; nj < jblk; ++nj) {
                  const GEMM_INT_TYPE col = jb + nj;
                  const GEMM_REAL_TYPE bval = ((p < K && col < N)
                    ? b[LIBXS_INDEX(tb, *ldb, p, col)] : (GEMM_REAL_TYPE)0);
                  double brecon = 0.0;
                  int si;
                  for (si = 0; si < ozaki_n; ++si) {
                    brecon += libxs_bf16_to_f64(bk_panel[b_idx][nj][si][kk]);
                  }
                  ozaki_store_block_pair(ref_blk, recon_blk, BLOCK_K, kk, nj,
                    (GEMM_REAL_TYPE)bval, (GEMM_REAL_TYPE)brecon);
                }
              }
              ozaki_accumulate_block_diff(&tdiff[tid], ref_blk, recon_blk, kblk, jblk, BLOCK_K, BLOCK_K);
            }

            { /* Diagonal-trim loop: iterate pairs (sa,sb) with sa+sb <= cutoff.
               * trim=0 means all pairs; larger values drop the least
               * significant diagonals (~8 bits each for BF16 slices). */
              const int trim = LIBXS_MIN(ozaki_trim, 2 * (nslices - 1));
              const int cutoff = 2 * (nslices - 1) - trim;
              LIBXS_PRAGMA_LOOP_COUNT(1, MAX_NSLICES, NSLICES_DEFAULT)
              for (slice_a = 0; slice_a < nslices && slice_a <= cutoff; ++slice_a) {
                const int sb_start = (0 != (ozaki_flags & OZ1_TRIANGULAR))
                  ? slice_a : 0;
                const int sb_end = LIBXS_MIN(nslices, cutoff + 1 - slice_a);
                for (slice_b = sb_start; slice_b < sb_end; ++slice_b) {
                  const int do_mirror = (0 != (ozaki_flags & OZ1_SYMMETRIZE))
                    && (slice_a != slice_b);
                  for (mi = 0; mi < iblk; ++mi) {
                    for (nj = 0; nj < jblk; ++nj) {
                      float dot = dot_bf16(ak_panel[a_idx][mi][slice_a], bk_panel[b_idx][nj][slice_b]);
                      if (do_mirror) {
                        dot += dot_bf16(ak_panel[a_idx][mi][slice_b], bk_panel[b_idx][nj][slice_a]);
                      }
                      if (0.0f != dot) {
                        cb[mi + nj * ldcv] += (GEMM_REAL_TYPE)((*alpha) * (double)dot);
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
  libxs_free(ak_panel);
  libxs_free(bk_panel);
  libxs_free(ref_panel);
}


LIBXS_API void gemm_oz3(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  OZAKI_GEMM_WRAPPER(gemm_oz3_diff)
}


#if defined(__LIBXSTREAM)
/**
 * Host preprocessing wrapper for A (rows) — scheme 3 (bf16 Dekker).
 * No exponent buffer (each bf16 value carries its own exponent).
 * Layout: slices[panel][BM][nslices][BK] (ushort elements).
 *   panel = ki * nblk + ib_idx.
 */
void oz3_host_preprocess_a(
    const void* matrix, int ld, int trans,
    int dim, int K, int kb_batch,
    int nkb, int nblk,
    int brc, int bk, int nslices_p,
    int kgroup_p, int use_xmx_p,
    void* slices, void* exp)
{
  const GEMM_REAL_TYPE* a = (const GEMM_REAL_TYPE*)matrix;
  int ki, ib_idx;
  (void)kgroup_p; (void)use_xmx_p; (void)exp;
  LIBXS_ASSERT(brc == BLOCK_M && bk == BLOCK_K);
  for (ki = 0; ki < nkb; ++ki) {
    for (ib_idx = 0; ib_idx < nblk; ++ib_idx) {
      const int panel = ki * nblk + ib_idx;
      const int ib = ib_idx * brc;
      const int kb = kb_batch + ki * bk;
      const GEMM_INT_TYPE iblk = (GEMM_INT_TYPE)LIBXS_MIN(brc, dim - ib);
      const GEMM_INT_TYPE kblk = (GEMM_INT_TYPE)LIBXS_MIN(bk, K - kb);
      libxs_bf16_t ak_tile[BLOCK_M][MAX_NSLICES][BLOCK_K];
      int mi;
      oz3_preprocess_rows(a, (GEMM_INT_TYPE)ld, trans,
        (GEMM_INT_TYPE)dim, (GEMM_INT_TYPE)K,
        (GEMM_INT_TYPE)ib, (GEMM_INT_TYPE)kb, iblk, kblk, ak_tile);
      for (mi = 0; mi < brc; ++mi) {
        memcpy((unsigned short*)slices + ((long)panel * brc + mi) * nslices_p * bk,
          &ak_tile[mi][0][0], (size_t)nslices_p * bk * sizeof(unsigned short));
      }
    }
  }
}


/**
 * Host preprocessing wrapper for B (columns) — scheme 3 (bf16 Dekker).
 * Layout (non-XMX): slices[panel][BN][nslices][BK] (ushort).
 * Layout (XMX):     slices[panel][nslices][BK][bn_pad] (ushort).
 *   panel = ki * nblk + jb_idx.
 */
void oz3_host_preprocess_b(
    const void* matrix, int ld, int trans,
    int dim, int K, int kb_batch,
    int nkb, int nblk,
    int brc, int bk, int nslices_p,
    int kgroup_p, int use_xmx_p,
    void* slices, void* exp)
{
  const GEMM_REAL_TYPE* b = (const GEMM_REAL_TYPE*)matrix;
  const int bn_pad = (0 != use_xmx_p) ? LIBXS_MAX(brc, 32) : brc;
  int ki, jb_idx;
  (void)kgroup_p; (void)exp;
  LIBXS_ASSERT(brc == BLOCK_N && bk == BLOCK_K);
  for (ki = 0; ki < nkb; ++ki) {
    for (jb_idx = 0; jb_idx < nblk; ++jb_idx) {
      const int panel = ki * nblk + jb_idx;
      const int jb = jb_idx * brc;
      const int kb = kb_batch + ki * bk;
      const GEMM_INT_TYPE jblk = (GEMM_INT_TYPE)LIBXS_MIN(brc, dim - jb);
      const GEMM_INT_TYPE kblk = (GEMM_INT_TYPE)LIBXS_MIN(bk, K - kb);
      libxs_bf16_t bk_tile[BLOCK_N][MAX_NSLICES][BLOCK_K];
      int nj, s, kk;
      oz3_preprocess_cols(b, (GEMM_INT_TYPE)ld, trans,
        (GEMM_INT_TYPE)dim, (GEMM_INT_TYPE)K,
        (GEMM_INT_TYPE)jb, (GEMM_INT_TYPE)kb, jblk, kblk, bk_tile);
      if (0 == use_xmx_p) {
        for (nj = 0; nj < brc; ++nj) {
          memcpy((unsigned short*)slices + ((long)panel * brc + nj) * nslices_p * bk,
            &bk_tile[nj][0][0], (size_t)nslices_p * bk * sizeof(unsigned short));
        }
      }
      else {
        unsigned short* dst = (unsigned short*)slices;
        for (s = 0; s < nslices_p; ++s) {
          for (kk = 0; kk < bk; ++kk) {
            long base = (((long)panel * nslices_p + s) * bk + kk) * bn_pad;
            for (nj = 0; nj < brc; ++nj) {
              dst[base + nj] = bk_tile[nj][s][kk];
            }
            for (nj = brc; nj < bn_pad; ++nj) {
              dst[base + nj] = 0;
            }
          }
        }
      }
    }
  }
}
#endif /* __LIBXSTREAM */
