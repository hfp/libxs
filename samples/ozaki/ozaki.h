/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"
#include <libxs_mhd.h>
#include <libxs_sync.h>


/*=== IEEE-754 format parameters derived from GEMM_REAL_TYPE ================*/
#if GEMM_IS_DOUBLE
# define OZ_MANT_BITS    52
# define OZ_EXP_BIAS     1023
#else /* single-precision */
# define OZ_MANT_BITS    23
# define OZ_EXP_BIAS     127
#endif
#define OZ_BIAS_PLUS_MANT (OZ_EXP_BIAS + OZ_MANT_BITS)


/*=== Block sizes ============================================================*/
#if !defined(BLOCK_M)
# define BLOCK_M 16
#endif
#if !defined(BLOCK_N)
# define BLOCK_N 16
#endif
#if !defined(BLOCK_K)
# define BLOCK_K 16
#endif


/*=== Scheme 1 (slice) limits ================================================*/
#if !defined(MAX_NSLICES)
# if GEMM_IS_DOUBLE
#   define MAX_NSLICES 16
# else
#   define MAX_NSLICES 8
# endif
#endif
#if !defined(NSLICES_DEFAULT)
# if GEMM_IS_DOUBLE
#   define NSLICES_DEFAULT 5
# else
#   define NSLICES_DEFAULT 4
# endif
#endif

/* Runtime flag-set controlling the Ozaki scheme 1 (GEMM_OZFLAGS env var).
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


/*=== Scheme 2 (CRT) limits =================================================*/
#if GEMM_IS_DOUBLE
# define OZ2_MAX_NPRIMES     16
# define OZ2_NPRIMES_DEFAULT 15
#else /* single-precision */
# define OZ2_MAX_NPRIMES    10
# define OZ2_NPRIMES_DEFAULT 7
#endif


/*=== Block-level helpers (shared by ozaki1.c and ozaki2.c) ==================*/

/** Scale a tile of C by beta, optionally capturing the pre-scaled block. */
LIBXS_API_INLINE void ozaki_scale_block_beta(GEMM_REAL_TYPE* mb, GEMM_INT_TYPE ldc,
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

/** Store a (reference, reconstructed) value pair into block buffers. */
LIBXS_API_INLINE void ozaki_store_block_pair(GEMM_REAL_TYPE* ref_blk,
  GEMM_REAL_TYPE* recon_blk, GEMM_INT_TYPE ld, GEMM_INT_TYPE row,
  GEMM_INT_TYPE col, GEMM_REAL_TYPE ref_val, GEMM_REAL_TYPE recon_val)
{
  recon_blk[row + col * ld] = recon_val;
  ref_blk[row + col * ld] = ref_val;
}

/** Compute matrix diff for one block and reduce into accumulator. */
LIBXS_API_INLINE void ozaki_accumulate_block_diff(libxs_matdiff_info_t* acc,
  const GEMM_REAL_TYPE* ref_blk, const GEMM_REAL_TYPE* tst_blk,
  GEMM_INT_TYPE bm, GEMM_INT_TYPE bn, GEMM_INT_TYPE ld_ref,
  GEMM_INT_TYPE ld_tst)
{
  libxs_matdiff_info_t block_diff;
  const int ild_ref = (int)ld_ref, ild_tst = (int)ld_tst;
  if (EXIT_SUCCESS == libxs_matdiff(&block_diff, LIBXS_DATATYPE(GEMM_REAL_TYPE),
    bm, bn, ref_blk, tst_blk, &ild_ref, &ild_tst))
  {
    libxs_matdiff_reduce(acc, &block_diff);
  }
}


/*=== Public API wrapper (shared verbose/dump/diff logic) ====================*/

/** Implement the public gemm_ozN function: call the _diff kernel,
 *  then handle verbose output, diff accumulation, and matrix dumps.
 *  DIFF_FN is the _diff kernel (gemm_oz1_diff or gemm_oz2_diff). */
#define OZAKI_GEMM_WRAPPER(DIFF_FN) \
  if (0 == gemm_verbose) { \
    DIFF_FN(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, \
      0, NULL); \
  } \
  else { \
    double epsilon; \
    libxs_matdiff_info_t diff; \
    libxs_matdiff_clear(&diff); \
    DIFF_FN(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, \
      LIBXS_ABS(gemm_wrap), &diff); \
    LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER); \
    libxs_matdiff_reduce(&gemm_diff, &diff); diff = gemm_diff; \
    LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER); \
    epsilon = libxs_matdiff_epsilon(&diff); \
    if (1 < gemm_verbose || 0 > gemm_verbose) { \
      const int nth = (0 < gemm_verbose ? gemm_verbose : 1); \
      if (0 == (diff.r % nth)) print_diff(stderr, &diff); \
    } \
    if (gemm_eps < epsilon || diff.rsq < gemm_rsq || 0 > gemm_verbose) { \
      if (0 != gemm_dump_inhibit) { \
        gemm_dump_inhibit = 2; \
      } \
      else { \
        gemm_dump_matrices(GEMM_ARGPASS, 1); \
      } \
    } \
  }

/* Wrap/real symbol definitions for real GEMM */
#define GEMM_WRAP LIBXS_CONCATENATE(__wrap_, GEMM)
#define GEMM_REAL LIBXS_CONCATENATE(__real_, GEMM)

/* Wrap/real symbol definitions for complex GEMM */
#define ZGEMM_WRAP LIBXS_CONCATENATE(__wrap_, ZGEMM)
#define ZGEMM_REAL LIBXS_CONCATENATE(__real_, ZGEMM)

/* Precision-specific type and variable names (enable dual-precision builds).
 * These macros redirect "friendly" names used throughout the implementation
 * to unique symbols, e.g. gemm_original -> dgemm_original (double) or
 * sgemm_original (float). Both precisions can coexist in one binary. */
#define gemm_function_t     LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ftype_t)
#define zgemm_function_t    LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm_ftype_t)
#define gemm_original       LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_original)
#define zgemm_original      LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm_original)
#define gemm_lock           LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_lock)
#define gemm_ozn            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ozn)
#define gemm_ozflags        LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ozflags)
#define gemm_wrap           LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_wrap)
#define gemm_eps            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_eps)
#define gemm_rsq            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_rsq)
#define ozaki_target_arch   LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_tarch)
#define gemm_oz1            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz1)
#define gemm_oz2            LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz2)
#define gemm_dump_inhibit   LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_dump_inhibit)
#define gemm_dump_matrices  LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_dump_mhd)
#define zgemm3m             LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm3m)
#define print_diff_atexit   LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_atexit)

/** Function type for GEMM (precision-specific). */
LIBXS_EXTERN_C typedef void (*gemm_function_t)(GEMM_ARGDECL);
/** Function type for complex GEMM (precision-specific). */
LIBXS_EXTERN_C typedef void (*zgemm_function_t)(GEMM_ARGDECL);

/** Function prototypes for wrapped / real / public GEMM and complex GEMM. */
LIBXS_API_INTERN void GEMM_WRAP(GEMM_ARGDECL);
LIBXS_API_INTERN void GEMM_REAL(GEMM_ARGDECL);
LIBXS_API_INTERN void ZGEMM_WRAP(GEMM_ARGDECL);
LIBXS_API_INTERN void ZGEMM_REAL(GEMM_ARGDECL);
LIBXS_API void ZGEMM(GEMM_ARGDECL);

/** Function prototype for GEMM using low-precision (Ozaki scheme 1). */
LIBXS_API void gemm_oz1(GEMM_ARGDECL);
/** Function prototype for GEMM using CRT modular arithmetic (Ozaki scheme 2). */
LIBXS_API void gemm_oz2(GEMM_ARGDECL);
/** Complex GEMM 3M (Karatsuba) implementation (internal). */
LIBXS_API_INTERN void zgemm3m(GEMM_ARGDECL);

LIBXS_APIVAR_PRIVATE(volatile LIBXS_ATOMIC_LOCKTYPE gemm_lock);
LIBXS_APIVAR_PRIVATE(gemm_function_t gemm_original);
LIBXS_APIVAR_PRIVATE(zgemm_function_t zgemm_original);
LIBXS_APIVAR_PRIVATE(int gemm_ozn);
LIBXS_APIVAR_PRIVATE(int gemm_ozflags);
LIBXS_APIVAR_PUBLIC(int gemm_wrap);
extern LIBXS_TLS int gemm_dump_inhibit;
LIBXS_APIVAR_PRIVATE(double gemm_eps);
LIBXS_APIVAR_PRIVATE(double gemm_rsq);
LIBXS_APIVAR_PRIVATE(int ozaki_target_arch);

/**
 * Dump A and B matrices as MHD files.
 * Works for both real (ncomponents=1) and complex (ncomponents=2) matrices.
 * Uses gemm_diff.r as the dump ID and updates gemm_eps/gemm_rsq thresholds
 * to avoid repeated dumps.
 */
LIBXS_API_INLINE void gemm_dump_matrices(GEMM_ARGDECL, size_t ncomponents)
{
  const size_t ext_size = sizeof(char) + sizeof(GEMM_INT_TYPE)
                        + ncomponents * sizeof(GEMM_REAL_TYPE);
  char extension[sizeof(char) + sizeof(GEMM_INT_TYPE) + 2 * sizeof(GEMM_REAL_TYPE)];
  libxs_mhd_info_t mhd_info;
  char fname[64];
  size_t size[2], pitch[2];
  FILE *file;
  int result = EXIT_SUCCESS;
  int dump_id;

  mhd_info.ndims = 2;
  mhd_info.ncomponents = ncomponents;
  mhd_info.type = LIBXS_DATATYPE(GEMM_REAL_TYPE);
  mhd_info.header_size = 0;

  LIBXS_ATOMIC_ACQUIRE(&gemm_lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
  dump_id = gemm_diff.r;

  LIBXS_SNPRINTF(fname, sizeof(fname), "gemm-%i-a.mhd", dump_id);
  file = fopen(fname, "rb");
  if (NULL == file) { /* Never overwrite an existing file */
    size[0] = *m; size[1] = *k; pitch[0] = *lda; pitch[1] = *k;
    *(char*)extension = *transa;
    memcpy(extension + sizeof(char), lda, sizeof(GEMM_INT_TYPE));
    memcpy(extension + sizeof(char) + sizeof(GEMM_INT_TYPE),
      alpha, ncomponents * sizeof(GEMM_REAL_TYPE));
    result |= libxs_mhd_write(fname, NULL/*offset*/, size, pitch,
      &mhd_info, a, NULL/*handler_info*/, NULL/*handler*/,
      NULL/*extension_header*/, extension, ext_size);
  }
  else fclose(file);

  LIBXS_SNPRINTF(fname, sizeof(fname), "gemm-%i-b.mhd", dump_id);
  file = fopen(fname, "rb");
  if (NULL == file) { /* Never overwrite an existing file */
    size[0] = *k; size[1] = *n; pitch[0] = *ldb; pitch[1] = *n;
    *(char*)extension = *transb;
    memcpy(extension + sizeof(char), ldb, sizeof(GEMM_INT_TYPE));
    memcpy(extension + sizeof(char) + sizeof(GEMM_INT_TYPE),
      beta, ncomponents * sizeof(GEMM_REAL_TYPE));
    result |= libxs_mhd_write(fname, NULL/*offset*/, size, pitch,
      &mhd_info, b, NULL/*handler_info*/, NULL/*handler*/,
      NULL/*extension_header*/, extension, ext_size);
  }
  else fclose(file);

  if (EXIT_SUCCESS == result) {
    print_gemm(stdout, GEMM_ARGPASS);
  }

  /* avoid repeated dumps */
  { const double epsilon = libxs_matdiff_epsilon(&gemm_diff);
    gemm_rsq = gemm_diff.rsq;
    gemm_eps = epsilon;
  }
  LIBXS_ATOMIC_RELEASE(&gemm_lock, LIBXS_ATOMIC_LOCKORDER);
}
