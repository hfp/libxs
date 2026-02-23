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
#define gemm_function_t   LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ftype_t)
#define zgemm_function_t  LIBXS_CONCATENATE(GEMM_ZPREFIX, gemm_ftype_t)
#define gemm_original     LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_original)
#define zgemm_original    LIBXS_CONCATENATE(GEMM_ZPREFIX, gemm_original)
#define gemm_lock         LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_lock)
#define gemm_ozn          LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ozn)
#define gemm_ozflags      LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ozflags)
#define gemm_diff_abc     LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_diff_abc)
#define gemm_eps          LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_eps)
#define gemm_rsq          LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_rsq)
#define ozaki_target_arch LIBXS_TPREFIX(GEMM_REAL_TYPE, ozaki_tarch)
#define gemm_oz1          LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz1)
#define gemm_oz2          LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_oz2)
#define gemm_dump_inhibit LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_dump_inhibit)
#define gemm_dump_matrices LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_dump_mhd)
#define zgemm3m           LIBXS_CONCATENATE(GEMM_ZPREFIX, gemm3m)
#define print_diff_atexit LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_atexit)

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
LIBXS_APIVAR_PRIVATE(int gemm_diff_abc);
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
