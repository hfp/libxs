/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_math.h>

#if !defined(GEMM_REAL_TYPE)
# define GEMM_REAL_TYPE double
#endif
#if !defined(GEMM_INT_TYPE)
# define GEMM_INT_TYPE int
#endif

/* Precision detection: GEMM_IS_DOUBLE expands to 1 for double, 0 for float/else */
#define GEMM_IS_DOUBLE LIBXS_TYPEORDER(LIBXS_DATATYPE_F32) \
                     < LIBXS_TYPEORDER(LIBXS_DATATYPE(GEMM_REAL_TYPE))
/* GEMM symbol (dgemm_ for double, sgemm_ for float) */
#if !defined(GEMM)
# define GEMM LIBXS_FSYMBOL(LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm))
#endif
/* Complex GEMM symbol (zgemm_ for double, cgemm_ for float) */
#if !defined(ZGEMM)
# define ZGEMM LIBXS_FSYMBOL(LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm))
#endif

/* Common GEMM argument list macros to reduce boilerplate */
#define GEMM_ARGDECL \
  const char* transa, const char* transb, \
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k, \
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, \
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb, \
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc
#define GEMM_ARGPASS transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc

/* Precision-specific name redirects for public/driver-visible symbols */
#define gemm_diff     LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_diff)
#define gemm_ozaki    LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ozaki)
#define gemm_stat     LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_stat)
#define gemm_verbose  LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_verbose)
#define print_gemm    LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_print)
#define print_diff    LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_print_diff)

/** Real GEMM entry point (dgemm_ or sgemm_). */
LIBXS_API void GEMM(GEMM_ARGDECL);
/** Complex GEMM entry point (zgemm_ or cgemm_). */
LIBXS_API void ZGEMM(GEMM_ARGDECL);

/** Print GEMM arguments. */
LIBXS_API void print_gemm(FILE* ostream, GEMM_ARGDECL);
/** Print statistics. */
LIBXS_API void print_diff(FILE* ostream, const libxs_matdiff_info_t* diff);

LIBXS_APIVAR_PUBLIC(libxs_matdiff_info_t gemm_diff);
LIBXS_APIVAR_PUBLIC(int gemm_ozaki);
LIBXS_APIVAR_PUBLIC(int gemm_verbose);
LIBXS_APIVAR_PUBLIC(int gemm_stat);
