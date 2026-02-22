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
/* Precision detection: GEMM_IS_DOUBLE expands to 1 for double, 0 for float */
#define GEMM_PREC_F64 1
#define GEMM_PREC_F32 0
#define GEMM_IS_DOUBLE LIBXS_CONCATENATE(GEMM_PREC_, LIBXS_TYPESYMBOL(GEMM_REAL_TYPE))
#if !defined(GEMM)
# define GEMM LIBXS_FSYMBOL(LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm))
#endif
#define GEMM_WRAP LIBXS_CONCATENATE(__wrap_, GEMM)
#define GEMM_REAL LIBXS_CONCATENATE(__real_, GEMM)


/** Function type for GEMM. */
LIBXS_EXTERN_C typedef void (*gemm_function_t)(const char*, const char*,
  const GEMM_INT_TYPE*, const GEMM_INT_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
                         const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, GEMM_REAL_TYPE*, const GEMM_INT_TYPE*);

/** Function prototype for wrapped GEMM. */
LIBXS_API_INTERN void GEMM_WRAP(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda
                             , const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc);

/** Function prototype for real GEMM. */
LIBXS_API_INTERN void GEMM_REAL(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc);

/** Function prototype for GEMM. */
LIBXS_API void GEMM(const char*, const char*,
  const GEMM_INT_TYPE*, const GEMM_INT_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
                         const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, GEMM_REAL_TYPE*, const GEMM_INT_TYPE*);

/** Function prototype for GEMM using low-precision (Ozaki scheme 1). */
LIBXS_API void gemm_oz1(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc);

/** Print GEMM arguments. */
LIBXS_API void print_gemm(FILE* ostream, const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc);

/** Print statistics (usually gemm_diff). */
LIBXS_API void print_diff(FILE* ostream, const libxs_matdiff_info_t* diff);

LIBXS_APIVAR_PUBLIC(libxs_matdiff_info_t gemm_diff);
LIBXS_APIVAR_PUBLIC(int gemm_verbose);

/** Original GEMM function (private). */
LIBXS_APIVAR_PRIVATE(gemm_function_t gemm_original);

/* Complex GEMM support: derive complex symbol from real type.
 * double -> zgemm_, float -> cgemm_ (BLAS convention). */
#define GEMM_ZPREFIX_double z
#define GEMM_ZPREFIX_float c
#define GEMM_ZPREFIX LIBXS_CONCATENATE(GEMM_ZPREFIX_, GEMM_REAL_TYPE)
#define ZGEMM LIBXS_FSYMBOL(LIBXS_CONCATENATE(GEMM_ZPREFIX, gemm))
#define ZGEMM_WRAP LIBXS_CONCATENATE(__wrap_, ZGEMM)
#define ZGEMM_REAL LIBXS_CONCATENATE(__real_, ZGEMM)

/** Function type for complex GEMM (interleaved real/imag pairs). */
LIBXS_EXTERN_C typedef void (*zgemm_function_t)(const char*, const char*,
  const GEMM_INT_TYPE*, const GEMM_INT_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
                         const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, GEMM_REAL_TYPE*, const GEMM_INT_TYPE*);

/** Function prototype for wrapped complex GEMM. */
LIBXS_API_INTERN void ZGEMM_WRAP(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc);

/** Function prototype for real complex GEMM. */
LIBXS_API_INTERN void ZGEMM_REAL(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc);

/** Function prototype for complex GEMM. */
LIBXS_API void ZGEMM(const char*, const char*,
  const GEMM_INT_TYPE*, const GEMM_INT_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
                         const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, GEMM_REAL_TYPE*, const GEMM_INT_TYPE*);

/** Original complex GEMM function (private). */
LIBXS_APIVAR_PRIVATE(zgemm_function_t zgemm_original);
