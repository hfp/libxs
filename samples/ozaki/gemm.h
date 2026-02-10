/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_macros.h>
#include <libxs_utils.h>

#if !defined(GEMM_REAL_TYPE)
# define GEMM_REAL_TYPE double
#endif
#if !defined(GEMM_INT_TYPE)
# define GEMM_INT_TYPE int
#endif
#if !defined(GEMM)
# define GEMM LIBXS_FSYMBOL(dgemm)
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

/** Original GEMM function. */
LIBXS_APIVAR_PRIVATE(gemm_function_t gemm_original);
