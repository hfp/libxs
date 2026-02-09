/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_macros.h>

#if !defined(GEMM_REAL_TYPE)
# define GEMM_REAL_TYPE double
#endif
#if !defined(GEMM_INT_TYPE)
# define GEMM_INT_TYPE int
#endif
#if !defined(GEMM)
# define GEMM dgemm_
#endif

/** Function type for GEMM. */
LIBXS_EXTERN_C typedef void (*gemm_function_t)(const char*, const char*, const GEMM_INT_TYPE*, const GEMM_INT_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*, const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, GEMM_REAL_TYPE*, const GEMM_INT_TYPE*);

/** Function prototype for GEMM. */
LIBXS_EXTERN_C void GEMM(const char*, const char*, const GEMM_INT_TYPE*, const GEMM_INT_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*, const GEMM_REAL_TYPE*, const GEMM_INT_TYPE*,
  const GEMM_REAL_TYPE*, GEMM_REAL_TYPE*, const GEMM_INT_TYPE*);
