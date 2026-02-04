/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_source.h>

/* must match definitions in headeronly.c */
#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif


LIBXS_EXTERN_C void* mmdispatch(int m, int n, int k);
LIBXS_EXTERN_C void* mmdispatch(int m, int n, int k)
{
  void* result = NULL;
#if 0
  const libxs_gemm_shape gemm_shape = libxs_create_gemm_shape(m, n, k, m/*lda*/, k/*ldb*/, m/*ldc*/,
    LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(OTYPE), LIBXS_DATATYPE(OTYPE));
  result = libxs_dispatch_gemm(gemm_shape, LIBXS_GEMM_FLAG_NONE,
    (libxs_bitfield)LIBXS_PREFETCH);
#endif
  return result;
}
