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


LIBXS_EXTERN_C libxs_gemmfunction mmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k);
LIBXS_EXTERN_C libxs_gemmfunction mmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k)
{
  libxs_gemmfunction result;
#if defined(__cplusplus) /* C++ by chance: test libxs_mmfunction<> wrapper */
  const libxs_mmfunction<ITYPE, OTYPE, LIBXS_PREFETCH> mmfunction(m, n, k);
  result = mmfunction.kernel();
#else
  const libxs_gemm_shape gemm_shape = libxs_create_gemm_shape(m, n, k, m/*lda*/, k/*ldb*/, m/*ldc*/,
    LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(OTYPE), LIBXS_DATATYPE(OTYPE));
  result = libxs_dispatch_gemm(gemm_shape, LIBXS_GEMM_FLAG_NONE,
    (libxs_bitfield)LIBXS_PREFETCH);
#endif
  return result;
}
