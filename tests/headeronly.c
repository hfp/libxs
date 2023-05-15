/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_source.h>

/* must match definitions in headeronly_aux.c */
#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif


LIBXS_EXTERN_C libxs_gemmfunction mmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k);


int main(void)
{
  const libxs_blasint m = LIBXS_MAX_M, n = LIBXS_MAX_N, k = LIBXS_MAX_K;
  const libxs_gemm_shape gemm_shape = libxs_create_gemm_shape(
    m, n, k, m/*lda*/, k/*ldb*/, m/*ldc*/,
    LIBXS_DATATYPE(ITYPE), LIBXS_DATATYPE(ITYPE),
    LIBXS_DATATYPE(OTYPE), LIBXS_DATATYPE(OTYPE));
  const libxs_gemmfunction fa = libxs_dispatch_gemm_v2(gemm_shape,
    LIBXS_GEMM_FLAG_NONE, (libxs_bitfield)LIBXS_PREFETCH);
  const libxs_gemmfunction fb = mmdispatch(m, n, k);
  int result = EXIT_SUCCESS;

  if (fa == fb) { /* test unregistering and freeing kernel */
    union {
      libxs_gemmfunction f;
      const void* p;
    } kernel;
    kernel.f = fa;
    libxs_release_kernel(kernel.p);
  }
  else {
    libxs_registry_info registry_info;
    result = libxs_get_registry_info(&registry_info);
    if (EXIT_SUCCESS == result && 2 != registry_info.size) {
      result = EXIT_FAILURE;
    }
  }
  return result;
}
