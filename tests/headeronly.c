/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
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


LIBXS_EXTERN_C LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE) mmdispatch(int m, int n, int k);


int main(void)
{
  const int m = LIBXS_MAX_M, n = LIBXS_MAX_N, k = LIBXS_MAX_K;
  const LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE) fa = LIBXS_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)(m, n, k,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*flags*/);
  const LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE) fb = mmdispatch(m, n, k);
  int result = EXIT_SUCCESS;

  if (fa == fb) { /* test unregistering and freeing kernel */
    union {
      LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE) f;
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

