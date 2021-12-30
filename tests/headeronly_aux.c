/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
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


LIBXS_EXTERN_C LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE) mmdispatch(int m, int n, int k);
LIBXS_EXTERN_C LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE) mmdispatch(int m, int n, int k)
{
  LIBXS_MMFUNCTION_TYPE2(ITYPE, OTYPE) result;
#if defined(__cplusplus) /* C++ by chance: test libxs_mmfunction<> wrapper */
  const libxs_mmfunction<ITYPE, OTYPE> mmfunction(m, n, k);
  result = mmfunction.kernel().LIBXS_TPREFIX2(ITYPE, OTYPE, mm);
#else
  result = LIBXS_MMDISPATCH_SYMBOL2(ITYPE, OTYPE)(m, n, k,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*flags*/);
#endif
  return result;
}

