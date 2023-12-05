/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_macros.h>

#if !defined(ELEMTYPE)
# define ELEMTYPE short
#endif

#define VLA_IJK_DECL(DIM, TYPE, ARRAY, DATA, S1, S2) LIBXS_VLA_DECL(  DIM, TYPE, ARRAY, DATA, S1, S2)
#define VLA_IJK_INDX(DIM, ARRAY, I0, I1, I2, S1, S2) LIBXS_VLA_ACCESS(DIM, ARRAY, I0, I1, I2, S1, S2)

#define VLA_IKJ_DECL(DIM, TYPE, ARRAY, DATA, S1, S2) LIBXS_VLA_DECL(  DIM, TYPE, ARRAY, DATA, S2, S1)
#define VLA_IKJ_INDX(DIM, ARRAY, I0, I1, I2, S1, S2) LIBXS_VLA_ACCESS(DIM, ARRAY, I0, I2, I1, S2, S1)


int main(int argc, char* argv[])
{
  int ni = 9, nj = 7, nk = 3, i, j, k, linear = 0, result = EXIT_SUCCESS;
  ELEMTYPE *const input = (ELEMTYPE*)malloc(sizeof(ELEMTYPE) * ni * nj * nk);
  LIBXS_VLA_DECL(1, const ELEMTYPE, in1, input);
  VLA_IJK_DECL(3, const ELEMTYPE, jk3, input, nj, nk);
  VLA_IKJ_DECL(3, const ELEMTYPE, kj3, input, nj, nk);
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  LIBXS_ASSERT(NULL != input);
  for (i = 0; i < (ni * nj * nk); ++i) input[i] = (ELEMTYPE)i;
  for (i = 0; i < ni && EXIT_SUCCESS == result; ++i) {
    for (j = 0; j < nj; ++j) {
      for (k = 0; k < nk; ++k) {
        const ELEMTYPE gold0 = input[linear];
        const ELEMTYPE test0a = VLA_IJK_INDX(3, jk3, i, j, k, nj, nk);
        const void *const vjk3 = LIBXS_CONCATENATE(jk3, LIBXS_VLA_POSTFIX);
        const void *const vpjk = LIBXS_ACCESS_RO(3, sizeof(ELEMTYPE), vjk3, i, j, k, nj, nk);
        const ELEMTYPE test0b = *(const ELEMTYPE*)vpjk;
        const ELEMTYPE test0c = *LIBXS_ACCESS(3, const ELEMTYPE, vjk3, i, j, k, nj, nk);
        const ELEMTYPE gold1 = VLA_IJK_INDX(3, kj3, i, k, j, nk, nj);
        const ELEMTYPE test1a = VLA_IKJ_INDX(3, kj3, i, j, k, nj, nk);
        const void *const vkj3 = LIBXS_CONCATENATE(kj3, LIBXS_VLA_POSTFIX);
        const void *const vpkj = LIBXS_ACCESS_RO(3, sizeof(ELEMTYPE), vkj3, i, k, j, nk, nj);
        const ELEMTYPE test1b = *(const ELEMTYPE*)vpkj;
        const ELEMTYPE test1c = *LIBXS_ACCESS(3, const ELEMTYPE, vkj3, i, k, j, nk, nj);
        if (gold0 != LIBXS_VLA_ACCESS(1, in1, linear) ||
            gold0 != test0a || gold1 != test1a ||
            test0a != test0b || test0b != test0c ||
            test1a != test1b || test1b != test1c)
        {
          result = EXIT_FAILURE;
          j = nj; break;
        }
        ++linear;
      }
    }
  }

  free(input);
  return result;
}
