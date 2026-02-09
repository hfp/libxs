/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_utils.h>


LIBXS_API int LIBXS_INTRINSICS_BITSCANFWD32_SW(unsigned int n)
{
  unsigned int i, r = 0;
  if (0 != n) for (i = 1; 0 == (n & i); i <<= 1) { ++r; }
  return r;
}


LIBXS_API int LIBXS_INTRINSICS_BITSCANFWD64_SW(unsigned long long n)
{
  unsigned int i, r = 0;
  if (0 != n) for (i = 1; 0 == (n & i); i <<= 1) { ++r; }
  return r;
}


#if defined(_WIN32) && !defined(LIBXS_INTRINSICS_NONE)

LIBXS_API unsigned int LIBXS_INTRINSICS_BITSCANFWD32(unsigned int n)
{
  unsigned long r = 0;
  _BitScanForward(&r, n);
  return (0 != n) * r;
}


LIBXS_API unsigned int LIBXS_INTRINSICS_BITSCANBWD32(unsigned int n)
{
  unsigned long r = 0;
  _BitScanReverse(&r, n);
  return r;
}


#if defined(_WIN64)
LIBXS_API unsigned int LIBXS_INTRINSICS_BITSCANFWD64(unsigned long long n)
{
  unsigned long r = 0;
  _BitScanForward64(&r, n);
  return (0 != n) * r;
}


LIBXS_API unsigned int LIBXS_INTRINSICS_BITSCANBWD64(unsigned long long n)
{
  unsigned long r = 0;
  _BitScanReverse64(&r, n);
  return r;
}
#endif

#endif


LIBXS_API unsigned int LIBXS_ILOG2(unsigned long long n)
{
  unsigned int result = 0;
  if (1 < n) {
    const unsigned int m = LIBXS_INTRINSICS_BITSCANBWD64(n);
    result = m + ((unsigned int)LIBXS_INTRINSICS_BITSCANBWD64(n - 1) == m);
  }
  return result;
}
