/******************************************************************************
** Copyright (c) 2017-2019, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
#include <libxs_math.h>
#include "libxs_diff.h"


LIBXS_API unsigned char libxs_diff_16(const void* a, const void* b, ...)
{
  LIBXS_DIFF_16_DECL(a16);
  LIBXS_DIFF_16_LOAD(a16, a);
  return LIBXS_DIFF_16(a16, b, 0/*dummy*/);
}


LIBXS_API unsigned char libxs_diff_32(const void* a, const void* b, ...)
{
  LIBXS_DIFF_32_DECL(a32);
  LIBXS_DIFF_32_LOAD(a32, a);
  return LIBXS_DIFF_32(a32, b, 0/*dummy*/);
}


LIBXS_API unsigned char libxs_diff_48(const void* a, const void* b, ...)
{
  LIBXS_DIFF_48_DECL(a48);
  LIBXS_DIFF_48_LOAD(a48, a);
  return LIBXS_DIFF_48(a48, b, 0/*dummy*/);
}


LIBXS_API unsigned char libxs_diff_64(const void* a, const void* b, ...)
{
  LIBXS_DIFF_64_DECL(a64);
  LIBXS_DIFF_64_LOAD(a64, a);
  return LIBXS_DIFF_64(a64, b, 0/*dummy*/);
}


LIBXS_API unsigned char libxs_diff(const void* a, const void* b, unsigned char size)
{
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  for (i = 0; i < (size & 0xF0); i += 16) {
    LIBXS_DIFF_16_DECL(a16);
    LIBXS_DIFF_16_LOAD(a16, a8 + i);
    if (LIBXS_DIFF_16(a16, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
}


LIBXS_API unsigned int libxs_diff_n(const void* a, const void* bn, unsigned char size,
  unsigned char stride, unsigned int hint, unsigned int n)
{
  unsigned int result;
  LIBXS_ASSERT(size <= stride);
  switch (size) {
    case 64: {
      LIBXS_DIFF_64_DECL(a64);
      LIBXS_DIFF_64_LOAD(a64, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_64, a64, bn, size, stride, hint, n);
    } break;
    case 48: {
      LIBXS_DIFF_48_DECL(a48);
      LIBXS_DIFF_48_LOAD(a48, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_48, a48, bn, size, stride, hint, n);
    } break;
    case 32: {
      LIBXS_DIFF_32_DECL(a32);
      LIBXS_DIFF_32_LOAD(a32, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_32, a32, bn, size, stride, hint, n);
    } break;
    case 16: {
      LIBXS_DIFF_16_DECL(a16);
      LIBXS_DIFF_16_LOAD(a16, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_16, a16, bn, size, stride, hint, n);
    } break;
    default: {
      LIBXS_DIFF_N(unsigned int, result, libxs_diff, a, bn, size, stride, hint, n);
    }
  }
  return result;
}

