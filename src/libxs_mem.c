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
#include <libxs_mem.h>
#include "libxs_hash.h"
#include "libxs_diff.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <stdio.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_DIFF_MEMCMP) && 0
# define LIBXS_DIFF_MEMCMP
#endif


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
#if defined(LIBXS_DIFF_MEMCMP)
  LIBXS_DIFF_N(unsigned int, result, memcmp, a, bn, size, stride, hint, n);
#else
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
#endif
  return result;
}


LIBXS_API int libxs_memcmp(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_DIFF_MEMCMP)
  return memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  LIBXS_DIFF_32_DECL(aa);
  size_t i;
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFE0); i += 32) {
    LIBXS_DIFF_32_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_32(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#endif
}


LIBXS_API unsigned int libxs_hash(const void* data, unsigned int size, unsigned int seed)
{
  LIBXS_INIT
  return libxs_crc32(seed, data, size);
}


LIBXS_API unsigned long long libxs_hash_string(const char* string)
{
  unsigned long long result;
  const size_t length = NULL != string ? strlen(string) : 0;
  if (sizeof(result) < length) {
    const size_t length2 = length / 2;
    unsigned int seed32 = 0; /* seed=0: match else-optimization */
    LIBXS_INIT
    seed32 = libxs_crc32(seed32, string, length2);
    result = libxs_crc32(seed32, string + length2, length - length2);
    result = (result << 32) | seed32;
  }
  else { /* reinterpret directly as hash value */
    char *const s = (char*)&result; signed char i;
    for (i = 0; i < (signed char)length; ++i) s[i] = string[i];
    for (; i < (signed char)sizeof(result); ++i) s[i] = 0;
  }
  return result;
}


#if defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_hash)(void* /*hash_seed*/, const void* /*data*/, const int* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_hash)(void* hash_seed, const void* data, const int* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != hash_seed && NULL != data && NULL != size && 0 <= *size)
#endif
  {
    unsigned int *const hash_seed_ui32 = (unsigned int*)hash_seed;
    *hash_seed_ui32 = (libxs_hash(data, (unsigned int)*size, *hash_seed_ui32) & 0x7FFFFFFF/*sign-bit*/);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_hash specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_hash_char)(void* /*hash_seed*/, const void* /*data*/, const int* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_hash_char)(void* hash_seed, const void* data, const int* size)
{
  LIBXS_FSYMBOL(libxs_hash)(hash_seed, data, size);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_hash_i8)(void* /*hash_seed*/, const void* /*data*/, const int* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_hash_i8)(void* hash_seed, const void* data, const int* size)
{
  LIBXS_FSYMBOL(libxs_hash)(hash_seed, data, size);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_hash_i32)(void* /*hash_seed*/, const void* /*data*/, const int* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_hash_i32)(void* hash_seed, const void* data, const int* size)
{
  LIBXS_FSYMBOL(libxs_hash)(hash_seed, data, size);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_hash_i64)(void* /*hash_seed*/, const void* /*data*/, const int* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_hash_i64)(void* hash_seed, const void* data, const int* size)
{
  LIBXS_FSYMBOL(libxs_hash)(hash_seed, data, size);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_diff)(int* /*result*/, const void* /*a*/, const void* /*b*/, const long long* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_diff)(int* result, const void* a, const void* b, const long long* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != result && NULL != a && NULL != b && NULL != size && 0 <= *size)
#endif
  {
    *result = libxs_memcmp(a, b, (size_t)*size);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_memcmp specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_diff_char)(int* /*result*/, const void* /*a*/, const void* /*b*/, const long long* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_diff_char)(int* result, const void* a, const void* b, const long long* size)
{
  LIBXS_FSYMBOL(libxs_diff)(result, a, b, size);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_diff_i8)(int* /*result*/, const void* /*a*/, const void* /*b*/, const long long* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_diff_i8)(int* result, const void* a, const void* b, const long long* size)
{
  LIBXS_FSYMBOL(libxs_diff)(result, a, b, size);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_diff_i32)(int* /*result*/, const void* /*a*/, const void* /*b*/, const long long* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_diff_i32)(int* result, const void* a, const void* b, const long long* size)
{
  LIBXS_FSYMBOL(libxs_diff)(result, a, b, size);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_diff_i64)(int* /*result*/, const void* /*a*/, const void* /*b*/, const long long* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_diff_i64)(int* result, const void* a, const void* b, const long long* size)
{
  LIBXS_FSYMBOL(libxs_diff)(result, a, b, size);
}

#endif /*defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))*/
