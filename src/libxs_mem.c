/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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

#if !defined(LIBXS_MEM_MEMCMP) && 0
# define LIBXS_MEM_MEMCMP
#endif
#if !defined(LIBXS_MEM_AVX512) && 0
# define LIBXS_MEM_AVX512
#endif


LIBXS_APIVAR_PRIVATE(int (*internal_memcmp_function)(const void*, const void*, size_t));


LIBXS_API_INLINE
int internal_memcmp_sw(const void* a, const void* b, size_t size)
{
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_32_DECL(aa);
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFE0); i += 32) {
    LIBXS_DIFF_32_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_32(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_SSE3)
int internal_memcmp_sse3(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_SSE3)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_SSE3_DECL(aa);
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFF0); i += 16) {
    LIBXS_DIFF_SSE3_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_SSE3(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_memcmp_sw(a, b, size);
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
int internal_memcmp_avx2(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_AVX2)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_AVX2_DECL(aa);
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFE0); i += 32) {
    LIBXS_DIFF_AVX2_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX2(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_memcmp_sw(a, b, size);
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int internal_memcmp_avx512(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_AVX512)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_AVX512_DECL(aa);
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFC0); i += 64) {
    LIBXS_DIFF_AVX512_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX512(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_memcmp_sw(a, b, size);
#endif
}


LIBXS_API_INTERN void libxs_memory_init(int target_arch)
{
#if defined(LIBXS_MEM_AVX512)
  if (LIBXS_X86_AVX512 <= target_arch) {
    internal_memcmp_function = internal_memcmp_avx512;
  }
  else
#endif
  if (LIBXS_X86_AVX2 <= target_arch) {
    internal_memcmp_function = internal_memcmp_avx2;
  }
  else if (LIBXS_X86_SSE3 <= target_arch) {
    internal_memcmp_function = internal_memcmp_sse3;
  }
  else {
    internal_memcmp_function = internal_memcmp_sw;
  }
  LIBXS_ASSERT(NULL != internal_memcmp_function);
}


LIBXS_API_INTERN void libxs_memory_finalize(void)
{
#if !defined(NDEBUG)
  internal_memcmp_function = NULL;
#endif
}


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
#if defined(LIBXS_MEM_MEMCMP)
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
#if defined(LIBXS_MEM_MEMCMP)
  return memcmp(a, b, size);
#elif (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH) && defined(LIBXS_MEM_AVX512)
  return internal_memcmp_avx512(a, b, size);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  return internal_memcmp_avx2(a, b, size);
#else /* pointer based function call */
  LIBXS_INIT
  LIBXS_ASSERT(NULL != internal_memcmp_function);
  return internal_memcmp_function(a, b, size);
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
