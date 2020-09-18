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
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <ctype.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_MEM_STDLIB) && 0
# define LIBXS_MEM_STDLIB
#endif
#if !defined(LIBXS_MEM_SW) && 0
# define LIBXS_MEM_SW
#endif


#if !defined(LIBXS_MEM_SW)
LIBXS_APIVAR_DEFINE(unsigned char (*internal_diff_function)(const void*, const void*, unsigned char));
LIBXS_APIVAR_DEFINE(int (*internal_memcmp_function)(const void*, const void*, size_t));
#endif


LIBXS_API_INLINE
unsigned char internal_diff_sw(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_MEM_STDLIB) && defined(LIBXS_MEM_SW)
  return (unsigned char)memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xF0); i += 16) {
    LIBXS_DIFF_16_DECL(aa);
    LIBXS_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_16(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_SSE3)
unsigned char internal_diff_sse3(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_SSE3) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xF0); i += 16) {
    LIBXS_DIFF_SSE3_DECL(aa);
    LIBXS_DIFF_SSE3_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_SSE3(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_diff_sw(a, b, size);
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
unsigned char internal_diff_avx2(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_AVX2) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xE0); i += 32) {
    LIBXS_DIFF_AVX2_DECL(aa);
    LIBXS_DIFF_AVX2_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX2(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_diff_sw(a, b, size);
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
unsigned char internal_diff_avx512(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_AVX512) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xC0); i += 64) {
    LIBXS_DIFF_AVX512_DECL(aa);
    LIBXS_DIFF_AVX512_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX512(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_diff_sw(a, b, size);
#endif
}


LIBXS_API_INLINE
int internal_memcmp_sw(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_MEM_STDLIB)
  return memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_16_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFF0); i += 16) {
    LIBXS_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_16(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_SSE3)
int internal_memcmp_sse3(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_SSE3) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_SSE3_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
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
#if defined(LIBXS_INTRINSICS_AVX2) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_AVX2_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
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
#if defined(LIBXS_INTRINSICS_AVX512) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_AVX512_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
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
#if defined(LIBXS_MEM_SW)
  LIBXS_UNUSED(target_arch);
#else
  if (LIBXS_X86_AVX512 <= target_arch) {
# if defined(LIBXS_DIFF_AVX512_ENABLED)
    internal_diff_function = internal_diff_avx512;
# else
    internal_diff_function = internal_diff_avx2;
# endif
# if defined(LIBXS_DIFF_AVX512_ENABLED)
    internal_memcmp_function = internal_memcmp_avx512;
# else
    internal_memcmp_function = internal_memcmp_avx2;
# endif
  }
  else if (LIBXS_X86_AVX2 <= target_arch) {
    internal_diff_function = internal_diff_avx2;
    internal_memcmp_function = internal_memcmp_avx2;
  }
  else if (LIBXS_X86_SSE3 <= target_arch) {
    internal_diff_function = internal_diff_sse3;
    internal_memcmp_function = internal_memcmp_sse3;
  }
  else {
    internal_diff_function = internal_diff_sw;
    internal_memcmp_function = internal_memcmp_sw;
  }
  LIBXS_ASSERT(NULL != internal_diff_function);
  LIBXS_ASSERT(NULL != internal_memcmp_function);
#endif
}


LIBXS_API_INTERN void libxs_memory_finalize(void)
{
#if !defined(NDEBUG) && !defined(LIBXS_MEM_SW)
  internal_diff_function = NULL;
  internal_memcmp_function = NULL;
#endif
}


LIBXS_API unsigned char libxs_diff_4(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 4);
#else
  LIBXS_DIFF_4_DECL(a4);
  LIBXS_DIFF_4_LOAD(a4, a);
  return LIBXS_DIFF_4(a4, b, 0/*dummy*/);
#endif
}


LIBXS_API unsigned char libxs_diff_8(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 8);
#else
  LIBXS_DIFF_8_DECL(a8);
  LIBXS_DIFF_8_LOAD(a8, a);
  return LIBXS_DIFF_8(a8, b, 0/*dummy*/);
#endif
}


LIBXS_API unsigned char libxs_diff_16(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 16);
#else
  LIBXS_DIFF_16_DECL(a16);
  LIBXS_DIFF_16_LOAD(a16, a);
  return LIBXS_DIFF_16(a16, b, 0/*dummy*/);
#endif
}


LIBXS_API unsigned char libxs_diff_32(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 32);
#else
  LIBXS_DIFF_32_DECL(a32);
  LIBXS_DIFF_32_LOAD(a32, a);
  return LIBXS_DIFF_32(a32, b, 0/*dummy*/);
#endif
}


LIBXS_API unsigned char libxs_diff_48(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 48);
#else
  LIBXS_DIFF_48_DECL(a48);
  LIBXS_DIFF_48_LOAD(a48, a);
  return LIBXS_DIFF_48(a48, b, 0/*dummy*/);
#endif
}


LIBXS_API unsigned char libxs_diff_64(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 64);
#else
  LIBXS_DIFF_64_DECL(a64);
  LIBXS_DIFF_64_LOAD(a64, a);
  return LIBXS_DIFF_64(a64, b, 0/*dummy*/);
#endif
}


LIBXS_API unsigned char libxs_diff(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_MEM_SW) && !defined(LIBXS_MEM_STDLIB)
  return internal_diff_sw(a, b, size);
#else
# if defined(LIBXS_MEM_STDLIB)
  return 0 != memcmp(a, b, size);
# elif (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH) && defined(LIBXS_DIFF_AVX512_ENABLED)
  return internal_diff_avx512(a, b, size);
# elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  return internal_diff_avx2(a, b, size);
# elif (LIBXS_X86_SSE3 <= LIBXS_STATIC_TARGET_ARCH)
# if (LIBXS_X86_AVX2 > LIBXS_MAX_STATIC_TARGET_ARCH)
  return internal_diff_sse3(a, b, size);
# else /* pointer based function call */
# if defined(LIBXS_INIT_COMPLETED)
  LIBXS_ASSERT(NULL != internal_diff_function);
  return internal_diff_function(a, b, size);
# else
  return (unsigned char)(NULL != internal_diff_function
    ? internal_diff_function(a, b, size)
    : internal_diff_sse3(a, b, size));
# endif
# endif
# else
  return internal_diff_sw(a, b, size);
# endif
#endif
}


LIBXS_API unsigned int libxs_diff_n(const void* a, const void* bn, unsigned char size,
  unsigned char stride, unsigned int hint, unsigned int n)
{
  unsigned int result;
  LIBXS_ASSERT(size <= stride);
#if defined(LIBXS_MEM_STDLIB) && !defined(LIBXS_MEM_SW)
  LIBXS_DIFF_N(unsigned int, result, memcmp, a, bn, size, stride, hint, n);
#else
# if !defined(LIBXS_MEM_SW)
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
    case 8: {
      LIBXS_DIFF_8_DECL(a8);
      LIBXS_DIFF_8_LOAD(a8, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_8, a8, bn, size, stride, hint, n);
    } break;
    case 4: {
      LIBXS_DIFF_4_DECL(a4);
      LIBXS_DIFF_4_LOAD(a4, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_4, a4, bn, size, stride, hint, n);
    } break;
    default:
# endif
    {
      LIBXS_DIFF_N(unsigned int, result, libxs_diff, a, bn, size, stride, hint, n);
    }
# if !defined(LIBXS_MEM_SW)
  }
# endif
#endif
  return result;
}


LIBXS_API int libxs_memcmp(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_MEM_SW) && !defined(LIBXS_MEM_STDLIB)
  return internal_memcmp_sw(a, b, size);
#else
# if defined(LIBXS_MEM_STDLIB)
  return memcmp(a, b, size);
# elif (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH) && defined(LIBXS_DIFF_AVX512_ENABLED)
  return internal_memcmp_avx512(a, b, size);
# elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  return internal_memcmp_avx2(a, b, size);
# elif (LIBXS_X86_SSE3 <= LIBXS_STATIC_TARGET_ARCH)
# if (LIBXS_X86_AVX2 > LIBXS_MAX_STATIC_TARGET_ARCH)
  return internal_memcmp_sse3(a, b, size);
# else /* pointer based function call */
# if defined(LIBXS_INIT_COMPLETED)
  LIBXS_ASSERT(NULL != internal_memcmp_function);
  return internal_memcmp_function(a, b, size);
# else
  return NULL != internal_memcmp_function
    ? internal_memcmp_function(a, b, size)
    : internal_memcmp_sse3(a, b, size);
# endif
# endif
# else
  return internal_memcmp_sw(a, b, size);
# endif
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
  const size_t length = (NULL != string ? strlen(string) : 0);
  if (sizeof(result) < length) {
    const size_t length2 = length / 2;
    unsigned int seed32 = 0; /* seed=0: match else-optimization */
    LIBXS_INIT
    seed32 = libxs_crc32(seed32, string, length2);
    result = libxs_crc32(seed32, string + length2, length - length2);
    result = (result << 32) | seed32;
  }
  else { /* reinterpret directly as hash value */
#if 1
    result = (unsigned long long)string;
#else
    char *const s = (char*)&result; signed char i;
    for (i = 0; i < (signed char)length; ++i) s[i] = string[i];
    for (; i < (signed char)sizeof(result); ++i) s[i] = 0;
#endif
  }
  return result;
}


LIBXS_API const char* libxs_stristr(const char* a, const char* b)
{
  const char* result = NULL;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b) {
    do {
      if (tolower(*a) != tolower(*b)) {
        ++a;
      }
      else {
        const char* c = b;
        result = a;
        while ('\0' != *++a && '\0' != *++c) {
          if (tolower(*a) != tolower(*c)) {
            result = NULL;
            break;
          }
        }
        if ('\0' == *c) break;
      }
    } while ('\0' != *a);
  }
  return result;
}


LIBXS_API int libxs_aligned(const void* ptr, const size_t* inc, int* alignment)
{
  const int minalign = 4 * libxs_cpuid_vlen32(libxs_target_archid);
  const uintptr_t address = (uintptr_t)ptr;
  int ptr_is_aligned;
  LIBXS_ASSERT(LIBXS_ISPOT(minalign));
  if (NULL == alignment) {
    ptr_is_aligned = !LIBXS_MOD2(address, (uintptr_t)minalign);
  }
  else {
    *alignment = (1 << LIBXS_INTRINSICS_BITSCANFWD64(address));
    ptr_is_aligned = (minalign <= *alignment);
  }
  return ptr_is_aligned && (NULL == inc || !LIBXS_MOD2(*inc, minalign));
}


#if defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xhash)(int* /*hash_seed*/, const void* /*data*/, const int* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xhash)(int* hash_seed, const void* data, const int* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != hash_seed && NULL != data && NULL != size && 0 <= *size)
#endif
  {
    *hash_seed = (int)(libxs_hash(data, (unsigned int)*size, (unsigned int)*hash_seed) & 0x7FFFFFFF/*sign-bit*/);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xhash specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xdiff)(int* /*result*/, const void* /*a*/, const void* /*b*/, const long long* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xdiff)(int* result, const void* a, const void* b, const long long* size)
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
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xdiff specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xclear)(void* /*dst*/, const int* /*size*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xclear)(void* dst, const int* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != dst && NULL != size && 0 <= *size && 128 > *size)
#endif
  {
    LIBXS_MEMSET127(dst, 0, *size);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xclear specified!\n");
  }
#endif
}


LIBXS_API void LIBXS_FSYMBOL(libxs_aligned)(int* /*result*/, const void* /*ptr*/, const int* /*inc*/, int* /*alignment*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_aligned)(int* result, const void* ptr, const int* inc, int* alignment)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != result)
#endif
  {
    const size_t next = (NULL != inc ? *inc : 0);
    *result = libxs_aligned(ptr, &next, alignment);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_aligned specified!\n");
  }
#endif
}

#endif /*defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))*/
