/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_mem.h>
#include "libxs_main.h"
#include <libxs_math.h>
#include "libxs_hash.h"
#include "libxs_diff.h"

#include <ctype.h>

#if !defined(LIBXS_MEM_STDLIB) && 0
# define LIBXS_MEM_STDLIB
#endif
#if !defined(LIBXS_MEM_SW) && 0
# define LIBXS_MEM_SW
#endif

#define LIBXS_MEM_SHUFFLE_COPRIME(N) libxs_coprime2(N)
#define LIBXS_MEM_SHUFFLE(INOUT, ELEMSIZE, COUNT, SHUFFLE, NREPEAT) do { \
  unsigned char *const LIBXS_RESTRICT data = (unsigned char*)(INOUT); \
  const size_t c = (COUNT) - 1, c2 = ((COUNT) + 1) / 2; \
  size_t i; \
  for (i = (0 != (NREPEAT) ? 0 : (COUNT)); i < (COUNT); ++i) { \
    size_t j = i, k = 0; \
    for (; k < (NREPEAT) || j < i; ++k) j = ((SHUFFLE) * j) % (COUNT); \
    if (i < j) LIBXS_MEMSWP( \
      data + (ELEMSIZE) * (c - j), \
      data + (ELEMSIZE) * (c - i), \
      ELEMSIZE); \
    if (c2 <= i) LIBXS_MEMSWP( \
      data + (ELEMSIZE) * (c - i), \
      data + (ELEMSIZE) * i, \
      ELEMSIZE); \
  } \
} while(0)


#if !defined(LIBXS_MEM_SW)
LIBXS_APIVAR_DEFINE(unsigned char (*internal_diff_function)(const void*, const void*, unsigned char));
LIBXS_APIVAR_DEFINE(int (*internal_memcmp_function)(const void*, const void*, size_t));
#endif


LIBXS_API size_t libxs_offset(size_t ndims, const size_t offset[], const size_t shape[], size_t* size)
{
  size_t result = 0, size1 = 0;
  if (0 != ndims && NULL != shape) {
    size_t i;
    result = (NULL != offset ? offset[0] : 0);
    size1 = shape[0];
    for (i = 1; i < ndims; ++i) {
      result += ((NULL != offset && 0 != offset[i]) ? (offset[i] - 1) : 0) * size1;
      size1 *= shape[i];
    }
  }
  if (NULL != size) *size = size1;
  return result;
}


LIBXS_API int libxs_aligned(const void* ptr, const size_t* inc, int* alignment)
{
  const int minalign = libxs_cpuid_vlen(libxs_cpuid(NULL));
  const uintptr_t address = (uintptr_t)ptr;
  int ptr_is_aligned;
  LIBXS_ASSERT(LIBXS_ISPOT(minalign));
  if (NULL == alignment) {
    ptr_is_aligned = !LIBXS_MOD2(address, (uintptr_t)minalign);
}
  else {
    const unsigned int nbits = LIBXS_INTRINSICS_BITSCANFWD64(address);
    *alignment = (32 > nbits ? (1 << nbits) : INT_MAX);
    ptr_is_aligned = (minalign <= *alignment);
  }
  return ptr_is_aligned && (NULL == inc || !LIBXS_MOD2(*inc, (size_t)minalign));
}


LIBXS_API_INLINE
unsigned char internal_diff_sw(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_MEM_STDLIB) && defined(LIBXS_MEM_SW)
  return (unsigned char)memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xF0); i += 16) {
    LIBXS_DIFF_16_DECL(aa);
    LIBXS_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_16(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
unsigned char internal_diff_sse(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_X86) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xF0); i += 16) {
    LIBXS_DIFF_SSE_DECL(aa);
    LIBXS_DIFF_SSE_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_SSE(aa, b8 + i, 0/*dummy*/)) return 1;
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
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xE0); i += 32) {
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


#if defined(LIBXS_DIFF_AVX512_ENABLED)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
unsigned char internal_diff_avx512(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_AVX512_SKX) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xC0); i += 64) {
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
#endif


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


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
int internal_memcmp_sse(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_X86) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_SSE_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFF0); i += 16) {
    LIBXS_DIFF_SSE_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_SSE(aa, b8 + i, 0/*dummy*/)) return 1;
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


#if defined(LIBXS_DIFF_AVX512_ENABLED)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int internal_memcmp_avx512(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_AVX512_SKX) && !defined(LIBXS_MEM_SW)
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
#endif


LIBXS_API_INTERN void libxs_memory_init(int target_arch)
{
  libxs_hash_init(target_arch);
#if !defined(LIBXS_MEM_SW)
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
  else if (LIBXS_X86_GENERIC <= target_arch) {
    internal_diff_function = internal_diff_sse;
    internal_memcmp_function = internal_memcmp_sse;
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
#if !defined(NDEBUG) && !defined(LIBXS_MEM_SW) && 0
  internal_diff_function = NULL;
  internal_memcmp_function = NULL;
#endif
}


LIBXS_API_INTERN unsigned char libxs_diff_4(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 4);
#else
  LIBXS_DIFF_4_DECL(a4);
  LIBXS_DIFF_4_LOAD(a4, a);
  return LIBXS_DIFF_4(a4, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char libxs_diff_8(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 8);
#else
  LIBXS_DIFF_8_DECL(a8);
  LIBXS_DIFF_8_LOAD(a8, a);
  return LIBXS_DIFF_8(a8, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char libxs_diff_16(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 16);
#else
  LIBXS_DIFF_16_DECL(a16);
  LIBXS_DIFF_16_LOAD(a16, a);
  return LIBXS_DIFF_16(a16, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char libxs_diff_32(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 32);
#else
  LIBXS_DIFF_32_DECL(a32);
  LIBXS_DIFF_32_LOAD(a32, a);
  return LIBXS_DIFF_32(a32, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char libxs_diff_48(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_diff_sw(a, b, 48);
#else
  LIBXS_DIFF_48_DECL(a48);
  LIBXS_DIFF_48_LOAD(a48, a);
  return LIBXS_DIFF_48(a48, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char libxs_diff_64(const void* a, const void* b, ...)
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
  return internal_diff_sse(a, b, size);
# else /* pointer based function call */
# if defined(LIBXS_INIT_COMPLETED)
  LIBXS_ASSERT(NULL != internal_diff_function);
  return (unsigned char)(64 <= size
    ? internal_diff_function(a, b, size)
    : internal_diff_sse(a, b, size));
# else
  return (unsigned char)((NULL != internal_diff_function && 64 <= size)
    ? internal_diff_function(a, b, size)
    : internal_diff_sse(a, b, size));
# endif
# endif
# else
  return internal_diff_sw(a, b, size);
# endif
#endif
}


LIBXS_API unsigned int libxs_diff_n(const void* a, const void* bn, unsigned char elemsize,
  unsigned char stride, unsigned int hint, unsigned int count)
{
  unsigned int result;
  LIBXS_ASSERT(elemsize <= stride);
#if defined(LIBXS_MEM_STDLIB) && !defined(LIBXS_MEM_SW)
  LIBXS_DIFF_N(unsigned int, result, memcmp, a, bn, elemsize, stride, hint, count);
#else
# if !defined(LIBXS_MEM_SW)
  switch (elemsize) {
    case 64: {
      LIBXS_DIFF_64_DECL(a64);
      LIBXS_DIFF_64_LOAD(a64, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_64, a64, bn, 64, stride, hint, count);
    } break;
    case 48: {
      LIBXS_DIFF_48_DECL(a48);
      LIBXS_DIFF_48_LOAD(a48, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_48, a48, bn, 48, stride, hint, count);
    } break;
    case 32: {
      LIBXS_DIFF_32_DECL(a32);
      LIBXS_DIFF_32_LOAD(a32, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_32, a32, bn, 32, stride, hint, count);
    } break;
    case 16: {
      LIBXS_DIFF_16_DECL(a16);
      LIBXS_DIFF_16_LOAD(a16, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_16, a16, bn, 16, stride, hint, count);
    } break;
    case 8: {
      LIBXS_DIFF_8_DECL(a8);
      LIBXS_DIFF_8_LOAD(a8, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_8, a8, bn, 8, stride, hint, count);
    } break;
    case 4: {
      LIBXS_DIFF_4_DECL(a4);
      LIBXS_DIFF_4_LOAD(a4, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_4, a4, bn, 4, stride, hint, count);
    } break;
    default:
# endif
    {
      LIBXS_DIFF_N(unsigned int, result, libxs_diff, a, bn, elemsize, stride, hint, count);
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
  return internal_memcmp_sse(a, b, size);
# else /* pointer based function call */
# if defined(LIBXS_INIT_COMPLETED)
  LIBXS_ASSERT(NULL != internal_memcmp_function);
  return (64 <= size
    ? internal_memcmp_function(a, b, size)
    : internal_memcmp_sse(a, b, size));
# else
  return ((NULL != internal_memcmp_function && 64 <= size)
    ? internal_memcmp_function(a, b, size)
    : internal_memcmp_sse(a, b, size));
# endif
# endif
# else
  return internal_memcmp_sw(a, b, size);
# endif
#endif
}


LIBXS_API unsigned int libxs_hash(const void* data, unsigned int size, unsigned int seed)
{
  /*LIBXS_INIT*/
  return libxs_crc32(seed, data, size);
}


LIBXS_API unsigned int libxs_hash8(unsigned int data)
{
  const unsigned int hash = libxs_hash16(data);
  uint8_t tmp_data = (uint8_t)hash;
  unsigned int tmp_seed = (unsigned int)(hash >> 8);
  return libxs_crc32_u8(tmp_seed, &tmp_data) & 0xFF;
}


LIBXS_API unsigned int libxs_hash16(unsigned int data)
{
  uint16_t tmp_data = (uint16_t)data;
  unsigned int tmp_seed = (unsigned int)(data >> 16);
  return libxs_crc32_u16(tmp_seed, &tmp_data) & 0xFFFF;
}


LIBXS_API unsigned int libxs_hash32(unsigned long long data)
{
  uint32_t tmp_data = (uint32_t)data;
  unsigned int tmp_seed = (unsigned int)(data >> 32);
  return libxs_crc32_u32(tmp_seed, &tmp_data) & 0xFFFFFFFF;
}


LIBXS_API unsigned long long libxs_hash_string(const char string[])
{
  unsigned long long result = 0;
  const size_t length = (NULL != string ? strlen(string) : 0);
  if (sizeof(result) < length) {
    const size_t length2 = LIBXS_MAX(length / 2, sizeof(result));
    unsigned int hash32, seed32 = 0; /* seed=0: match else-optimization */
    /*LIBXS_INIT*/
    seed32 = libxs_crc32(seed32, string, length2);
    hash32 = libxs_crc32(seed32, string + length2, length - length2);
    result = hash32; result = (result << 32) | seed32;
  }
  else { /* length <= sizeof(result) */
    char *const s = (char*)&result; signed char i;
    for (i = 0; i < (signed char)length; ++i) s[i] = string[i];
    for (; i < (signed char)sizeof(result); ++i) s[i] = 0;
  }
  return result;
}


LIBXS_API const char* libxs_stristrn(const char a[], const char b[], size_t maxlen)
{
  const char* result = NULL;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b && 0 != maxlen) {
    do {
      if (tolower(*a) != tolower(*b)) ++a;
      else {
        const char* c = b;
        size_t i = 0;
        result = a;
        while ('\0' != c[++i] && i != maxlen && '\0' != *++a) {
          if (tolower(*a) != tolower(c[i])) {
            result = NULL;
            break;
          }
        }
        if ('\0' != c[i] && '\0' != c[i + 1] && c[i] != c[i + 1] && i != maxlen) {
          result = NULL;
        }
        else break;
      }
    } while ('\0' != *a);
  }
  return result;
}


LIBXS_API const char* libxs_stristr(const char a[], const char b[])
{
  return libxs_stristrn(a, b, (size_t)-1);
}


LIBXS_API int libxs_strimatch(const char a[], const char b[], const char delims[], int* count)
{
  int result = 0, na = 0, nb = 0;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b) {
    const char* const sep = ((NULL == delims || '\0' == *delims) ? " \t;,:-" : delims);
    const char *c, *tmp;
    char s[2] = {'\0'};
    size_t m, n;
    for (;;) {
      while (*s = *b, NULL != strpbrk(s, sep)) ++b; /* left-trim */
      if ('\0' != *b && '[' != *b) ++nb; /* count words */
      else break;
      tmp = b;
      while ('\0' != *tmp && (*s = *tmp, NULL == strpbrk(s, sep))) ++tmp;
      m = tmp - b;
      c = libxs_stristrn(a, b, LIBXS_MIN(1, m));
      if (NULL != c) {
        const char* d = c;
        while ('\0' != *d && (*s = *d, NULL == strpbrk(s, sep))) ++d;
        n = d - c;
        if (1 >= n || NULL != libxs_stristrn(c, b, LIBXS_MIN(m, n))) ++result;
      }
      b = tmp;
    }
    for (;;) { /* count number of words */
      while (*s = *a, NULL != strpbrk(s, sep)) ++a; /* left-trim */
      if ('\0' != *a && '[' != *a) ++na; /* count words */
      else break;
      while ('\0' != *a && (*s = *a, NULL == strpbrk(s, sep))) ++a;
    }
    if (na < result) result = na;
  }
  else result = -1;
  if (NULL != count) *count = LIBXS_MAX(na, nb);
  return result;
}


LIBXS_API size_t libxs_format_value(char buffer[32],
  int buffer_size, size_t nbytes, const char scale[], const char* unit, int base)
{
  const int len = (NULL != scale ? ((int)strlen(scale)) : 0);
  const int m = LIBXS_INTRINSICS_BITSCANBWD64(nbytes) / base, n = LIBXS_MIN(m, len);
  int i;
  buffer[0] = 0; /* clear */
  LIBXS_ASSERT(NULL != unit && 0 <= base);
  for (i = 0; i < n; ++i) nbytes >>= base;
  LIBXS_SNPRINTF(buffer, buffer_size, "%i %c%s",
    (int)nbytes, 0 < n ? scale[n-1] : *unit, 0 < n ? unit : "");
  return nbytes;
}


LIBXS_API int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat)
{
  int result;
  if (NULL != inout || 0 == elemsize || 0 == count) {
    const size_t s = (NULL == shuffle ? LIBXS_MEM_SHUFFLE_COPRIME(count) : *shuffle);
    const size_t n = (NULL == nrepeat ? 1 : *nrepeat);
    switch (elemsize) {
      case 8:   LIBXS_MEM_SHUFFLE(inout, 8, count, s, n); break;
      case 4:   LIBXS_MEM_SHUFFLE(inout, 4, count, s, n); break;
      case 2:   LIBXS_MEM_SHUFFLE(inout, 2, count, s, n); break;
      case 1:   LIBXS_MEM_SHUFFLE(inout, 1, count, s, n); break;
      default:  LIBXS_MEM_SHUFFLE(inout, elemsize, count, s, n);
    }
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXS_API int libxs_shuffle2(void* dst, const void* src, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat)
{
  const unsigned char *const LIBXS_RESTRICT inp = (const unsigned char*)src;
  unsigned char *const LIBXS_RESTRICT out = (unsigned char*)dst;
  const size_t size = elemsize * count;
  int result;
  if ((NULL != inp && NULL != out && ((out + size) <= inp || (inp + size) <= out)) || 0 == size) {
    const size_t s = (NULL == shuffle ? LIBXS_MEM_SHUFFLE_COPRIME(count) : *shuffle);
    size_t i = 0, j = 1;
    if (NULL == nrepeat || 1 == *nrepeat) {
      if (elemsize < 128) {
        switch (elemsize) {
          case 8: for (; i < size; i += 8, j += s) {
            if (count < j) j -= count;
            *(unsigned long long*)(out + i) = *(const unsigned long long*)(inp + size - 8 * j);
          } break;
          case 4: for (; i < size; i += 4, j += s) {
            if (count < j) j -= count;
            *(unsigned int*)(out + i) = *(const unsigned int*)(inp + size - 4 * j);
          } break;
          case 2: for (; i < size; i += 2, j += s) {
            if (count < j) j -= count;
            *(unsigned short*)(out + i) = *(const unsigned short*)(inp + size - 2 * j);
          } break;
          case 1: for (; i < size; ++i, j += s) {
            if (count < j) j -= count;
            out[i] = inp[size-j];
          } break;
          default: for (; i < size; i += elemsize, j += s) {
            if (count < j) j -= count;
            LIBXS_MEMCPY(out + i, inp + size - elemsize * j, elemsize);
          }
        }
      }
      else { /* generic path */
        for (; i < size; i += elemsize, j += s) {
          if (count < j) j -= count;
          memcpy(out + i, inp + size - elemsize * j, elemsize);
        }
      }
    }
    else if (0 != *nrepeat) { /* generic path */
      const size_t c = count - 1;
      for (; i < count; ++i) {
        size_t k = 0;
        LIBXS_ASSERT(NULL != inp && NULL != out);
        for (j = i; k < *nrepeat; ++k) j = c - ((s * j) % count);
        memcpy(out + elemsize * i, inp + elemsize * j, elemsize);
      }
    }
    else { /* ordinary copy */
      memcpy(out, inp, size);
    }
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXS_API size_t libxs_unshuffle(size_t count, const size_t* shuffle)
{
  size_t result = 0;
  if (0 < count) {
    const size_t n = (NULL == shuffle ? LIBXS_MEM_SHUFFLE_COPRIME(count) : *shuffle);
    size_t c = count - 1, j = c, d = 0;
    for (; result < count; ++result, j = c - d) {
      d = (j * n) % count;
      if (0 == d) break;
    }
  }
  assert(result <= count);
  return result;
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
  else if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
    /*LIBXS_INIT*/
    if (0 != libxs_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xhash specified!\n");
    }
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
  else if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
    /*LIBXS_INIT*/
    if (0 != libxs_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xdiff specified!\n");
    }
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
  { const int s = *size;
    LIBXS_MEMSET(dst, 0, s);
  }
#if !defined(NDEBUG)
  else if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
    /*LIBXS_INIT*/
    if (0 != libxs_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xclear specified!\n");
    }
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
  else if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
    /*LIBXS_INIT*/
    if (0 != libxs_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_aligned specified!\n");
    }
  }
#endif
}

#endif /*defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))*/
