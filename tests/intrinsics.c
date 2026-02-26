/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_utils.h>

#if defined(_DEBUG) || 1
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#if (LIBXS_MAX_STATIC_TARGET_ARCH < LIBXS_STATIC_TARGET_ARCH)
# error "LIBXS_MAX_STATIC_TARGET_ARCH < LIBXS_STATIC_TARGET_ARCH"
#endif


/* SSE4.2: _mm_crc32_u32 (needs target("sse4.2") when baseline is lower) */
#if defined(LIBXS_INTRINSICS_SSE42)
LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_SSE42)
int test_intrinsics_sse42(void)
{
  /* CRC32C of 0x12345678 with initial 0 - deterministic */
  const unsigned int crc = _mm_crc32_u32(0, 0x12345678u);
  return (0 != crc) ? 0 : EXIT_FAILURE; /* crc is non-zero */
}
#endif


/* AVX: 256-bit float add */
#if defined(LIBXS_INTRINSICS_AVX)
LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX)
int test_intrinsics_avx(void)
{
  LIBXS_ALIGNED(float buf[8], 32);
  const __m256 a = _mm256_set1_ps(1.5f);
  const __m256 b = _mm256_set1_ps(2.5f);
  const __m256 c = _mm256_add_ps(a, b);
  int i;
  _mm256_store_ps(buf, c);
  for (i = 0; i < 8; ++i) if (4.0f != buf[i]) return EXIT_FAILURE;
  return 0;
}
#endif


/* AVX2: 256-bit integer add */
#if defined(LIBXS_INTRINSICS_AVX2)
LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
int test_intrinsics_avx2(void)
{
  LIBXS_ALIGNED(int buf[8], 32);
  const __m256i a = _mm256_set1_epi32(3);
  const __m256i b = _mm256_set1_epi32(4);
  const __m256i c = _mm256_add_epi32(a, b);
  int i;
  _mm256_store_si256((__m256i*)buf, c);
  for (i = 0; i < 8; ++i) if (7 != buf[i]) return EXIT_FAILURE;
  return 0;
}
#endif


/* AVX-512: 512-bit integer add */
#if defined(LIBXS_INTRINSICS_AVX512)
LIBXS_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int test_intrinsics_avx512(void)
{
  LIBXS_ALIGNED(int buf[16], 64);
  const __m512i a = _mm512_set1_epi32(11);
  const __m512i b = _mm512_set1_epi32(22);
  const __m512i c = _mm512_add_epi32(a, b);
  int i;
  _mm512_store_si512(buf, c);
  for (i = 0; i < 16; ++i) if (33 != buf[i]) return EXIT_FAILURE;
  return 0;
}
#endif


int main(int argc, char* argv[])
{
  const int cpuid = libxs_cpuid(NULL);
  int highest = LIBXS_TARGET_ARCH_UNKNOWN;
  int nerrors = 0;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  /* macro sanity */
  if (LIBXS_MAX_STATIC_TARGET_ARCH < LIBXS_STATIC_TARGET_ARCH) {
    FPRINTF(stderr, "ERROR: MAX_STATIC_TARGET_ARCH < STATIC_TARGET_ARCH\n");
    ++nerrors;
  }
#if !defined(LIBXS_INTRINSICS_X86) && defined(LIBXS_PLATFORM_X86)
# if !defined(__NO_INTRINSICS)
  FPRINTF(stderr, "ERROR: x86 platform but LIBXS_INTRINSICS_X86 not defined\n");
  ++nerrors;
# endif
#endif

  /* ISA kernels: call only when compiler can generate AND CPU supports */
#if defined(LIBXS_INTRINSICS_SSE42)
  if (LIBXS_X86_SSE42 <= cpuid) {
    if (0 != test_intrinsics_sse42()) {
      FPRINTF(stderr, "ERROR: test_intrinsics_sse42\n");
      ++nerrors;
    }
  }
#endif
#if defined(LIBXS_INTRINSICS_AVX)
  if (LIBXS_X86_AVX <= cpuid) {
    if (0 != test_intrinsics_avx()) {
      FPRINTF(stderr, "ERROR: test_intrinsics_avx\n");
      ++nerrors;
    }
  }
#endif
#if defined(LIBXS_INTRINSICS_AVX2)
  if (LIBXS_X86_AVX2 <= cpuid) {
    if (0 != test_intrinsics_avx2()) {
      FPRINTF(stderr, "ERROR: test_intrinsics_avx2\n");
      ++nerrors;
    }
  }
#endif
#if defined(LIBXS_INTRINSICS_AVX512)
  if (LIBXS_X86_AVX512 <= cpuid) {
    if (0 != test_intrinsics_avx512()) {
      FPRINTF(stderr, "ERROR: test_intrinsics_avx512\n");
      ++nerrors;
    }
  }
#endif

  /* determine highest ISA level that compiled AND ran successfully */
  if (0 == nerrors) {
    highest = LIBXS_STATIC_TARGET_ARCH;
#if defined(LIBXS_INTRINSICS_SSE42)
    if (LIBXS_X86_SSE42 <= cpuid) {
      highest = LIBXS_X86_SSE42;
    }
#endif
#if defined(LIBXS_INTRINSICS_AVX)
    if (LIBXS_X86_AVX <= cpuid) {
      highest = LIBXS_X86_AVX;
    }
#endif
#if defined(LIBXS_INTRINSICS_AVX2)
    if (LIBXS_X86_AVX2 <= cpuid) {
      highest = LIBXS_X86_AVX2;
    }
#endif
#if defined(LIBXS_INTRINSICS_AVX512)
    if (LIBXS_X86_AVX512 <= cpuid) {
      highest = LIBXS_X86_AVX512;
    }
#endif
  }

  if (highest < LIBXS_MAX_STATIC_TARGET_ARCH && LIBXS_MAX_STATIC_TARGET_ARCH <= cpuid) {
    FPRINTF(stderr, "ERROR: cannot reach LIBXS_MAX_STATIC_TARGET_ARCH (%i < %i)\n",
      highest, LIBXS_MAX_STATIC_TARGET_ARCH);
    ++nerrors;
  }

  fprintf(stderr, "static=%s cpuid=%s target=%s\n",
    libxs_cpuid_name(LIBXS_STATIC_TARGET_ARCH),
    libxs_cpuid_name(libxs_cpuid(NULL)),
    libxs_cpuid_name(highest));

  return (0 != nerrors) ? EXIT_FAILURE : EXIT_SUCCESS;
}
