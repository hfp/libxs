/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DIFF_H
#define LIBXS_DIFF_H

#include <libxs_intrinsics_x86.h>

#if !defined(LIBXS_DIFF_AVX512_ENABLED) && 0
# define LIBXS_DIFF_AVX512_ENABLED
#endif

#define LIBXS_DIFF_4_DECL(A) const uint32_t */*const*/ A = NULL
#define LIBXS_DIFF_4_ASSIGN(A, B) (A) = (B)
#define LIBXS_DIFF_4_LOAD(A, SRC) A = (const uint32_t*)(SRC)
#define LIBXS_DIFF_4(A, B, ...) ((unsigned char)(0 != (*(A) ^ (*(const uint32_t*)(B)))))

#define LIBXS_DIFF_8_DECL(A) const uint64_t */*const*/ A = NULL
#define LIBXS_DIFF_8_ASSIGN(A, B) (A) = (B)
#define LIBXS_DIFF_8_LOAD(A, SRC) A = (const uint64_t*)(SRC)
#define LIBXS_DIFF_8(A, B, ...) ((unsigned char)(0 != (*(A) ^ (*(const uint64_t*)(B)))))

#define LIBXS_DIFF_SSE_DECL(A) __m128i A = LIBXS_INTRINSICS_MM_UNDEFINED_SI128()
#define LIBXS_DIFF_SSE_ASSIGN(A, B) (A) = (B)
#define LIBXS_DIFF_SSE_LOAD(A, SRC) A = LIBXS_INTRINSICS_LDDQU_SI128((const __m128i*)(SRC))
#define LIBXS_DIFF_SSE(A, B, ...) ((unsigned char)(0xFFFF != _mm_movemask_epi8(_mm_cmpeq_epi8( \
  A, LIBXS_INTRINSICS_LDDQU_SI128((const __m128i*)(B))))))

#if (LIBXS_X86_GENERIC <= LIBXS_STATIC_TARGET_ARCH) /*|| defined(LIBXS_INTRINSICS_TARGET)*/
# define LIBXS_DIFF_16_DECL LIBXS_DIFF_SSE_DECL
# define LIBXS_DIFF_16_ASSIGN LIBXS_DIFF_SSE_ASSIGN
# define LIBXS_DIFF_16_LOAD LIBXS_DIFF_SSE_LOAD
# define LIBXS_DIFF_16 LIBXS_DIFF_SSE
#else
# define LIBXS_DIFF_16_DECL(A) const uint64_t */*const*/ A = NULL
# define LIBXS_DIFF_16_ASSIGN(A, B) (A) = (B)
# define LIBXS_DIFF_16_LOAD(A, SRC) A = (const uint64_t*)(SRC)
# define LIBXS_DIFF_16(A, B, ...) ((unsigned char)(0 != (((A)[0] ^ (*(const uint64_t*)(B))) | \
    ((A)[1] ^ ((const uint64_t*)(B))[1]))))
#endif

#define LIBXS_DIFF_AVX2_DECL(A) __m256i A = _mm256_undefined_si256()
#define LIBXS_DIFF_AVX2_ASSIGN(A, B) (A) = (B)
#define LIBXS_DIFF_AVX2_LOAD(A, SRC) A = _mm256_loadu_si256((const __m256i*)(SRC))
#define LIBXS_DIFF_AVX2(A, B, ...) ((unsigned char)(-1 != _mm256_movemask_epi8(_mm256_cmpeq_epi8( \
  A, _mm256_loadu_si256((const __m256i*)(B))))))

#if (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
# define LIBXS_DIFF_32_DECL LIBXS_DIFF_AVX2_DECL
# define LIBXS_DIFF_32_ASSIGN LIBXS_DIFF_AVX2_ASSIGN
# define LIBXS_DIFF_32_LOAD LIBXS_DIFF_AVX2_LOAD
# define LIBXS_DIFF_32 LIBXS_DIFF_AVX2
#else
# define LIBXS_DIFF_32_DECL(A) LIBXS_DIFF_16_DECL(A); LIBXS_DIFF_16_DECL(LIBXS_CONCATENATE3(libxs_diff_32_, A, _))
# define LIBXS_DIFF_32_ASSIGN(A, B) LIBXS_DIFF_16_ASSIGN(A, B); LIBXS_DIFF_16_ASSIGN(LIBXS_CONCATENATE3(libxs_diff_32_, A, _), LIBXS_CONCATENATE3(libxs_diff_32_, B, _))
# define LIBXS_DIFF_32_LOAD(A, SRC) LIBXS_DIFF_16_LOAD(A, SRC); LIBXS_DIFF_16_LOAD(LIBXS_CONCATENATE3(libxs_diff_32_, A, _), (const uint64_t*)(SRC) + 2)
# define LIBXS_DIFF_32(A, B, ...) ((unsigned char)(0 != LIBXS_DIFF_16(A, B, __VA_ARGS__) ? 1 : LIBXS_DIFF_16(LIBXS_CONCATENATE3(libxs_diff_32_, A, _), (const uint64_t*)(B) + 2, __VA_ARGS__)))
#endif

#define LIBXS_DIFF_48_DECL(A) LIBXS_DIFF_16_DECL(A); LIBXS_DIFF_32_DECL(LIBXS_CONCATENATE3(libxs_diff_48_, A, _))
#define LIBXS_DIFF_48_ASSIGN(A, B) LIBXS_DIFF_16_ASSIGN(A, B); LIBXS_DIFF_32_ASSIGN(LIBXS_CONCATENATE3(libxs_diff_48_, A, _), LIBXS_CONCATENATE3(libxs_diff_48_, B, _))
#define LIBXS_DIFF_48_LOAD(A, SRC) LIBXS_DIFF_16_LOAD(A, SRC); LIBXS_DIFF_32_LOAD(LIBXS_CONCATENATE3(libxs_diff_48_, A, _), (const uint64_t*)(SRC) + 2)
#define LIBXS_DIFF_48(A, B, ...) ((unsigned char)(0 != LIBXS_DIFF_16(A, B, __VA_ARGS__) ? 1 : LIBXS_DIFF_32(LIBXS_CONCATENATE3(libxs_diff_48_, A, _), (const uint64_t*)(B) + 2, __VA_ARGS__)))

#define LIBXS_DIFF_64SW_DECL(A) LIBXS_DIFF_32_DECL(A); LIBXS_DIFF_32_DECL(LIBXS_CONCATENATE3(libxs_diff_64_, A, _))
#define LIBXS_DIFF_64SW_ASSIGN(A, B) LIBXS_DIFF_32_ASSIGN(A, B); LIBXS_DIFF_32_ASSIGN(LIBXS_CONCATENATE3(libxs_diff_64_, A, _), LIBXS_CONCATENATE3(libxs_diff_64_, B, _))
#define LIBXS_DIFF_64SW_LOAD(A, SRC) LIBXS_DIFF_32_LOAD(A, SRC); LIBXS_DIFF_32_LOAD(LIBXS_CONCATENATE3(libxs_diff_64_, A, _), (const uint64_t*)(SRC) + 4)
#define LIBXS_DIFF_64SW(A, B, ...) ((unsigned char)(0 != LIBXS_DIFF_32(A, B, __VA_ARGS__) ? 1 : LIBXS_DIFF_32(LIBXS_CONCATENATE3(libxs_diff_64_, A, _), (const uint64_t*)(B) + 4, __VA_ARGS__)))

#if defined(LIBXS_DIFF_AVX512_ENABLED)
# define LIBXS_DIFF_AVX512_DECL(A) __m512i A = LIBXS_INTRINSICS_MM512_UNDEFINED_EPI32()
# define LIBXS_DIFF_AVX512_ASSIGN(A, B) (A) = (B)
# define LIBXS_DIFF_AVX512_LOAD(A, SRC) A = _mm512_loadu_si512((const __m512i*)(SRC))
# define LIBXS_DIFF_AVX512(A, B, ...) ((unsigned char)(0xFFFF != (unsigned int)/*_cvtmask16_u32*/(_mm512_cmpeq_epi32_mask( \
    A, _mm512_loadu_si512((const __m512i*)(B))))))
#else
# define LIBXS_DIFF_AVX512_DECL LIBXS_DIFF_64SW_DECL
# define LIBXS_DIFF_AVX512_ASSIGN LIBXS_DIFF_64SW_ASSIGN
# define LIBXS_DIFF_AVX512_LOAD LIBXS_DIFF_64SW_LOAD
# define LIBXS_DIFF_AVX512 LIBXS_DIFF_64SW
#endif

#if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
# define LIBXS_DIFF_64_DECL LIBXS_DIFF_AVX512_DECL
# define LIBXS_DIFF_64_ASSIGN LIBXS_DIFF_AVX512_ASSIGN
# define LIBXS_DIFF_64_LOAD LIBXS_DIFF_AVX512_LOAD
# define LIBXS_DIFF_64 LIBXS_DIFF_AVX512
#else
# define LIBXS_DIFF_64_DECL LIBXS_DIFF_64SW_DECL
# define LIBXS_DIFF_64_ASSIGN LIBXS_DIFF_64SW_ASSIGN
# define LIBXS_DIFF_64_LOAD LIBXS_DIFF_64SW_LOAD
# define LIBXS_DIFF_64 LIBXS_DIFF_64SW
#endif

#define LIBXS_DIFF_DECL(N, A) LIBXS_CONCATENATE3(LIBXS_DIFF_, N, _DECL)(A)
#define LIBXS_DIFF_LOAD(N, A, SRC) LIBXS_CONCATENATE3(LIBXS_DIFF_, N, _LOAD)(A, SRC)
#define LIBXS_DIFF(N) LIBXS_CONCATENATE(LIBXS_DIFF_, N)

#define LIBXS_DIFF_N(TYPE, RESULT, DIFF, A, BN, ELEMSIZE, STRIDE, HINT, N) { \
  const char* libxs_diff_b_ = (const char*)(BN) + (size_t)(HINT) * (STRIDE); \
  for (RESULT = (HINT); (RESULT) < (N); ++(RESULT)) { \
    if (0 == DIFF(A, libxs_diff_b_, ELEMSIZE)) break; \
    libxs_diff_b_ += (STRIDE); \
  } \
  if ((N) == (RESULT)) { /* wrong hint */ \
    TYPE libxs_diff_r_ = 0; \
    libxs_diff_b_ = (const char*)(BN); /* reset */ \
    for (; libxs_diff_r_ < (HINT); ++libxs_diff_r_) { \
      if (0 == DIFF(A, libxs_diff_b_, ELEMSIZE)) { \
        RESULT = libxs_diff_r_; \
        break; \
      } \
      libxs_diff_b_ += (STRIDE); \
    } \
  } \
}


/** Function type representing the diff-functionality. */
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE unsigned int (*libxs_diff_function)(
  const void* /*a*/, const void* /*b*/, ... /*size*/);

/** Compare two data blocks of 4 Byte each. */
LIBXS_API unsigned char libxs_diff_4(const void* a, const void* b, ...);
/** Compare two data blocks of 8 Byte each. */
LIBXS_API unsigned char libxs_diff_8(const void* a, const void* b, ...);
/** Compare two data blocks of 16 Byte each. */
LIBXS_API unsigned char libxs_diff_16(const void* a, const void* b, ...);
/** Compare two data blocks of 32 Byte each. */
LIBXS_API unsigned char libxs_diff_32(const void* a, const void* b, ...);
/** Compare two data blocks of 48 Byte each. */
LIBXS_API unsigned char libxs_diff_48(const void* a, const void* b, ...);
/** Compare two data blocks of 64 Byte each. */
LIBXS_API unsigned char libxs_diff_64(const void* a, const void* b, ...);

#endif /*LIBXS_DIFF_H*/
