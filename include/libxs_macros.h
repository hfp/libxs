/******************************************************************************
** Copyright (c) 2013-2017, Intel Corporation                                **
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
#ifndef LIBXS_MACROS_H
#define LIBXS_MACROS_H

#include "libxs_config.h"

#define LIBXS_STRINGIFY2(SYMBOL) #SYMBOL
#define LIBXS_STRINGIFY(SYMBOL) LIBXS_STRINGIFY2(SYMBOL)
#define LIBXS_TOSTRING(SYMBOL) LIBXS_STRINGIFY(SYMBOL)
#define LIBXS_CONCATENATE2(A, B) A##B
#define LIBXS_CONCATENATE(A, B) LIBXS_CONCATENATE2(A, B)
#define LIBXS_FSYMBOL(SYMBOL) LIBXS_CONCATENATE2(SYMBOL, _)
#define LIBXS_UNIQUE(NAME) LIBXS_CONCATENATE(NAME, __LINE__)
#define LIBXS_EXPAND(A) A

#define LIBXS_VERSION2(MAJOR, MINOR) ((MAJOR) * 10000 + (MINOR) * 100)
#define LIBXS_VERSION3(MAJOR, MINOR, UPDATE) (LIBXS_VERSION2(MAJOR, MINOR) + (UPDATE))
#define LIBXS_VERSION4(MAJOR, MINOR, UPDATE, PATCH) ((MAJOR) * 100000000 + (MINOR) * 1000000 + (UPDATE) * 10000 + (PATCH))

#if defined(__cplusplus)
# define LIBXS_VARIADIC ...
# define LIBXS_EXTERN extern "C"
# define LIBXS_INLINE_KEYWORD inline
# define LIBXS_INLINE LIBXS_INLINE_KEYWORD
# define LIBXS_CALLER __FUNCTION__
#else
# define LIBXS_VARIADIC
# define LIBXS_EXTERN extern
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
#   define LIBXS_PRAGMA(DIRECTIVE) _Pragma(LIBXS_STRINGIFY(DIRECTIVE))
#   define LIBXS_CALLER __func__
#   define LIBXS_RESTRICT restrict
#   define LIBXS_INLINE_KEYWORD inline
# elif defined(_MSC_VER)
#   define LIBXS_CALLER __FUNCTION__
#   define LIBXS_INLINE_KEYWORD __inline
#   define LIBXS_INLINE_FIXUP
# elif defined(__GNUC__) && !defined(__STRICT_ANSI__)
#   define LIBXS_CALLER __FUNCTION__
# endif
# if !defined(LIBXS_INLINE_KEYWORD)
#   define LIBXS_INLINE_KEYWORD
#   define LIBXS_INLINE_FIXUP
# endif
# if !defined(LIBXS_CALLER)
#   define LIBXS_CALLER 0
# endif
# define LIBXS_INLINE static LIBXS_INLINE_KEYWORD
#endif /*__cplusplus*/
#if !defined(LIBXS_INTERNAL_API)
# define LIBXS_INTERNAL_API LIBXS_EXTERN
#endif
#if !defined(LIBXS_INTERNAL_API_DEFINITION)
# define LIBXS_INTERNAL_API_DEFINITION LIBXS_INTERNAL_API
#endif
#if defined(LIBXS_BUILD)
# define LIBXS_INTERNAL_API_INLINE LIBXS_INTERNAL_API
#else
# define LIBXS_INTERNAL_API_INLINE LIBXS_INLINE
#endif

#define LIBXS_API LIBXS_INTERNAL_API LIBXS_RETARGETABLE
#define LIBXS_API_DEFINITION LIBXS_INTERNAL_API_DEFINITION LIBXS_RETARGETABLE
#define LIBXS_API_INLINE LIBXS_INTERNAL_API_INLINE LIBXS_RETARGETABLE

/* Some definitions kept for compatibility with earlier versions */
#if !defined(LIBXS_EXTERN_C)
# if defined(__cplusplus)
#   define LIBXS_EXTERN_C LIBXS_EXTERN
# else
#   define LIBXS_EXTERN_C
# endif
#endif

#if !defined(LIBXS_RESTRICT)
# if ((defined(__GNUC__) && !defined(__CYGWIN32__)) || defined(__INTEL_COMPILER)) && !defined(_WIN32)
#   define LIBXS_RESTRICT __restrict__
# elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#   define LIBXS_RESTRICT __restrict
# else
#   define LIBXS_RESTRICT
# endif
#endif /*LIBXS_RESTRICT*/
#if !defined(LIBXS_PRAGMA)
# if defined(__INTEL_COMPILER) || defined(_MSC_VER)
#   define LIBXS_PRAGMA(DIRECTIVE) __pragma(DIRECTIVE)
# else
#   define LIBXS_PRAGMA(DIRECTIVE)
# endif
#endif /*LIBXS_PRAGMA*/
#if defined(_MSC_VER)
# define LIBXS_MESSAGE(MSG) LIBXS_PRAGMA(message(MSG))
#elif LIBXS_VERSION3(4, 4, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__) \
   && LIBXS_VERSION3(5, 0, 0) >  LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
# define LIBXS_MESSAGE(MSG) LIBXS_PRAGMA(message MSG)
#else
# define LIBXS_MESSAGE(MSG)
#endif

#if defined(__INTEL_COMPILER)
# define LIBXS_ATTRIBUTE_SIMD
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(simd reduction(EXPRESSION))
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(simd collapse(N))
# define LIBXS_PRAGMA_SIMD_PRIVATE(...) LIBXS_PRAGMA(simd private(__VA_ARGS__))
# define LIBXS_PRAGMA_SIMD LIBXS_PRAGMA(simd)
# define LIBXS_PRAGMA_NOVECTOR LIBXS_PRAGMA(novector)
#elif (defined(_OPENMP) && (201307 <= _OPENMP)) /*OpenMP 4.0*/ || (defined(LIBXS_OPENMP_SIMD) \
  && LIBXS_VERSION3(4, 9, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
# define LIBXS_ATTRIBUTE_SIMD LIBXS_PRAGMA(optimize("openmp-simd"))
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(omp simd reduction(EXPRESSION))
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(omp simd collapse(N))
# define LIBXS_PRAGMA_SIMD_PRIVATE(...) LIBXS_PRAGMA(omp simd private(__VA_ARGS__))
# define LIBXS_PRAGMA_SIMD LIBXS_PRAGMA(omp simd)
# define LIBXS_PRAGMA_NOVECTOR
#else
# define LIBXS_ATTRIBUTE_SIMD
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION)
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N)
# define LIBXS_PRAGMA_SIMD_PRIVATE(...)
# define LIBXS_PRAGMA_SIMD
# define LIBXS_PRAGMA_NOVECTOR
#endif

#if defined(__INTEL_COMPILER)
# define LIBXS_PRAGMA_NONTEMPORAL_VARS(...) LIBXS_PRAGMA(vector nontemporal(__VA_ARGS__))
# define LIBXS_PRAGMA_NONTEMPORAL LIBXS_PRAGMA(vector nontemporal)
# define LIBXS_PRAGMA_VALIGNED_VARS(...) LIBXS_PRAGMA(vector aligned(__VA_ARGS__))
# define LIBXS_PRAGMA_VALIGNED LIBXS_PRAGMA(vector aligned)
# define LIBXS_PRAGMA_FORCEINLINE LIBXS_PRAGMA(forceinline)
# define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG) LIBXS_PRAGMA(loop_count min(MIN) max(MAX) avg(AVG))
# define LIBXS_PRAGMA_UNROLL_N(N) LIBXS_PRAGMA(unroll(N))
# define LIBXS_PRAGMA_UNROLL LIBXS_PRAGMA(unroll)
/*# define LIBXS_UNUSED(VARIABLE) LIBXS_PRAGMA(unused(VARIABLE))*/
#else
# define LIBXS_PRAGMA_NONTEMPORAL_VARS(...)
# define LIBXS_PRAGMA_NONTEMPORAL
# define LIBXS_PRAGMA_VALIGNED_VARS(...)
# define LIBXS_PRAGMA_VALIGNED
# define LIBXS_PRAGMA_FORCEINLINE
# define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG)
# define LIBXS_PRAGMA_UNROLL_N(N)
# define LIBXS_PRAGMA_UNROLL
#endif

#if defined(__clang__) && !defined(__INTEL_COMPILER)
# define LIBXS_PRAGMA_OPTIMIZE_OFF LIBXS_PRAGMA(clang optimize off)
# define LIBXS_PRAGMA_OPTIMIZE_ON  LIBXS_PRAGMA(clang optimize on)
#else
# define LIBXS_PRAGMA_OPTIMIZE_OFF
# define LIBXS_PRAGMA_OPTIMIZE_ON
#endif

#if defined(_OPENMP) && (200805 <= _OPENMP) /*OpenMP 3.0*/
# define LIBXS_OPENMP_COLLAPSE(N) collapse(N)
#else
# define LIBXS_OPENMP_COLLAPSE(N)
#endif

#define LIBXS_REPEAT_1(A) A
#define LIBXS_REPEAT_2(A) LIBXS_REPEAT_1(A); A
#define LIBXS_REPEAT_3(A) LIBXS_REPEAT_2(A); A
#define LIBXS_REPEAT_4(A) LIBXS_REPEAT_3(A); A
#define LIBXS_REPEAT_5(A) LIBXS_REPEAT_4(A); A
#define LIBXS_REPEAT_6(A) LIBXS_REPEAT_5(A); A
#define LIBXS_REPEAT_7(A) LIBXS_REPEAT_6(A); A
#define LIBXS_REPEAT_8(A) LIBXS_REPEAT_7(A); A
#define LIBXS_REPEAT(N, A) LIBXS_CONCATENATE(LIBXS_REPEAT_, N)(A)

/*Based on Stackoverflow's NBITSx macro.*/
#define LIBXS_LOG2_02(N) (0 != ((N) & 0x2/*0b10*/) ? 1 : 0)
#define LIBXS_LOG2_04(N) (0 != ((N) & 0xC/*0b1100*/) ? (2 | LIBXS_LOG2_02((N) >> 2)) : LIBXS_LOG2_02(N))
#define LIBXS_LOG2_08(N) (0 != ((N) & 0xF0/*0b11110000*/) ? (4 | LIBXS_LOG2_04((N) >> 4)) : LIBXS_LOG2_04(N))
#define LIBXS_LOG2_16(N) (0 != ((N) & 0xFF00) ? (8 | LIBXS_LOG2_08((N) >> 8)) : LIBXS_LOG2_08(N))
#define LIBXS_LOG2_32(N) (0 != ((N) & 0xFFFF0000) ? (16 | LIBXS_LOG2_16((N) >> 16)) : LIBXS_LOG2_16(N))
#define LIBXS_LOG2_64(N) (0 != ((N) & 0xFFFFFFFF00000000) ? (32 | LIBXS_LOG2_32((N) >> 32)) : LIBXS_LOG2_32(N))
#define LIBXS_LOG2(N) LIBXS_MAX(LIBXS_LOG2_64((unsigned long long)(N)), 1)

#define LIBXS_DEFAULT(DEFAULT, VALUE) (0 < (VALUE) ? (VALUE) : (DEFAULT))
#define LIBXS_ABS(A) (0 <= (A) ? (A) : -(A))
#define LIBXS_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXS_MAX(A, B) ((A) < (B) ? (B) : (A))
#define LIBXS_CLMP(VALUE, LO, HI) ((LO) < (VALUE) ? ((HI) > (VALUE) ? (VALUE) : LIBXS_MAX(HI, VALUE)) : LIBXS_MIN(LO, VALUE))
#define LIBXS_MOD2(N, NPOT) ((N) & ((NPOT) - 1))
#define LIBXS_MUL2(N, NPOT) ((N) << LIBXS_LOG2(NPOT))
#define LIBXS_DIV2(N, NPOT) ((N) >> LIBXS_LOG2(NPOT))
#define LIBXS_SQRT2(N) (1 << (LIBXS_LOG2((N << 1) - 1) >> 1))
#define LIBXS_UP2(N, NPOT) (((N) + ((NPOT) - 1)) & ~((NPOT) - 1))
#define LIBXS_UP(N, UP) ((((N) + (UP) - 1) / (UP)) * (UP))
/* compares floating point values but avoids warning about unreliable comparison */
#define LIBXS_FEQ(A, B) (!((A) < (B) || (A) > (B)))

#if defined(_WIN32) && !defined(__GNUC__)
# define LIBXS_ATTRIBUTE(A) __declspec(A)
# if defined(__cplusplus)
#   define LIBXS_INLINE_ALWAYS __forceinline
# else
#   define LIBXS_INLINE_ALWAYS static __forceinline
# endif
# define LIBXS_ALIGNED(DECL, N) LIBXS_ATTRIBUTE(align(N)) DECL
# define LIBXS_CDECL __cdecl
#elif defined(__GNUC__)
# define LIBXS_ATTRIBUTE(A) __attribute__((A))
# define LIBXS_INLINE_ALWAYS LIBXS_ATTRIBUTE(always_inline) LIBXS_INLINE
# define LIBXS_ALIGNED(DECL, N) DECL LIBXS_ATTRIBUTE(aligned(N))
# define LIBXS_CDECL LIBXS_ATTRIBUTE(cdecl)
#else
# define LIBXS_ATTRIBUTE(A)
# define LIBXS_INLINE_ALWAYS LIBXS_INLINE
# define LIBXS_ALIGNED(DECL, N)
# define LIBXS_CDECL
#endif

#if defined(__INTEL_COMPILER)
# define LIBXS_ASSUME_ALIGNED(A, N) __assume_aligned(A, N);
# define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION);
#else
# define LIBXS_ASSUME_ALIGNED(A, N)
# if defined(_MSC_VER)
#   define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION);
# elif (LIBXS_VERSION3(4, 5, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#   define LIBXS_ASSUME(EXPRESSION) do { if (!(EXPRESSION)) __builtin_unreachable(); } while(0);
# else
#   define LIBXS_ASSUME(EXPRESSION)
# endif
#endif
#define LIBXS_ALIGN(POINTER, ALIGNMENT/*POT*/) ((POINTER) + (LIBXS_UP2((uintptr_t)(POINTER), ALIGNMENT) - ((uintptr_t)(POINTER))) / sizeof(*(POINTER)))
#define LIBXS_HASH_VALUE(N) ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) ^ ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) >> 27))
#define LIBXS_HASH2(POINTER, ALIGNMENT/*POT*/, NPOT) LIBXS_MOD2(LIBXS_HASH_VALUE(LIBXS_DIV2((uintptr_t)(POINTER), ALIGNMENT)), NPOT)

#if defined(_MSC_VER) /* account for incorrect handling of __VA_ARGS__ */
# define LIBXS_SELECT_ELEMENT(INDEX1/*one-based*/, .../*elements*/) LIBXS_CONCATENATE(LIBXS_SELECT_ELEMENT_, INDEX1)LIBXS_EXPAND((__VA_ARGS__))
#else
# define LIBXS_SELECT_ELEMENT(INDEX1/*one-based*/, .../*elements*/) LIBXS_CONCATENATE(LIBXS_SELECT_ELEMENT_, INDEX1)(__VA_ARGS__)
#endif
#define LIBXS_SELECT_ELEMENT_1(E0, E1, E2, E3, E4, E5, E6, E7) E0
#define LIBXS_SELECT_ELEMENT_2(E0, E1, E2, E3, E4, E5, E6, E7) E1
#define LIBXS_SELECT_ELEMENT_3(E0, E1, E2, E3, E4, E5, E6, E7) E2
#define LIBXS_SELECT_ELEMENT_4(E0, E1, E2, E3, E4, E5, E6, E7) E3
#define LIBXS_SELECT_ELEMENT_5(E0, E1, E2, E3, E4, E5, E6, E7) E4
#define LIBXS_SELECT_ELEMENT_6(E0, E1, E2, E3, E4, E5, E6, E7) E5
#define LIBXS_SELECT_ELEMENT_7(E0, E1, E2, E3, E4, E5, E6, E7) E6
#define LIBXS_SELECT_ELEMENT_8(E0, E1, E2, E3, E4, E5, E6, E7) E7

/**
 * For VLAs, check EXACTLY for C99 since a C11-conforming compiler may not provide VLAs.
 * However, some compilers (Intel) may signal support for VLA even with strict ANSI (C89).
 * To ultimately disable VLA-support, define LIBXS_NO_VLA (make VLA=0).
 * VLA-support is signaled by LIBXS_VLA.
 */
#if !defined(LIBXS_VLA) && !defined(LIBXS_NO_VLA) && ((defined(__STDC_VERSION__) && (199901L/*C99*/ == __STDC_VERSION__ || \
   (!defined(__STDC_NO_VLA__)&& 199901L/*C99*/ < __STDC_VERSION__))) || (defined(__INTEL_COMPILER) && !defined(_WIN32)) || \
    (defined(__GNUC__) && !defined(__STRICT_ANSI__) && !defined(__cplusplus))/*depends on above C99-check*/)
# define LIBXS_VLA
#endif

/**
 * LIBXS_INDEX1 calculates the linear address for a given set of (multiple) indexes/bounds.
 * Syntax: LIBXS_INDEX1(<ndims>, <i0>, ..., <i(ndims-1)>, <s1>, ..., <s(ndims-1)>).
 * Please note that the leading dimension (s0) is omitted in the above syntax!
 * TODO: support leading dimension (pitch/stride).
 */
#if defined(_MSC_VER) /* account for incorrect handling of __VA_ARGS__ */
# define LIBXS_INDEX1(NDIMS, ...) LIBXS_CONCATENATE(LIBXS_INDEX1_, NDIMS)LIBXS_EXPAND((__VA_ARGS__))
#else
# define LIBXS_INDEX1(NDIMS, ...) LIBXS_CONCATENATE(LIBXS_INDEX1_, NDIMS)(__VA_ARGS__)
#endif
#define LIBXS_INDEX1_1(I0) (1ULL * (I0))
#define LIBXS_INDEX1_2(I0, I1, S1) (LIBXS_INDEX1_1(I0) * (S1) + I1)
#define LIBXS_INDEX1_3(I0, I1, I2, S1, S2) (LIBXS_INDEX1_2(I0, I1, S1) * (S2) + (I2))
#define LIBXS_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) (LIBXS_INDEX1_3(I0, I1, I2, S1, S2) * (S3) + (I3))
#define LIBXS_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) (LIBXS_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) * (S4) + (I4))
#define LIBXS_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) (LIBXS_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) * (S5) + (I5))
#define LIBXS_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) (LIBXS_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) * (S6) + (I6))
#define LIBXS_INDEX1_8(I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, S7) (LIBXS_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) * (S7) + (I7))

/**
 * LIBXS_VLA_DECL declares an array according to the given set of (multiple) bounds.
 * Syntax: LIBXS_VLA_DECL(<ndims>, <elem-type>, <var-name>, <init>, <s1>, ..., <s(ndims-1)>).
 * The element type can be "const" or otherwise qualified; initial value must be (const)element-type*.
 * Please note that the syntax is similar to LIBXS_INDEX1, and the leading dimension (s0) is omitted!
 *
 * LIBXS_VLA_ACCESS gives the array element according to the given set of (multiple) indexes/bounds.
 * Syntax: LIBXS_VLA_ACCESS(<ndims>, <array>, <i0>, ..., <i(ndims-1)>, <s1>, ..., <s(ndims-1)>).
 * Please note that the syntax is similar to LIBXS_INDEX1, and the leading dimension (s0) is omitted!
 */
#if defined(LIBXS_VLA)
# define LIBXS_VLA_ACCESS(NDIMS, ARRAY, ...) LIBXS_VLA_ACCESS_Z(NDIMS, ARRAY, LIBXS_VLA_ACCESS_X, __VA_ARGS__)
# define LIBXS_VLA_ACCESS_X(S) + 0 * (S)
# define LIBXS_VLA_ACCESS_Y(...)
# define LIBXS_VLA_ACCESS_Z(NDIMS, ARRAY, XY, ...) LIBXS_CONCATENATE(LIBXS_VLA_ACCESS_, NDIMS)(ARRAY, XY, __VA_ARGS__)
# define LIBXS_VLA_ACCESS_0(ARRAY, XY, ...) (ARRAY)/*scalar*/
# define LIBXS_VLA_ACCESS_1(ARRAY, XY, I0, ...) ((ARRAY)[I0])
# define LIBXS_VLA_ACCESS_2(ARRAY, XY, I0, I1, ...) (((ARRAY) XY(__VA_ARGS__))[I0][I1])
# define LIBXS_VLA_ACCESS_3(ARRAY, XY, I0, I1, I2, S1, ...) (((ARRAY) XY(S1) XY(__VA_ARGS__))[I0][I1][I2])
# define LIBXS_VLA_ACCESS_4(ARRAY, XY, I0, I1, I2, I3, S1, S2, ...) (((ARRAY) XY(S1) XY(S2) XY(__VA_ARGS__))[I0][I1][I2][I3])
# define LIBXS_VLA_ACCESS_5(ARRAY, XY, I0, I1, I2, I3, I4, S1, S2, S3, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(__VA_ARGS__))[I0][I1][I2][I3][I4])
# define LIBXS_VLA_ACCESS_6(ARRAY, XY, I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5])
# define LIBXS_VLA_ACCESS_7(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6])
# define LIBXS_VLA_ACCESS_8(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(S6) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6][I7])
# define LIBXS_VLA_DECL(NDIMS, ELEMENT_TYPE, VARIABLE_NAME, INIT_VALUE, .../*bounds*/) \
    ELEMENT_TYPE LIBXS_VLA_ACCESS_Z(LIBXS_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7), *LIBXS_RESTRICT VARIABLE_NAME, LIBXS_VLA_ACCESS_Y, __VA_ARGS__/*bounds*/, __VA_ARGS__/*dummy*/) = \
   (ELEMENT_TYPE LIBXS_VLA_ACCESS_Z(LIBXS_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7), *, LIBXS_VLA_ACCESS_Y, __VA_ARGS__/*bounds*/, __VA_ARGS__/*dummy*/))(INIT_VALUE)
#else /* calculate linear index */
# define LIBXS_VLA_ACCESS(NDIMS, ARRAY, ...) ((ARRAY)[LIBXS_INDEX1(NDIMS, __VA_ARGS__)])
# define LIBXS_VLA_DECL(NDIMS, ELEMENT_TYPE, VARIABLE_NAME, INIT_VALUE, .../*bounds*/) \
    ELEMENT_TYPE *LIBXS_RESTRICT VARIABLE_NAME = /*(ELEMENT_TYPE*)*/(INIT_VALUE)
#endif

#if !defined(LIBXS_UNUSED)
# if 0 /*defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)*/
#   define LIBXS_UNUSED(VARIABLE) LIBXS_PRAGMA(unused(VARIABLE))
# else
#   define LIBXS_UNUSED(VARIABLE) (void)(VARIABLE)
# endif
#endif

#if defined(__GNUC__) && defined(LIBXS_BUILD)
# define LIBXS_VISIBILITY_HIDDEN LIBXS_ATTRIBUTE(visibility("hidden"))
# define LIBXS_VISIBILITY_INTERNAL LIBXS_ATTRIBUTE(visibility("internal"))
#else
# define LIBXS_VISIBILITY_HIDDEN
# define LIBXS_VISIBILITY_INTERNAL
#endif

#if (defined(__GNUC__) || defined(__clang__)) && !defined(__CYGWIN__)
# define LIBXS_ATTRIBUTE_WEAK_IMPORT LIBXS_ATTRIBUTE(weak_import)
# define LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE(weak)
#else
# define LIBXS_ATTRIBUTE_WEAK
# define LIBXS_ATTRIBUTE_WEAK_IMPORT
#endif

#if defined(__GNUC__)
# define LIBXS_ATTRIBUTE_CTOR LIBXS_ATTRIBUTE(constructor)
# define LIBXS_ATTRIBUTE_DTOR LIBXS_ATTRIBUTE(destructor)
#else
# define LIBXS_ATTRIBUTE_CTOR
# define LIBXS_ATTRIBUTE_DTOR
#endif

#if defined(__GNUC__) || (defined(__INTEL_COMPILER) && !defined(_WIN32))
# define LIBXS_ATTRIBUTE_UNUSED LIBXS_ATTRIBUTE(unused)
#else
# define LIBXS_ATTRIBUTE_UNUSED
#endif
#if defined(__GNUC__)
# define LIBXS_MAY_ALIAS LIBXS_ATTRIBUTE(__may_alias__)
#else
# define LIBXS_MAY_ALIAS
#endif

#if defined(_WIN32)
# define LIBXS_SNPRINTF(S, N, ...) _snprintf_s(S, N, _TRUNCATE, __VA_ARGS__)
# define LIBXS_FLOCK(FILE) _lock_file(FILE)
# define LIBXS_FUNLOCK(FILE) _unlock_file(FILE)
#else
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__)
#   define LIBXS_SNPRINTF(S, N, ...) snprintf(S, N, __VA_ARGS__)
# else
#   define LIBXS_SNPRINTF(S, N, ...) sprintf(S, __VA_ARGS__); LIBXS_UNUSED(N)
# endif
# if !defined(__CYGWIN__)
#   define LIBXS_FLOCK(FILE) flockfile(FILE)
#   define LIBXS_FUNLOCK(FILE) funlockfile(FILE)
# else /* Only available with __CYGWIN__ *and* C++0x. */
#   define LIBXS_FLOCK(FILE)
#   define LIBXS_FUNLOCK(FILE)
# endif
#endif

/** Below group is to fixup some platform/compiler specifics. */
#if defined(_WIN32)
# if !defined(_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES)
#   define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
# endif
# if !defined(_CRT_SECURE_NO_DEPRECATE)
#   define _CRT_SECURE_NO_DEPRECATE 1
# endif
# if !defined(_USE_MATH_DEFINES)
#   define _USE_MATH_DEFINES 1
# endif
# if !defined(WIN32_LEAN_AND_MEAN)
#   define WIN32_LEAN_AND_MEAN 1
# endif
# if !defined(NOMINMAX)
#   define NOMINMAX 1
# endif
# if defined(__INTEL_COMPILER) && (190023506 <= _MSC_FULL_VER)
#   define __builtin_huge_val() HUGE_VAL
#   define __builtin_huge_valf() HUGE_VALF
#   define __builtin_nan nan
#   define __builtin_nanf nanf
#   define __builtin_nans nan
#   define __builtin_nansf nanf
# endif
# if defined(LIBXS_BUILD)
#   if !defined(__STATIC) && !defined(_WINDLL)
#     define __STATIC
#   endif
# endif
#endif
#if defined(__GNUC__)
# if !defined(_GNU_SOURCE)
#   define _GNU_SOURCE
# endif
#endif
#if defined(__clang__) && !defined(__extern_always_inline)
# define __extern_always_inline LIBXS_INLINE
#endif
#if defined(LIBXS_INLINE_FIXUP) && !defined(inline)
# define inline LIBXS_INLINE_KEYWORD
#endif

#if defined(LIBXS_OFFLOAD_BUILD) && \
  defined(__INTEL_OFFLOAD) && (!defined(_WIN32) || (1400 <= __INTEL_COMPILER))
# define LIBXS_OFFLOAD(A) LIBXS_ATTRIBUTE(target(A))
# define LIBXS_NO_OFFLOAD(RTYPE, FN, ...) ((RTYPE (*)(LIBXS_VARIADIC))(FN))(__VA_ARGS__)
# if !defined(LIBXS_OFFLOAD_TARGET)
#   define LIBXS_OFFLOAD_TARGET mic
# endif
#else
# define LIBXS_OFFLOAD(A)
# define LIBXS_NO_OFFLOAD(RTYPE, FN, ...) (FN)(__VA_ARGS__)
#endif
#define LIBXS_RETARGETABLE LIBXS_OFFLOAD(LIBXS_OFFLOAD_TARGET)

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdint.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/* Implementation is taken from an anonymous GiHub Gist. */
LIBXS_INLINE LIBXS_RETARGETABLE unsigned int libxs_icbrt(unsigned long long n) {
  unsigned long long b; unsigned int y = 0; int s;
  for (s = 63; s >= 0; s -= 3) {
    y += y; b = 3 * y * ((unsigned long long)y + 1) + 1;
    if (b <= (n >> s)) { n -= b << s; ++y; }
  }
  return y;
}

/** Similar to LIBXS_UNUSED, this helper "sinks" multiple arguments. */
LIBXS_INLINE LIBXS_RETARGETABLE int libxs_sink(int rvalue, ...) { return rvalue; }

#endif /*LIBXS_MACROS_H*/
