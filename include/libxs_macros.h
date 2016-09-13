/******************************************************************************
** Copyright (c) 2013-2016, Intel Corporation                                **
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

#define LIBXS_STRINGIFY2(SYMBOL) #SYMBOL
#define LIBXS_STRINGIFY(SYMBOL) LIBXS_STRINGIFY2(SYMBOL)
#define LIBXS_TOSTRING(SYMBOL) LIBXS_STRINGIFY(SYMBOL)
#define LIBXS_CONCATENATE2(A, B) A##B
#define LIBXS_CONCATENATE(A, B) LIBXS_CONCATENATE2(A, B)
#define LIBXS_FSYMBOL(SYMBOL) LIBXS_CONCATENATE2(SYMBOL, _)
#define LIBXS_UNIQUE(NAME) LIBXS_CONCATENATE(NAME, __LINE__)

#define LIBXS_VERSION3(MAJOR, MINOR, UPDATE) ((MAJOR) * 10000 + (MINOR) * 100 + (UPDATE))
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
# elif defined(__GNUC__)
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
#if !defined(LIBXS_EXTERN_C) && defined(__cplusplus)
# define LIBXS_EXTERN_C LIBXS_EXTERN
#else
# define LIBXS_EXTERN_C
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
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(simd reduction(EXPRESSION))
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(simd collapse(N))
# define LIBXS_PRAGMA_SIMD_PRIVATE(...) LIBXS_PRAGMA(simd private(__VA_ARGS__))
# define LIBXS_PRAGMA_SIMD LIBXS_PRAGMA(simd)
# define LIBXS_PRAGMA_NOVECTOR LIBXS_PRAGMA(novector)
#elif defined(_OPENMP) && (201307 <= _OPENMP) /*OpenMP 4.0*/
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(omp simd reduction(EXPRESSION))
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(omp simd collapse(N))
# define LIBXS_PRAGMA_SIMD_PRIVATE(...) LIBXS_PRAGMA(omp simd private(__VA_ARGS__))
# define LIBXS_PRAGMA_SIMD LIBXS_PRAGMA(omp simd)
# define LIBXS_PRAGMA_NOVECTOR
#else
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

/* For VLAs, check EXACTLY for C99 since a C11-conformant compiler may not provide VLAs */
#if !defined(LIBXS_VLA) && ((defined(__STDC_VERSION__) && (199901L/*C99*/ == __STDC_VERSION__ || \
   (!defined(__STDC_NO_VLA__)&& 199901L/*C99*/ < __STDC_VERSION__))) || defined(__INTEL_COMPILER) || \
    (defined(__GNUC__) && !defined(__STRICT_ANSI__))/*depends on above C99-check*/)
# define LIBXS_VLA
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
#define LIBXS_CLMP(VALUE, LO, HI) ((LO) < (VALUE) ? ((HI) > (VALUE) ? (VALUE) : (HI)) : (LO))
#define LIBXS_MOD2(N, NPOT) ((N) & ((NPOT) - 1))
#define LIBXS_MUL2(N, NPOT) ((N) << LIBXS_LOG2(NPOT))
#define LIBXS_DIV2(N, NPOT) ((N) >> LIBXS_LOG2(NPOT))
#define LIBXS_SQRT2(N) (1 << (LIBXS_LOG2((N << 1) - 1) >> 1))
#define LIBXS_UP2(N, NPOT) LIBXS_MUL2(LIBXS_DIV2((N) + (NPOT) - 1, NPOT), NPOT)
#define LIBXS_UP(N, UP) ((((N) + (UP) - 1) / (UP)) * (UP))
/* compares floating point values but avoids warning about unreliable comparison */
#define LIBXS_FEQ(A, B) (!((A) < (B) || (A) > (B)))

#if defined(_WIN32) && !defined(__GNUC__)
# define LIBXS_ATTRIBUTE(...) __declspec(__VA_ARGS__)
# if defined(__cplusplus)
#   define LIBXS_INLINE_ALWAYS __forceinline
# else
#   define LIBXS_INLINE_ALWAYS static __forceinline
# endif
# define LIBXS_ALIGNED(DECL, N) LIBXS_ATTRIBUTE(align(N)) DECL
# define LIBXS_CDECL __cdecl
#elif defined(__GNUC__)
# define LIBXS_ATTRIBUTE(...) __attribute__((__VA_ARGS__))
# define LIBXS_INLINE_ALWAYS LIBXS_ATTRIBUTE(always_inline) LIBXS_INLINE
# define LIBXS_ALIGNED(DECL, N) DECL LIBXS_ATTRIBUTE(aligned(N))
# define LIBXS_CDECL LIBXS_ATTRIBUTE(cdecl)
#else
# define LIBXS_ATTRIBUTE(...)
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
#define LIBXS_ALIGN_VALUE(N, TYPESIZE, ALIGNMENT/*POT*/) (LIBXS_UP2((N) * (TYPESIZE), ALIGNMENT) / (TYPESIZE))
#define LIBXS_ALIGN_VALUE2(N, POTSIZE, ALIGNMENT/*POT*/) LIBXS_DIV2(LIBXS_UP2(LIBXS_MUL2(N, POTSIZE), ALIGNMENT), POTSIZE)
#define LIBXS_ALIGN(POINTER, ALIGNMENT/*POT*/) ((POINTER) + (LIBXS_ALIGN_VALUE((unsigned long long)(POINTER), 1, ALIGNMENT) - ((unsigned long long)(POINTER))) / sizeof(*(POINTER)))
#define LIBXS_ALIGN2(POINTPOT, ALIGNMENT/*POT*/) ((POINTPOT) + LIBXS_DIV2(LIBXS_ALIGN_VALUE2((unsigned long long)(POINTPOT), 1, ALIGNMENT) - ((unsigned long long)(POINTPOT)), sizeof(*(POINTPOT))))

#define LIBXS_HASH_VALUE(N) ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) ^ ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) >> 27))
#define LIBXS_HASH2(POINTER, ALIGNMENT/*POT*/, NPOT) LIBXS_MOD2(LIBXS_HASH_VALUE(LIBXS_DIV2((unsigned long long)(POINTER), ALIGNMENT)), NPOT)

#define LIBXS_CALC_SIZE1(TYPE, VARIABLE, NDIMS, SHAPE, INIT) { \
  unsigned int libxs_calc_size1_i_ = 0; \
  VARIABLE = LIBXS_MAX(INIT, 1); \
  LIBXS_REPEAT(NDIMS, \
    VARIABLE *= (TYPE)((SHAPE)[libxs_calc_size1_i_]); \
    ++libxs_calc_size1_i_;) \
}
/* TODO: LIBXS_CALC_INDEX1 plus PITCH */
#define LIBXS_CALC_INDEX1(TYPE, VARIABLE, NDIMS, INDEXN, SHAPE) { \
  unsigned int libxs_calc_index1_i_ = 0; \
  TYPE libxs_calc_index1_size_ = 1; \
  VARIABLE = 0; \
  LIBXS_REPEAT(NDIMS, \
    VARIABLE += libxs_calc_index1_size_ * ((TYPE)(INDEXN)[libxs_calc_index1_i_]); \
    libxs_calc_index1_size_ *= (TYPE)((SHAPE)[libxs_calc_index1_i_]); \
    ++libxs_calc_index1_i_;) \
}

#if !defined(LIBXS_UNUSED)
# if 0 /*defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)*/
#   define LIBXS_UNUSED(VARIABLE) LIBXS_PRAGMA(unused(VARIABLE))
# else
#   define LIBXS_UNUSED(VARIABLE) (void)(VARIABLE)
# endif
#endif

#if defined(__GNUC__) || (defined(__INTEL_COMPILER) && !defined(_WIN32))
# define LIBXS_UNUSED_ARG LIBXS_ATTRIBUTE(unused)
#else
# define LIBXS_UNUSED_ARG
#endif

#if defined(__GNUC__) && defined(LIBXS_BUILD)
# define LIBXS_VISIBILITY_HIDDEN LIBXS_ATTRIBUTE(visibility("hidden"))
# define LIBXS_VISIBILITY_INTERNAL LIBXS_ATTRIBUTE(visibility("internal"))
#else
# define LIBXS_VISIBILITY_HIDDEN
# define LIBXS_VISIBILITY_INTERNAL
#endif

#if (defined(__GNUC__) || defined(__clang__))
# define LIBXS_ATTRIBUTE_WEAK_IMPORT LIBXS_ATTRIBUTE(weak_import)
# if defined(__CYGWIN__)
#   define LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE(weakref)
#else
#   define LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE(weak)
# endif
#else
# define LIBXS_ATTRIBUTE_WEAK
# define LIBXS_ATTRIBUTE_WEAK_IMPORT
#endif

#if defined(NDEBUG)
# define LIBXS_NDEBUG NDEBUG
# define LIBXS_DEBUG(...)
#else
# define LIBXS_DEBUG(...) __VA_ARGS__
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

#endif /*LIBXS_MACROS_H*/
