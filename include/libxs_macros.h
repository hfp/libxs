/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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

#define LIBXS_STRINGIFY(SYMBOL) #SYMBOL
#define LIBXS_CONCATENATE(A, B) A##B
#define LIBXS_FSYMBOL(SYMBOL) LIBXS_CONCATENATE(SYMBOL, _)

#define LIBXS_BLASPREC(PREFIX, REAL, FUNCTION) LIBXS_BLASPREC_##REAL(PREFIX, FUNCTION)
#define LIBXS_BLASPREC_double(PREFIX, FUNCTION) PREFIX##d##FUNCTION
#define LIBXS_BLASPREC_float(PREFIX, FUNCTION) PREFIX##s##FUNCTION

#if defined(__cplusplus)
# define LIBXS_EXTERN_C extern "C"
# define LIBXS_INLINE inline
#else
# define LIBXS_EXTERN_C
# if (199901L <= __STDC_VERSION__)
#   define LIBXS_PRAGMA(DIRECTIVE) _Pragma(LIBXS_STRINGIFY(DIRECTIVE))
#   define LIBXS_RESTRICT restrict
#   define LIBXS_INLINE inline
# else
#   define LIBXS_INLINE static
# endif /*C99*/
#endif /*__cplusplus*/
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
#if defined(__INTEL_COMPILER)
# define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG) LIBXS_PRAGMA(loop_count min(MIN) max(MAX) avg(AVG))
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(simd reduction(EXPRESSION))
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(simd collapse(N))
#elif (201307 <= _OPENMP) // V4.0
# define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG)
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(omp simd reduction(EXPRESSION))
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(omp simd collapse(N))
#else
# define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG)
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION)
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N)
#endif

#define LIBXS_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXS_MAX(A, B) ((A) < (B) ? (B) : (A))

#if defined(_WIN32) && !defined(__GNUC__)
# define LIBXS_ATTRIBUTE(A) __declspec(A)
# define LIBXS_ALIGNED(DECL, N) LIBXS_ATTRIBUTE(align(N)) DECL
#elif defined(__GNUC__)
# define LIBXS_ATTRIBUTE(A) __attribute__((A))
# define LIBXS_ALIGNED(DECL, N) DECL LIBXS_ATTRIBUTE(aligned(N))
#else
# define LIBXS_ATTRIBUTE(A)
# define LIBXS_ALIGNED(DECL, N)
#endif

#if defined(__INTEL_OFFLOAD) && (!defined(_WIN32) || (1400 <= __INTEL_COMPILER))
# define LIBXS_OFFLOAD 1
# define LIBXS_TARGET(A) LIBXS_ATTRIBUTE(target(A))
#else
/*# define LIBXS_OFFLOAD 0*/
# define LIBXS_TARGET(A)
#endif

#if defined(__INTEL_COMPILER)
# define LIBXS_ASSUME_ALIGNED(A, N) __assume_aligned(A, N)
# define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION)
#else
# define LIBXS_ASSUME_ALIGNED(A, N)
# if defined(_MSC_VER)
#   define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION)
# elif (40500 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__))
#   define LIBXS_ASSUME(EXPRESSION) do { if (!(EXPRESSION)) __builtin_unreachable(); } while(0)
# endif
#endif
#define LIBXS_ALIGN_VALUE(DST_TYPE, SRC_TYPE, VALUE, ALIGNMENT) ((DST_TYPE)((-( \
  -((intptr_t)(VALUE) * ((intptr_t)sizeof(SRC_TYPE))) & \
  -((intptr_t)(LIBXS_MAX(ALIGNMENT, 1))))) / sizeof(SRC_TYPE)))
#define LIBXS_ALIGN(TYPE, PTR, ALIGNMENT) LIBXS_ALIGN_VALUE(TYPE, char, PTR, ALIGNMENT)

#if defined(LIBXMM_OFFLOAD)
# pragma offload_attribute(push,target(mic))
# include <stdint.h>
# pragma offload_attribute(pop)
#else
# include <stdint.h>
#endif

#endif /*LIBXS_MACROS_H*/
