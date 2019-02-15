/******************************************************************************
** Copyright (c) 2013-2019, Intel Corporation                                **
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

/** Parameters the library was built for. */
#define LIBXS_CACHELINE LIBXS_CONFIG_CACHELINE
#define LIBXS_ALIGNMENT LIBXS_CONFIG_ALIGNMENT
#define LIBXS_ILP64 LIBXS_CONFIG_ILP64
#define LIBXS_SYNC LIBXS_CONFIG_SYNC
#define LIBXS_JIT LIBXS_CONFIG_JIT

/** Parameters of GEMM domain (static kernels, etc). */
#define LIBXS_PREFETCH LIBXS_CONFIG_PREFETCH
#define LIBXS_MAX_MNK LIBXS_CONFIG_MAX_MNK
#define LIBXS_MAX_M LIBXS_CONFIG_MAX_M
#define LIBXS_MAX_N LIBXS_CONFIG_MAX_N
#define LIBXS_MAX_K LIBXS_CONFIG_MAX_K
#define LIBXS_AVG_M LIBXS_CONFIG_AVG_M
#define LIBXS_AVG_N LIBXS_CONFIG_AVG_N
#define LIBXS_AVG_K LIBXS_CONFIG_AVG_K
#define LIBXS_FLAGS LIBXS_CONFIG_FLAGS
#define LIBXS_ALPHA LIBXS_CONFIG_ALPHA
#define LIBXS_BETA LIBXS_CONFIG_BETA
#define LIBXS_WRAP LIBXS_CONFIG_WRAP

#if (defined(__SIZEOF_PTRDIFF_T__) && 4 < (__SIZEOF_PTRDIFF_T__)) || \
    (defined(__SIZE_MAX__) && (4294967295U < (__SIZE_MAX__))) || \
    (defined(__GNUC__) && defined(_CRAYC)) || defined(_WIN64) || \
    (defined(__x86_64__) && 0 != (__x86_64__))
# define LIBXS_BITS 64
#elif defined(NDEBUG) /* not for production use! */
# error LIBXS is only supported on a 64-bit platform!
#else /* JIT-generated code (among other issues) is not supported! */
# define LIBXS_BITS 32
#endif

#define LIBXS_STRINGIFY2(SYMBOL) #SYMBOL
#define LIBXS_STRINGIFY(SYMBOL) LIBXS_STRINGIFY2(SYMBOL)
#define LIBXS_TOSTRING(SYMBOL) LIBXS_STRINGIFY(SYMBOL)
#define LIBXS_CONCATENATE1(A, B) A##B
#define LIBXS_CONCATENATE(A, B) LIBXS_CONCATENATE1(A, B)
#define LIBXS_CONCATENATE2(A, B, C) LIBXS_CONCATENATE(LIBXS_CONCATENATE(A, B), C)
#define LIBXS_CONCATENATE3(A, B, C, D) LIBXS_CONCATENATE(LIBXS_CONCATENATE2(A, B, C), D)
#define LIBXS_FSYMBOL(SYMBOL) LIBXS_CONCATENATE(SYMBOL, _)
#define LIBXS_UNIQUE(NAME) LIBXS_CONCATENATE(NAME, __LINE__)
#define LIBXS_EXPAND(...) __VA_ARGS__
#define LIBXS_ELIDE(...)

#define LIBXS_VERSION2(MAJOR, MINOR) ((MAJOR) * 10000 + (MINOR) * 100)
#define LIBXS_VERSION3(MAJOR, MINOR, UPDATE) (LIBXS_VERSION2(MAJOR, MINOR) + (UPDATE))
#define LIBXS_VERSION4(MAJOR, MINOR, UPDATE, PATCH) ((MAJOR) * 100000000 + (MINOR) * 1000000 + (UPDATE) * 10000 + (PATCH))

#if defined(__cplusplus)
# define LIBXS_VARIADIC ...
# define LIBXS_EXTERN_KEYWORD extern "C"
# define LIBXS_EXTERN LIBXS_EXTERN_KEYWORD
# define LIBXS_EXTERN_C LIBXS_EXTERN_KEYWORD
# define LIBXS_INLINE_KEYWORD inline
# define LIBXS_INLINE LIBXS_INLINE_KEYWORD
# if defined(__GNUC__)
#   define LIBXS_CALLER_ID __PRETTY_FUNCTION__
# elif defined(_MSC_VER)
#   define LIBXS_CALLER_ID __FUNCDNAME__
#   define LIBXS_CALLER __FUNCTION__
# else
#   define LIBXS_CALLER_ID __FUNCNAME__
# endif
#else
# define LIBXS_VARIADIC
# define LIBXS_EXTERN_KEYWORD extern
# define LIBXS_EXTERN LIBXS_EXTERN_KEYWORD
# define LIBXS_EXTERN_C
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
#   define LIBXS_PRAGMA(DIRECTIVE) _Pragma(LIBXS_STRINGIFY(DIRECTIVE))
#   define LIBXS_CALLER_ID __func__
#   define LIBXS_RESTRICT restrict
#   define LIBXS_INLINE_KEYWORD inline
# elif defined(_MSC_VER)
#   define LIBXS_CALLER_ID __FUNCDNAME__
#   define LIBXS_CALLER __FUNCTION__
#   define LIBXS_INLINE_KEYWORD __inline
#   define LIBXS_INLINE_FIXUP
# elif defined(__GNUC__) && !defined(__STRICT_ANSI__)
#   define LIBXS_CALLER_ID __PRETTY_FUNCTION__
# endif
# if !defined(LIBXS_INLINE_KEYWORD)
#   define LIBXS_INLINE_KEYWORD
#   define LIBXS_INLINE_FIXUP
# endif
# define LIBXS_INLINE static LIBXS_INLINE_KEYWORD
#endif /*__cplusplus*/
#if !defined(LIBXS_CALLER_ID)
# define LIBXS_CALLER_ID NULL
#endif
#if !defined(LIBXS_CALLER)
# define LIBXS_CALLER LIBXS_CALLER_ID
#endif

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
# define LIBXS_ALIGNED(DECL, N) DECL
# define LIBXS_CDECL
#endif

#if defined(__INTEL_COMPILER)
# define LIBXS_INTEL_COMPILER __INTEL_COMPILER
#elif defined(__INTEL_COMPILER_BUILD_DATE)
# define LIBXS_INTEL_COMPILER ((__INTEL_COMPILER_BUILD_DATE / 10000 - 2000) * 100)
#endif

#if defined(LIBXS_OFFLOAD_BUILD) && \
  defined(__INTEL_OFFLOAD) && (!defined(_WIN32) || (1400 <= LIBXS_INTEL_COMPILER))
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

#if !defined(__STATIC) && !defined(_WINDLL) && !defined(__MINGW32__) && (defined(_WIN32) || defined(__CYGWIN__))
# define __STATIC
#endif

/* may include Clang and other compatible compilers */
#if defined(__GNUC__) && !defined(_WIN32) && !defined(__CYGWIN__)
# define LIBXS_VISIBILITY_INTERNAL LIBXS_ATTRIBUTE(visibility("internal"))
# define LIBXS_VISIBILITY_HIDDEN LIBXS_ATTRIBUTE(visibility("hidden"))
# define LIBXS_VISIBILITY_PUBLIC LIBXS_ATTRIBUTE(visibility("default"))
#endif
#if !defined(LIBXS_VISIBILITY_INTERNAL)
# define LIBXS_VISIBILITY_INTERNAL
#endif
#if !defined(LIBXS_VISIBILITY_HIDDEN)
# define LIBXS_VISIBILITY_HIDDEN
#endif
#if !defined(LIBXS_VISIBILITY_PUBLIC)
# define LIBXS_VISIBILITY_PUBLIC
#endif
#if !defined(LIBXS_VISIBILITY_PRIVATE)
# define LIBXS_VISIBILITY_PRIVATE LIBXS_VISIBILITY_HIDDEN
#endif

/* Windows Dynamic Link Library (DLL) */
#if !defined(__STATIC) && (defined(_WIN32) || defined(__CYGWIN__))
# define LIBXS_VISIBILITY_EXPORT LIBXS_ATTRIBUTE(dllexport)
# define LIBXS_VISIBILITY_IMPORT LIBXS_ATTRIBUTE(dllimport)
#endif
#if !defined(LIBXS_VISIBILITY_EXPORT)
# define LIBXS_VISIBILITY_EXPORT LIBXS_EXTERN_C LIBXS_VISIBILITY_PUBLIC
#endif
#if !defined(LIBXS_VISIBILITY_IMPORT)
# define LIBXS_VISIBILITY_IMPORT LIBXS_EXTERN_KEYWORD LIBXS_VISIBILITY_PUBLIC
#endif

#if defined(LIBXS_API) /* header-only mode */
# define LIBXS_API_INTERN LIBXS_API
# define LIBXS_API_EXPORT LIBXS_API
# define LIBXS_APIVAR_PUBLIC(DECL) LIBXS_EXTERN_C LIBXS_RETARGETABLE DECL; LIBXS_RETARGETABLE DECL
# define LIBXS_APIVAR(DECL) LIBXS_APIVAR_PUBLIC(DECL)
# define LIBXS_EXTVAR(DECL) LIBXS_APIVAR(DECL)
# define LIBXS_APIEXT_INTERN LIBXS_API_INTERN
# define LIBXS_APIEXT LIBXS_APIEXT_INTERN
#else /* classic ABI */
# if defined(LIBXS_BUILD_EXT)
#   define LIBXS_API LIBXS_VISIBILITY_IMPORT LIBXS_RETARGETABLE
#   define LIBXS_API_INTERN LIBXS_VISIBILITY_PRIVATE LIBXS_RETARGETABLE
#   define LIBXS_API_EXPORT LIBXS_EXTERN_C LIBXS_API
#   define LIBXS_APIVAR_PUBLIC(DECL) LIBXS_API DECL; LIBXS_API DECL
#   define LIBXS_APIVAR(DECL) LIBXS_EXTERN_C LIBXS_VISIBILITY_PRIVATE LIBXS_RETARGETABLE DECL; \
                                LIBXS_EXTERN_C LIBXS_VISIBILITY_PRIVATE LIBXS_RETARGETABLE DECL
#   define LIBXS_EXTVAR(DECL) LIBXS_APIVAR(DECL)
#   define LIBXS_APIEXT LIBXS_VISIBILITY_EXPORT LIBXS_RETARGETABLE
#   define LIBXS_APIEXT_INTERN LIBXS_API_INTERN
# elif defined(LIBXS_BUILD)
#   define LIBXS_API LIBXS_VISIBILITY_EXPORT LIBXS_RETARGETABLE
#   define LIBXS_API_INTERN LIBXS_VISIBILITY_PRIVATE LIBXS_RETARGETABLE
#   define LIBXS_API_EXPORT LIBXS_API
#   define LIBXS_APIVAR_PUBLIC(DECL) LIBXS_VISIBILITY_EXPORT LIBXS_RETARGETABLE DECL; \
                                       LIBXS_VISIBILITY_EXPORT LIBXS_RETARGETABLE DECL
#   define LIBXS_APIVAR(DECL) LIBXS_EXTERN_C LIBXS_VISIBILITY_PRIVATE LIBXS_RETARGETABLE DECL; \
                                LIBXS_EXTERN_C LIBXS_VISIBILITY_PRIVATE LIBXS_RETARGETABLE DECL
#   define LIBXS_APIEXT LIBXS_VISIBILITY_IMPORT LIBXS_RETARGETABLE
# else /* import */
#   define LIBXS_API LIBXS_VISIBILITY_IMPORT LIBXS_RETARGETABLE
#   define LIBXS_API_INTERN LIBXS_API
#   define LIBXS_API_EXPORT LIBXS_API
#   define LIBXS_APIVAR_PUBLIC(DECL) LIBXS_VISIBILITY_IMPORT LIBXS_RETARGETABLE DECL; \
                                       LIBXS_VISIBILITY_IMPORT LIBXS_RETARGETABLE DECL
#   define LIBXS_APIVAR(DECL) LIBXS_EXTERN_C LIBXS_VISIBILITY_PRIVATE LIBXS_RETARGETABLE DECL; \
                                LIBXS_EXTERN_C LIBXS_VISIBILITY_PRIVATE LIBXS_RETARGETABLE DECL
#   define LIBXS_APIEXT LIBXS_API
# endif
#endif
#if !defined(_WIN32)
# define LIBXS_APIVAR_ALIGNED(DECL) LIBXS_APIVAR_PUBLIC(LIBXS_ALIGNED(DECL, 32))
#else /* Windows */
# define LIBXS_APIVAR_ALIGNED LIBXS_APIVAR_PUBLIC
#endif
#define LIBXS_API_INLINE LIBXS_EXTERN_C LIBXS_INLINE LIBXS_RETARGETABLE

#if !defined(LIBXS_RESTRICT)
# if ((defined(__GNUC__) && !defined(__CYGWIN32__)) || defined(LIBXS_INTEL_COMPILER)) && !defined(_WIN32)
#   define LIBXS_RESTRICT __restrict__
# elif defined(_MSC_VER) || defined(LIBXS_INTEL_COMPILER)
#   define LIBXS_RESTRICT __restrict
# else
#   define LIBXS_RESTRICT
# endif
#endif /*LIBXS_RESTRICT*/
#if !defined(LIBXS_PRAGMA)
# if defined(LIBXS_INTEL_COMPILER) || defined(_MSC_VER)
#   define LIBXS_PRAGMA(DIRECTIVE) __pragma(DIRECTIVE)
# else
#   define LIBXS_PRAGMA(DIRECTIVE)
# endif
#endif /*LIBXS_PRAGMA*/
#if !defined(LIBXS_OPENMP_SIMD) && (defined(_OPENMP) && (201307 <= _OPENMP)) /*OpenMP 4.0*/
# if defined(LIBXS_INTEL_COMPILER)
#   if (1500 <= LIBXS_INTEL_COMPILER)
#     define LIBXS_OPENMP_SIMD
#   endif
# elif defined(__GNUC__)
#   if LIBXS_VERSION3(4, 9, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#     define LIBXS_OPENMP_SIMD
#   endif
# else
#   define LIBXS_OPENMP_SIMD
# endif
#endif

#if !defined(LIBXS_INTEL_COMPILER) || (LIBXS_INTEL_COMPILER < 9900)
# if defined(LIBXS_OPENMP_SIMD)
#   define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(omp simd reduction(EXPRESSION))
#   define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(omp simd collapse(N))
#   define LIBXS_PRAGMA_SIMD_PRIVATE(A, ...) LIBXS_PRAGMA(omp simd private(A, __VA_ARGS__))
#   define LIBXS_PRAGMA_SIMD LIBXS_PRAGMA(omp simd)
# elif defined(LIBXS_INTEL_COMPILER)
#   define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(simd reduction(EXPRESSION))
#   define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(simd collapse(N))
#   define LIBXS_PRAGMA_SIMD_PRIVATE(A, ...) LIBXS_PRAGMA(simd private(A, __VA_ARGS__))
#   define LIBXS_PRAGMA_SIMD LIBXS_PRAGMA(simd)
# endif
#endif
#if !defined(LIBXS_PRAGMA_SIMD)
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION)
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N)
# define LIBXS_PRAGMA_SIMD_PRIVATE(A, ...)
# define LIBXS_PRAGMA_SIMD
#endif

#if defined(__INTEL_COMPILER)
# define LIBXS_PRAGMA_NONTEMPORAL_VARS(A, ...) LIBXS_PRAGMA(vector nontemporal(A, __VA_ARGS__))
# define LIBXS_PRAGMA_NONTEMPORAL LIBXS_PRAGMA(vector nontemporal)
# define LIBXS_PRAGMA_VALIGNED LIBXS_PRAGMA(vector aligned)
# define LIBXS_PRAGMA_NOVECTOR LIBXS_PRAGMA(novector)
# define LIBXS_PRAGMA_FORCEINLINE LIBXS_PRAGMA(forceinline)
# define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG) LIBXS_PRAGMA(loop_count min=MIN max=MAX avg=AVG)
# define LIBXS_PRAGMA_UNROLL_AND_JAM(N) LIBXS_PRAGMA(unroll_and_jam(N))
# define LIBXS_PRAGMA_UNROLL_N(N) LIBXS_PRAGMA(unroll(N))
# define LIBXS_PRAGMA_UNROLL LIBXS_PRAGMA(unroll)
# define LIBXS_PRAGMA_VALIGNED_VAR(A) LIBXS_ASSUME_ALIGNED(A, LIBXS_ALIGNMENT);
/*# define LIBXS_UNUSED(VARIABLE) LIBXS_PRAGMA(unused(VARIABLE))*/
#else
# define LIBXS_PRAGMA_NONTEMPORAL_VARS(A, ...)
# define LIBXS_PRAGMA_NONTEMPORAL
# define LIBXS_PRAGMA_VALIGNED_VAR(A)
# define LIBXS_PRAGMA_VALIGNED
# define LIBXS_PRAGMA_NOVECTOR
# define LIBXS_PRAGMA_FORCEINLINE
# define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG)
# define LIBXS_PRAGMA_UNROLL_AND_JAM(N)
# define LIBXS_PRAGMA_UNROLL_N(N)
# define LIBXS_PRAGMA_UNROLL
#endif

#if defined(LIBXS_INTEL_COMPILER)
# define LIBXS_PRAGMA_OPTIMIZE_OFF LIBXS_PRAGMA(optimize("", off))
# define LIBXS_PRAGMA_OPTIMIZE_ON  LIBXS_PRAGMA(optimize("", on))
#elif defined(__clang__)
# define LIBXS_PRAGMA_OPTIMIZE_OFF LIBXS_PRAGMA(clang optimize off)
# define LIBXS_PRAGMA_OPTIMIZE_ON  LIBXS_PRAGMA(clang optimize on)
#elif defined(__GNUC__)
# define LIBXS_PRAGMA_OPTIMIZE_OFF LIBXS_PRAGMA(GCC push_options) LIBXS_PRAGMA(GCC optimize("O0"))
# define LIBXS_PRAGMA_OPTIMIZE_ON  LIBXS_PRAGMA(GCC pop_options)
#else
# define LIBXS_PRAGMA_OPTIMIZE_OFF
# define LIBXS_PRAGMA_OPTIMIZE_ON
#endif

#if defined(_OPENMP) && (200805 <= _OPENMP) /*OpenMP 3.0*/
# define LIBXS_OPENMP_COLLAPSE(N) collapse(N)
#else
# define LIBXS_OPENMP_COLLAPSE(N)
#endif

/** LIBXS_NBITS determines the minimum number of bits needed to represent N. */
#define LIBXS_NBITS(N) (LIBXS_INTRINSICS_BITSCANBWD64(N) + LIBXS_MIN(1, N))
/** LIBXS_ILOG2 definition matches ceil(log2(N)). */
#define LIBXS_ILOG2(N) (1 < (N) ? (LIBXS_INTRINSICS_BITSCANBWD64(N) + \
  (LIBXS_INTRINSICS_BITSCANBWD64((N) - 1) != LIBXS_INTRINSICS_BITSCANBWD64(N) ? 0 : 1)) : 0)

/** LIBXS_UP2POT rounds up to the next power of two (POT). */
#define LIBXS_UP2POT_01(N) ((N) | ((N) >> 1))
#define LIBXS_UP2POT_02(N) (LIBXS_UP2POT_01(N) | (LIBXS_UP2POT_01(N) >> 2))
#define LIBXS_UP2POT_04(N) (LIBXS_UP2POT_02(N) | (LIBXS_UP2POT_02(N) >> 4))
#define LIBXS_UP2POT_08(N) (LIBXS_UP2POT_04(N) | (LIBXS_UP2POT_04(N) >> 8))
#define LIBXS_UP2POT_16(N) (LIBXS_UP2POT_08(N) | (LIBXS_UP2POT_08(N) >> 16))
#define LIBXS_UP2POT_32(N) (LIBXS_UP2POT_16(N) | (LIBXS_UP2POT_16(N) >> 32))
#define LIBXS_UP2POT(N) (LIBXS_UP2POT_32((unsigned long long)(N) - LIBXS_MIN(1, N)) + LIBXS_MIN(1, N))
#define LIBXS_LO2POT(N) (LIBXS_UP2POT_32((unsigned long long)(N) >> 1) + LIBXS_MIN(1, N))

#define LIBXS_UP2(N, NPOT) ((((uintptr_t)N) + ((NPOT) - 1)) & ~((NPOT) - 1))
#define LIBXS_UP(N, UP) ((((N) + (UP) - 1) / (UP)) * (UP))
#define LIBXS_ABS(A) (0 <= (A) ? (A) : -(A))
#define LIBXS_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXS_MAX(A, B) ((A) < (B) ? (B) : (A))
#define LIBXS_MOD(A, N) ((A) % (N))
#define LIBXS_MOD2(A, NPOT) ((A) & ((NPOT) - 1))
#define LIBXS_DIFF(T0, T1) ((T0) < (T1) ? ((T1) - (T0)) : ((T0) - (T1)))
#define LIBXS_CLMP(VALUE, LO, HI) ((LO) < (VALUE) ? ((VALUE) <= (HI) ? (VALUE) : LIBXS_MIN(VALUE, HI)) : LIBXS_MAX(LO, VALUE))
#define LIBXS_SQRT2(N) ((unsigned int)((1ULL << (LIBXS_NBITS(N) >> 1)) /*+ LIBXS_MIN(1, N)*/))
#define LIBXS_HASH2(N) ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) ^ ((((N) ^ ((N) >> 12)) ^ (((N) ^ ((N) >> 12)) << 25)) >> 27))
#define LIBXS_SIZEOF(START, LAST) (((const char*)(LAST)) - ((const char*)(START)) + sizeof(*LAST))
#define LIBXS_FEQ(A, B) ((A) == (B))
#define LIBXS_NEQ(A, B) ((A) != (B))
#define LIBXS_ISNAN(A)  LIBXS_NEQ(A, A)
#define LIBXS_NOTNAN(A) LIBXS_FEQ(A, A)
#define LIBXS_ROUNDX(TYPE, A) ((TYPE)((long long)(0 <= (A) ? ((double)(A) + 0.5) : ((double)(A) - 0.5))))

/** Makes some functions available independent of C99 support. */
#if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
# define LIBXS_FREXPF(A, B) frexpf(A, B)
# define LIBXS_POWF(A, B) powf(A, B)
# define LIBXS_ROUNDF(A) roundf(A)
# define LIBXS_ROUND(A) round(A)
# define LIBXS_TANHF(A) tanhf(A)
# define LIBXS_LOG2(A) log2(A)
# define LIBXS_LOGF(A) logf(A)
#else
# define LIBXS_FREXPF(A, B) ((float)frexp((double)(A), B))
# define LIBXS_POWF(A, B) ((float)pow((double)(A), (double)(B)))
# define LIBXS_ROUNDF(A) LIBXS_ROUNDX(float, A)
# define LIBXS_ROUND(A) LIBXS_ROUNDX(double, A)
# define LIBXS_TANHF(A) ((float)tanh((double)(A)))
# define LIBXS_LOG2(A) (log(A) * (1.0 / (M_LN2)))
# define LIBXS_LOGF(A) ((float)log((double)(A)))
#endif

#if defined(LIBXS_INTEL_COMPILER)
# if (1600 <= LIBXS_INTEL_COMPILER)
#   define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION)
# else
#   define LIBXS_ASSUME(EXPRESSION) assert(EXPRESSION)
# endif
#elif defined(_MSC_VER)
# define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION)
#elif defined(__GNUC__) && !defined(_CRAYC) && (LIBXS_VERSION3(4, 5, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
# define LIBXS_ASSUME(EXPRESSION) do { if (!(EXPRESSION)) __builtin_unreachable(); } while(0)
#else
# define LIBXS_ASSUME(EXPRESSION) assert(EXPRESSION)
#endif

#if defined(LIBXS_INTEL_COMPILER)
# define LIBXS_ASSUME_ALIGNED(A, N) __assume_aligned(A, N)
#else
# define LIBXS_ASSUME_ALIGNED(A, N) assert(0 == ((uintptr_t)(A)) % (N))
#endif
#define LIBXS_ALIGN(POINTER, ALIGNMENT/*POT*/) ((POINTER) + (LIBXS_UP2((uintptr_t)(POINTER), ALIGNMENT) - ((uintptr_t)(POINTER))) / sizeof(*(POINTER)))
#define LIBXS_FOLD2(POINTER, ALIGNMENT, NPOT) LIBXS_MOD2(((uintptr_t)(POINTER) / (ALIGNMENT)), NPOT)

#if defined(_MSC_VER) && !defined(__clang__) && !defined(LIBXS_INTEL_COMPILER) /* account for incorrect handling of __VA_ARGS__ */
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
#define LIBXS_SELECT_HEAD(A, ...) A
#define LIBXS_SELECT_TAIL(A, ...) __VA_ARGS__

/**
 * For VLAs, check EXACTLY for C99 since a C11-conforming compiler may not provide VLAs.
 * However, some compilers (Intel) may signal support for VLA even with strict ANSI (C89).
 * To ultimately disable VLA-support, define LIBXS_NO_VLA (make VLA=0).
 * VLA-support is signaled by LIBXS_VLA.
 */
#if !defined(LIBXS_VLA) && !defined(LIBXS_NO_VLA) && !defined(__PGI) && ((defined(__STDC_VERSION__) && (199901L/*C99*/ == __STDC_VERSION__ || \
   (!defined(__STDC_NO_VLA__) && 199901L/*C99*/ < __STDC_VERSION__))) || (defined(LIBXS_INTEL_COMPILER) && !defined(_WIN32) && !defined(__cplusplus)) || \
    (defined(__INTEL_COMPILER) && !defined(_WIN32)) || (defined(__GNUC__) && !defined(__STRICT_ANSI__) && !defined(__cplusplus))/*depends on prior C99-check*/)
# define LIBXS_VLA
#endif

/**
 * LIBXS_INDEX1 calculates the linear address for a given set of (multiple) indexes/bounds.
 * Syntax: LIBXS_INDEX1(<ndims>, <i0>, ..., <i(ndims-1)>, <s1>, ..., <s(ndims-1)>).
 * Please note that the leading dimension (s0) is omitted in the above syntax!
 * TODO: support leading dimension (pitch/stride).
 */
#if defined(_MSC_VER) && !defined(__clang__) /* account for incorrect handling of __VA_ARGS__ */
# define LIBXS_INDEX1(NDIMS, ...) LIBXS_CONCATENATE(LIBXS_INDEX1_, NDIMS)LIBXS_EXPAND((__VA_ARGS__))
#else
# define LIBXS_INDEX1(NDIMS, ...) LIBXS_CONCATENATE(LIBXS_INDEX1_, NDIMS)(__VA_ARGS__)
#endif
#define LIBXS_INDEX1_1(I0) ((size_t)I0)
#define LIBXS_INDEX1_2(I0, I1, S1) (LIBXS_INDEX1_1(I0) * ((size_t)S1) + (size_t)I1)
#define LIBXS_INDEX1_3(I0, I1, I2, S1, S2) (LIBXS_INDEX1_2(I0, I1, S1) * ((size_t)S2) + (size_t)I2)
#define LIBXS_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) (LIBXS_INDEX1_3(I0, I1, I2, S1, S2) * ((size_t)S3) + (size_t)I3)
#define LIBXS_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) (LIBXS_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) * ((size_t)S4) + (size_t)I4)
#define LIBXS_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) (LIBXS_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) * ((size_t)S5) + (size_t)I5)
#define LIBXS_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) (LIBXS_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) * ((size_t)S6) + (size_t)I6)
#define LIBXS_INDEX1_8(I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, S7) (LIBXS_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) * ((size_t)S7) + (size_t)I7)

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
#if !defined(LIBXS_VLA_POSTFIX)
# define LIBXS_VLA_POSTFIX _
#endif
#if defined(LIBXS_VLA)
# define LIBXS_VLA_ACCESS(NDIMS, ARRAY, ...) LIBXS_VLA_ACCESS_Z(NDIMS, LIBXS_CONCATENATE(ARRAY, LIBXS_VLA_POSTFIX), LIBXS_VLA_ACCESS_X, __VA_ARGS__)
# define LIBXS_VLA_ACCESS_X(S) + 0 * (S)
# define LIBXS_VLA_ACCESS_Y(...)
# define LIBXS_VLA_ACCESS_Z(NDIMS, ARRAY, XY, ...) LIBXS_CONCATENATE(LIBXS_VLA_ACCESS_, NDIMS)(ARRAY, XY, __VA_ARGS__)
# define LIBXS_VLA_ACCESS_0(ARRAY, XY, ...) (ARRAY)/*scalar*/
# define LIBXS_VLA_ACCESS_1(ARRAY, XY, ...) ((ARRAY)[LIBXS_SELECT_HEAD(__VA_ARGS__, 0/*dummy*/)])
# define LIBXS_VLA_ACCESS_2(ARRAY, XY, I0, I1, ...) (((ARRAY) XY(__VA_ARGS__))[I0][I1])
# define LIBXS_VLA_ACCESS_3(ARRAY, XY, I0, I1, I2, S1, ...) (((ARRAY) XY(S1) XY(__VA_ARGS__))[I0][I1][I2])
# define LIBXS_VLA_ACCESS_4(ARRAY, XY, I0, I1, I2, I3, S1, S2, ...) (((ARRAY) XY(S1) XY(S2) XY(__VA_ARGS__))[I0][I1][I2][I3])
# define LIBXS_VLA_ACCESS_5(ARRAY, XY, I0, I1, I2, I3, I4, S1, S2, S3, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(__VA_ARGS__))[I0][I1][I2][I3][I4])
# define LIBXS_VLA_ACCESS_6(ARRAY, XY, I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5])
# define LIBXS_VLA_ACCESS_7(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6])
# define LIBXS_VLA_ACCESS_8(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(S6) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6][I7])
# define LIBXS_VLA_DECL(NDIMS, ELEMENT_TYPE, ARRAY_VAR, INIT_VALUE, .../*bounds*/) \
    ELEMENT_TYPE LIBXS_VLA_ACCESS_Z(LIBXS_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7), *LIBXS_RESTRICT LIBXS_CONCATENATE(ARRAY_VAR, LIBXS_VLA_POSTFIX), LIBXS_VLA_ACCESS_Y, __VA_ARGS__/*bounds*/, __VA_ARGS__/*dummy*/) = \
   (ELEMENT_TYPE LIBXS_VLA_ACCESS_Z(LIBXS_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7), *, LIBXS_VLA_ACCESS_Y, __VA_ARGS__/*bounds*/, __VA_ARGS__/*dummy*/))(INIT_VALUE)
#else /* calculate linear index */
# define LIBXS_VLA_ACCESS(NDIMS, ARRAY, ...) (LIBXS_CONCATENATE(ARRAY, LIBXS_VLA_POSTFIX)[LIBXS_INDEX1(NDIMS, __VA_ARGS__)])
# define LIBXS_VLA_DECL(NDIMS, ELEMENT_TYPE, ARRAY_VAR, INIT_VALUE, .../*bounds*/) \
    ELEMENT_TYPE *LIBXS_RESTRICT LIBXS_CONCATENATE(ARRAY_VAR, LIBXS_VLA_POSTFIX) = /*(ELEMENT_TYPE*)*/(INIT_VALUE)
#endif

/** Access an array of TYPE with Byte-measured stride. */
#define LIBXS_ACCESS(TYPE, ARRAY, STRIDE) (*(TYPE*)((char*)(ARRAY) + (STRIDE)))

#if !defined(LIBXS_UNUSED)
# if 0
#   define LIBXS_UNUSED(VARIABLE) LIBXS_PRAGMA(unused(VARIABLE))
# else
#   define LIBXS_UNUSED(VARIABLE) (void)(VARIABLE)
# endif
#endif

#if defined(_OPENMP)
# define LIBXS_PRAGMA_OMP(...) LIBXS_PRAGMA(omp __VA_ARGS__)
#else
# define LIBXS_PRAGMA_OMP(...)
#endif
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(LIBXS_INTEL_COMPILER)
# define LIBXS_OMP_VAR(A) LIBXS_UNUSED(A) /* suppress warning about "unused" variable */
#else
# define LIBXS_OMP_VAR(A)
#endif

#if (defined(__GNUC__) || defined(__clang__)) && !defined(__CYGWIN__) && !defined(__MINGW32__)
# define LIBXS_ATTRIBUTE_WEAK_IMPORT LIBXS_ATTRIBUTE(weak_import)
# define LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE(weak)
#else
# define LIBXS_ATTRIBUTE_WEAK
# define LIBXS_ATTRIBUTE_WEAK_IMPORT
#endif

#if !defined(LIBXS_NO_CTOR) && defined(__GNUC__) && !defined(LIBXS_CTOR)
# define LIBXS_ATTRIBUTE_CTOR LIBXS_ATTRIBUTE(constructor)
# define LIBXS_ATTRIBUTE_DTOR LIBXS_ATTRIBUTE(destructor)
# if defined(LIBXS_BUILD) && !defined(__STATIC)
#   define LIBXS_CTOR
# endif
#else
# define LIBXS_ATTRIBUTE_CTOR
# define LIBXS_ATTRIBUTE_DTOR
#endif

#if defined(__GNUC__) || (defined(LIBXS_INTEL_COMPILER) && !defined(_WIN32))
# define LIBXS_ATTRIBUTE_UNUSED LIBXS_ATTRIBUTE(unused)
#else
# define LIBXS_ATTRIBUTE_UNUSED
#endif
#if defined(__GNUC__)
# define LIBXS_MAY_ALIAS LIBXS_ATTRIBUTE(__may_alias__)
#else
# define LIBXS_MAY_ALIAS
#endif

#if !defined(LIBXS_MKTEMP_PATTERN)
# define LIBXS_MKTEMP_PATTERN "XXXXXX"
#endif
#if defined(_WIN32) && 0
# define LIBXS_SNPRINTF(S, N, ...) _snprintf_s(S, N, _TRUNCATE, __VA_ARGS__)
# define setenv(NAME, VALUE, OVERWRITE) _putenv(NAME "=" VALUE)
#elif defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__ || defined(__GNUC__))
# define LIBXS_SNPRINTF(S, N, ...) snprintf(S, N, __VA_ARGS__)
#else
# define LIBXS_SNPRINTF(S, N, ...) sprintf(S, __VA_ARGS__); LIBXS_UNUSED(N)
#endif
#if (0 == LIBXS_SYNC)
# define LIBXS_FLOCK(FILE)
# define LIBXS_FUNLOCK(FILE)
#elif defined(_WIN32)
# define LIBXS_FLOCK(FILE) _lock_file(FILE)
# define LIBXS_FUNLOCK(FILE) _unlock_file(FILE)
#elif !defined(__CYGWIN__)
# define LIBXS_FLOCK(FILE) flockfile(FILE)
# define LIBXS_FUNLOCK(FILE) funlockfile(FILE)
#else /* Only available with __CYGWIN__ *and* C++0x. */
# define LIBXS_FLOCK(FILE)
# define LIBXS_FUNLOCK(FILE)
#endif

/** Synchronize console output */
#define LIBXS_STDIO_ACQUIRE() LIBXS_FLOCK(stdout); LIBXS_FLOCK(stderr)
#define LIBXS_STDIO_RELEASE() LIBXS_FUNLOCK(stderr); LIBXS_FUNLOCK(stdout)

/** Determines whether constant-folding is available or not. */
#if !defined(LIBXS_STRING_POOLING)
# if defined(__GNUC__) /*&& !defined(_MSC_VER)*/
#   define LIBXS_STRING_POOLING
# endif
#endif

/** Below group is to fix-up some platform/compiler specifics. */
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
#   if defined(__cplusplus)
#     define _CMATH_
#   endif
# endif
#endif
#if defined(__GNUC__) && !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif
#if !defined(__STDC_FORMAT_MACROS)
# define __STDC_FORMAT_MACROS
#endif
#if defined(__clang__) && !defined(__extern_always_inline)
# define __extern_always_inline LIBXS_INLINE
#endif
#if defined(LIBXS_INLINE_FIXUP) && !defined(inline)
# define inline LIBXS_INLINE_KEYWORD
#endif

#if (0 != LIBXS_SYNC) && !defined(_REENTRANT)
# define _REENTRANT
#endif

/* _Float128 was introduced with GNU GCC 7.0. */
#if !defined(_Float128) && defined(__GNUC__) && !defined(__cplusplus) && defined(__linux__) \
  && (LIBXS_VERSION3(7, 0, 0) > LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__) \
  || (defined(LIBXS_INTEL_COMPILER) && defined(LIBXS_INTEL_COMPILER_UPDATE) && ( \
        ((1800 <= ((LIBXS_INTEL_COMPILER) + (LIBXS_INTEL_COMPILER_UPDATE))) \
      && (1801  > ((LIBXS_INTEL_COMPILER) + (LIBXS_INTEL_COMPILER_UPDATE)))) || \
        ((1706  > ((LIBXS_INTEL_COMPILER) + (LIBXS_INTEL_COMPILER_UPDATE))) \
      &&    (0 != ((LIBXS_INTEL_COMPILER) + (LIBXS_INTEL_COMPILER_UPDATE)))))))
# define _Float128 __float128
#endif
#if !defined(LIBXS_GLIBC_FPTYPES) && defined(__GNUC__) && !defined(__cplusplus) && defined(__linux__) \
  && (LIBXS_VERSION3(7, 0, 0) > LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__) \
  || defined(LIBXS_INTEL_COMPILER)) /* TODO */
# define LIBXS_GLIBC_FPTYPES
#endif
#if !defined(_Float128X) && defined(LIBXS_GLIBC_FPTYPES)
# define _Float128X _Float128
#endif
#if !defined(_Float32) && defined(LIBXS_GLIBC_FPTYPES)
# define _Float32 float
#endif
#if !defined(_Float32x) && defined(LIBXS_GLIBC_FPTYPES)
# define _Float32x _Float32
#endif
#if !defined(_Float64) && defined(LIBXS_GLIBC_FPTYPES)
# define _Float64 double
#endif
#if !defined(_Float64x) && defined(LIBXS_GLIBC_FPTYPES)
# define _Float64x _Float64
#endif
#if /* !LIBXS_INTEL_COMPILER */defined(__INTEL_COMPILER) && !defined(__clang__) /* TODO */
# define __has_feature(A) 0
# define __has_builtin(A) 0
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if !defined(LIBXS_ASSERT)
# include <assert.h>
# if defined(NDEBUG)
#   define LIBXS_ASSERT(EXPR) LIBXS_ASSUME(EXPR)
# else
#   define LIBXS_ASSERT(EXPR) assert(EXPR)
# endif
#endif
#if !defined(LIBXS_ASSERT_MSG)
# define LIBXS_ASSERT_MSG(EXPR, MSG) assert((EXPR) && (0 != *(MSG)))
#endif
#if !defined(LIBXS_EXPECT)
# if defined(NDEBUG)
#   define LIBXS_EXPECT(RESULT, EXPR) (EXPR)
# else
#   define LIBXS_EXPECT(RESULT, EXPR) LIBXS_ASSERT((RESULT) == (EXPR))
# endif
#endif
#if !defined(LIBXS_EXPECT_NOT)
# if defined(NDEBUG)
#   define LIBXS_EXPECT_NOT(RESULT, EXPR) (EXPR)
# else
#   define LIBXS_EXPECT_NOT(RESULT, EXPR) LIBXS_ASSERT((RESULT) != (EXPR))
# endif
#endif
#if defined(LIBXS_GLIBC_FPTYPES)
# if defined(__cplusplus)
#   undef __USE_MISC
#   include <math.h>
#   if !defined(_DEFAULT_SOURCE)
#     define _DEFAULT_SOURCE
#   endif
#   if !defined(_BSD_SOURCE)
#     define _BSD_SOURCE
#   endif
# elif !defined(__PURE_INTEL_C99_HEADERS__)
#   define __PURE_INTEL_C99_HEADERS__
# endif
#endif
#include <stddef.h>
#include <stdint.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#endif /*LIBXS_MACROS_H*/
