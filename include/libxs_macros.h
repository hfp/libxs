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

#if defined(__cplusplus)
# define LIBXS_EXTERN_C extern "C"
# define LIBXS_INLINE inline
# define LIBXS_VARIADIC ...
#else
# define LIBXS_EXTERN_C
# define LIBXS_VARIADIC
# if defined(__STDC_VERSION__) && (199901L <= (__STDC_VERSION__))
#   define LIBXS_PRAGMA(DIRECTIVE) _Pragma(LIBXS_STRINGIFY(DIRECTIVE))
#   define LIBXS_RESTRICT restrict
#   define LIBXS_INLINE static inline
# elif defined(_MSC_VER)
#   define LIBXS_INLINE static __inline
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
#if defined(_MSC_VER)
# define LIBXS_MESSAGE(MSG) LIBXS_PRAGMA(message(MSG))
#elif (40400 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__))
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
# define LIBXS_PRAGMA_FORCEINLINE LIBXS_PRAGMA(forceinline recursive)
# define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG) LIBXS_PRAGMA(loop_count min(MIN) max(MAX) avg(AVG))
# define LIBXS_PRAGMA_UNROLL_N(N) LIBXS_PRAGMA(unroll(N))
# define LIBXS_PRAGMA_UNROLL LIBXS_PRAGMA(unroll)
/*# define LIBXS_UNUSED(VARIABLE) LIBXS_PRAGMA(unused(VARIABLE))*/
#else
# define LIBXS_PRAGMA_FORCEINLINE
# define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG)
# define LIBXS_PRAGMA_UNROLL_N(N)
# define LIBXS_PRAGMA_UNROLL
#endif

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

/*Based on Stackoverflow's NBITSx macro.*/
#define LIBXS_NBITS02(N) (0 != ((N) & 2/*0b10*/) ? 1 : 0)
#define LIBXS_NBITS04(N) (0 != ((N) & 0xC/*0b1100*/) ? (2 + LIBXS_NBITS02((N) >> 2)) : LIBXS_NBITS02(N))
#define LIBXS_NBITS08(N) (0 != ((N) & 0xF0/*0b11110000*/) ? (4 + LIBXS_NBITS04((N) >> 4)) : LIBXS_NBITS04(N))
#define LIBXS_NBITS16(N) (0 != ((N) & 0xFF00) ? (8 + LIBXS_NBITS08((N) >> 8)) : LIBXS_NBITS08(N))
#define LIBXS_NBITS32(N) (0 != ((N) & 0xFFFF0000) ? (16 + LIBXS_NBITS16((N) >> 16)) : LIBXS_NBITS16(N))
#define LIBXS_NBITS64(N) (0 != ((N) & 0xFFFFFFFF00000000) ? (32 + LIBXS_NBITS32((N) >> 32)) : LIBXS_NBITS32(N))
#define LIBXS_NBITS(N) (0 != (N) ? (LIBXS_NBITS64((unsigned long long)(N)) + 1) : 1)

#define LIBXS_DEFAULT(DEFAULT, VALUE) (0 < (VALUE) ? (VALUE) : (DEFAULT))
#define LIBXS_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXS_MAX(A, B) ((A) < (B) ? (B) : (A))
#define LIBXS_MOD2(N, NPOT) ((N) & ((NPOT) - 1))
#define LIBXS_MUL2(N, NPOT) ((N) << (LIBXS_NBITS(NPOT) - 1))
#define LIBXS_DIV2(N, NPOT) ((N) >> (LIBXS_NBITS(NPOT) - 1))
#define LIBXS_UP2(N, NPOT) LIBXS_MUL2(LIBXS_DIV2((N) + (NPOT) - 1, NPOT), NPOT)
#define LIBXS_UP(N, UP) ((((N) + (UP) - 1) / (UP)) * (UP))

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
# define LIBXS_ASSUME_ALIGNED(A, N) __assume_aligned(A, N)
# define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION)
#else
# define LIBXS_ASSUME_ALIGNED(A, N)
# if defined(_MSC_VER)
#   define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION)
# elif (40500 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__))
#   define LIBXS_ASSUME(EXPRESSION) do { if (!(EXPRESSION)) __builtin_unreachable(); } while(0)
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

#if defined(LIBXS_NOSYNC)
# undef _REENTRANT
#elif !defined(_REENTRANT)
# define _REENTRANT
#endif

#if defined(_REENTRANT)
# if defined(_WIN32) && !defined(__GNUC__)
#   define LIBXS_TLS LIBXS_ATTRIBUTE(thread)
# elif defined(__GNUC__) || defined(__clang__)
#   define LIBXS_TLS __thread
# elif defined(__cplusplus)
#   define LIBXS_TLS thread_local
# else
#   error Missing TLS support!
# endif
#else
# define LIBXS_TLS
#endif

#if defined(__GNUC__)
# define LIBXS_VISIBILITY_HIDDEN LIBXS_ATTRIBUTE(visibility("hidden"))
# define LIBXS_VISIBILITY_INTERNAL LIBXS_ATTRIBUTE(visibility("internal"))
#else
# define LIBXS_VISIBILITY_HIDDEN
# define LIBXS_VISIBILITY_INTERNAL
#endif

/** Execute the CPUID, and receive results (EAX, EBX, ECX, EDX) for requested FUNCTION. */
#if defined(__GNUC__)
# define LIBXS_CPUID(FUNCTION, EAX, EBX, ECX, EDX) \
    __asm__ __volatile__ ("cpuid" : "=a"(EAX), "=b"(EBX), "=c"(ECX), "=d"(EDX) : "a"(FUNCTION))
#else
# define LIBXS_CPUID(FUNCTION, EAX, EBX, ECX, EDX) { \
    int libxs_cpuid_[4]; \
    __cpuid(libxs_cpuid_, FUNCTION); \
    EAX = libxs_cpuid_[0]; \
    EBX = libxs_cpuid_[1]; \
    ECX = libxs_cpuid_[2]; \
    EDX = libxs_cpuid_[3]; \
  }
#endif

/** Execute the XGETBV, and receive results (EAX, EDX) for req. eXtended Control Register (XCR). */
#if defined(__GNUC__)
# define LIBXS_XGETBV(XCR, EAX, EDX) __asm__ __volatile__( \
    ".byte 0x0f, 0x01, 0xd0" /*xgetbv*/ : "=a"(EAX), "=d"(EDX) : "c"(XCR) \
  )
#else
# define LIBXS_XGETBV(XCR, EAX, EDX) { \
    unsigned long long libxs_xgetbv_ = _xgetbv(XCR); \
    EAX = (int)libxs_xgetbv_; \
    EDX = (int)(libxs_xgetbv_ >> 32); \
  }
#endif

/**
 * Below group of preprocessor symbols are used to fixup some platform specifics.
 */
#if !defined(_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES)
# define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif
#if !defined(_CRT_SECURE_NO_DEPRECATE)
# define _CRT_SECURE_NO_DEPRECATE 1
#endif
#if !defined(_USE_MATH_DEFINES)
# define _USE_MATH_DEFINES 1
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
# define WIN32_LEAN_AND_MEAN 1
#endif
#if !defined(NOMINMAX)
# define NOMINMAX 1
#endif
#if defined(_WIN32)
# define LIBXS_SNPRINTF(S, N, ...) _snprintf_s(S, N, _TRUNCATE, __VA_ARGS__)
# define LIBXS_FLOCK(FILE) _lock_file(FILE)
# define LIBXS_FUNLOCK(FILE) _unlock_file(FILE)
#else
# if defined(__STDC_VERSION__) && (199901L <= (__STDC_VERSION__))
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

#if defined(_WIN32) /*TODO*/
# define LIBXS_LOCK_TYPE HANDLE
# define LIBXS_LOCK_CONSTRUCT 0
# define LIBXS_LOCK_DESTROY(LOCK) CloseHandle(LOCK)
# define LIBXS_LOCK_ACQUIRE(LOCK) WaitForSingleObject(LOCK, INFINITE)
# define LIBXS_LOCK_RELEASE(LOCK) ReleaseMutex(LOCK)
#else /* PThreads: include <pthread.h> */
# define LIBXS_LOCK_TYPE pthread_mutex_t
# define LIBXS_LOCK_CONSTRUCT PTHREAD_MUTEX_INITIALIZER
# define LIBXS_LOCK_DESTROY(LOCK) pthread_mutex_destroy(&(LOCK))
# define LIBXS_LOCK_ACQUIRE(LOCK) pthread_mutex_lock(&(LOCK))
# define LIBXS_LOCK_RELEASE(LOCK) pthread_mutex_unlock(&(LOCK))
#endif

#endif /*LIBXS_MACROS_H*/
