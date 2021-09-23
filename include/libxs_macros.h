/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MACROS_H
#define LIBXS_MACROS_H

#include "libxs_config.h"

/** Parameters the library was built for. */
#define LIBXS_CACHELINE LIBXS_CONFIG_CACHELINE
#define LIBXS_ALIGNMENT LIBXS_CONFIG_ALIGNMENT
#define LIBXS_MALLOC LIBXS_CONFIG_MALLOC
#define LIBXS_ILP64 LIBXS_CONFIG_ILP64
#define LIBXS_SYNC LIBXS_CONFIG_SYNC
#define LIBXS_JIT LIBXS_CONFIG_JIT

/** Parameters of GEMM domain (static kernels, etc). */
#define LIBXS_PREFETCH LIBXS_CONFIG_PREFETCH
#define LIBXS_MAX_MNK LIBXS_CONFIG_MAX_MNK
#define LIBXS_MAX_DIM LIBXS_CONFIG_MAX_DIM
#define LIBXS_MAX_M LIBXS_CONFIG_MAX_M
#define LIBXS_MAX_N LIBXS_CONFIG_MAX_N
#define LIBXS_MAX_K LIBXS_CONFIG_MAX_K
#define LIBXS_FLAGS LIBXS_CONFIG_FLAGS
#define LIBXS_ALPHA LIBXS_CONFIG_ALPHA
#define LIBXS_BETA LIBXS_CONFIG_BETA

/**
 * Use "make PLATFORM=1" to disable platform checks.
 * The platform check is to bail-out with an error
 * message for an attempt to build an upstream package
 * and subsequently to list LIBXS as "broken" on
 * that platform.
 * Note: successful compilation on an unsupported
 * platform is desired, but only fallback code is
 * present at best.
 */
#if !defined(LIBXS_PLATFORM_FORCE) && 0
# define LIBXS_PLATFORM_FORCE
#endif

#if !defined(LIBXS_PLATFORM_X86) && ( \
    (defined(__x86_64__) && 0 != (__x86_64__)) || \
    (defined(__amd64__) && 0 != (__amd64__)) || \
    (defined(_M_X64) || defined(_M_AMD64)) || \
    (defined(__i386__) && 0 != (__i386__)) || \
    (defined(_M_IX86)))
# define LIBXS_PLATFORM_X86
#endif
#if !defined(LIBXS_PLATFORM_AARCH64) && \
    (defined(__aarch64__) || defined(__arm64__))
# define LIBXS_PLATFORM_AARCH64
#endif
#if !defined(LIBXS_PLATFORM_SUPPORTED)
# if defined(LIBXS_PLATFORM_X86) || defined(LIBXS_PLATFORM_AARCH64)
#   define LIBXS_PLATFORM_SUPPORTED
# elif !defined(LIBXS_PLATFORM_FORCE)
#   error LIBXS requires X86_64, AArch64, or compatible CPUs!
# endif
#endif
#if !defined(LIBXS_BITS)
# if  (defined(__SIZEOF_PTRDIFF_T__) && 4 < (__SIZEOF_PTRDIFF_T__)) || \
      (defined(__SIZE_MAX__) && (4294967295U < (__SIZE_MAX__))) || \
      (defined(__x86_64__) && 0 != (__x86_64__)) || \
      (defined(__amd64__) && 0 != (__amd64__)) || \
      (defined(_M_X64) || defined(_M_AMD64)) || \
      (defined(_WIN64)) || \
      (defined(__powerpc64)) || \
      (defined(__aarch64__))
#   define LIBXS_UNLIMITED 0xFFFFFFFFFFFFFFFF
#   define LIBXS_BITS 64
# elif !defined(LIBXS_PLATFORM_FORCE) && defined(NDEBUG)
#   error LIBXS is only supported on 64-bit platforms!
# else /* JIT-generated code (among other issues) is not supported! */
#   define LIBXS_UNLIMITED 0xFFFFFFFF
#   define LIBXS_BITS 32
# endif
#endif

#define LIBXS_STRINGIFY2(SYMBOL) #SYMBOL
#define LIBXS_STRINGIFY(SYMBOL) LIBXS_STRINGIFY2(SYMBOL)
#define LIBXS_TOSTRING(SYMBOL) LIBXS_STRINGIFY(SYMBOL)
#define LIBXS_CONCATENATE2(A, B) A##B
#define LIBXS_CONCATENATE3(A, B, C) LIBXS_CONCATENATE(LIBXS_CONCATENATE(A, B), C)
#define LIBXS_CONCATENATE4(A, B, C, D) LIBXS_CONCATENATE(LIBXS_CONCATENATE3(A, B, C), D)
#define LIBXS_CONCATENATE(A, B) LIBXS_CONCATENATE2(A, B)
#define LIBXS_FSYMBOL(SYMBOL) LIBXS_CONCATENATE(SYMBOL, _)
#define LIBXS_UNIQUE(NAME) LIBXS_CONCATENATE(NAME, __LINE__)
#define LIBXS_EXPAND(...) __VA_ARGS__
#define LIBXS_ELIDE(...)

/**
 * Check given value against type-range (assertion).
 * Note: allows "-1" for unsigned types.
 */
#if !defined(NDEBUG)
# define LIBXS_CHECK_ULLONG(VALUE) assert(-1 <= (VALUE) && (VALUE) <= ULLONG_MAX)
# define LIBXS_CHECK_LLONG(VALUE) assert(ULLONG_MIN <= (VALUE) && (VALUE) <= LLONG_MAX)
# define LIBXS_CHECK_ULONG(VALUE) assert(-1 <= (VALUE) && (VALUE) <= ULONG_MAX)
# define LIBXS_CHECK_LONG(VALUE) assert(LONG_MIN <= (VALUE) && (VALUE) <= LONG_MAX)
# define LIBXS_CHECK_USHORT(VALUE) assert(-1 <= (VALUE) && (VALUE) <= USHRT_MAX)
# define LIBXS_CHECK_SHORT(VALUE) assert(SHRT_MIN <= (VALUE) && (VALUE) <= SHRT_MAX)
# define LIBXS_CHECK_UCHAR(VALUE) assert(-1 <= (VALUE) && (VALUE) <= UCHAR_MAX)
# define LIBXS_CHECK_ICHAR(VALUE) assert(SCHAR_MIN <= (VALUE) && (VALUE) <= SCHAR_MAX)
# define LIBXS_CHECK_UINT(VALUE) assert(-1 <= (VALUE) && (VALUE) <= UINT_MAX)
# define LIBXS_CHECK_INT(VALUE) assert(INT_MIN <= (VALUE) && (VALUE) <= INT_MAX)
#else
# define LIBXS_CHECK_ULLONG(VALUE) 0/*dummy*/
# define LIBXS_CHECK_LLONG(VALUE) 0/*dummy*/
# define LIBXS_CHECK_ULONG(VALUE) 0/*dummy*/
# define LIBXS_CHECK_LONG(VALUE) 0/*dummy*/
# define LIBXS_CHECK_USHORT(VALUE) 0/*dummy*/
# define LIBXS_CHECK_SHORT(VALUE) 0/*dummy*/
# define LIBXS_CHECK_UCHAR(VALUE) 0/*dummy*/
# define LIBXS_CHECK_ICHAR(VALUE) 0/*dummy*/
# define LIBXS_CHECK_UINT(VALUE) 0/*dummy*/
# define LIBXS_CHECK_INT(VALUE) 0/*dummy*/
#endif

/**
 * Perform verbose type-cast with following two advantages:
 * (1) Make it easy to locate/find the type-cast.
 * (2) Range-check to ensure fitting into type.
 */
#define LIBXS_CAST_ULLONG(VALUE) (LIBXS_CHECK_ULLONG(VALUE), (unsigned long long)(VALUE))
#define LIBXS_CAST_LLONG(VALUE) (LIBXS_CHECK_LLONG(VALUE), (/*signed*/long long)(VALUE))
#define LIBXS_CAST_ULONG(VALUE) (LIBXS_CHECK_ULONG(VALUE), (unsigned long)(VALUE))
#define LIBXS_CAST_LONG(VALUE) (LIBXS_CHECK_LONG(VALUE), (/*signed*/long)(VALUE))
#define LIBXS_CAST_USHORT(VALUE) (LIBXS_CHECK_USHORT(VALUE), (unsigned short)(VALUE))
#define LIBXS_CAST_SHORT(VALUE) (LIBXS_CHECK_SHORT(VALUE), (/*signed*/short)(VALUE))
#define LIBXS_CAST_UCHAR(VALUE) (LIBXS_CHECK_UCHAR(VALUE), (unsigned char)(VALUE))
#define LIBXS_CAST_ICHAR(VALUE) (LIBXS_CHECK_ICHAR(VALUE), (signed char)(VALUE))
#define LIBXS_CAST_UINT(VALUE) (LIBXS_CHECK_UINT(VALUE), (unsigned int)(VALUE))
#define LIBXS_CAST_INT(VALUE) (LIBXS_CHECK_INT(VALUE), (/*signed*/int)(VALUE))

/** Use LIBXS_VERSION2 instead of LIBXS_VERSION3, e.g., if __GNUC_PATCHLEVEL__ or __clang_patchlevel__ is zero (0). */
#define LIBXS_VERSION2(MAJOR, MINOR) ((MAJOR) * 10000 + (MINOR) * 100)
#define LIBXS_VERSION3(MAJOR, MINOR, UPDATE) (LIBXS_VERSION2(MAJOR, MINOR) + (UPDATE))
#define LIBXS_VERSION4(MAJOR, MINOR, UPDATE, PATCH) \
  (((0x7F & (MAJOR)) << 24) | ((0x1F & (MINOR)) << 19) | ((0x1F & (UPDATE)) << 14) | (0x3FFF & (PATCH)))
#define LIBXS_VERSION41(VERSION) (((VERSION) >> 24))
#define LIBXS_VERSION42(VERSION) (((VERSION) >> 19) & 0x1F)
#define LIBXS_VERSION43(VERSION) (((VERSION) >> 14) & 0x1F)
#define LIBXS_VERSION44(VERSION) (((VERSION)) & 0x3FFF)

#if !defined(LIBXS_UNPACKED) && (defined(_CRAYC) || defined(LIBXS_OFFLOAD_BUILD) || \
  (0 == LIBXS_SYNC)/*Windows: missing pack(pop) error*/)
# define LIBXS_UNPACKED
#endif
#if defined(_WIN32) && !defined(__GNUC__) && !defined(__clang__)
# define LIBXS_ATTRIBUTE(A) __declspec(A)
# if defined(__cplusplus)
#   define LIBXS_INLINE_ALWAYS __forceinline
# else
#   define LIBXS_INLINE_ALWAYS static __forceinline
# endif
# define LIBXS_ALIGNED(DECL, N) LIBXS_ATTRIBUTE(align(N)) DECL
# if !defined(LIBXS_UNPACKED)
#   define LIBXS_PACKED(TYPE) LIBXS_PRAGMA(pack(1)) TYPE
# endif
# define LIBXS_CDECL __cdecl
#elif (defined(__GNUC__) || defined(__clang__) || defined(__PGI))
# define LIBXS_ATTRIBUTE(A) __attribute__((A))
# define LIBXS_INLINE_ALWAYS LIBXS_ATTRIBUTE(always_inline) LIBXS_INLINE
# define LIBXS_ALIGNED(DECL, N) LIBXS_ATTRIBUTE(aligned(N)) DECL
# if !defined(LIBXS_UNPACKED)
#   define LIBXS_PACKED(TYPE) TYPE LIBXS_ATTRIBUTE(__packed__)
# endif
# define LIBXS_CDECL LIBXS_ATTRIBUTE(cdecl)
#else
# define LIBXS_ATTRIBUTE(A)
# define LIBXS_INLINE_ALWAYS LIBXS_INLINE
# define LIBXS_ALIGNED(DECL, N) DECL
# define LIBXS_CDECL
#endif
#if !defined(LIBXS_PACKED)
# define LIBXS_PACKED(TYPE) TYPE
# if !defined(LIBXS_UNPACKED)
#   define LIBXS_UNPACKED
# endif
#endif
#if !defined(LIBXS_UNPACKED) && 0
/* no braces around EXPR */
# define LIBXS_PAD(EXPR) EXPR;
#endif
#if !defined(LIBXS_PAD)
# define LIBXS_PAD(EXPR)
#endif

#if defined(__INTEL_COMPILER)
# if !defined(__INTEL_COMPILER_UPDATE)
#   define LIBXS_INTEL_COMPILER __INTEL_COMPILER
# else
#   define LIBXS_INTEL_COMPILER (__INTEL_COMPILER + __INTEL_COMPILER_UPDATE)
# endif
#elif defined(__INTEL_COMPILER_BUILD_DATE)
# define LIBXS_INTEL_COMPILER ((__INTEL_COMPILER_BUILD_DATE / 10000 - 2000) * 100)
#endif

/* LIBXS_ATTRIBUTE_USED: mark library functions as used to avoid warning */
#if defined(__GNUC__) || defined(__clang__) || (defined(__INTEL_COMPILER) && !defined(_WIN32))
# if !defined(__cplusplus) || !defined(__clang__)
#   define LIBXS_ATTRIBUTE_COMMON LIBXS_ATTRIBUTE(common)
# else
#   define LIBXS_ATTRIBUTE_COMMON
# endif
# define LIBXS_ATTRIBUTE_MALLOC LIBXS_ATTRIBUTE(malloc)
# define LIBXS_ATTRIBUTE_UNUSED LIBXS_ATTRIBUTE(unused)
# define LIBXS_ATTRIBUTE_USED LIBXS_ATTRIBUTE(used)
#else
# if defined(_WIN32)
#   define LIBXS_ATTRIBUTE_COMMON LIBXS_ATTRIBUTE(selectany)
# else
#   define LIBXS_ATTRIBUTE_COMMON
# endif
# define LIBXS_ATTRIBUTE_MALLOC
# define LIBXS_ATTRIBUTE_UNUSED
# define LIBXS_ATTRIBUTE_USED
#endif
#if !defined(__INTEL_COMPILER) && (defined(__clang__) || defined(__PGLLVM__))
# define LIBXS_ATTRIBUTE_NO_SANITIZE(KIND) LIBXS_ATTRIBUTE(no_sanitize(LIBXS_STRINGIFY(KIND)))
#elif defined(__GNUC__) && LIBXS_VERSION2(4, 8) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__) \
  && !defined(__INTEL_COMPILER)
# define LIBXS_ATTRIBUTE_NO_SANITIZE(KIND) LIBXS_ATTRIBUTE(LIBXS_CONCATENATE(no_sanitize_, KIND))
#else
# define LIBXS_ATTRIBUTE_NO_SANITIZE(KIND)
#endif

#if defined(__cplusplus)
# define LIBXS_VARIADIC ...
# define LIBXS_EXTERN extern "C"
# define LIBXS_EXTERN_C extern "C"
# define LIBXS_INLINE_KEYWORD inline
# define LIBXS_INLINE LIBXS_INLINE_KEYWORD
# if defined(__GNUC__) || defined(_CRAYC)
#   define LIBXS_CALLER __PRETTY_FUNCTION__
# elif defined(_MSC_VER)
#   define LIBXS_CALLER __FUNCDNAME__
#   define LIBXS_FUNCNAME __FUNCTION__
# else
#   define LIBXS_CALLER __FUNCNAME__
# endif
#else /* C */
# define LIBXS_VARIADIC
# define LIBXS_EXTERN extern
# define LIBXS_EXTERN_C
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
#   define LIBXS_PRAGMA(DIRECTIVE) _Pragma(LIBXS_STRINGIFY(DIRECTIVE))
#   define LIBXS_CALLER __func__
#   define LIBXS_RESTRICT restrict
#   define LIBXS_INLINE_KEYWORD inline
# elif defined(_MSC_VER)
#   define LIBXS_CALLER __FUNCDNAME__
#   define LIBXS_FUNCNAME __FUNCTION__
#   define LIBXS_INLINE_KEYWORD __inline
#   define LIBXS_INLINE_FIXUP
# elif defined(__GNUC__) && !defined(__STRICT_ANSI__)
#   define LIBXS_CALLER __PRETTY_FUNCTION__
# endif
# if !defined(LIBXS_INLINE_KEYWORD)
#   define LIBXS_INLINE_KEYWORD
#   define LIBXS_INLINE_FIXUP
# endif
/* LIBXS_ATTRIBUTE_USED: increases compile-time of header-only by a large factor */
# define LIBXS_INLINE static LIBXS_INLINE_KEYWORD LIBXS_ATTRIBUTE_UNUSED
#endif /*__cplusplus*/
#if !defined(LIBXS_CALLER)
# define LIBXS_CALLER NULL
#endif
#if !defined(LIBXS_FUNCNAME)
# define LIBXS_FUNCNAME LIBXS_CALLER
#endif
#if !defined(LIBXS_CALLER_ID)
# if defined(__GNUC__) || 1
#   define LIBXS_CALLER_ID ((const void*)((uintptr_t)libxs_hash_string(LIBXS_CALLER)))
# else /* assume no string-pooling (perhaps unsafe) */
#   define LIBXS_CALLER_ID LIBXS_CALLER
# endif
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

#if !defined(__STATIC) && !defined(_WINDLL) && (defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__))
# define __STATIC
#endif

/* may include Clang and other compatible compilers */
#if defined(__GNUC__) && !defined(_WIN32) && !defined(__CYGWIN__) && !defined(__MINGW32__)
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
#if !defined(__STATIC) && (defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__))
# define LIBXS_VISIBILITY_EXPORT LIBXS_ATTRIBUTE(dllexport)
# define LIBXS_VISIBILITY_IMPORT LIBXS_ATTRIBUTE(dllimport)
#endif
#if !defined(LIBXS_VISIBILITY_EXPORT)
# define LIBXS_VISIBILITY_EXPORT LIBXS_VISIBILITY_PUBLIC
#endif
#if !defined(LIBXS_VISIBILITY_IMPORT)
# define LIBXS_VISIBILITY_IMPORT LIBXS_VISIBILITY_PUBLIC
#endif

#if defined(LIBXS_SOURCE_H) /* header-only mode */
# define LIBXS_API_VISIBILITY_EXPORT
# define LIBXS_API_VISIBILITY_IMPORT
# define LIBXS_API_VISIBILITY_INTERN
# define LIBXS_API_COMMON LIBXS_RETARGETABLE LIBXS_ATTRIBUTE_COMMON
# define LIBXS_API_TARGET LIBXS_API_INLINE
# define LIBXS_API_EXTERN LIBXS_EXTERN_C
#else /* classic ABI */
# if defined(LIBXS_BUILD_EXT)
#   define LIBXS_API_VISIBILITY_EXPORT LIBXS_VISIBILITY_IMPORT
#   define LIBXS_API_VISIBILITY_IMPORT LIBXS_VISIBILITY_EXPORT
#   define LIBXS_API_VISIBILITY_INTERN LIBXS_VISIBILITY_PRIVATE
# elif defined(LIBXS_BUILD)
#   define LIBXS_API_VISIBILITY_EXPORT LIBXS_VISIBILITY_EXPORT
#   define LIBXS_API_VISIBILITY_IMPORT LIBXS_VISIBILITY_IMPORT
#   define LIBXS_API_VISIBILITY_INTERN LIBXS_VISIBILITY_PRIVATE
# else /* import */
#   define LIBXS_API_VISIBILITY_EXPORT LIBXS_VISIBILITY_IMPORT
#   define LIBXS_API_VISIBILITY_IMPORT LIBXS_VISIBILITY_IMPORT
#   define LIBXS_API_VISIBILITY_INTERN
# endif
# define LIBXS_API_COMMON LIBXS_RETARGETABLE
# define LIBXS_API_TARGET LIBXS_RETARGETABLE
# define LIBXS_API_EXTERN LIBXS_EXTERN
#endif

#define LIBXS_API_VISIBILITY(VISIBILITY) LIBXS_CONCATENATE(LIBXS_API_VISIBILITY_, VISIBILITY)
#define LIBXS_APIVAR(DECL, VISIBILITY, EXTERN) EXTERN LIBXS_API_COMMON LIBXS_API_VISIBILITY(VISIBILITY) DECL
#define LIBXS_API_INLINE LIBXS_INLINE LIBXS_RETARGETABLE
#define LIBXS_API_DEF

#if (!defined(__INTEL_COMPILER) || !defined(_WIN32))
#define LIBXS_APIVAR_ALIGNED(DECL, VISIBILITY) LIBXS_ALIGNED(LIBXS_APIVAR(DECL, VISIBILITY, LIBXS_API_DEF), LIBXS_CONFIG_CACHELINE)
#else
#define LIBXS_APIVAR_ALIGNED(DECL, VISIBILITY) LIBXS_APIVAR(DECL, VISIBILITY, LIBXS_API_DEF)
#endif

/** Public variable declaration (without definition) located in header file. */
#define LIBXS_APIVAR_PUBLIC(DECL) LIBXS_APIVAR(DECL, EXPORT, LIBXS_API_EXTERN)
/** Public variable definition (complements declaration) located in source file. */
#define LIBXS_APIVAR_PUBLIC_DEF(DECL) LIBXS_APIVAR_ALIGNED(DECL, EXPORT)
/** Private variable declaration (without definition) located in header file. */
#define LIBXS_APIVAR_PRIVATE(DECL) LIBXS_APIVAR(DECL, INTERN, LIBXS_API_EXTERN)
/** Private variable definition (complements declaration) located in source file. */
#define LIBXS_APIVAR_PRIVATE_DEF(DECL) LIBXS_APIVAR_ALIGNED(DECL, INTERN)
/** Private variable (declaration and definition) located in source file. */
#define LIBXS_APIVAR_DEFINE(DECL) LIBXS_APIVAR_PRIVATE(DECL); LIBXS_APIVAR_PRIVATE_DEF(DECL)
/** Function decoration used for private functions. */
#define LIBXS_API_INTERN LIBXS_API_EXTERN LIBXS_API_TARGET LIBXS_API_VISIBILITY(INTERN)
/** Function decoration used for public functions of LIBXSext library. */
#define LIBXS_APIEXT LIBXS_API_EXTERN LIBXS_API_TARGET LIBXS_API_VISIBILITY(IMPORT)
/** Function decoration used for public functions of LIBXS library. */
#define LIBXS_API LIBXS_API_EXTERN LIBXS_API_TARGET LIBXS_API_VISIBILITY(EXPORT)

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
#   define LIBXS_PRAGMA(DIRECTIVE) __pragma(LIBXS_EXPAND(DIRECTIVE))
# else
#   define LIBXS_PRAGMA(DIRECTIVE)
# endif
#endif /*LIBXS_PRAGMA*/
#if !defined(LIBXS_OPENMP_SIMD)
# if defined(LIBXS_INTEL_COMPILER) && (1500 <= LIBXS_INTEL_COMPILER)
#   define LIBXS_OPENMP_SIMD
# elif defined(_OPENMP) && (201307/*v4.0*/ <= _OPENMP)
#   define LIBXS_OPENMP_SIMD
# endif
#endif

#if !defined(LIBXS_INTEL_COMPILER) || (LIBXS_INTEL_COMPILER < 9900)
# if defined(LIBXS_OPENMP_SIMD)
#   define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(omp simd reduction(EXPRESSION))
#   define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(omp simd collapse(N))
#   define LIBXS_PRAGMA_SIMD_PRIVATE(...) LIBXS_PRAGMA(omp simd private(__VA_ARGS__))
#   define LIBXS_PRAGMA_SIMD LIBXS_PRAGMA(omp simd)
# elif defined(__INTEL_COMPILER)
#   define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION) LIBXS_PRAGMA(simd reduction(EXPRESSION))
#   define LIBXS_PRAGMA_SIMD_COLLAPSE(N) LIBXS_PRAGMA(simd collapse(N))
#   define LIBXS_PRAGMA_SIMD_PRIVATE(...) LIBXS_PRAGMA(simd private(__VA_ARGS__))
#   define LIBXS_PRAGMA_SIMD LIBXS_PRAGMA(simd)
# endif
#endif
#if !defined(LIBXS_PRAGMA_SIMD)
# define LIBXS_PRAGMA_SIMD_REDUCTION(EXPRESSION)
# define LIBXS_PRAGMA_SIMD_COLLAPSE(N)
# define LIBXS_PRAGMA_SIMD_PRIVATE(...)
# define LIBXS_PRAGMA_SIMD
#endif

#if defined(__INTEL_COMPILER)
# define LIBXS_PRAGMA_NONTEMPORAL(...) LIBXS_PRAGMA(vector nontemporal(__VA_ARGS__))
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
# if defined(LIBXS_OPENMP_SIMD) && (201811/*v5.0*/ <= _OPENMP)
#   define LIBXS_PRAGMA_NONTEMPORAL(...) LIBXS_PRAGMA(omp simd nontemporal(__VA_ARGS__))
# else
#   define LIBXS_PRAGMA_NONTEMPORAL(...)
# endif
# if defined(__clang__)
#   define LIBXS_PRAGMA_VALIGNED_VAR(A)
#   define LIBXS_PRAGMA_VALIGNED
#   define LIBXS_PRAGMA_NOVECTOR LIBXS_PRAGMA(clang loop vectorize(disable))
#   define LIBXS_PRAGMA_FORCEINLINE
#   define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG) LIBXS_PRAGMA(unroll(AVG))
#   define LIBXS_PRAGMA_UNROLL_AND_JAM(N) LIBXS_PRAGMA(unroll(N))
#   define LIBXS_PRAGMA_UNROLL_N(N) LIBXS_PRAGMA(unroll(N))
#   define LIBXS_PRAGMA_UNROLL LIBXS_PRAGMA_UNROLL_N(4)
# else
#   define LIBXS_PRAGMA_VALIGNED_VAR(A)
#   define LIBXS_PRAGMA_VALIGNED
#   define LIBXS_PRAGMA_NOVECTOR
#   define LIBXS_PRAGMA_FORCEINLINE
#   define LIBXS_PRAGMA_LOOP_COUNT(MIN, MAX, AVG)
#   define LIBXS_PRAGMA_UNROLL_AND_JAM(N)
#   define LIBXS_PRAGMA_UNROLL
# endif
#endif
#if !defined(LIBXS_PRAGMA_UNROLL_N)
# if defined(__GNUC__) && (LIBXS_VERSION2(8, 3) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__))
#   define LIBXS_PRAGMA_UNROLL_N(N) LIBXS_PRAGMA(GCC unroll N)
# else
#   define LIBXS_PRAGMA_UNROLL_N(N)
# endif
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

#if defined(_OPENMP) && (200805/*v3.0*/ <= _OPENMP) \
 && defined(NDEBUG) /* CCE complains for debug builds */
# define LIBXS_OPENMP_COLLAPSE(N) collapse(N)
#else
# define LIBXS_OPENMP_COLLAPSE(N)
#endif

/** LIBXS_UP2POT rounds up to the next power of two (POT). */
#define LIBXS_UP2POT_01(N) ((N) | ((N) >> 1))
#define LIBXS_UP2POT_02(N) (LIBXS_UP2POT_01(N) | (LIBXS_UP2POT_01(N) >> 2))
#define LIBXS_UP2POT_04(N) (LIBXS_UP2POT_02(N) | (LIBXS_UP2POT_02(N) >> 4))
#define LIBXS_UP2POT_08(N) (LIBXS_UP2POT_04(N) | (LIBXS_UP2POT_04(N) >> 8))
#define LIBXS_UP2POT_16(N) (LIBXS_UP2POT_08(N) | (LIBXS_UP2POT_08(N) >> 16))
#define LIBXS_UP2POT_32(N) (LIBXS_UP2POT_16(N) | (LIBXS_UP2POT_16(N) >> 32))
#define LIBXS_UP2POT(N) (LIBXS_UP2POT_32((unsigned long long)(N) - LIBXS_MIN(1, N)) + LIBXS_MIN(1, N))
#define LIBXS_LO2POT(N) (LIBXS_UP2POT_32((unsigned long long)(N) >> 1) + LIBXS_MIN(1, N))

#define LIBXS_UPDIV(N, MULT) (((N) + ((MULT) - 1)) / (MULT))
#define LIBXS_UP(N, MULT) (LIBXS_UPDIV(N, MULT) * (MULT))
#define LIBXS_UP2(N, NPOT) (((N) + ((NPOT) - 1)) & ~((NPOT) - 1))
#define LIBXS_ABS(A) (0 <= (A) ? (A) : -(A))
#define LIBXS_MIN(A, B) ((A) < (B) ? (A) : (B))
#define LIBXS_MAX(A, B) ((A) < (B) ? (B) : (A))
#define LIBXS_MOD(A, N) ((A) % (N))
#define LIBXS_MOD2(A, NPOT) ((A) & ((NPOT) - 1))
#define LIBXS_DELTA(T0, T1) ((T0) < (T1) ? ((T1) - (T0)) : ((T0) - (T1)))
#define LIBXS_CLMP(VALUE, LO, HI) ((LO) < (VALUE) ? ((VALUE) <= (HI) ? (VALUE) : LIBXS_MIN(VALUE, HI)) : LIBXS_MAX(LO, VALUE))
#define LIBXS_SIZEOF(START, LAST) (((const char*)(LAST)) - ((const char*)(START)) + sizeof(*LAST))
#define LIBXS_FEQ(A, B) ((A) == (B))
#define LIBXS_NEQ(A, B) ((A) != (B))
#define LIBXS_ISPOT(A) (0 != (A) && !((A) & ((A) - 1)))
#define LIBXS_ISWAP(A, B) (((A) ^= (B)), ((B) ^= (A)), ((A) ^= (B)))
#define LIBXS_ISNAN(A)  LIBXS_NEQ(A, A)
#define LIBXS_NOTNAN(A) LIBXS_FEQ(A, A)
#define LIBXS_ROUNDX(TYPE, A) ((TYPE)((long long)(0 <= (A) ? ((double)(A) + 0.5) : ((double)(A) - 0.5))))
#define LIBXS_CONST_VOID_PTR(A) *((const void**)&(A))

/** Makes some functions available independent of C99 support. */
#if defined(__STDC_VERSION__) && (199901L/*C99*/ <= __STDC_VERSION__)
# if defined(__PGI)
#   define LIBXS_POWF(A, B) ((float)pow((float)(A), (float)(B)))
# else
#   define LIBXS_POWF(A, B) powf(A, B)
# endif
# define LIBXS_FREXPF(A, B) frexpf(A, B)
# define LIBXS_ROUNDF(A) roundf(A)
# define LIBXS_ROUND(A) round(A)
# define LIBXS_TANHF(A) tanhf(A)
# define LIBXS_SQRTF(A) sqrtf(A)
# define LIBXS_EXP2F(A) exp2f(A)
# define LIBXS_LOG2F(A) log2f(A)
# define LIBXS_ERFF(A) erff(A)
# define LIBXS_EXP2(A) exp2(A)
# define LIBXS_LOG2(A) log2(A)
# define LIBXS_EXPF(A) expf(A)
# define LIBXS_LOGF(A) logf(A)
#else
# define LIBXS_POWF(A, B) ((float)pow((float)(A), (float)(B)))
# define LIBXS_FREXPF(A, B) ((float)frexp((float)(A), B))
# define LIBXS_ROUNDF(A) LIBXS_ROUNDX(float, A)
# define LIBXS_ROUND(A) LIBXS_ROUNDX(double, A)
# define LIBXS_TANHF(A) ((float)tanh((float)(A)))
# define LIBXS_SQRTF(A) ((float)sqrt((float)(A)))
# define LIBXS_EXP2F(A) LIBXS_POWF(2, A)
# define LIBXS_LOG2F(A) ((float)LIBXS_LOG2((float)(A)))
# define LIBXS_ERFF(A) ((float)erf((float)(A)))
# define LIBXS_EXP2(A) pow(2.0, A)
# define LIBXS_LOG2(A) (log(A) * (1.0 / (M_LN2)))
# define LIBXS_EXPF(A) ((float)exp((float)(A)))
# define LIBXS_LOGF(A) ((float)log((float)(A)))
#endif

#if defined(LIBXS_INTEL_COMPILER)
# if (1700 <= LIBXS_INTEL_COMPILER)
#   define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION)
# else
#   define LIBXS_ASSUME(EXPRESSION) assert(EXPRESSION)
# endif
#elif defined(_MSC_VER)
# define LIBXS_ASSUME(EXPRESSION) __assume(EXPRESSION)
#elif defined(__GNUC__) && !defined(_CRAYC) && (LIBXS_VERSION2(4, 5) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__))
# define LIBXS_ASSUME(EXPRESSION) do { if (!(EXPRESSION)) __builtin_unreachable(); } while(0)
#else
# define LIBXS_ASSUME(EXPRESSION) assert(EXPRESSION)
#endif

#if defined(__INTEL_COMPILER)
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
#define  LIBXS_SELECT_ELEMENT_1(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E0
#define  LIBXS_SELECT_ELEMENT_2(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E1
#define  LIBXS_SELECT_ELEMENT_3(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E2
#define  LIBXS_SELECT_ELEMENT_4(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E3
#define  LIBXS_SELECT_ELEMENT_5(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E4
#define  LIBXS_SELECT_ELEMENT_6(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E5
#define  LIBXS_SELECT_ELEMENT_7(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E6
#define  LIBXS_SELECT_ELEMENT_8(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E7
#define  LIBXS_SELECT_ELEMENT_9(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E8
#define LIBXS_SELECT_ELEMENT_10(E0, E1, E2, E3, E4, E5, E6, E7, E8, E9) E9
#define LIBXS_SELECT_HEAD_AUX(A, ...) (A)
#define LIBXS_SELECT_HEAD(...) LIBXS_EXPAND(LIBXS_SELECT_HEAD_AUX(__VA_ARGS__, 0/*dummy*/))
#define LIBXS_SELECT_TAIL(A, ...) __VA_ARGS__

/**
 * For VLAs, check EXACTLY for C99 since a C11-conforming compiler may not provide VLAs.
 * However, some compilers (Intel) may signal support for VLA even with strict ANSI (C89).
 * To ultimately disable VLA-support, define LIBXS_NO_VLA (make VLA=0).
 * VLA-support is signaled by LIBXS_VLA.
 */
#if !defined(LIBXS_VLA) && !defined(LIBXS_NO_VLA) && !defined(__PGI) && ( \
    (defined(__STDC_VERSION__) && (199901L/*C99*/ == __STDC_VERSION__ || (!defined(__STDC_NO_VLA__) && 199901L/*C99*/ < __STDC_VERSION__))) || \
    (defined(__GNUC__) && LIBXS_VERSION2(5, 0) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__) && !defined(__STRICT_ANSI__) && !defined(__cplusplus)) || \
    (defined(LIBXS_INTEL_COMPILER) && !defined(_WIN32) && !defined(__cplusplus)) || \
    (defined(__INTEL_COMPILER) && !defined(_WIN32)))
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
#define  LIBXS_INDEX1_1(...) ((size_t)LIBXS_SELECT_HEAD(__VA_ARGS__))
#define  LIBXS_INDEX1_2(I0, I1, S1) (LIBXS_INDEX1_1(I0) * ((size_t)S1) + (size_t)I1)
#define  LIBXS_INDEX1_3(I0, I1, I2, S1, S2) (LIBXS_INDEX1_2(I0, I1, S1) * ((size_t)S2) + (size_t)I2)
#define  LIBXS_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) (LIBXS_INDEX1_3(I0, I1, I2, S1, S2) * ((size_t)S3) + (size_t)I3)
#define  LIBXS_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) (LIBXS_INDEX1_4(I0, I1, I2, I3, S1, S2, S3) * ((size_t)S4) + (size_t)I4)
#define  LIBXS_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) (LIBXS_INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) * ((size_t)S5) + (size_t)I5)
#define  LIBXS_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) (LIBXS_INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) * ((size_t)S6) + (size_t)I6)
#define  LIBXS_INDEX1_8(I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, S7) (LIBXS_INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) * ((size_t)S7) + (size_t)I7)
#define  LIBXS_INDEX1_9(I0, I1, I2, I3, I4, I5, I6, I7, I8, S1, S2, S3, S4, S5, S6, S7, S8) (LIBXS_INDEX1_8(I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, S7) * ((size_t)S8) + (size_t)I8)
#define LIBXS_INDEX1_10(I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, S1, S2, S3, S4, S5, S6, S7, S8, S9) (LIBXS_INDEX1_9(I0, I1, I2, I3, I4, I5, I6, I7, I8, S1, S2, S3, S4, S5, S6, S7, S8) * ((size_t)S9) + (size_t)I9)

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
LIBXS_API_INLINE int libxs_nonconst_int(int i) { return i; }
# define  LIBXS_VLA_ACCESS(NDIMS, ARRAY, ...) LIBXS_VLA_ACCESS_ND(NDIMS, LIBXS_CONCATENATE(ARRAY, LIBXS_VLA_POSTFIX), LIBXS_VLA_ACCESS_SINK, __VA_ARGS__)
# define  LIBXS_VLA_ACCESS_SINK(S) + 0 * (S)
# define  LIBXS_VLA_ACCESS_NONCONST(I) libxs_nonconst_int(I)
# define  LIBXS_VLA_ACCESS_ND(NDIMS, ARRAY, XY, ...) LIBXS_CONCATENATE3(LIBXS_VLA_ACCESS_, NDIMS, D)(ARRAY, XY, __VA_ARGS__)
# define  LIBXS_VLA_ACCESS_0D(ARRAY, XY, ...) (ARRAY)/*scalar*/
# define  LIBXS_VLA_ACCESS_1D(ARRAY, XY, ...) ((ARRAY)[LIBXS_VLA_ACCESS_NONCONST(LIBXS_SELECT_HEAD(__VA_ARGS__))])
# define  LIBXS_VLA_ACCESS_2D(ARRAY, XY, I0, I1, ...) (((ARRAY) XY(__VA_ARGS__))[I0][LIBXS_VLA_ACCESS_NONCONST(I1)])
# define  LIBXS_VLA_ACCESS_3D(ARRAY, XY, I0, I1, I2, S1, ...) (((ARRAY) XY(S1) XY(__VA_ARGS__))[I0][I1][LIBXS_VLA_ACCESS_NONCONST(I2)])
# define  LIBXS_VLA_ACCESS_4D(ARRAY, XY, I0, I1, I2, I3, S1, S2, ...) (((ARRAY) XY(S1) XY(S2) XY(__VA_ARGS__))[I0][I1][I2][LIBXS_VLA_ACCESS_NONCONST(I3)])
# define  LIBXS_VLA_ACCESS_5D(ARRAY, XY, I0, I1, I2, I3, I4, S1, S2, S3, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(__VA_ARGS__))[I0][I1][I2][I3][LIBXS_VLA_ACCESS_NONCONST(I4)])
# define  LIBXS_VLA_ACCESS_6D(ARRAY, XY, I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][LIBXS_VLA_ACCESS_NONCONST(I5)])
# define  LIBXS_VLA_ACCESS_7D(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][LIBXS_VLA_ACCESS_NONCONST(I6)])
# define  LIBXS_VLA_ACCESS_8D(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(S6) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6][LIBXS_VLA_ACCESS_NONCONST(I7)])
# define  LIBXS_VLA_ACCESS_9D(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, I7, I8, S1, S2, S3, S4, S5, S6, S7, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(S6) XY(S7) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6][I7][LIBXS_VLA_ACCESS_NONCONST(I8)])
# define LIBXS_VLA_ACCESS_10D(ARRAY, XY, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, S1, S2, S3, S4, S5, S6, S7, S8, ...) (((ARRAY) XY(S1) XY(S2) XY(S3) XY(S4) XY(S5) XY(S6) XY(S7) XY(S8) XY(__VA_ARGS__))[I0][I1][I2][I3][I4][I5][I6][I7][I8][LIBXS_VLA_ACCESS_NONCONST(I9)])
# define LIBXS_VLA_DECL(NDIMS, ELEMENT_TYPE, ARRAY_VAR, .../*initial value, and bounds*/) \
    ELEMENT_TYPE LIBXS_VLA_ACCESS_ND(LIBXS_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9), *LIBXS_RESTRICT LIBXS_CONCATENATE(ARRAY_VAR, LIBXS_VLA_POSTFIX), \
      LIBXS_ELIDE, LIBXS_SELECT_TAIL(__VA_ARGS__, 0)/*bounds*/, LIBXS_SELECT_TAIL(__VA_ARGS__, 0)/*dummy*/) = \
   (ELEMENT_TYPE LIBXS_VLA_ACCESS_ND(LIBXS_SELECT_ELEMENT(NDIMS, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9), *, \
      LIBXS_ELIDE, LIBXS_SELECT_TAIL(__VA_ARGS__, 0)/*bounds*/, LIBXS_SELECT_TAIL(__VA_ARGS__, 0)/*dummy*/))LIBXS_SELECT_HEAD(__VA_ARGS__)
#else /* calculate linear index */
# define LIBXS_VLA_ACCESS(NDIMS, ARRAY, ...) LIBXS_CONCATENATE(ARRAY, LIBXS_VLA_POSTFIX)[LIBXS_INDEX1(NDIMS, __VA_ARGS__)]
# define LIBXS_VLA_DECL(NDIMS, ELEMENT_TYPE, ARRAY_VAR, .../*initial value, and bounds*/) \
    ELEMENT_TYPE *LIBXS_RESTRICT LIBXS_CONCATENATE(ARRAY_VAR, LIBXS_VLA_POSTFIX) = /*(ELEMENT_TYPE*)*/LIBXS_SELECT_HEAD(__VA_ARGS__) \
    + 0 * LIBXS_INDEX1(NDIMS, LIBXS_SELECT_TAIL(__VA_ARGS__, LIBXS_SELECT_TAIL(__VA_ARGS__, 0))) /* dummy-shift to "sink" unused arguments */
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
#if !defined(NDEBUG)
# define LIBXS_UNUSED_DEBUG(VARIABLE) LIBXS_UNUSED(VARIABLE)
#else
# define LIBXS_UNUSED_DEBUG(VARIABLE)
#endif

#if defined(_OPENMP)
# define LIBXS_PRAGMA_OMP(...) LIBXS_PRAGMA(omp __VA_ARGS__)
# if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#   define LIBXS_OMP_VAR(A) LIBXS_UNUSED(A) /* suppress warning about "unused" variable */
# elif defined(__clang__)
#   define LIBXS_OMP_VAR(A) (A) = 0
# else
# define LIBXS_OMP_VAR(A)
# endif
#else
# define LIBXS_PRAGMA_OMP(...)
# define LIBXS_OMP_VAR(A)
#endif

#if defined(LIBXS_BUILD) && (defined(__GNUC__) || defined(__clang__)) && !defined(__CYGWIN__) && !defined(__MINGW32__)
# define LIBXS_ATTRIBUTE_WEAK_IMPORT LIBXS_ATTRIBUTE(weak_import)
# define LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE(weak)
#else
# define LIBXS_ATTRIBUTE_WEAK
# define LIBXS_ATTRIBUTE_WEAK_IMPORT
#endif

#if !defined(LIBXS_NO_CTOR) && !defined(LIBXS_CTOR) && \
    (defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__)) && \
    (defined(LIBXS_BUILD) && !defined(__STATIC)) && \
    (defined(__GNUC__) || defined(__clang__))
# define LIBXS_ATTRIBUTE_CTOR LIBXS_ATTRIBUTE(constructor)
# define LIBXS_ATTRIBUTE_DTOR LIBXS_ATTRIBUTE(destructor)
# define LIBXS_CTOR
#else
# define LIBXS_ATTRIBUTE_CTOR
# define LIBXS_ATTRIBUTE_DTOR
#endif

#if defined(__GNUC__) && !defined(__PGI) && !defined(__ibmxl__)
# define LIBXS_ATTRIBUTE_NO_TRACE LIBXS_ATTRIBUTE(no_instrument_function)
#else
# define LIBXS_ATTRIBUTE_NO_TRACE
#endif

#if defined(__GNUC__)
# define LIBXS_MAY_ALIAS LIBXS_ATTRIBUTE(__may_alias__)
#else
# define LIBXS_MAY_ALIAS
#endif

#if !defined(LIBXS_MKTEMP_PATTERN)
# define LIBXS_MKTEMP_PATTERN "XXXXXX"
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
#if !defined(_GNU_SOURCE) && defined(LIBXS_BUILD)
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

#if (0 != LIBXS_SYNC)
# if !defined(_REENTRANT)
#   define _REENTRANT
# endif
# if defined(__PGI)
#   if defined(__GCC_ATOMIC_TEST_AND_SET_TRUEVAL)
#     undef __GCC_ATOMIC_TEST_AND_SET_TRUEVAL
#   endif
#   define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
# endif
#endif

#if !defined(__has_feature) && !defined(__clang__)
# define __has_feature(A) 0
#endif
#if !defined(__has_builtin) && !defined(__clang__)
# define __has_builtin(A) 0
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif

#if (0 != LIBXS_SYNC)
# if defined(_WIN32) || defined(__CYGWIN__)
#   include <windows.h>
# else
#   include <pthread.h>
# endif
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
# define LIBXS_ASSERT_MSG(EXPR, MSG) assert((EXPR) && *MSG)
#endif
#if !defined(LIBXS_EXPECT_ELIDE)
# define LIBXS_EXPECT_ELIDE(EXPR) do { \
    /*const*/ int libxs_expect_elide_ = (EXPR); \
    LIBXS_UNUSED(libxs_expect_elide_); \
  } while(0)
#endif
#if defined(NDEBUG)
# define LIBXS_EXPECT LIBXS_EXPECT_ELIDE
#else
# define LIBXS_EXPECT LIBXS_ASSERT
#endif
#if defined(_DEBUG)
# define LIBXS_EXPECT_DEBUG LIBXS_EXPECT
#else
# define LIBXS_EXPECT_DEBUG LIBXS_EXPECT_ELIDE
#endif
#if defined(_OPENMP) && defined(LIBXS_SYNC_OMP)
# include <omp.h>
#endif
#include <inttypes.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <stdio.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(FLT_MAX)
# if !defined(__FLT_MAX__)
#   define FLT_MAX 3.40282346638528859811704183484516925e+38F
# else
#   define FLT_MAX __FLT_MAX__
# endif
#endif
#if !defined(FLT_MIN)
# if !defined(__FLT_MIN__)
#   define FLT_MIN 1.17549435082228750796873653722224568e-38F
# else
#   define FLT_MIN __FLT_MIN__
# endif
#endif
#if defined(_WIN32) && 0
# define LIBXS_SNPRINTF(S, N, ...) _snprintf_s(S, N, _TRUNCATE, __VA_ARGS__)
#elif defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__ || defined(__GNUC__))
# define LIBXS_SNPRINTF(S, N, ...) snprintf(S, N, __VA_ARGS__)
#else
# define LIBXS_SNPRINTF(S, N, ...) sprintf((S) + /*unused*/(N) * 0, __VA_ARGS__)
#endif

#if defined(__THROW) && defined(__cplusplus)
# define LIBXS_THROW __THROW
#endif
#if !defined(LIBXS_THROW)
# define LIBXS_THROW
#endif
#if defined(__GNUC__) && LIBXS_VERSION2(4, 2) == LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__) && \
  !defined(__clang__) && !defined(__PGI) && !defined(__INTEL_COMPILER) && !defined(_CRAYC)
# define LIBXS_NOTHROW LIBXS_THROW
#else
# define LIBXS_NOTHROW
#endif
#if defined(__cplusplus)
# if (__cplusplus > 199711L)
#   define LIBXS_NOEXCEPT noexcept
# else
#   define LIBXS_NOEXCEPT throw()
# endif
#else
# define LIBXS_NOEXCEPT LIBXS_NOTHROW
#endif

#if defined(_WIN32)
# define LIBXS_PUTENV(A) _putenv(A)
#else
# define LIBXS_PUTENV(A) putenv(A)
#endif

/* block must be after including above header files */
#if (defined(__GLIBC__) && defined(__GLIBC_MINOR__) && LIBXS_VERSION2(__GLIBC__, __GLIBC_MINOR__) < LIBXS_VERSION2(2, 26)) \
  || (defined(LIBXS_INTEL_COMPILER) && (1802 >= LIBXS_INTEL_COMPILER) && !defined(__cplusplus) && defined(__linux__))
/* _Float128 was introduced with GNU GCC 7.0. */
# if !defined(_Float128) && !defined(__SIZEOF_FLOAT128__) && defined(__GNUC__) && !defined(__cplusplus) && defined(__linux__)
#   define _Float128 __float128
# endif
# if !defined(LIBXS_GLIBC_FPTYPES) && defined(__GNUC__) && !defined(__cplusplus) && defined(__linux__) \
  && (LIBXS_VERSION2(7, 0) > LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__) || \
     (defined(LIBXS_INTEL_COMPILER) && (1802 >= LIBXS_INTEL_COMPILER)))
#   define LIBXS_GLIBC_FPTYPES
# endif
# if !defined(_Float128X) && defined(LIBXS_GLIBC_FPTYPES)
#   define _Float128X _Float128
# endif
# if !defined(_Float32) && defined(LIBXS_GLIBC_FPTYPES)
#   define _Float32 float
# endif
# if !defined(_Float32x) && defined(LIBXS_GLIBC_FPTYPES)
#   define _Float32x _Float32
# endif
# if !defined(_Float64) && defined(LIBXS_GLIBC_FPTYPES)
#   define _Float64 double
# endif
# if !defined(_Float64x) && defined(LIBXS_GLIBC_FPTYPES)
#   define _Float64x _Float64
# endif
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if defined(LIBXS_GLIBC_FPTYPES)
# if defined(__cplusplus)
#   undef __USE_MISC
#   if !defined(_DEFAULT_SOURCE)
#     define _DEFAULT_SOURCE
#   endif
#   if !defined(_BSD_SOURCE)
#     define _BSD_SOURCE
#   endif
# else
#   if !defined(__PURE_INTEL_C99_HEADERS__)
#     define __PURE_INTEL_C99_HEADERS__
#   endif
# endif
#endif
#if !defined(LIBXS_NO_LIBM)
# if (defined(LIBXS_INTEL_COMPILER) && (1800 <= LIBXS_INTEL_COMPILER)) \
  && !defined(_WIN32) /* error including dfp754.h */
#   include <mathimf.h>
# endif
# include <math.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#endif /*LIBXS_MACROS_H*/
