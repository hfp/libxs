/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_UTILS_H
#define LIBXS_UTILS_H

#include "libxs_cpuid.h"

/** Macro evaluates to LIBXS_ATTRIBUTE_TARGET_xxx (see below). */
#define LIBXS_ATTRIBUTE_TARGET(TARGET) LIBXS_CONCATENATE(LIBXS_ATTRIBUTE_TARGET_, TARGET)

#if !defined(__NO_INTRINSICS) && !defined(LIBXS_PLATFORM_X86)
# define __NO_INTRINSICS
#endif
#if !defined(__NO_INTRINSICS) && ( \
    /*no intrinsics: tested with 17.x and 18.x*/ \
    (defined(__PGI) && LIBXS_VERSION2(19, 0) > LIBXS_VERSION2(__PGIC__, __PGIC_MINOR__)) \
 || /*legacy*/(defined(_CRAYC) && !defined(__GNUC__)) \
 || (defined(__GNUC__) && !defined(__clang__) && !defined(LIBXS_INTEL_COMPILER) && !defined(_CRAYC) && \
      LIBXS_VERSION2(4, 4) > LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__)) /* GCC 4.4 */ \
 || (defined(__clang__) && !defined(__APPLE__) && \
      LIBXS_VERSION2(3, 7) > LIBXS_VERSION2(__clang_major__, __clang_minor__)) \
 || (defined(__APPLE__) && defined(__MACH__) && !defined(LIBXS_INTEL_COMPILER) && defined(__clang__) && \
      LIBXS_VERSION2(9, 0) > LIBXS_VERSION2(__clang_major__, __clang_minor__)))
# define __NO_INTRINSICS
#endif

#if !defined(__NO_INTRINSICS)
  /* AVX10.1/512: compiler defines __AVX10_1_512__ (GCC >= 14, Clang >= 19) */
# if defined(__AVX10_1_512__) && defined(__AVX2__) && defined(__FMA__) && defined(__SSE4_2__) && defined(__SSE3__)
#   if !defined(LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX10_512
#   endif
#   define LIBXS_INTRINSICS_INCLUDE
  /* AVX-512 + AVX-VNNI-INT8: compiler defines __AVXVNNIINT8__ (GCC >= 12, Clang >= 16) */
# elif defined(__AVX512F__) && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   &&   defined(__AVXVNNIINT8__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(LIBXS_INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXS_VERSION2(12, 0) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__))) \
   && (!defined(__clang__) || (LIBXS_VERSION2(16, 0) <= LIBXS_VERSION2(__clang_major__, __clang_minor__)))
#   if !defined(LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX512_INT8
#   endif
#   define LIBXS_INTRINSICS_INCLUDE
  /* AVX-512 baseline: avx512f+cd+dq+bw+vl+vnni */
# elif defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(LIBXS_INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXS_VERSION2(8, 0) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__))) \
   && (!defined(__clang__) || (LIBXS_VERSION2(8, 1) <= LIBXS_VERSION2(__clang_major__, __clang_minor__))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXS_VERSION2(9, 0) <= LIBXS_VERSION2(__clang_major__, __clang_minor__))
#   if !defined(LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX512
#   endif
#   define LIBXS_INTRINSICS_INCLUDE
  /* AVX10.1/256 without AVX-512: compiler defines __AVX10_1_256__ or __AVX10_1__ */
# elif (defined(__AVX10_1_256__) || defined(__AVX10_1__)) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   if !defined(LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX10_256
#   endif
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   if !defined(LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#   endif
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   if !defined(LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX
#   endif
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   if !defined(LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_SSE42
#   endif
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(__SSE3__)
#   if !defined(LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_SSE3
#   endif
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(LIBXS_PLATFORM_X86)
#   if !defined(LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_GENERIC
#   endif
#   if defined(__GNUC__)
#     define LIBXS_INTRINSICS_INCLUDE
#   endif
# endif
# if defined(LIBXS_STATIC_TARGET_ARCH)
#   if defined(__TARGET_ARCH) && !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
#     define LIBXS_MAX_STATIC_TARGET_ARCH __TARGET_ARCH
#   endif
#   if defined(__INTEL_COMPILER)
#     if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
        /* TODO: compiler version check for LIBXS_MAX_STATIC_TARGET_ARCH */
#       if 1500 <= (LIBXS_INTEL_COMPILER)
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512
#       else
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#       endif
#     endif
#     define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXS_INTRINSICS_INCLUDE
#   elif defined(_CRAYC) && defined(__GNUC__)
      /* TODO: version check, e.g., LIBXS_VERSION2(11, 5) <= LIBXS_VERSION2(_RELEASE, _RELEASE_MINOR) */
#     if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX
#     endif
#     define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXS_INTRINSICS_INCLUDE
#   elif defined(_MSC_VER) && !defined(__clang__)
      /* TODO: compiler version check for LIBXS_MAX_STATIC_TARGET_ARCH */
#     if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     endif
#     define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXS_INTRINSICS_INCLUDE
#   elif (!defined(__GNUC__)  || LIBXS_VERSION2(4, 9) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__)) \
      && (!defined(__clang__) || LIBXS_VERSION2(4, 0) <= LIBXS_VERSION2(__clang_major__, __clang_minor__)) \
      && (!defined(__APPLE__) || !defined(__MACH__)) && !defined(__PGI) && !defined(_MSC_VER)
#     if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
#       if defined(__CYGWIN__) /* Cygwin: invalid register for .seh_savexmm */
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
          /* GCC >= 14 / Clang >= 19: target("avx10.1-512") available */
#       elif (defined(__GNUC__)  && LIBXS_VERSION2(14, 1) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__)) \
          || (defined(__clang__) && LIBXS_VERSION2(19, 0) <= LIBXS_VERSION2(__clang_major__, __clang_minor__))
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX10_512
          /* GCC >= 12 / Clang >= 16: target("avxvnniint8") available */
#       elif (defined(__GNUC__)  && LIBXS_VERSION2(12, 0) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__)) \
          || (defined(__clang__) && LIBXS_VERSION2(16, 0) <= LIBXS_VERSION2(__clang_major__, __clang_minor__))
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_INT8
          /* GCC >= 8 / Clang >= 8.1: target("avx512f,...,avx512vnni") available */
#       elif (defined(__GNUC__)  && LIBXS_VERSION2(8, 0) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__)) \
          || (defined(__clang__) && LIBXS_VERSION2(8, 1) <= LIBXS_VERSION2(__clang_major__, __clang_minor__))
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512
#       else
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#       endif
#     endif
#     define LIBXS_INTRINSICS_INCLUDE
#   else /* GCC/legacy incl. Clang */
#     if defined(__clang__) && !(defined(__APPLE__) && defined(__MACH__)) && !defined(_WIN32)
#       if (LIBXS_VERSION2(7, 0) <= LIBXS_VERSION2(__clang_major__, __clang_minor__)) /* TODO */
          /* no limitations */
#       elif (LIBXS_VERSION2(4, 0) <= LIBXS_VERSION2(__clang_major__, __clang_minor__))
#         if !defined(LIBXS_MAX_STATIC_TARGET_ARCH) && (LIBXS_STATIC_TARGET_ARCH < LIBXS_X86_AVX2/*workaround*/)
#           define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#         endif
#       elif !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#       endif
#       if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
#         if defined(__CYGWIN__) /* Cygwin: invalid register for .seh_savexmm */
#           define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#         else
#           define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512
#         endif
#       endif
#     else /* fallback */
#       if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#       endif
#     endif
#     if !defined(LIBXS_INTRINSICS_INCLUDE) && (!defined(__PGI) || LIBXS_VERSION2(19, 0) <= LIBXS_VERSION2(__PGIC__, __PGIC_MINOR__))
#       define LIBXS_INTRINSICS_INCLUDE
#     endif
#   endif /* GCC/legacy incl. Clang */
#   if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
#     error "LIBXS_MAX_STATIC_TARGET_ARCH not defined!"
#   endif
#   if defined(LIBXS_INTRINSICS_INCLUDE) && !defined(__NO_INTRINSICS)
#     include <immintrin.h>
#   endif /*defined(LIBXS_INTRINSICS_INCLUDE)*/
#   if !defined(LIBXS_INTRINSICS)
#     if (LIBXS_MAX_STATIC_TARGET_ARCH > LIBXS_STATIC_TARGET_ARCH)
#       define LIBXS_INTRINSICS(TARGET) LIBXS_ATTRIBUTE(LIBXS_ATTRIBUTE_TARGET(TARGET))
        /* LIBXS_ATTRIBUTE_TARGET_xxx is required to literally match the CPUID (libxs_cpuid.h)! */
#       define LIBXS_ATTRIBUTE_TARGET_1002 target("sse2") /* LIBXS_X86_GENERIC (64-bit ABI) */
#       if (LIBXS_X86_SSE3 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1003 target("sse3")
#       else
#         define LIBXS_ATTRIBUTE_TARGET_1003 LIBXS_ATTRIBUTE_TARGET_1002
#       endif
#       if (LIBXS_X86_SSE42 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1004 target("sse4.1,sse4.2")
#       else
#         define LIBXS_ATTRIBUTE_TARGET_1004 LIBXS_ATTRIBUTE_TARGET_1003
#       endif
#       if (LIBXS_X86_AVX <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1005 target("avx")
#       else
#         define LIBXS_ATTRIBUTE_TARGET_1005 LIBXS_ATTRIBUTE_TARGET_1004
#       endif
#       if (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1006 target("avx2,fma")
#       else
#         define LIBXS_ATTRIBUTE_TARGET_1006 LIBXS_ATTRIBUTE_TARGET_1005
#       endif
#       if (LIBXS_X86_AVX10_256 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1050 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl,avx512vnni,avxvnniint8")
#       else
#         define LIBXS_ATTRIBUTE_TARGET_1050 LIBXS_ATTRIBUTE_TARGET_1006
#       endif
#       if (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1100 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl,avx512vnni")
#       else
#         define LIBXS_ATTRIBUTE_TARGET_1100 LIBXS_ATTRIBUTE_TARGET_1050
#       endif
#       if (LIBXS_X86_AVX512_INT8 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1110 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl,avx512vnni,avxvnniint8")
#       else
#         define LIBXS_ATTRIBUTE_TARGET_1110 LIBXS_ATTRIBUTE_TARGET_1100
#       endif
#       if (LIBXS_X86_AVX10_512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1200 target("avx10.1-512")
#       else
#         define LIBXS_ATTRIBUTE_TARGET_1200 LIBXS_ATTRIBUTE_TARGET_1110
#       endif
#     else
#       define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     endif
#   elif !defined(LIBXS_INTRINSICS_TARGET)
#     define LIBXS_INTRINSICS_TARGET
#   endif /*!defined(LIBXS_INTRINSICS)*/
# endif /*defined(LIBXS_STATIC_TARGET_ARCH)*/
#endif /*!defined(__NO_INTRINSICS)*/
#if !defined(LIBXS_STATIC_TARGET_ARCH)
# if !defined(__NO_INTRINSICS)
#   define __NO_INTRINSICS
# endif
# define LIBXS_STATIC_TARGET_ARCH LIBXS_TARGET_ARCH_GENERIC
#endif

#if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#elif (LIBXS_MAX_STATIC_TARGET_ARCH < LIBXS_STATIC_TARGET_ARCH)
# undef LIBXS_MAX_STATIC_TARGET_ARCH
# define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#endif

#if !defined(LIBXS_INTRINSICS)
# define LIBXS_INTRINSICS(TARGET)
#endif

/**
 * Target attribution
 */
/** LIBXS_INTRINSICS_X86 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_X86) && !defined(__NO_INTRINSICS) && \
    (LIBXS_X86_GENERIC <= LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_INTRINSICS_X86
#endif
/** LIBXS_INTRINSICS_SSE3 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_SSE3) && !defined(__NO_INTRINSICS) && \
    (LIBXS_X86_SSE3 <= LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_INTRINSICS_SSE3
#endif
/** LIBXS_INTRINSICS_SSE42 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_SSE42) && !defined(__NO_INTRINSICS) && \
    (LIBXS_X86_SSE42 <= LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_INTRINSICS_SSE42
#endif
/** LIBXS_INTRINSICS_AVX is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX) && !defined(__NO_INTRINSICS) && \
    (LIBXS_X86_AVX <= LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_INTRINSICS_AVX
#endif
/** LIBXS_INTRINSICS_AVX2 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX2) && !defined(__NO_INTRINSICS) && \
    (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_INTRINSICS_AVX2
#endif
/** LIBXS_INTRINSICS_AVX512 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512) && !defined(__NO_INTRINSICS) && \
    (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_INTRINSICS_AVX512
#endif

/** Include basic x86 intrinsics such as __rdtsc. */
#if defined(LIBXS_INTRINSICS_INCLUDE)
# if defined(_WIN32)
#   include <intrin.h>
# elif defined(LIBXS_INTEL_COMPILER) || defined(_CRAYC) || defined(__clang__) || defined(__PGI)
#   include <x86intrin.h>
# elif defined(__GNUC__) && (LIBXS_VERSION2(4, 4) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__))
#   include <x86intrin.h>
# endif
# include <xmmintrin.h>
# if defined(__SSE3__)
#   include <pmmintrin.h>
# endif
#endif

/**
 * Intrinsic-specific fix-ups
 */
# define LIBXS_INTRINSICS_LOADU_SI128(A) _mm_loadu_si128(A)
# define LIBXS_INTRINSICS_MM512_LOAD_PS(A) _mm512_loadu_ps((const float*)(A))
# define LIBXS_INTRINSICS_MM512_LOAD_PD(A) _mm512_loadu_pd((const double*)(A))
# define LIBXS_INTRINSICS_MM512_STREAM_SI512(A, B) _mm512_stream_si512((__m512i*)(A), (B))
# define LIBXS_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_stream_ps(A, B)
# define LIBXS_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_stream_pd(A, B)
#if !defined(LIBXS_INTEL_COMPILER) || (defined(__clang__) && ( \
      (LIBXS_VERSION2(8, 0) > LIBXS_VERSION2(__clang_major__, __clang_minor__)))) \
   || (defined(__APPLE__) && defined(__MACH__)) || defined(__GNUC__)
# define LIBXS_INTRINSICS_MM256_STORE_EPI32(A, B) _mm256_storeu_si256((__m256i*)(A), B)
#else
# define LIBXS_INTRINSICS_MM256_STORE_EPI32(A, B) _mm256_storeu_epi32(A, B)
#endif
#if defined(LIBXS_INTEL_COMPILER)
# if 1600 <= (LIBXS_INTEL_COMPILER)
#   define LIBXS_INTRINSICS_MM512_SET_EPI16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                        E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0) \
                             _mm512_set_epi16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                        E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0)
# else
#   define LIBXS_INTRINSICS_MM512_SET_EPI16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                        E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0) \
         _mm512_castps_si512(_mm512_set_epi16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                        E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0))
# endif
#else
# define LIBXS_INTRINSICS_MM512_SET_EPI16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                      E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0) \
               _mm512_set_epi32(((E31) << 16) | (E30), ((E29) << 16) | (E28), ((E27) << 16) | (E26), ((E25) << 16) | (E24), \
                                ((E23) << 16) | (E22), ((E21) << 16) | (E20), ((E19) << 16) | (E18), ((E17) << 16) | (E16), \
                                ((E15) << 16) | (E14), ((E13) << 16) | (E12), ((E11) << 16) | (E10),  ((E9) << 16) |  (E8), \
                                 ((E7) << 16) |  (E6),  ((E5) << 16) |  (E4),  ((E3) << 16) |  (E2),  ((E1) << 16) |  (E0))
#endif
#if (defined(LIBXS_INTEL_COMPILER) \
  || (defined(__GNUC__) && LIBXS_VERSION2(7, 0) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__)) \
  || (defined(__clang__) && (!defined(__APPLE__) || !defined(__MACH__)) \
      && LIBXS_VERSION2(4, 0) <= LIBXS_VERSION2(__clang_major__, __clang_minor__))) \
  && defined(NDEBUG) /* avoid warning "maybe-uninitialized" due to undefined value init */
# define LIBXS_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_mask_i32gather_epi32(A, B, C, D, E)
# define LIBXS_INTRINSICS_MM512_EXTRACTI64X4_EPI64(A, B) _mm512_extracti64x4_epi64(A, B)
# define LIBXS_INTRINSICS_MM512_ABS_PS(A) _mm512_abs_ps(A)
# define LIBXS_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_undefined_epi32()
# define LIBXS_INTRINSICS_MM512_UNDEFINED() _mm512_undefined()
# define LIBXS_INTRINSICS_MM256_UNDEFINED_SI256() _mm256_undefined_si256()
# define LIBXS_INTRINSICS_MM_UNDEFINED_SI128() _mm_undefined_si128()
# define LIBXS_INTRINSICS_MM_UNDEFINED_PD() _mm_undefined_pd()
#else
# define LIBXS_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_castps_si512(_mm512_mask_i32gather_ps( \
                           _mm512_castsi512_ps(A), B, C, (const float*)(D), E))
# define LIBXS_INTRINSICS_MM512_EXTRACTI64X4_EPI64(A, B) _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(A), B))
# define LIBXS_INTRINSICS_MM512_ABS_PS(A) _mm512_castsi512_ps(_mm512_and_epi32( \
                           _mm512_castps_si512(A), _mm512_set1_epi32(0x7FFFFFFF)))
# define LIBXS_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_set1_epi32(0)
# define LIBXS_INTRINSICS_MM512_UNDEFINED() _mm512_set1_ps(0)
# define LIBXS_INTRINSICS_MM256_UNDEFINED_SI256() _mm256_set1_epi32(0)
# define LIBXS_INTRINSICS_MM_UNDEFINED_SI128() _mm_set1_epi32(0)
# define LIBXS_INTRINSICS_MM_UNDEFINED_PD() _mm_set1_pd(0)
#endif
#if (defined(LIBXS_INTEL_COMPILER) && (1800 <= (LIBXS_INTEL_COMPILER))) \
  || (!defined(LIBXS_INTEL_COMPILER) && defined(__GNUC__) \
      && LIBXS_VERSION2(7, 0) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__)) \
  || ((!defined(__APPLE__) || !defined(__MACH__)) && defined(__clang__) \
      && LIBXS_VERSION2(8, 0) <= LIBXS_VERSION2(__clang_major__, __clang_minor__))
# define LIBXS_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, NBITS) \
    LIBXS_CONCATENATE(_store_mask, NBITS)((LIBXS_CONCATENATE(__mmask, NBITS)*)(DST_PTR), SRC)
# define LIBXS_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, NBITS) \
    LIBXS_CONCATENATE(_load_mask, NBITS)((/*const*/ LIBXS_CONCATENATE(__mmask, NBITS)*)(SRC_PTR))
# define LIBXS_INTRINSICS_MM512_CVTU32_MASK(A, NBITS) LIBXS_CONCATENATE(_cvtu32_mask, NBITS)((unsigned int)(A))
#elif defined(LIBXS_INTEL_COMPILER)
# define LIBXS_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, NBITS) \
    (*(LIBXS_CONCATENATE(__mmask, NBITS)*)(DST_PTR) = (LIBXS_CONCATENATE(__mmask, NBITS))(SRC))
# define LIBXS_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, NBITS) \
    ((LIBXS_CONCATENATE(__mmask, NBITS))_mm512_mask2int(*(const __mmask16*)(SRC_PTR)))
# define LIBXS_INTRINSICS_MM512_CVTU32_MASK(A, NBITS) LIBXS_CONCATENATE(LIBXS_INTRINSICS_MM512_CVTU32_MASK_, NBITS)(A)
# define LIBXS_INTRINSICS_MM512_CVTU32_MASK_32(A) ((__mmask32)(A))
# define LIBXS_INTRINSICS_MM512_CVTU32_MASK_16(A) _mm512_int2mask((int)(A))
# define LIBXS_INTRINSICS_MM512_CVTU32_MASK_8(A) ((__mmask8)(A))
#else
# define LIBXS_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, NBITS) \
    (*(LIBXS_CONCATENATE(__mmask, NBITS)*)(DST_PTR) = (LIBXS_CONCATENATE(__mmask, NBITS))(SRC))
# define LIBXS_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, NBITS) (*(const LIBXS_CONCATENATE(__mmask, NBITS)*)(SRC_PTR))
# define LIBXS_INTRINSICS_MM512_CVTU32_MASK(A, NBITS) ((LIBXS_CONCATENATE(__mmask, NBITS))(A))
#endif
#define LIBXS_INTRINSICS_MM512_STORE_MASK64(DST_PTR, SRC) LIBXS_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, 64)
#define LIBXS_INTRINSICS_MM512_STORE_MASK32(DST_PTR, SRC) LIBXS_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, 32)
#define LIBXS_INTRINSICS_MM512_STORE_MASK16(DST_PTR, SRC) LIBXS_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, 16)
#define LIBXS_INTRINSICS_MM512_STORE_MASK8(DST_PTR, SRC) LIBXS_INTRINSICS_MM512_STORE_MASK(DST_PTR, SRC, 8)
#define LIBXS_INTRINSICS_MM512_LOAD_MASK64(SRC_PTR) LIBXS_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, 64)
#define LIBXS_INTRINSICS_MM512_LOAD_MASK32(SRC_PTR) LIBXS_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, 32)
#define LIBXS_INTRINSICS_MM512_LOAD_MASK16(SRC_PTR) LIBXS_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, 16)
#define LIBXS_INTRINSICS_MM512_LOAD_MASK8(SRC_PTR) LIBXS_INTRINSICS_MM512_LOAD_MASK(SRC_PTR, 8)
#define LIBXS_INTRINSICS_MM512_CVTU32_MASK32(A) LIBXS_INTRINSICS_MM512_CVTU32_MASK(A, 32)
#define LIBXS_INTRINSICS_MM512_CVTU32_MASK16(A) LIBXS_INTRINSICS_MM512_CVTU32_MASK(A, 16)
#define LIBXS_INTRINSICS_MM512_CVTU32_MASK8(A) LIBXS_INTRINSICS_MM512_CVTU32_MASK(A, 8)

/**
 * Pseudo intrinsics for portability
 */
LIBXS_API int LIBXS_INTRINSICS_BITSCANFWD32_SW(unsigned int n);
LIBXS_API int LIBXS_INTRINSICS_BITSCANFWD64_SW(unsigned long long n);

/** Binary Logarithm (based on Stackoverflow's NBITSx macro). */
#define LIBXS_INTRINSICS_BITSCANBWD_SW02(N) (0 != ((N) & 0x2/*0b10*/) ? 1 : 0)
#define LIBXS_INTRINSICS_BITSCANBWD_SW04(N) (0 != ((N) & 0xC/*0b1100*/) ? (2 | LIBXS_INTRINSICS_BITSCANBWD_SW02((N) >> 2)) : LIBXS_INTRINSICS_BITSCANBWD_SW02(N))
#define LIBXS_INTRINSICS_BITSCANBWD_SW08(N) (0 != ((N) & 0xF0/*0b11110000*/) ? (4 | LIBXS_INTRINSICS_BITSCANBWD_SW04((N) >> 4)) : LIBXS_INTRINSICS_BITSCANBWD_SW04(N))
#define LIBXS_INTRINSICS_BITSCANBWD_SW16(N) (0 != ((N) & 0xFF00) ? (8 | LIBXS_INTRINSICS_BITSCANBWD_SW08((N) >> 8)) : LIBXS_INTRINSICS_BITSCANBWD_SW08(N))
#define LIBXS_INTRINSICS_BITSCANBWD_SW32(N) (0 != ((N) & 0xFFFF0000) ? (16 | LIBXS_INTRINSICS_BITSCANBWD_SW16((N) >> 16)) : LIBXS_INTRINSICS_BITSCANBWD_SW16(N))
#define LIBXS_INTRINSICS_BITSCANBWD_SW64(N) (0 != ((N) & 0xFFFFFFFF00000000) ? (32 | LIBXS_INTRINSICS_BITSCANBWD_SW32((N) >> 32)) : LIBXS_INTRINSICS_BITSCANBWD_SW32(N))
#define LIBXS_INTRINSICS_BITSCANBWD32_SW(N) LIBXS_INTRINSICS_BITSCANBWD_SW32((unsigned int)(N))
#define LIBXS_INTRINSICS_BITSCANBWD64_SW(N) LIBXS_INTRINSICS_BITSCANBWD_SW64((unsigned long long)(N))

#if defined(_WIN32) && !defined(__NO_INTRINSICS)
LIBXS_API unsigned int LIBXS_INTRINSICS_BITSCANFWD32(unsigned int n);
LIBXS_API unsigned int LIBXS_INTRINSICS_BITSCANBWD32(unsigned int n);
# if defined(_WIN64)
LIBXS_API unsigned int LIBXS_INTRINSICS_BITSCANFWD64(unsigned long long n);
LIBXS_API unsigned int LIBXS_INTRINSICS_BITSCANBWD64(unsigned long long n);
# else
# define LIBXS_INTRINSICS_BITSCANFWD64 LIBXS_INTRINSICS_BITSCANFWD64_SW
# define LIBXS_INTRINSICS_BITSCANBWD64 LIBXS_INTRINSICS_BITSCANBWD64_SW
# endif
#elif defined(__GNUC__) && !defined(__NO_INTRINSICS)
# define LIBXS_INTRINSICS_BITSCANFWD32(N) (0 != (N) ? __builtin_ctz(N) : 0)
# define LIBXS_INTRINSICS_BITSCANFWD64(N) (0 != (N) ? __builtin_ctzll(N) : 0)
# define LIBXS_INTRINSICS_BITSCANBWD32(N) (0 != (N) ? (31 - __builtin_clz(N)) : 0)
# define LIBXS_INTRINSICS_BITSCANBWD64(N) (0 != (N) ? (63 - __builtin_clzll(N)) : 0)
#else /* fallback implementation */
# define LIBXS_INTRINSICS_BITSCANFWD32 LIBXS_INTRINSICS_BITSCANFWD32_SW
# define LIBXS_INTRINSICS_BITSCANFWD64 LIBXS_INTRINSICS_BITSCANFWD64_SW
# define LIBXS_INTRINSICS_BITSCANBWD32 LIBXS_INTRINSICS_BITSCANBWD32_SW
# define LIBXS_INTRINSICS_BITSCANBWD64 LIBXS_INTRINSICS_BITSCANBWD64_SW
#endif

/** LIBXS_NBITS determines the minimum number of bits needed to represent N. */
#define LIBXS_NBITS(N) (LIBXS_INTRINSICS_BITSCANBWD64(N) + LIBXS_MIN(1, N))
#define LIBXS_ISQRT2(N) ((unsigned int)((1ULL << (LIBXS_NBITS(N) >> 1)) /*+ LIBXS_MIN(1, N)*/))
/** LIBXS_ILOG2 definition matches ceil(log2(N)). */
LIBXS_API unsigned int LIBXS_ILOG2(unsigned long long n);


/**
 * AVX-512 unsigned 32-bit high-multiply: floor(a * b / 2^32) for 16 lanes.
 * Emulates the missing _mm512_mulhi_epu32 via even/odd _mm512_mul_epu32.
 */
#if defined(LIBXS_INTRINSICS_AVX512)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
__m512i libxs_mulhi_epu32(__m512i a, __m512i b)
{
  const __m512i even = _mm512_srli_epi64(_mm512_mul_epu32(a, b), 32);
  const __m512i a_odd = _mm512_srli_epi64(a, 32);
  const __m512i b_odd = _mm512_srli_epi64(b, 32);
  const __m512i odd = _mm512_srli_epi64(_mm512_mul_epu32(a_odd, b_odd), 32);
  return _mm512_or_si512(even, _mm512_slli_epi64(odd, 32));
}

/**
 * AVX-512 vectorized Barrett reduction: x mod p for 16 uint32 lanes.
 * rcp = floor(2^32 / p) as returned by libxs_barrett_rcp().
 * This is the SIMD counterpart of the scalar libxs_mod_u32.
 */
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
__m512i libxs_mod_u32x16(__m512i x, unsigned int p, unsigned int rcp)
{
  const __m512i vp = _mm512_set1_epi32((int)p);
  const __m512i vrcp = _mm512_set1_epi32((int)rcp);
  const __m512i q = libxs_mulhi_epu32(x, vrcp);
  __m512i r = _mm512_sub_epi32(x, _mm512_mullo_epi32(q, vp));
  { const __mmask16 ge = _mm512_cmpge_epu32_mask(r, vp);
    r = _mm512_mask_sub_epi32(r, ge, r, vp);
  }
  return r;
}
#endif /* LIBXS_INTRINSICS_AVX512 */

#endif /*LIBXS_UTILS_H*/
