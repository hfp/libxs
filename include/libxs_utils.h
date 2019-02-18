/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
#ifndef LIBXS_INTRINSICS_X86_H
#define LIBXS_INTRINSICS_X86_H

#include "libxs_cpuid.h"

/** Macro evaluates to LIBXS_ATTRIBUTE_TARGET_xxx (see below). */
#define LIBXS_ATTRIBUTE_TARGET(TARGET) LIBXS_CONCATENATE(LIBXS_ATTRIBUTE_TARGET_, TARGET)
/** Helper macro to access 2048-bit RNG-state. */
#define LIBXS_INTRINSICS_RNG_STATE(IDX) LIBXS_CONCATENATE(libxs_rng_state_, IDX)

#if defined(__PGI) /* no intrinsics: tested with 17.x and 18.x */
# if !defined(LIBXS_INTRINSICS_NONE)
#   define LIBXS_INTRINSICS_NONE
# endif
#elif !defined(LIBXS_INTRINSICS_STATIC) && /* GCC 4.4 (target-attribute) */ \
    (defined(__GNUC__) && !defined(__clang__) && !defined(LIBXS_INTEL_COMPILER) && !defined(_CRAYC) && \
     LIBXS_VERSION3(4, 4, 0) > LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) \
 || (defined(__clang__) && LIBXS_VERSION3(3, 7, 0) > LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
 || (defined(__APPLE__) && defined(__MACH__) && !defined(LIBXS_INTEL_COMPILER) && defined(__clang__) && \
     LIBXS_VERSION3(9, 0, 0) > LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
# define LIBXS_INTRINSICS_STATIC
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif

#if defined(__MIC__) && !defined(LIBXS_INTRINSICS_NONE)
# define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_IMCI
# define LIBXS_INTRINSICS(TARGET)
# define LIBXS_INTRINSICS_INCLUDE
#elif !defined(LIBXS_INTRINSICS_NONE) /*!defined(__MIC__)*/
# if    defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(__INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXS_VERSION3(5, 0, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))) \
   && (!defined(__clang__) || (LIBXS_VERSION3(4, 0, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) \
                           || (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXS_VERSION3(8, 1, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX512_ICL
#   define LIBXS_INTRINSICS_INCLUDE
# elif  defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(__INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXS_VERSION3(5, 0, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))) \
   && (!defined(__clang__) || (LIBXS_VERSION3(4, 0, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) \
                           || (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXS_VERSION3(8, 1, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
#   define LIBXS_INTRINSICS_INCLUDE
# elif  defined(__AVX512F__) && defined(__AVX512CD__) \
   &&   defined(__AVX512PF__) && defined(__AVX512ER__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(__INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXS_VERSION3(5, 0, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))) \
   && (!defined(__clang__) || (LIBXS_VERSION3(4, 0, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) \
                           || (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXS_VERSION3(8, 1, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
#   define LIBXS_INTRINSICS_INCLUDE
# elif  defined(__AVX512F__) && defined(__AVX512CD__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && (!defined(__GNUC__)  || defined(__clang__) || defined(__INTEL_COMPILER) || defined(_CRAYC) \
                           || (LIBXS_VERSION3(5, 0, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))) \
   && (!defined(__clang__) || (LIBXS_VERSION3(4, 0, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) \
                           || (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)))) \
   && (!defined(__APPLE__) || !defined(__MACH__) || LIBXS_VERSION3(8, 1, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX512
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_SSE4
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(__SSE3__)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_SSE3
#   define LIBXS_INTRINSICS_INCLUDE
# elif defined(__x86_64__) || defined(_WIN32) || defined(_WIN64)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_GENERIC
# endif
# if defined(LIBXS_STATIC_TARGET_ARCH) && !defined(LIBXS_INTRINSICS_STATIC)
#   if defined(__INTEL_COMPILER)
      /* TODO: compiler version check for LIBXS_MAX_STATIC_TARGET_ARCH */
#     if 1500 <= (LIBXS_INTEL_COMPILER)
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
#     elif 1400 <= (LIBXS_INTEL_COMPILER)
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
#     else
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     endif
#     define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXS_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   elif defined(_CRAYC) && defined(__GNUC__)
      /* TODO: version check e.g., LIBXS_VERSION2(11, 5) <= LIBXS_VERSION2(_RELEASE, _RELEASE_MINOR) */
#     define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX
#     define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXS_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   elif defined(_MSC_VER) && !defined(__clang__)
      /* TODO: compiler version check for LIBXS_MAX_STATIC_TARGET_ARCH */
#     define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     define LIBXS_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   elif (defined(__GNUC__) && LIBXS_VERSION3(5, 1, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) \
      && !defined(__PGI)
      /* AVX-512 pseudo intrinsics are missing e.g., reductions */
#     if !defined(LIBXS_INTRINSICS_AVX512_NOREDUCTIONS)
#       define LIBXS_INTRINSICS_AVX512_NOREDUCTIONS
#     endif
#     if !defined(__CYGWIN__)
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
#     else /* Error: invalid register for .seh_savexmm */
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     endif
#     define LIBXS_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   elif (defined(__GNUC__) && LIBXS_VERSION3(4, 9, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) \
      && !defined(__PGI)
      /* too many AVX-512 (pseudo-)intrinsics are missing e.g., reductions, or casts (_mm512_castps_si512) */
#     define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     define LIBXS_INTRINSICS_INCLUDE
#     include <immintrin.h>
#   else /* GCC/legacy incl. Clang */
#     if defined(__clang__) && !(defined(__APPLE__) && defined(__MACH__))
#       if (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) /* devel */ || \
           (LIBXS_VERSION3(7, 0, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) /* TODO */
          /* no limitations */
#       elif (LIBXS_VERSION3(4, 0, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#         if !defined(LIBXS_INTRINSICS_STATIC) && (LIBXS_STATIC_TARGET_ARCH < LIBXS_X86_AVX2/*workaround*/)
#           define LIBXS_INTRINSICS_STATIC
#         endif
#       elif !defined(LIBXS_INTRINSICS_STATIC)
#         define LIBXS_INTRINSICS_STATIC
#       endif
#       if !defined(__CYGWIN__)
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
#         if (LIBXS_MAX_STATIC_TARGET_ARCH < LIBXS_STATIC_TARGET_ARCH)
#           undef LIBXS_STATIC_TARGET_ARCH /* account for compiler issues */
#           define LIBXS_STATIC_TARGET_ARCH LIBXS_MAX_STATIC_TARGET_ARCH
#         endif
#       else /* Error: invalid register for .seh_savexmm */
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#       endif
#     else /* fall-back */
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#       if !defined(LIBXS_INTRINSICS_STATIC) && (LIBXS_STATIC_TARGET_ARCH < LIBXS_X86_AVX2/*workaround*/)
#         define LIBXS_INTRINSICS_STATIC
#       endif
#     endif
#     if !defined(LIBXS_INTRINSICS_INCLUDE) && !defined(__PGI)
#       define LIBXS_INTRINSICS_INCLUDE
#     endif
#     if defined(LIBXS_INTRINSICS_INCLUDE) && !defined(LIBXS_INTRINSICS_NONE) && !defined(LIBXS_INTRINSICS_STATIC)
#       if !defined(__SSE3__)
#         define __SSE3__ 1
#       endif
#       if !defined(__SSSE3__)
#         define __SSSE3__ 1
#       endif
#       if !defined(__SSE4_1__)
#         define __SSE4_1__ 1
#       endif
#       if !defined(__SSE4_2__)
#         define __SSE4_2__ 1
#       endif
#       if !defined(__AVX__)
#         define __AVX__ 1
#       endif
#       if !defined(__AVX2__)
#         define __AVX2__ 1
#       endif
#       if !defined(__FMA__)
#         define __FMA__ 1
#       endif
#       if !defined(__AVX512F__)
#         define __AVX512F__ 1
#       endif
#       if !defined(__AVX512CD__)
#         define __AVX512CD__ 1
#       endif
#       if !defined(__AVX512PF__)
#         define __AVX512PF__ 1
#       endif
#       if !defined(__AVX512ER__)
#         define __AVX512ER__ 1
#       endif
#       if !defined(__AVX512DQ__)
#         define __AVX512DQ__ 1
#       endif
#       if !defined(__AVX512BW__)
#         define __AVX512BW__ 1
#       endif
#       if !defined(__AVX512VL__)
#         define __AVX512VL__ 1
#       endif
#       if !defined(__AVX512VNNI__)
#         define __AVX512VNNI__ 1
#       endif
#       if defined(__GNUC__) && !defined(__clang__)
#         pragma GCC push_options
#         if (LIBXS_X86_AVX < LIBXS_MAX_STATIC_TARGET_ARCH)
#           pragma GCC target("avx2,fma")
#         else
#           pragma GCC target("avx")
#         endif
#       endif
#       include <immintrin.h>
#       if defined(__GNUC__) && !defined(__clang__)
#         pragma GCC pop_options
#       endif
#       if (LIBXS_X86_SSE3 > (LIBXS_STATIC_TARGET_ARCH))
#         undef __SSE3__
#       endif
#       if (LIBXS_X86_SSE4 > (LIBXS_STATIC_TARGET_ARCH))
#         undef __SSSE3__
#         undef __SSE4_1__
#         undef __SSE4_2__
#       endif
#       if (LIBXS_X86_AVX > (LIBXS_STATIC_TARGET_ARCH))
#         undef __AVX__
#       endif
#       if (LIBXS_X86_AVX2 > (LIBXS_STATIC_TARGET_ARCH))
#         undef __AVX2__
#         undef __FMA__
#       endif
#       if (LIBXS_X86_AVX512 > (LIBXS_STATIC_TARGET_ARCH))
#         undef __AVX512F__
#         undef __AVX512CD__
#       endif
#       if (LIBXS_X86_AVX512_MIC > (LIBXS_STATIC_TARGET_ARCH))
#         undef __AVX512F__
#         undef __AVX512CD__
#         undef __AVX512PF__
#         undef __AVX512ER__
#       endif
#       if (LIBXS_X86_AVX512_CORE > (LIBXS_STATIC_TARGET_ARCH))
#         undef __AVX512F__
#         undef __AVX512CD__
#         undef __AVX512DQ__
#         undef __AVX512BW__
#         undef __AVX512VL__
#       endif
#       if (LIBXS_X86_AVX512_ICL > (LIBXS_STATIC_TARGET_ARCH))
#         undef __AVX512VNNI__
#       endif
#     endif /*defined(LIBXS_INTRINSICS_INCLUDE)*/
#   endif /* GCC/legacy incl. Clang */
#   if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
#     error "LIBXS_MAX_STATIC_TARGET_ARCH not defined!"
#   endif
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
#       if (LIBXS_X86_SSE4 <= LIBXS_MAX_STATIC_TARGET_ARCH)
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
#       if (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1007 target("avx2,fma,avx512f,avx512cd")
#       else
#         define LIBXS_ATTRIBUTE_TARGET_1007 LIBXS_ATTRIBUTE_TARGET_1006
#       endif
#       if (LIBXS_X86_AVX512_MIC <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1010 target("avx2,fma,avx512f,avx512cd,avx512pf,avx512er")
#       else /* LIBXS_X86_AVX512 */
#         define LIBXS_ATTRIBUTE_TARGET_1010 LIBXS_ATTRIBUTE_TARGET_1007
#       endif
#       if (LIBXS_X86_AVX512_KNM <= LIBXS_MAX_STATIC_TARGET_ARCH) /* TODO: add compiler flags */
#         define LIBXS_ATTRIBUTE_TARGET_1011 target("avx2,fma,avx512f,avx512cd,avx512pf,avx512er")
#       else /* LIBXS_X86_AVX512_MIC */
#         define LIBXS_ATTRIBUTE_TARGET_1011 LIBXS_ATTRIBUTE_TARGET_1010
#       endif
#       if (LIBXS_X86_AVX512_CORE <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1020 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl")
#       else /* LIBXS_X86_AVX512 */
#         define LIBXS_ATTRIBUTE_TARGET_1020 LIBXS_ATTRIBUTE_TARGET_1007
#       endif
#       if (LIBXS_X86_AVX512_ICL <= LIBXS_MAX_STATIC_TARGET_ARCH)
#         define LIBXS_ATTRIBUTE_TARGET_1022 target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl,avx512vnni")
#       else /* LIBXS_X86_AVX512_CORE */
#         define LIBXS_ATTRIBUTE_TARGET_1022 LIBXS_ATTRIBUTE_TARGET_1020
#       endif
#     else
#       define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     endif
#   endif /*!defined(LIBXS_INTRINSICS)*/
# endif /*defined(LIBXS_STATIC_TARGET_ARCH)*/
#endif /*!defined(LIBXS_INTRINSICS_NONE)*/
#if !defined(LIBXS_STATIC_TARGET_ARCH)
# if !defined(LIBXS_INTRINSICS_NONE)
#   define LIBXS_INTRINSICS_NONE
# endif
# define LIBXS_STATIC_TARGET_ARCH LIBXS_TARGET_ARCH_GENERIC
#endif

#if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#endif

#if !defined(LIBXS_INTRINSICS)
# define LIBXS_INTRINSICS(TARGET)
#endif

/** Include basic x86 intrinsics such as __rdtsc. */
#if defined(LIBXS_INTRINSICS_INCLUDE)
# if defined(_WIN32)
#   include <intrin.h>
# elif defined(LIBXS_INTEL_COMPILER) || defined(_CRAYC) || defined(__clang__) || defined(__PGI)
#   include <x86intrin.h>
# elif defined(__GNUC__) && (LIBXS_VERSION3(4, 4, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#   include <x86intrin.h>
# endif
# include <xmmintrin.h>
# if defined(__SSE3__)
#   include <pmmintrin.h>
# endif
#endif

#if !defined(LIBXS_INTRINSICS_NONE)
# if defined(_WIN32)
#   include <malloc.h>
# else
#   include <mm_malloc.h>
# endif
#endif

/**
 * Intrinsic-specific fix-ups
 */
#if defined(__clang__)
# define LIBXS_INTRINSICS_LDDQU_SI128(A) _mm_loadu_si128(A)
#else
# define LIBXS_INTRINSICS_LDDQU_SI128(A) _mm_lddqu_si128(A)
#endif
#if defined(__clang__) && ( \
      (LIBXS_VERSION3(3, 9, 0)  > LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) && \
       LIBXS_VERSION3(0, 0, 0) != LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
   || (LIBXS_VERSION3(7, 3, 0)  > LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__) && \
       defined(__APPLE__) && defined(__MACH__)))
/* prototypes with incorrect signature: _mm512_load_ps takes DP*, _mm512_load_pd takes SP* (checked with v3.8.1) */
# define LIBXS_INTRINSICS_MM512_LOAD_PS(A) _mm512_load_ps((const double*)(A))
# define LIBXS_INTRINSICS_MM512_LOAD_PD(A) _mm512_load_pd((const float*)(A))
/* Clang misses _mm512_stream_p? (checked with v3.8.1). */
# define LIBXS_INTRINSICS_MM512_STREAM_SI512(A, B) _mm512_store_si512((A), (B))
# define LIBXS_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_store_ps((A), (B))
# define LIBXS_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_store_pd(A, B)
#else
# define LIBXS_INTRINSICS_MM512_LOAD_PS(A) _mm512_load_ps((const float*)(A))
# define LIBXS_INTRINSICS_MM512_LOAD_PD(A) _mm512_load_pd((const double*)(A))
# define LIBXS_INTRINSICS_MM512_STREAM_SI512(A, B) _mm512_stream_si512((__m512i*)(A), (B))
# define LIBXS_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_stream_ps((A), (B))
# define LIBXS_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_stream_pd((A), (B))
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
# define LIBXS_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_mask_i32gather_epi32(A, B, C, D, E)
# define LIBXS_INTRINSICS_MM512_EXTRACTI64X4_EPI64(A, B) _mm512_extracti64x4_epi64(A, B)
# define LIBXS_INTRINSICS_MM512_ABS_PS(A) _mm512_abs_ps(A)
# define LIBXS_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_undefined_epi32()
# define LIBXS_INTRINSICS_MM512_UNDEFINED() _mm512_undefined()
# define LIBXS_INTRINSICS_MM_UNDEFINED_PD() _mm_undefined_pd()
#else
# define LIBXS_INTRINSICS_MM512_SET_EPI16(E31, E30, E29, E28, E27, E26, E25, E24, E23, E22, E21, E20, E19, E18, E17, E16, \
                                                      E15, E14, E13, E12, E11, E10, E9, E8, E7, E6, E5, E4, E3, E2, E1, E0) \
               _mm512_set_epi32(((E31) << 16) | (E30), ((E29) << 16) | (E28), ((E27) << 16) | (E26), ((E25) << 16) | (E24), \
                                ((E23) << 16) | (E22), ((E21) << 16) | (E20), ((E19) << 16) | (E18), ((E17) << 16) | (E16), \
                                ((E15) << 16) | (E14), ((E13) << 16) | (E12), ((E11) << 16) | (E10),  ((E9) << 16) |  (E8), \
                                 ((E7) << 16) |  (E6),  ((E5) << 16) |  (E4),  ((E3) << 16) |  (E2),  ((E1) << 16) |  (E0))
# define LIBXS_INTRINSICS_MM512_MASK_I32GATHER_EPI32(A, B, C, D, E) _mm512_castps_si512(_mm512_mask_i32gather_ps( \
                           _mm512_castsi512_ps(A), B, C, (const float*)(D), E))
# define LIBXS_INTRINSICS_MM512_EXTRACTI64X4_EPI64(A, B) _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(A), B))
# define LIBXS_INTRINSICS_MM512_ABS_PS(A) _mm512_castsi512_ps(_mm512_and_epi32( \
                           _mm512_castps_si512(A), _mm512_set1_epi32(0x7FFFFFFF)))
# define LIBXS_INTRINSICS_MM512_UNDEFINED_EPI32() _mm512_set1_epi32(0)
# define LIBXS_INTRINSICS_MM512_UNDEFINED() _mm512_set1_ps(0)
# define LIBXS_INTRINSICS_MM_UNDEFINED_PD() _mm_set1_pd(0)
#endif

/**
 * Pseudo intrinsics for portability
 */
LIBXS_API_INLINE int LIBXS_INTRINSICS_BITSCANFWD32_SW(unsigned int n) {
  unsigned int i, r = 0; if (0 != n) for (i = 1; 0 == (n & i); i <<= 1) { ++r; } return r;
}
LIBXS_API_INLINE int LIBXS_INTRINSICS_BITSCANFWD64_SW(unsigned long long n) {
  unsigned int i, r = 0; if (0 != n) for (i = 1; 0 == (n & i); i <<= 1) { ++r; } return r;
}

/** Binary Logarithm (based on Stackoverflow's NBITSx macro). */
#define LIBXS_INTRINSICS_BITSCANBWD_SW02(N) (0 != ((N) & 0x2/*0b10*/) ? 1 : 0)
#define LIBXS_INTRINSICS_BITSCANBWD_SW04(N) (0 != ((N) & 0xC/*0b1100*/) ? (2 | LIBXS_INTRINSICS_BITSCANBWD_SW02((N) >> 2)) : LIBXS_INTRINSICS_BITSCANBWD_SW02(N))
#define LIBXS_INTRINSICS_BITSCANBWD_SW08(N) (0 != ((N) & 0xF0/*0b11110000*/) ? (4 | LIBXS_INTRINSICS_BITSCANBWD_SW04((N) >> 4)) : LIBXS_INTRINSICS_BITSCANBWD_SW04(N))
#define LIBXS_INTRINSICS_BITSCANBWD_SW16(N) (0 != ((N) & 0xFF00) ? (8 | LIBXS_INTRINSICS_BITSCANBWD_SW08((N) >> 8)) : LIBXS_INTRINSICS_BITSCANBWD_SW08(N))
#define LIBXS_INTRINSICS_BITSCANBWD_SW32(N) (0 != ((N) & 0xFFFF0000) ? (16 | LIBXS_INTRINSICS_BITSCANBWD_SW16((N) >> 16)) : LIBXS_INTRINSICS_BITSCANBWD_SW16(N))
#define LIBXS_INTRINSICS_BITSCANBWD_SW64(N) (0 != ((N) & 0xFFFFFFFF00000000) ? (32 | LIBXS_INTRINSICS_BITSCANBWD_SW32((N) >> 32)) : LIBXS_INTRINSICS_BITSCANBWD_SW32(N))
#define LIBXS_INTRINSICS_BITSCANBWD32_SW(N) LIBXS_INTRINSICS_BITSCANBWD_SW32((unsigned int)(N))
#define LIBXS_INTRINSICS_BITSCANBWD64_SW(N) LIBXS_INTRINSICS_BITSCANBWD_SW64((unsigned long long)(N))

#if defined(_WIN32) && !defined(LIBXS_INTRINSICS_NONE)
  LIBXS_API_INLINE unsigned int LIBXS_INTRINSICS_BITSCANFWD32(unsigned int n) {
    unsigned long r = 0; _BitScanForward(&r, n); return (0 != n) * r;
  }
  LIBXS_API_INLINE unsigned int LIBXS_INTRINSICS_BITSCANBWD32(unsigned int n) {
    unsigned long r = 0; _BitScanReverse(&r, n); return r;
  }
# if defined(_WIN64)
    LIBXS_API_INLINE unsigned int LIBXS_INTRINSICS_BITSCANFWD64(unsigned long long n) {
      unsigned long r = 0; _BitScanForward64(&r, n); return (0 != n) * r;
    }
    LIBXS_API_INLINE unsigned int LIBXS_INTRINSICS_BITSCANBWD64(unsigned long long n) {
      unsigned long r = 0; _BitScanReverse64(&r, n); return r;
    }
# else
#   define LIBXS_INTRINSICS_BITSCANFWD64 LIBXS_INTRINSICS_BITSCANFWD64_SW
#   define LIBXS_INTRINSICS_BITSCANBWD64 LIBXS_INTRINSICS_BITSCANBWD64_SW
# endif
#elif defined(__GNUC__) && !defined(LIBXS_INTRINSICS_NONE)
# define LIBXS_INTRINSICS_BITSCANFWD32(N) ((0 != (N)) * __builtin_ctz(N))
# define LIBXS_INTRINSICS_BITSCANFWD64(N) ((0 != (N)) * __builtin_ctzll(N))
# define LIBXS_INTRINSICS_BITSCANBWD32(N) ((0 != (N)) * (31 - __builtin_clz(N)))
# define LIBXS_INTRINSICS_BITSCANBWD64(N) ((0 != (N)) * (63 - __builtin_clzll(N)))
#else /* fall-back implementation */
# define LIBXS_INTRINSICS_BITSCANFWD32 LIBXS_INTRINSICS_BITSCANFWD32_SW
# define LIBXS_INTRINSICS_BITSCANFWD64 LIBXS_INTRINSICS_BITSCANFWD64_SW
# define LIBXS_INTRINSICS_BITSCANBWD32 LIBXS_INTRINSICS_BITSCANBWD32_SW
# define LIBXS_INTRINSICS_BITSCANBWD64 LIBXS_INTRINSICS_BITSCANBWD64_SW
#endif

/**
 * Target attribution
 */
#if !defined(LIBXS_INTRINSICS_KNC) && !defined(LIBXS_INTRINSICS_NONE) && defined(__MIC__)
# define LIBXS_INTRINSICS_KNC
#endif
/** LIBXS_INTRINSICS_X86 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_X86) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_GENERIC <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_GENERIC <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_X86
#endif
/** LIBXS_INTRINSICS_SSE3 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_SSE3) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_SSE3 <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_SSE3 <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_SSE3
#endif
/** LIBXS_INTRINSICS_SSE4 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_SSE4) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_SSE4 <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_SSE4 <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_SSE4
#endif
/** LIBXS_INTRINSICS_AVX is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_AVX <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_AVX <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX
#endif
/** LIBXS_INTRINSICS_AVX2 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX2) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX2
#endif
/** LIBXS_INTRINSICS_AVX512 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX512
#endif
/** LIBXS_INTRINSICS_AVX512_MIC is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512_MIC) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_AVX512_MIC <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_AVX512_MIC <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX512_MIC
#endif
/** LIBXS_INTRINSICS_AVX512_KNM is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512_KNM) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_AVX512_KNM <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_AVX512_KNM <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX512_KNM
#endif
/** LIBXS_INTRINSICS_AVX512_CORE is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512_CORE) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_AVX512_CORE <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX512_CORE
#endif
/** LIBXS_INTRINSICS_AVX512_ICL is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512_ICL) && !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_AVX512_ICL <= LIBXS_STATIC_TARGET_ARCH || \
   (!defined(LIBXS_INTRINSICS_STATIC) && LIBXS_X86_AVX512_ICL <= LIBXS_MAX_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX512_ICL
#endif

/** 2048-bit state for RNG */
LIBXS_APIVAR(unsigned int libxs_rng_state_0[16]);
LIBXS_APIVAR(unsigned int libxs_rng_state_1[16]);
LIBXS_APIVAR(unsigned int libxs_rng_state_2[16]);
LIBXS_APIVAR(unsigned int libxs_rng_state_3[16]);

/**
 * Pseudo intrinsics (AVX-512)
 */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
# define LIBXS_INTRINSICS_MM512_QUANTIZE_NEAR_PS_EPI16( A, B ) _mm512_cvtepi32_epi16(_mm512_cvt_roundps_epi32( \
    _mm512_mul_ps(LIBXS_INTRINSICS_MM512_LOAD_PS(A), B), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))

LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512) __m512i LIBXS_INTRINSICS_MM512_ROUNDNE_BF16(__m512 a) {
  const __m512i vnaninf = _mm512_set1_epi32(0x7f800000), vrneadd = _mm512_set1_epi32(0x00007fff);
  const __m512i vfixup = _mm512_set1_epi32(0x00000001), vfixupmask = _mm512_set1_epi32(0x00010000);
  const __m512i mm512_roundbf16rne_a_ = _mm512_castps_si512(a);
  const __mmask16 mm512_roundbf16rne_mask1_ = _mm512_cmp_epi32_mask(_mm512_and_epi32(mm512_roundbf16rne_a_, vnaninf), vnaninf, _MM_CMPINT_NE);
  const __mmask16 mm512_roundbf16rne_mask2_ = _mm512_cmp_epi32_mask(_mm512_and_epi32(mm512_roundbf16rne_a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
  return _mm512_mask_add_epi32(mm512_roundbf16rne_a_, mm512_roundbf16rne_mask1_, mm512_roundbf16rne_a_, _mm512_mask_add_epi32(vrneadd, mm512_roundbf16rne_mask2_, vrneadd, vfixup));
}

/** SVML-intrinsics */
#if defined(LIBXS_INTEL_COMPILER)
# define LIBXS_INTRINSICS_MM512_TANH_PS(A) _mm512_tanh_ps(A)
#else
# include <math.h>
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512) __m512 LIBXS_INTRINSICS_MM512_TANH_PS(__m512 a) {
  float a16[16]; int i;
  _mm512_store_ps(a16, a);
  for (i = 0; i < 16; ++i) a16[i] = LIBXS_TANHF(a16[i]);
  return _mm512_loadu_ps(a16);
}
#endif /* SVML */
/** Helper macro to access 2048-bit RNG-state. */
#define LIBXS_INTRINSICS_MM512_RNG_STATE(IDX) (*(__m512i*)LIBXS_CONCATENATE(libxs_rng_state_, IDX))

/** Generate random number in the interval [0, 1); not thread-safe. */
LIBXS_API_INLINE __m512 LIBXS_INTRINSICS_MM512_RNG_PS(void) {
  const __m512i rng_mantissa = _mm512_srli_epi32(_mm512_add_epi32(LIBXS_INTRINSICS_MM512_RNG_STATE(0), LIBXS_INTRINSICS_MM512_RNG_STATE(3)), 9);
  const __m512i s = _mm512_slli_epi32(LIBXS_INTRINSICS_MM512_RNG_STATE(1), 9);
  const __m512i t = _mm512_slli_epi32(LIBXS_INTRINSICS_MM512_RNG_STATE(3), 11);

  LIBXS_INTRINSICS_MM512_RNG_STATE(2) = _mm512_xor_epi32(LIBXS_INTRINSICS_MM512_RNG_STATE(2), LIBXS_INTRINSICS_MM512_RNG_STATE(0));
  LIBXS_INTRINSICS_MM512_RNG_STATE(3) = _mm512_xor_epi32(LIBXS_INTRINSICS_MM512_RNG_STATE(3), LIBXS_INTRINSICS_MM512_RNG_STATE(1));
  LIBXS_INTRINSICS_MM512_RNG_STATE(1) = _mm512_xor_epi32(LIBXS_INTRINSICS_MM512_RNG_STATE(1), LIBXS_INTRINSICS_MM512_RNG_STATE(2));
  LIBXS_INTRINSICS_MM512_RNG_STATE(0) = _mm512_xor_epi32(LIBXS_INTRINSICS_MM512_RNG_STATE(0), LIBXS_INTRINSICS_MM512_RNG_STATE(3));
  LIBXS_INTRINSICS_MM512_RNG_STATE(2) = _mm512_xor_epi32(LIBXS_INTRINSICS_MM512_RNG_STATE(2), s);
  LIBXS_INTRINSICS_MM512_RNG_STATE(3) = _mm512_or_epi32(t, _mm512_srli_epi32(LIBXS_INTRINSICS_MM512_RNG_STATE(3), 32 - 11));

  return _mm512_sub_ps(_mm512_castsi512_ps(_mm512_or_epi32(_mm512_set1_epi32(0x3f800000), rng_mantissa)),
    _mm512_set1_ps(1.0f));
}
#endif /*__AVX512F__*/
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#endif /*LIBXS_INTRINSICS_X86_H*/
