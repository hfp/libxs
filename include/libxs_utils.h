/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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

/** Macro evaluates to LIBXS_ATTRIBUTE_TARGET_xxx (see below) */
#define LIBXS_ATTRIBUTE_TARGET(TARGET) LIBXS_CONCATENATE2(LIBXS_ATTRIBUTE_TARGET_, TARGET)

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif

#if defined(__MIC__) && !defined(LIBXS_INTRINSICS_NONE)
# define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_IMCI
# define LIBXS_INTRINSICS(TARGET)
#elif !defined(LIBXS_INTRINSICS_NONE) /*!defined(__MIC__)*/
# if    defined(__AVX512F__)  && defined(__AVX512CD__) \
   &&   defined(__AVX512DQ__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && !(defined(__APPLE__) && defined(__MACH__)) \
   && (!defined(__clang__) || ((LIBXS_VERSION3(3, 9, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
   || (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))))
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
# elif  defined(__AVX512F__) && defined(__AVX512CD__) \
   &&   defined(__AVX512PF__) && defined(__AVX512ER__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && !(defined(__APPLE__) && defined(__MACH__)) \
   && (!defined(__clang__) || ((LIBXS_VERSION3(3, 5, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
   || (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))))
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
# elif  defined(__AVX512F__) && defined(__AVX512CD__) \
   &&   defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__) \
   && !(defined(__APPLE__) && defined(__MACH__)) \
   && (!defined(__clang__) || ((LIBXS_VERSION3(3, 5, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
   || (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))))
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX512
# elif defined(__AVX2__) && defined(__FMA__) && defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX2
# elif defined(__AVX__) && defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_AVX
# elif defined(__SSE4_2__) && defined(__SSE4_1__) && defined(__SSE3__)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_SSE4
# elif defined(__SSE3__)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_SSE3
# elif defined(__x86_64__) || defined(_WIN32) || defined(_WIN64)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_GENERIC
# endif
# if defined(LIBXS_STATIC_TARGET_ARCH) && !defined(LIBXS_INTRINSICS_STATIC)
#   if defined(__INTEL_COMPILER)
      /* TODO: compiler version check for LIBXS_MAX_STATIC_TARGET_ARCH */
#     if 1500 <= (__INTEL_COMPILER)
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
#     elif 1400 <= (__INTEL_COMPILER)
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
#     else
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     endif
#     define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     include <immintrin.h>
#   elif defined(_CRAYC) && defined(__GNUC__)
      /* TODO: version check e.g., LIBXS_VERSION2(11, 5) <= LIBXS_VERSION2(_RELEASE, _RELEASE_MINOR) */
#     define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX
#     define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     include <immintrin.h>
#   elif defined(_MSC_VER)
      /* TODO: compiler version check for LIBXS_MAX_STATIC_TARGET_ARCH */
#     define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     include <immintrin.h>
#   elif (LIBXS_VERSION3(5, 1, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
      /* AVX-512 pseudo intrinsics are missing e.g., reductions */
#     if !defined(LIBXS_INTRINSICS_AVX512_NOREDUCTIONS)
#       define LIBXS_INTRINSICS_AVX512_NOREDUCTIONS
#     endif
#     if !defined(__CYGWIN__)
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
#     else /* Error: invalid register for .seh_savexmm */
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     endif
#     include <immintrin.h>
#   elif (LIBXS_VERSION3(4, 9, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
      /* AVX-512 pseudo intrinsics are missing e.g., reductions */
#     if !defined(LIBXS_INTRINSICS_AVX512_NOREDUCTIONS)
#       define LIBXS_INTRINSICS_AVX512_NOREDUCTIONS
#     endif
      /* AVX-512 mask register support is missing */
#     if !defined(LIBXS_INTRINSICS_AVX512_NOMASK)
#       define LIBXS_INTRINSICS_AVX512_NOMASK
#     endif
#     if !defined(__CYGWIN__)
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
#     else /* Error: invalid register for .seh_savexmm */
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     endif
#     include <immintrin.h>
#   else /* GCC/legacy incl. Clang */
#     if defined(__clang__) && !(defined(__APPLE__) && defined(__MACH__)) \
        && ((LIBXS_VERSION3(3, 9, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
         || (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))) /* devel */
        /* AVX-512 pseudo intrinsics are missing e.g., reductions */
#       if !defined(LIBXS_INTRINSICS_AVX512_NOREDUCTIONS)
#         define LIBXS_INTRINSICS_AVX512_NOREDUCTIONS
#       endif
        /* AVX-512 mask register support is missing */
#       if !defined(LIBXS_INTRINSICS_AVX512_NOMASK)
#         define LIBXS_INTRINSICS_AVX512_NOMASK
#       endif
#       if !defined(__CYGWIN__)
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
#       else /* Error: invalid register for .seh_savexmm */
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#       endif
#     elif defined(__clang__) && !(defined(__APPLE__) && defined(__MACH__)) \
        && (LIBXS_VERSION3(3, 5, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
        /* AVX-512 pseudo intrinsics are missing e.g., reductions */
#       if !defined(LIBXS_INTRINSICS_AVX512_NOREDUCTIONS)
#         define LIBXS_INTRINSICS_AVX512_NOREDUCTIONS
#       endif
        /* AVX-512 mask register support is missing */
#       if !defined(LIBXS_INTRINSICS_AVX512_NOMASK)
#         define LIBXS_INTRINSICS_AVX512_NOMASK
#       endif
#       if !defined(__CYGWIN__)
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
#       else /* Error: invalid register for .seh_savexmm */
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#       endif
#     elif (defined(__clang__)  && defined(__APPLE__) && defined(__MACH__)) \
        || (defined(__GNUC__)   && LIBXS_VERSION3(4, 7, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#     elif (defined(__GNUC__)   && LIBXS_VERSION3(4, 4, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX
#     else /* fall-back */
#       define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#       if !defined(LIBXS_INTRINSICS_NONE)
#         define LIBXS_INTRINSICS_NONE
#       endif
#     endif
#     if !defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_STATIC_TARGET_ARCH < LIBXS_X86_AVX2/*workaround*/)
#       define LIBXS_INTRINSICS_LEGACY
#     endif
#     if !defined(LIBXS_INTRINSICS_PATCH)
#       define LIBXS_INTRINSICS_PATCH
#     endif
#   endif /* GCC/legacy incl. Clang */
#   if defined(LIBXS_INTRINSICS_PATCH) && !defined(LIBXS_INTRINSICS_NONE)
#     if !defined(__SSE3__)
#       define __SSE3__ 1
#     endif
#     if !defined(__SSSE3__)
#       define __SSSE3__ 1
#     endif
#     if !defined(__SSE4_1__)
#       define __SSE4_1__ 1
#     endif
#     if !defined(__SSE4_2__)
#       define __SSE4_2__ 1
#     endif
#     if !defined(__AVX__)
#       define __AVX__ 1
#     endif
#     if !defined(__AVX2__)
#       define __AVX2__ 1
#     endif
#     if !defined(__FMA__)
#       define __FMA__ 1
#     endif
#     if !defined(__AVX512F__)
#       define __AVX512F__ 1
#     endif
#     if !defined(__AVX512CD__)
#       define __AVX512CD__ 1
#     endif
#     if !defined(__AVX512PF__)
#       define __AVX512PF__ 1
#     endif
#     if !defined(__AVX512ER__)
#       define __AVX512ER__ 1
#     endif
#     if !defined(__AVX512DQ__)
#       define __AVX512DQ__ 1
#     endif
#     if !defined(__AVX512BW__)
#       define __AVX512BW__ 1
#     endif
#     if !defined(__AVX512VL__)
#       define __AVX512VL__ 1
#     endif
#     if defined(__GNUC__) && !defined(__clang__)
#       pragma GCC push_options
#       if (LIBXS_X86_AVX < LIBXS_MAX_STATIC_TARGET_ARCH)
#         pragma GCC target("avx2,fma")
#       else
#         pragma GCC target("avx")
#       endif
#     endif
#     include <immintrin.h>
#     if defined(__GNUC__) && !defined(__clang__)
#       pragma GCC pop_options
#     endif
#     if (LIBXS_X86_SSE3 > (LIBXS_STATIC_TARGET_ARCH))
#       undef __SSE3__
#     endif
#     if (LIBXS_X86_SSE4 > (LIBXS_STATIC_TARGET_ARCH))
#       undef __SSSE3__
#       undef __SSE4_1__
#       undef __SSE4_2__
#     endif
#     if (LIBXS_X86_AVX > (LIBXS_STATIC_TARGET_ARCH))
#       undef __AVX__
#     endif
#     if (LIBXS_X86_AVX2 > (LIBXS_STATIC_TARGET_ARCH))
#       undef __AVX2__
#       undef __FMA__
#     endif
#     if (LIBXS_X86_AVX512 > (LIBXS_STATIC_TARGET_ARCH))
#       undef __AVX512F__
#       undef __AVX512CD__
#     endif
#     if (LIBXS_X86_AVX512_MIC > (LIBXS_STATIC_TARGET_ARCH))
#       undef __AVX512F__
#       undef __AVX512CD__
#       undef __AVX512PF__
#       undef __AVX512ER__
#     endif
#     if (LIBXS_X86_AVX512_CORE > (LIBXS_STATIC_TARGET_ARCH))
#       undef __AVX512F__
#       undef __AVX512CD__
#       undef __AVX512DQ__
#       undef __AVX512BW__
#       undef __AVX512VL__
#     endif
#   endif /*defined(LIBXS_INTRINSICS_PATCH)*/
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
#     else
#       define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#     endif
#   endif /*!defined(LIBXS_INTRINSICS)*/
# elif defined(LIBXS_STATIC_TARGET_ARCH)
#   include <immintrin.h>
# endif /*defined(LIBXS_STATIC_TARGET_ARCH)*/
#endif /*!defined(LIBXS_INTRINSICS_NONE)*/
#if !defined(LIBXS_STATIC_TARGET_ARCH)
# define LIBXS_STATIC_TARGET_ARCH LIBXS_TARGET_ARCH_GENERIC
#endif

#if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#endif

/** Include basic x86 intrinsics such as __rdtsc. */
#if defined(LIBXS_INTRINSICS) && !defined(LIBXS_INTRINSICS_NONE)
# if defined(_WIN32)
#   include <intrin.h>
# else
#   include <x86intrin.h>
# endif
# include <xmmintrin.h>
# if defined(__SSE3__)
#   include <pmmintrin.h>
# endif
#else
# if !defined(LIBXS_INTRINSICS_NONE)
#   define LIBXS_INTRINSICS_NONE
# endif
# define LIBXS_INTRINSICS(TARGET)
#endif

#if !defined(LIBXS_INTRINSICS_NONE)
# if defined(_WIN32)
#   include <malloc.h>
# else
#   include <mm_malloc.h>
# endif
/** Intrinsic-specific fixups */
# if defined(__clang__)
#   define LIBXS_INTRINSICS_LDDQU_SI128(A) _mm_loadu_si128(A)
# else
#   define LIBXS_INTRINSICS_LDDQU_SI128(A) _mm_lddqu_si128(A)
# endif
/* Clang misses _mm512_stream_p? (checked with v3.8.1). */
# if defined(__clang__) && (LIBXS_VERSION3(3, 9, 0) > LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXS_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_store_ps(A, B)
#   define LIBXS_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_store_pd(A, B)
# else
#   define LIBXS_INTRINSICS_MM512_STREAM_PS(A, B) _mm512_stream_ps(A, B)
#   define LIBXS_INTRINSICS_MM512_STREAM_PD(A, B) _mm512_stream_pd(A, B)
# endif
/* at least Clang 3.8.1 declares prototypes with incorrect signature (_mm512_load_ps takes DP*, _mm512_load_pd takes SP*) */
# if defined(__clang__) && (LIBXS_VERSION3(3, 9, 0) >  LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
                        && (LIBXS_VERSION3(0, 0, 0) != LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#   define LIBXS_INTRINSICS_MM512_LOAD_PS(A) _mm512_load_ps((const double*)(A))
#   define LIBXS_INTRINSICS_MM512_LOAD_PD(A) _mm512_load_pd((const float*)(A))
# elif defined(__clang__)
#   define LIBXS_INTRINSICS_MM512_LOAD_PS(A) _mm512_load_ps((const float*)(A))
#   define LIBXS_INTRINSICS_MM512_LOAD_PD(A) _mm512_load_pd((const double*)(A))
# else
#   define LIBXS_INTRINSICS_MM512_LOAD_PS(A) _mm512_load_ps(A)
#   define LIBXS_INTRINSICS_MM512_LOAD_PD(A) _mm512_load_pd(A)
# endif
#endif /*!defined(LIBXS_INTRINSICS_NONE)*/
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if (defined(__INTEL_COMPILER) || defined(_CRAYC)) && !defined(LIBXS_INTRINSICS_NONE)
# define LIBXS_INTRINSICS_BITSCANFWD(N) _bit_scan_forward(N)
#elif defined(__GNUC__) && !defined(_CRAYC) && !defined(LIBXS_INTRINSICS_NONE)
# define LIBXS_INTRINSICS_BITSCANFWD(N) (__builtin_ffs(N) - 1)
#else /* fall-back implementation */
  LIBXS_API_INLINE int libxs_bitscanfwd(int n) {
    int i, r = 0; for (i = 1; 0 == (n & i) ; i <<= 1) { ++r; } return r;
  }
# define LIBXS_INTRINSICS_BITSCANFWD(N) libxs_bitscanfwd(N)
#endif

#if !defined(LIBXS_INTRINSICS_KNC) && !defined(LIBXS_INTRINSICS_NONE) && defined(__MIC__)
# define LIBXS_INTRINSICS_KNC
#endif

/** LIBXS_INTRINSICS_X86 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_X86) && !defined(LIBXS_INTRINSICS_NONE) && ( \
     (!defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_X86_GENERIC <= LIBXS_MAX_STATIC_TARGET_ARCH)) \
  || (defined(__clang__) && LIBXS_X86_GENERIC <= LIBXS_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_X86
#endif

/** LIBXS_INTRINSICS_SSE3 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_SSE3) && !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_INTRINSICS_X86) && ( \
     (!defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_X86_SSE3 <= LIBXS_MAX_STATIC_TARGET_ARCH)) \
  || (defined(__clang__) && LIBXS_X86_SSE3 <= LIBXS_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_SSE3
#endif

/** LIBXS_INTRINSICS_SSE4 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_SSE4) && !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_INTRINSICS_SSE3) && ( \
     (!defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_X86_SSE4 <= LIBXS_MAX_STATIC_TARGET_ARCH)) \
  || (defined(__clang__) && LIBXS_X86_SSE4 <= LIBXS_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_SSE4
#endif

/** LIBXS_INTRINSICS_AVX is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX) && !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_INTRINSICS_SSE4) && ( \
     (!defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_X86_AVX <= LIBXS_MAX_STATIC_TARGET_ARCH)) \
  || (defined(__clang__) && LIBXS_X86_AVX <= LIBXS_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX
#endif

/** LIBXS_INTRINSICS_AVX2 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX2) && !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_INTRINSICS_AVX) && ( \
     (!defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)) \
  || (defined(__clang__) && LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX2
#endif

/** LIBXS_INTRINSICS_AVX512 is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512) && !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_INTRINSICS_AVX2) && ( \
     (!defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)) \
  || (defined(__clang__) && LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX512
#endif

/** LIBXS_INTRINSICS_AVX512_MIC is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512_MIC) && !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_INTRINSICS_AVX512) && ( \
     (!defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_X86_AVX512_MIC <= LIBXS_MAX_STATIC_TARGET_ARCH)) \
  || (defined(__clang__) && LIBXS_X86_AVX512_MIC <= LIBXS_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX512_MIC
#endif

/** LIBXS_INTRINSICS_AVX512_KNM is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512_KNM) && !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_INTRINSICS_AVX512_MIC) && ( \
     (!defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_X86_AVX512_KNM <= LIBXS_MAX_STATIC_TARGET_ARCH)) \
  || (defined(__clang__) && LIBXS_X86_AVX512_KNM <= LIBXS_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX512_KNM
#endif

/** LIBXS_INTRINSICS_AVX512_CORE is defined only if the compiler is able to generate this code without special flags. */
#if !defined(LIBXS_INTRINSICS_AVX512_CORE) && !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_INTRINSICS_AVX512) && ( \
     (!defined(LIBXS_INTRINSICS_LEGACY) && (LIBXS_X86_AVX512_CORE <= LIBXS_MAX_STATIC_TARGET_ARCH)) \
  || (defined(__clang__) && LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH))
# define LIBXS_INTRINSICS_AVX512_CORE
#endif

#endif /*LIBXS_INTRINSICS_X86_H*/
