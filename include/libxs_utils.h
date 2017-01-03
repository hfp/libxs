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

/** The following set of macros are required to literally match the CPUID (libxs_cpuid.h)! */
#define LIBXS_ATTRIBUTE_TARGET_1009 /* LIBXS_X86_AVX512_CORE */ \
  target("avx2,fma,avx512f,avx512cd,avx512dq,avx512bw,avx512vl")
#define LIBXS_ATTRIBUTE_TARGET_1008 /* LIBXS_X86_AVX512_MIC */ \
  target("avx2,fma,avx512f,avx512cd,avx512pf,avx512er")
#define LIBXS_ATTRIBUTE_TARGET_1007 /* LIBXS_X86_AVX512 */ \
  target("avx2,fma,avx512f,avx512cd")
#define LIBXS_ATTRIBUTE_TARGET_1006 /* LIBXS_X86_AVX2 */ \
  target("avx2,fma")
#define LIBXS_ATTRIBUTE_TARGET_1005 /* LIBXS_X86_AVX */ \
  target("avx")
#define LIBXS_ATTRIBUTE_TARGET_1004 /* LIBXS_X86_SSE4 */ \
  target("sse2,sse3,ssse3,sse4.1,sse4.2")
#define LIBXS_ATTRIBUTE_TARGET_1003 /* LIBXS_X86_SSE3 */ \
  target("sse3")
#define LIBXS_ATTRIBUTE_TARGET_1002 /* LIBXS_X86_GENERIC */ \
  target("sse2") /* 64-bit ABI */
/** Macro evaluates to LIBXS_ATTRIBUTE_TARGET_xxx (see above) */
#define LIBXS_ATTRIBUTE_TARGET(TARGET) LIBXS_CONCATENATE2(LIBXS_ATTRIBUTE_TARGET_, TARGET)

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif

#if defined(__MIC__)
# define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_IMCI
# define LIBXS_INTRINSICS(TARGET)
#else
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
# elif defined(__x86_64__)
#   define LIBXS_STATIC_TARGET_ARCH LIBXS_X86_GENERIC
# endif
# if defined(__INTEL_COMPILER)
    /* TODO: compiler version check for LIBXS_MAX_STATIC_TARGET_ARCH */
#   if 1500 <= (__INTEL_COMPILER)
#     define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
#   elif 1300 <= (__INTEL_COMPILER)
#     define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
#   else
#     define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#   endif
#   define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#   include <immintrin.h>
# elif defined(_CRAYC) && defined(__GNUC__)
    /* TODO: version check e.g., LIBXS_VERSION2(11, 5) <= LIBXS_VERSION2(_RELEASE, _RELEASE_MINOR) */
#   define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX
#   define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#   include <immintrin.h>
# elif defined(_MSC_VER)
    /* TODO: compiler version check for LIBXS_MAX_STATIC_TARGET_ARCH */
#   define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#   define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#   include <immintrin.h>
# else
#   if defined(__clang__)
#     if !defined(__SSE3__)
#       define __SSE3__ 1
#     endif
#     if !defined(__SSSE3__)
#       define LIBXS_UNDEF_SSSE
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
#     if !defined(LIBXS_INTRINSICS_INCOMPLETE_AVX512) /* some AVX-512 pseudo intrinsics are missing in Clang e.g., reductions */
#       define LIBXS_INTRINSICS_INCOMPLETE_AVX512
#     endif
#     if defined(__APPLE__) && defined(__MACH__)
#       if (LIBXS_X86_AVX2 > LIBXS_STATIC_TARGET_ARCH)
#         define LIBXS_INTRINSICS(TARGET) LIBXS_ATTRIBUTE(LIBXS_ATTRIBUTE_TARGET(TARGET))
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#         undef  LIBXS_ATTRIBUTE_TARGET_1009 /* LIBXS_X86_AVX512_CORE */
#         define LIBXS_ATTRIBUTE_TARGET_1009 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#         undef  LIBXS_ATTRIBUTE_TARGET_1008 /* LIBXS_X86_AVX512_MIC */
#         define LIBXS_ATTRIBUTE_TARGET_1008 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#         undef  LIBXS_ATTRIBUTE_TARGET_1007 /* LIBXS_X86_AVX512 */
#         define LIBXS_ATTRIBUTE_TARGET_1007 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#       else
#         define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#       endif
#     else
#       if ((LIBXS_VERSION3(3, 9, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
         || (LIBXS_VERSION3(0, 0, 0) == LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))) /* Clang/Development */ \
         && !defined(__CYGWIN__) /* Error: invalid register for .seh_savexmm */
#         if (LIBXS_X86_AVX512_CORE > LIBXS_STATIC_TARGET_ARCH)
#           define LIBXS_INTRINSICS(TARGET) LIBXS_ATTRIBUTE(LIBXS_ATTRIBUTE_TARGET(TARGET))
#           define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
#         else
#           define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#         endif
#       elif (LIBXS_VERSION3(3, 5, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) \
         && !defined(__CYGWIN__) /* Error: invalid register for .seh_savexmm */
#         if (LIBXS_X86_AVX512_MIC > LIBXS_STATIC_TARGET_ARCH)
#           define LIBXS_INTRINSICS(TARGET) LIBXS_ATTRIBUTE(LIBXS_ATTRIBUTE_TARGET(TARGET))
#           define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
#           undef  LIBXS_ATTRIBUTE_TARGET_1009 /* LIBXS_X86_AVX512_CORE */
#           define LIBXS_ATTRIBUTE_TARGET_1009 LIBXS_ATTRIBUTE_TARGET(LIBXS_X86_AVX512/*common*/)
#         else
#           define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#         endif
#       else
#         if (LIBXS_X86_AVX/*2*/ > LIBXS_STATIC_TARGET_ARCH)
#           define LIBXS_INTRINSICS(TARGET) LIBXS_ATTRIBUTE(LIBXS_ATTRIBUTE_TARGET(TARGET))
#           define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX/*2*/
#           undef  LIBXS_ATTRIBUTE_TARGET_1009 /* LIBXS_X86_AVX512_CORE */
#           define LIBXS_ATTRIBUTE_TARGET_1009 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#           undef  LIBXS_ATTRIBUTE_TARGET_1008 /* LIBXS_X86_AVX512_MIC */
#           define LIBXS_ATTRIBUTE_TARGET_1008 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#           undef  LIBXS_ATTRIBUTE_TARGET_1007 /* LIBXS_X86_AVX512 */
#           define LIBXS_ATTRIBUTE_TARGET_1007 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#         else
#           define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#         endif
#       endif
#     endif
#     if (LIBXS_X86_AVX512_MIC <= LIBXS_MAX_STATIC_TARGET_ARCH) /* Common */
#       if !defined(__AVX512F__)
#         define __AVX512F__ 1
#       endif
#       if !defined(__AVX512CD__)
#         define __AVX512CD__ 1
#       endif
#     endif
#     if (LIBXS_X86_AVX512_MIC == LIBXS_MAX_STATIC_TARGET_ARCH) /* MIC */
#       if !defined(__AVX512PF__)
#         define __AVX512PF__ 1
#       endif
#       if !defined(__AVX512ER__)
#         define __AVX512ER__ 1
#       endif
#     endif
#     if (LIBXS_X86_AVX512_CORE <= LIBXS_MAX_STATIC_TARGET_ARCH) /* Core */
#       if !defined(__AVX512DQ__)
#         define __AVX512DQ__ 1
#       endif
#       if !defined(__AVX512BW__)
#         define __AVX512BW__ 1
#       endif
#       if !defined(__AVX512VL__)
#         define __AVX512VL__ 1
#       endif
#     endif
#     if !defined(__AVX2__)
#       define __AVX2__ 1
#     endif
#     if !defined(__FMA__)
#       define __FMA__ 1
#     endif
#   elif defined(__GNUC__) && (LIBXS_VERSION3(4, 4, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#     if !defined(LIBXS_INTRINSICS_INCOMPLETE_AVX512) /* some AVX-512 pseudo intrinsics are missing in GCC e.g., reductions */
#       define LIBXS_INTRINSICS_INCOMPLETE_AVX512
#     endif
#     if !defined(LIBXS_INTRINSICS_INCOMPLETE_AVX) /* some AVX2 intrinsics issues in GCC e.g., _mm256_testnzc_si256 */
#       define LIBXS_INTRINSICS_INCOMPLETE_AVX
#     endif
#     if (LIBXS_VERSION3(5, 1, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) \
        && !defined(__CYGWIN__) /* Error: invalid register for .seh_savexmm */
#       if (LIBXS_X86_AVX512_CORE > LIBXS_STATIC_TARGET_ARCH)
#         define LIBXS_INTRINSICS(TARGET) LIBXS_ATTRIBUTE(LIBXS_ATTRIBUTE_TARGET(TARGET))
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_CORE
#       else
#         define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#       endif
      /* TODO: AVX-512 in GCC appears to be incomplete (missing at _mm512_mask_reduce_or_epi32, and some pseudo intrinsics) */
#     elif (LIBXS_VERSION3(4, 9, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)) \
        && !defined(__CYGWIN__) /* Error: invalid register for .seh_savexmm */
#       if (LIBXS_X86_AVX512_MIC > LIBXS_STATIC_TARGET_ARCH)
#         define LIBXS_INTRINSICS(TARGET) LIBXS_ATTRIBUTE(LIBXS_ATTRIBUTE_TARGET(TARGET))
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX512_MIC
#         undef  LIBXS_ATTRIBUTE_TARGET_1009 /* LIBXS_X86_AVX512_CORE */
#         define LIBXS_ATTRIBUTE_TARGET_1009 LIBXS_ATTRIBUTE_TARGET(LIBXS_X86_AVX512/*common*/)
#       else
#         define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#       endif
#     else /* GCC/legacy */
#       if !defined(__SSE3__)
#         define __SSE3__ 1
#       endif
#       if !defined(__SSSE3__)
#         define LIBXS_UNDEF_SSSE
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
#       if defined(__GNUC__) && (LIBXS_VERSION3(4, 7, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#         if (LIBXS_X86_AVX2 > LIBXS_STATIC_TARGET_ARCH)
#           define LIBXS_INTRINSICS(TARGET) LIBXS_ATTRIBUTE(LIBXS_ATTRIBUTE_TARGET(TARGET))
#           define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX2
#           undef  LIBXS_ATTRIBUTE_TARGET_1009 /* LIBXS_X86_AVX512_CORE */
#           define LIBXS_ATTRIBUTE_TARGET_1009 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#           undef  LIBXS_ATTRIBUTE_TARGET_1008 /* LIBXS_X86_AVX512_MIC */
#           define LIBXS_ATTRIBUTE_TARGET_1008 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#           undef  LIBXS_ATTRIBUTE_TARGET_1007 /* LIBXS_X86_AVX512 */
#           define LIBXS_ATTRIBUTE_TARGET_1007 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#         else
#           define LIBXS_INTRINSICS(TARGET)/*no need for target flags*/
#         endif
#       elif (LIBXS_X86_AVX > LIBXS_STATIC_TARGET_ARCH)
#         define LIBXS_INTRINSICS(TARGET) LIBXS_ATTRIBUTE(LIBXS_ATTRIBUTE_TARGET(TARGET))
#         define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_X86_AVX
#         undef  LIBXS_ATTRIBUTE_TARGET_1009 /* LIBXS_X86_AVX512_CORE */
#         define LIBXS_ATTRIBUTE_TARGET_1009 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#         undef  LIBXS_ATTRIBUTE_TARGET_1008 /* LIBXS_X86_AVX512_MIC */
#         define LIBXS_ATTRIBUTE_TARGET_1008 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#         undef  LIBXS_ATTRIBUTE_TARGET_1007 /* LIBXS_X86_AVX512 */
#         define LIBXS_ATTRIBUTE_TARGET_1007 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#         undef  LIBXS_ATTRIBUTE_TARGET_1006 /* LIBXS_X86_AVX2 */
#         define LIBXS_ATTRIBUTE_TARGET_1006 LIBXS_ATTRIBUTE_TARGET(LIBXS_MAX_STATIC_TARGET_ARCH)
#       endif
#     endif
#   endif
#   include <immintrin.h>
#   if !defined(LIBXS_STATIC_TARGET_ARCH) || (LIBXS_X86_SSE3 > (LIBXS_STATIC_TARGET_ARCH))
#     undef __SSE3__
#   endif
#   if defined(LIBXS_UNDEF_SSSE)
#     undef LIBXS_UNDEF_SSSE
#     undef __SSSE3__
#   endif
#   if !defined(LIBXS_STATIC_TARGET_ARCH) || (LIBXS_X86_SSE4 > (LIBXS_STATIC_TARGET_ARCH))
#     undef __SSE4_1__
#     undef __SSE4_2__
#   endif
#   if !defined(LIBXS_STATIC_TARGET_ARCH) || (LIBXS_X86_AVX > (LIBXS_STATIC_TARGET_ARCH))
#     undef __AVX__
#   endif
#   if !defined(LIBXS_STATIC_TARGET_ARCH) || (LIBXS_X86_AVX2 > (LIBXS_STATIC_TARGET_ARCH))
#     undef __AVX2__
#     undef __FMA__
#   endif
#   if !defined(LIBXS_STATIC_TARGET_ARCH) || (LIBXS_X86_AVX512 > (LIBXS_STATIC_TARGET_ARCH))
#     undef __AVX512F__
#     undef __AVX512CD__
#   endif
#   if !defined(LIBXS_STATIC_TARGET_ARCH) || (LIBXS_X86_AVX512_MIC > (LIBXS_STATIC_TARGET_ARCH))
#     undef __AVX512F__
#     undef __AVX512CD__
#     undef __AVX512PF__
#     undef __AVX512ER__
#   endif
#   if !defined(LIBXS_STATIC_TARGET_ARCH) || (LIBXS_X86_AVX512_CORE > (LIBXS_STATIC_TARGET_ARCH))
#     undef __AVX512F__
#     undef __AVX512CD__
#     undef __AVX512DQ__
#     undef __AVX512BW__
#     undef __AVX512VL__
#   endif
# endif
#endif

#if !defined(LIBXS_STATIC_TARGET_ARCH)
# define LIBXS_STATIC_TARGET_ARCH LIBXS_TARGET_ARCH_GENERIC
#endif

#if !defined(LIBXS_MAX_STATIC_TARGET_ARCH)
# define LIBXS_MAX_STATIC_TARGET_ARCH LIBXS_STATIC_TARGET_ARCH
#endif

/** Include basic x86 intrinsics such as __rdtsc. */
#if defined(LIBXS_INTRINSICS)
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
/** Intrinsic-specifc fixups */
# if defined(__clang__)
#   define LIBXS_INTRINSICS_LDDQU_SI128(A) _mm_loadu_si128(A)
# else
#   define LIBXS_INTRINSICS_LDDQU_SI128(A) _mm_lddqu_si128(A)
# endif
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if (defined(__INTEL_COMPILER) || defined(_CRAYC)) && !defined(LIBXS_INTRINSICS_NONE)
# define LIBXS_INTRINSICS_BITSCANFWD(N) _bit_scan_forward(N)
#elif defined(__GNUC__) && !defined(_CRAYC) && !defined(LIBXS_INTRINSICS_NONE)
# define LIBXS_INTRINSICS_BITSCANFWD(N) (__builtin_ffs(N) - 1)
#else /* fall-back implementation */
  LIBXS_INLINE LIBXS_RETARGETABLE int libxs_bitscanfwd(int n) {
    int i, r = 0; for (i = 1; 0 == (n & i) ; i <<= 1) { ++r; } return r;
  }
# define LIBXS_INTRINSICS_BITSCANFWD(N) libxs_bitscanfwd(N)
#endif

#endif /*LIBXS_INTRINSICS_X86_H*/
