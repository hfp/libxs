/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#ifndef LIBXS_INTRINSICS_H
#define LIBXS_INTRINSICS_H

#include <libxs_macros.h>

#if defined(__MIC__)
# include <immintrin.h>
#else
# if defined(__AVX512F__)
#   define LIBXS_AVX 3
# elif defined(__AVX2__)
#   define LIBXS_AVX 2
# elif defined(__AVX__)
#   define LIBXS_AVX 1
# endif
# if defined(LIBXS_AVX)
#   define LIBXS_SSE 5
# elif defined(__SSE4_2__)
#   define LIBXS_SSE 4
# elif defined(__SSE3__)
#   define LIBXS_SSE 3
# endif
# if defined(__AVX2__) || defined(__INTEL_COMPILER) || defined(_WIN32)
#   define LIBXS_AVX_MAX 2
#   define LIBXS_SSE_MAX 5
#   include <immintrin.h>
# elif defined(__clang__)
#   if (defined(__APPLE__) && defined(__MACH__) && (LIBXS_VERSION3(6, 2, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))) \
    || (!(defined(__APPLE__) && defined(__MACH__)) && LIBXS_VERSION3(3, 7, 0) <= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
#     define LIBXS_INTRINSICS LIBXS_ATTRIBUTE(target("sse3,sse4.1,sse4.2,avx,avx2"))
#     define LIBXS_AVX_MAX 2
#     define LIBXS_SSE_MAX 5
#     define __AVX2__ 1
#     if !defined(__AVX__)
#       define __AVX__ 1
#     endif
#     if !defined(__SSE4_1__) && !defined(__SSE4_2__)
#       define __SSE4_1__ 1
#       define __SSE4_2__ 1
#     endif
#     if !defined(__SSSE3__)
#       define LIBXS_UNDEF_SSSE
#       define __SSSE3__ 1
#     endif
#     if !defined(__SSE3__)
#       define __SSE3__ 1
#     endif
#     include <immintrin.h>
#     if !defined(LIBXS_SSE) || (3 > (LIBXS_SSE))
#       undef __SSE3__
#     endif
#     if defined(LIBXS_UNDEF_SSSE)
#       undef __SSSE3__
#     endif
#     if !defined(LIBXS_SSE) || (4 > (LIBXS_SSE))
#       undef __SSE4_1__
#       undef __SSE4_2__
#     endif
#     if !defined(LIBXS_AVX)
#       undef __AVX__
#     endif
#     undef __AVX2__
#   else
#     include <immintrin.h>
#   endif
# elif defined(__GNUC__)
#   if (LIBXS_VERSION3(4, 7, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#     define LIBXS_INTRINSICS LIBXS_ATTRIBUTE(target("sse3,sse4.1,sse4.2,avx,avx2"))
#     pragma GCC push_options
#     pragma GCC target("sse3,sse4.1,sse4.2,avx,avx2")
#     include <immintrin.h>
#     pragma GCC pop_options
#     define LIBXS_AVX_MAX 2
#     define LIBXS_SSE_MAX 5
#   elif (LIBXS_VERSION3(4, 4, 0) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#     define LIBXS_INTRINSICS LIBXS_ATTRIBUTE(target("sse3,sse4.1,sse4.2,avx"))
#     pragma GCC push_options
#     pragma GCC target("sse3,sse4.1,sse4.2,avx")
#     include <immintrin.h>
#     pragma GCC pop_options
#     define LIBXS_AVX_MAX 1
#     define LIBXS_SSE_MAX 5
#   else
#     include <immintrin.h>
#   endif
# elif defined(__AVX__)
#   define LIBXS_AVX_MAX 1
#   define LIBXS_SSE_MAX 5
#   include <immintrin.h>
# elif defined(__SSE4_2__)
#   define LIBXS_SSE_MAX 4
#   include <immintrin.h>
# elif defined(__SSE3__)
#   define LIBXS_SSE_MAX 3
#   include <immintrin.h>
# else
#   include <immintrin.h>
# endif
#endif
#if !defined(LIBXS_INTRINSICS)
# define LIBXS_INTRINSICS
#endif

#endif /*LIBXS_INTRINSICS_H*/
