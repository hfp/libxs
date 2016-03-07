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
#ifndef LIBXS_GEMM_EXT_H
#define LIBXS_GEMM_EXT_H

#include <libxs.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_GEMM_EXTWRAP) && defined(__GNUC__) && !defined(_WIN32) && !(defined(__APPLE__) && defined(__MACH__) && \
  LIBXS_VERSION3(6, 1, 0) >= LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) && !defined(__CYGWIN__)
# if defined(__STATIC) /* -Wl,--wrap=xgemm_ */
#   define LIBXS_GEMM_EXTWRAP
#   define LIBXS_GEMM_EXTWRAP_SGEMM LIBXS_FSYMBOL(__wrap_sgemm)
#   define LIBXS_GEMM_EXTWRAP_DGEMM LIBXS_FSYMBOL(__wrap_dgemm)
    LIBXS_EXTERN_C LIBXS_RETARGETABLE LIBXS_ATTRIBUTE(weak) void LIBXS_FSYMBOL(__real_sgemm)(
      const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
      const float*, const float*, const libxs_blasint*, const float* b, const libxs_blasint*,
      const float* beta, float*, const libxs_blasint*);
    LIBXS_EXTERN_C LIBXS_RETARGETABLE LIBXS_ATTRIBUTE(weak) void LIBXS_FSYMBOL(__real_dgemm)(
      const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
      const double*, const double*, const libxs_blasint*, const double* b, const libxs_blasint*,
      const double* beta, double*, const libxs_blasint*);
    LIBXS_EXTERN_C LIBXS_RETARGETABLE LIBXS_ATTRIBUTE(weak) void LIBXS_FSYMBOL(__real_mkl_sgemm)(
      const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
      const float*, const float*, const libxs_blasint*, const float* b, const libxs_blasint*,
      const float* beta, float*, const libxs_blasint*);
    LIBXS_EXTERN_C LIBXS_RETARGETABLE LIBXS_ATTRIBUTE(weak) void LIBXS_FSYMBOL(__real_mkl_dgemm)(
      const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
      const double*, const double*, const libxs_blasint*, const double* b, const libxs_blasint*,
      const double* beta, double*, const libxs_blasint*);
    /* mute warning about external function definition with no prior declaration */
    LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_GEMM_EXTWRAP_SGEMM(
      const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
      const float*, const float*, const libxs_blasint*, const float* b, const libxs_blasint*,
      const float* beta, float*, const libxs_blasint*);
    /* mute warning about external function definition with no prior declaration */
    LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_GEMM_EXTWRAP_DGEMM(
      const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
      const double*, const double*, const libxs_blasint*, const double* b, const libxs_blasint*,
      const double* beta, double*, const libxs_blasint*);
# elif !defined(__CYGWIN__) /* LD_PRELOAD */
#   define LIBXS_GEMM_EXTWRAP
#   define LIBXS_GEMM_WEAK LIBXS_ATTRIBUTE(weak)
#   define LIBXS_GEMM_EXTWRAP_SGEMM LIBXS_FSYMBOL(sgemm)
#   define LIBXS_GEMM_EXTWRAP_DGEMM LIBXS_FSYMBOL(dgemm)
# endif
#endif /*defined(LIBXS_GEMM_EXTWRAP)*/
#if !defined(LIBXS_GEMM_WEAK)
# define LIBXS_GEMM_WEAK
#endif


/** INTERNAL: configuration table containing the tile sizes separate for DP and SP. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_internal_tile_size[/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/];

/** INTERNAL: number of threads per core */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_internal_num_nt;

/** INTERNAL: kind of GEMM (0: small gemm, 1: sequential, 2: parallelized) */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_internal_gemm;

/**
 * INTERNAL pre-initialization step called by libxs_gemm_init,
 * e.g. configures the tile sizes for multithreaded GEMM functions.
 */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_gemm_configure(const char* archid, int gemm_kind);

#endif /*LIBXS_GEMM_EXT_H*/
