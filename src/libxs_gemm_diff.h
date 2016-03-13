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
#ifndef LIBXS_GEMM_DIFF_H
#define LIBXS_GEMM_DIFF_H

#include <libxs.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <libxs_generator.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/** Enable masked load of reference (A) descriptor (disabled: descriptor must be SIMD-padded!) */
#if !defined(LIBXS_GEMM_DIFF_MASK_A)
# define LIBXS_GEMM_DIFF_MASK_A
#endif
/** Enable generic implementation */
#if !defined(LIBXS_GEMM_DIFF_SW) /*&& defined(__MIC__)*/
# define LIBXS_GEMM_DIFF_SW
#endif


/** Function type representing the gemm_diff functionality. */
typedef LIBXS_RETARGETABLE unsigned int (*libxs_gemm_diff_function)(const libxs_gemm_descriptor*, const libxs_gemm_descriptor*);
/** Function type representing the gemm_diffn functionality. */
typedef LIBXS_RETARGETABLE unsigned int (*libxs_gemm_diffn_function)(const libxs_gemm_descriptor*, const libxs_gemm_descriptor*, unsigned int, int);


/** Initialize GEMM/DIFF module; not thread-safe. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_gemm_diff_init(const char* archid, int has_sse);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_gemm_diff_finalize(void);

/** Dispatched implementation which may (or may not) use a SIMD extension. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_gemm_diff(const libxs_gemm_descriptor* reference, const libxs_gemm_descriptor* desc);
/** Generic implementation which is only relying on high-level constructs. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_gemm_diff_sw(const libxs_gemm_descriptor* reference, const libxs_gemm_descriptor* desc);
/** Collection of implementations which are using specific instruction set extensions. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_gemm_diff_sse(const libxs_gemm_descriptor* reference, const libxs_gemm_descriptor* desc);
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_gemm_diff_avx(const libxs_gemm_descriptor* reference, const libxs_gemm_descriptor* desc);
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_gemm_diff_avx2(const libxs_gemm_descriptor* reference, const libxs_gemm_descriptor* desc);
#if defined(__MIC__)
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_gemm_diff_imci(const libxs_gemm_descriptor* reference, const libxs_gemm_descriptor* desc);
#endif

/** Compare a number of descriptors (array) against a reference descriptor. Returns the index of the first match (or ndesc in case of no match). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_gemm_diffn(const libxs_gemm_descriptor* reference,
  /** Array of descriptors with ndesc elements. */
  const libxs_gemm_descriptor* desc, unsigned int ndesc,
  /** Number of bytes until the next descriptor is reached (stride). */
  int nbytes);
/** Generic implementation of libxs_gemm_diffn which is only relying on high-level constructs. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_gemm_diffn_sw(const libxs_gemm_descriptor* reference,
  const libxs_gemm_descriptor* desc, unsigned int ndesc, int nbytes);

#if defined(LIBXS_BUILD) && !defined(LIBXS_GEMM_DIFF_NOINLINE)
# include "libxs_gemm_diff.c"
#endif

#endif /*LIBXS_GEMM_DIFF_H*/
