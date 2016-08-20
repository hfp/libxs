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
#include "libxs_trans.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdio.h>
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(LIBXS_TRANS_EXTERNAL)
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
#   include <mkl_trans.h>
# elif defined(__OPENBLAS)
#   include <openblas/cblas.h>
# endif
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_TRANS_CACHESIZE)
# define LIBXS_TRANS_CACHESIZE 32768
#endif
#if !defined(LIBXS_TRANS_CHUNKSIZE)
# if defined(__MIC__)
#   define LIBXS_TRANS_CHUNKSIZE 8
# else
#   define LIBXS_TRANS_CHUNKSIZE 32
# endif
#endif


/* Based on the cache-oblivious transpose by Frigo et.al. with optimization for a loop with bounds which are known at compile-time. */
LIBXS_INLINE LIBXS_RETARGETABLE void internal_trans_oop(void *LIBXS_RESTRICT out, const void *LIBXS_RESTRICT in,
  unsigned int typesize, libxs_blasint m0, libxs_blasint m1, libxs_blasint n0, libxs_blasint n1,
  libxs_blasint ld, libxs_blasint ldo)
{
  const libxs_blasint m = m1 - m0, n = n1 - n0;
  if (m * n * typesize <= ((LIBXS_TRANS_CACHESIZE) / 2)) {
    switch(typesize) {
      case 1: {
        LIBXS_TRANS_OOP(char, LIBXS_TRANS_CHUNKSIZE, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      case 2: {
        LIBXS_TRANS_OOP(short, LIBXS_TRANS_CHUNKSIZE, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      case 4: {
        LIBXS_TRANS_OOP(float, LIBXS_TRANS_CHUNKSIZE, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      case 8: {
        LIBXS_TRANS_OOP(double, LIBXS_TRANS_CHUNKSIZE, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      case 16: {
        typedef struct dvec2_t { double value[2]; } dvec2_t;
        LIBXS_TRANS_OOP(dvec2_t, LIBXS_TRANS_CHUNKSIZE, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      default: {
#if !defined(NDEBUG) /* library code is expected to be mute */
        fprintf(stderr, "LIBXS: unsupported element type in transpose!\n");
#endif
        assert(0);
      }
    }
  }
  else if (m >= n) {
    const libxs_blasint mi = (m0 + m1) / 2;
    internal_trans_oop(out, in, typesize, m0, mi, n0, n1, ld, ldo);
    internal_trans_oop(out, in, typesize, mi, m1, n0, n1, ld, ldo);
  }
  else {
#if (0 < LIBXS_TRANS_CHUNKSIZE)
    if (LIBXS_TRANS_CHUNKSIZE < n) {
      const libxs_blasint ni = n0 + LIBXS_TRANS_CHUNKSIZE;
      internal_trans_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
      internal_trans_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
    }
    else
#endif
    {
      const libxs_blasint ni = (n0 + n1) / 2;
      internal_trans_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
      internal_trans_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
    }
  }
}


LIBXS_API_DEFINITION void libxs_transpose_oop(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
#if !defined(NDEBUG) /* library code is expected to be mute */
  if (ld < m && ldo < n) {
    fprintf(stderr, "LIBXS: the leading dimensions of the transpose are too small!\n");
  }
  else if (ld < m) {
    fprintf(stderr, "LIBXS: the leading dimension of the transpose input is too small!\n");
  }
  else if (ldo < n) {
    fprintf(stderr, "LIBXS: the leading dimension of the transpose output is too small!\n");
  }
#endif
#if defined(LIBXS_TRANS_EXTERNAL)
  if (8 == typesize) { /* hopefully the actual type is not complex-SP (or alpha-multiplication is not performed) */
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
    mkl_domatcopy('C', 'T', m, n, 1.0, (const double*)in, ld, (double*)out, ldo);
# elif defined(__OPENBLAS) /* tranposes are not really covered by the common CBLAS interface */
    cblas_domatcopy(CblasColMajor, CblasTrans, m, n, 1.0, (const double*)in, ld, (double*)out, ldo);
# endif
  }
  else if (4 == typesize) {
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
    mkl_somatcopy('C', 'T', m, n, 1.f, (const float*)in, ld, (float*)out, ldo);
# elif defined(__OPENBLAS) /* tranposes are not really covered by the common CBLAS interface */
    cblas_somatcopy(CblasColMajor, CblasTrans, m, n, 1.f, (const float*)in, ld, (float*)out, ldo);
# endif
  }
  else if (16 == typesize) {
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
    const MKL_Complex16 one = { 1.0/*real*/, 0.0/*imag*/ };
    mkl_zomatcopy('C', 'T', m, n, one, (const MKL_Complex16*)in, ld, (MKL_Complex16*)out, ldo);
# elif defined(__OPENBLAS) /* tranposes are not really covered by the common CBLAS interface */
    cblas_zomatcopy(CblasColMajor, CblasTrans, m, n, 1.0, (const double*)in, ld, (double*)out, ldo);
# endif
  }
  else
#endif
  {
    internal_trans_oop(out, in, typesize, 0, m, 0, n, ld, ldo);
  }
}


LIBXS_API_DEFINITION void libxs_transpose_inp(void* inout, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  LIBXS_UNUSED(inout); LIBXS_UNUSED(typesize); LIBXS_UNUSED(m); LIBXS_UNUSED(n); LIBXS_UNUSED(ld);
  assert(0/*Not yet implemented!*/);
}


#if defined(LIBXS_BUILD)

LIBXS_API_DEFINITION void libxs_stranspose_oop(float* out, const float* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  libxs_transpose_oop(out, in, sizeof(float), m, n, ld, ldo);
}


LIBXS_API_DEFINITION void libxs_dtranspose_oop(double* out, const double* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  libxs_transpose_oop(out, in, sizeof(double), m, n, ld, ldo);
}


LIBXS_API_DEFINITION void libxs_stranspose_inp(float* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  libxs_transpose_inp(inout, sizeof(float), m, n, ld);
}


LIBXS_API_DEFINITION void libxs_dtranspose_inp(double* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  libxs_transpose_inp(inout, sizeof(double), m, n, ld);
}

#endif /*defined(LIBXS_BUILD)*/
