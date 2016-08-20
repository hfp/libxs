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
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdio.h>
#if !defined(NDEBUG)
# include <assert.h>
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


LIBXS_INLINE LIBXS_RETARGETABLE void internal_trans_oop(void *LIBXS_RESTRICT out, const void *LIBXS_RESTRICT in,
  unsigned int typesize, libxs_blasint m0, libxs_blasint m1, libxs_blasint n0, libxs_blasint n1,
  libxs_blasint ld, libxs_blasint ldo)
{
  LIBXS_TRANS_OOP_MAIN(LIBXS_SEQUENTIAL, LIBXS_JOIN, LIBXS_NOOP, LIBXS_NOOP,
    internal_trans_oop, out, in, typesize, LIBXS_TRANS_CHUNKSIZE, m0, m1, n0, n1, ld, ldo);
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
  internal_trans_oop(out, in, typesize, 0, m, 0, n, ld, ldo);
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
