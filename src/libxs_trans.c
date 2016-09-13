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
#include <libxs_sync.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API_DEFINITION void libxs_trans_init(int archid)
{
  libxs_trans_chunksize = LIBXS_TRANS_MAX_CHUNKSIZE;
#if defined(__MIC__)
  LIBXS_UNUSED(archid);
#else
  if (LIBXS_X86_AVX512_MIC == archid)
#endif
  {
    libxs_trans_chunksize = LIBXS_TRANS_MIN_CHUNKSIZE;
  }
}


LIBXS_API_DEFINITION void libxs_trans_finalize(void)
{
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_otrans(void *LIBXS_RESTRICT out, const void *LIBXS_RESTRICT in,
  unsigned int typesize, libxs_blasint m0, libxs_blasint m1, libxs_blasint n0, libxs_blasint n1,
  libxs_blasint ld, libxs_blasint ldo)
{
  LIBXS_OTRANS_MAIN(LIBXS_NOOP_ARGS, internal_otrans, out, in, typesize, m0, m1, n0, n1, ld, ldo);
}


LIBXS_API_DEFINITION int libxs_otrans(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG) /* library code is expected to be mute */
  static LIBXS_TLS int trans_error = 0;
#endif
  LIBXS_INIT

  if (ld >= m && ldo >= n) {
    if (out != in) {
      internal_otrans(out, in, typesize, 0, m, 0, n, ld, ldo);
    }
    else if (ld == ldo) {
      libxs_itrans(out, typesize, m, n, ld);
    }
    else {
#if !defined(NDEBUG) /* library code is expected to be mute */
      if (0 == trans_error) {
        fprintf(stderr, "LIBXS: output location of the transpose must be different from the input!\n");
        trans_error = 1;
      }
#endif
      result = EXIT_FAILURE;
    }
  }
  else {
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (0 == trans_error) {
      if (ld < m && ldo < n) {
        fprintf(stderr, "LIBXS: the leading dimensions of the transpose are too small!\n");
      }
      else if (ld < m) {
        fprintf(stderr, "LIBXS: the leading dimension of the transpose input is too small!\n");
      }
      else {
        assert(ldo < n);
        fprintf(stderr, "LIBXS: the leading dimension of the transpose output is too small!\n");
      }
      trans_error = 1;
    }
#endif
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXS_API_DEFINITION int libxs_itrans(void* inout, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  LIBXS_UNUSED(inout); LIBXS_UNUSED(typesize); LIBXS_UNUSED(m); LIBXS_UNUSED(n); LIBXS_UNUSED(ld);
  assert(0/*TODO: not yet implemented!*/);
  LIBXS_INIT
  return EXIT_FAILURE;
}


#if defined(LIBXS_BUILD)

LIBXS_API void libxsf_otrans(void*, const void*, unsigned int, libxs_blasint, libxs_blasint, libxs_blasint, libxs_blasint);
LIBXS_API_DEFINITION void libxsf_otrans(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  libxs_otrans(out, in, typesize, m, n, ld, ldo);
}


/** code variant for the Fortran interface, which does not produce a return value */
LIBXS_API void libxsf_sotrans(float*, const float*, libxs_blasint, libxs_blasint, libxs_blasint, libxs_blasint);
LIBXS_API_DEFINITION void libxsf_sotrans(float* out, const float* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  libxsf_otrans(out, in, sizeof(float), m, n, ld, ldo);
}


LIBXS_API_DEFINITION int libxs_sotrans(float* out, const float* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  return libxs_otrans(out, in, sizeof(float), m, n, ld, ldo);
}


/** code variant for the Fortran interface, which does not produce a return value */
LIBXS_API void libxsf_dotrans(double*, const double*, libxs_blasint, libxs_blasint, libxs_blasint, libxs_blasint);
LIBXS_API_DEFINITION void libxsf_dotrans(double* out, const double* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  libxsf_otrans(out, in, sizeof(double), m, n, ld, ldo);
}


LIBXS_API_DEFINITION int libxs_dotrans(double* out, const double* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ldo)
{
  return libxs_otrans(out, in, sizeof(double), m, n, ld, ldo);
}


LIBXS_API_DEFINITION int libxs_sitrans(float* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  return libxs_itrans(inout, sizeof(float), m, n, ld);
}


LIBXS_API_DEFINITION int libxs_ditrans(double* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  return libxs_itrans(inout, sizeof(double), m, n, ld);
}

#endif /*defined(LIBXS_BUILD)*/
