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
#include <libxs_cpuid.h>

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
  libxs_blasint ldi, libxs_blasint ldo)
{
  LIBXS_OTRANS_MAIN(LIBXS_NOOP_ARGS, internal_otrans, out, in, typesize, m0, m1, n0, n1, ldi, ldo);
}


LIBXS_API_DEFINITION int libxs_otrans(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG) /* library code is expected to be mute */
  static int error_once = 0;
#endif
  assert(0 < typesize);
  if (ldi >= m && ldo >= n && 0 != out && 0 != in) {
    LIBXS_INIT
    if (out != in) {
      internal_otrans(out, in, typesize, 0, m, 0, n, ldi, ldo);
    }
    else if (ldi == ldo) {
      result = libxs_itrans(out, typesize, m, n, ldi);
    }
    else {
#if !defined(NDEBUG) /* library code is expected to be mute */
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS: output location of the transpose must be different from the input!\n");
      }
#endif
      result = EXIT_FAILURE;
    }
  }
  else {
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      if (0 == out || 0 == in) {
        fprintf(stderr, "LIBXS: the transpose input and/or output is NULL!\n");
      }
      else if (ldi < m && ldo < n) {
        fprintf(stderr, "LIBXS: the leading dimensions of the transpose are too small!\n");
      }
      else if (ldi < m) {
        fprintf(stderr, "LIBXS: the leading dimension of the transpose input is too small!\n");
      }
      else {
        assert(ldo < n);
        fprintf(stderr, "LIBXS: the leading dimension of the transpose output is too small!\n");
      }
    }
#endif
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXS_API_DEFINITION int libxs_itrans(void* inout, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG) /* library code is expected to be mute */
  static int error_once = 0;
#endif
  if (0 != inout) {
    LIBXS_INIT
    if (m == n) { /* some fallback; still warned as "not implemented" */
      libxs_blasint i, j;
      for (i = 0; i < n; ++i) {
        for (j = 0; j < i; ++j) {
          char *const a = ((char*)inout) + (i * ldi + j) * typesize;
          char *const b = ((char*)inout) + (j * ldi + i) * typesize;
          unsigned int k;
          for (k = 0; k < typesize; ++k) {
            const char tmp = a[k];
            a[k] = b[k];
            b[k] = tmp;
          }
        }
      }
    }
    else {
#if !defined(NDEBUG) /* library code is expected to be mute */
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS: in-place transpose is not fully implemented!\n");
      }
#endif
      assert(0/*TODO: proper implementation is pending*/);
      result = EXIT_FAILURE;
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: performance warning - in-place transpose is not fully implemented!\n");
    }
#endif
  }
  else {
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: the transpose input/output is NULL!\n");
    }
#endif
    result = EXIT_FAILURE;
  }

  return result;
}


#if defined(LIBXS_BUILD)

LIBXS_API_DEFINITION int libxs_sotrans(float* out, const float* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  return libxs_otrans(out, in, sizeof(float), m, n, ldi, ldo);
}


LIBXS_API_DEFINITION int libxs_dotrans(double* out, const double* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  return libxs_otrans(out, in, sizeof(double), m, n, ldi, ldo);
}


LIBXS_API_DEFINITION int libxs_sitrans(float* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi)
{
  return libxs_itrans(inout, sizeof(float), m, n, ldi);
}


LIBXS_API_DEFINITION int libxs_ditrans(double* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi)
{
  return libxs_itrans(inout, sizeof(double), m, n, ldi);
}


/** code variant for the Fortran interface, which does not produce a return value */
LIBXS_API void libxsf_otrans(void*, const void*, unsigned int, libxs_blasint, libxs_blasint, libxs_blasint, libxs_blasint);
LIBXS_API_DEFINITION void libxsf_otrans(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  libxs_otrans(out, in, typesize, m, n, ldi, ldo);
}


/** code variant for the Fortran interface, which does not produce a return value */
LIBXS_API void libxsf_sotrans(float*, const float*, libxs_blasint, libxs_blasint, libxs_blasint, libxs_blasint);
LIBXS_API_DEFINITION void libxsf_sotrans(float* out, const float* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  libxs_sotrans(out, in, m, n, ldi, ldo);
}


/** code variant for the Fortran interface, which does not produce a return value */
LIBXS_API void libxsf_dotrans(double*, const double*, libxs_blasint, libxs_blasint, libxs_blasint, libxs_blasint);
LIBXS_API_DEFINITION void libxsf_dotrans(double* out, const double* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  libxs_dotrans(out, in, m, n, ldi, ldo);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_otrans)(void*, const void*, const unsigned int*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*);
LIBXS_API_DEFINITION void LIBXS_FSYMBOL(libxs_otrans)(void* out, const void* in, const unsigned int* typesize,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* ldi, const libxs_blasint* ldo)
{
  libxs_blasint ldx;
  assert(0 != typesize && 0 != m);
  ldx = *(ldi ? ldi : m);
  libxs_otrans(out, in, *typesize, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_sotrans)(float*, const float*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*);
LIBXS_API_DEFINITION void LIBXS_FSYMBOL(libxs_sotrans)(float* out, const float* in,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* ldi, const libxs_blasint* ldo)
{
  libxs_blasint ldx;
  assert(0 != m);
  ldx = *(ldi ? ldi : m);
  libxs_sotrans(out, in, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_dotrans)(double*, const double*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*);
LIBXS_API_DEFINITION void LIBXS_FSYMBOL(libxs_dotrans)(double* out, const double* in,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* ldi, const libxs_blasint* ldo)
{
  libxs_blasint ldx;
  assert(0 != m);
  ldx = *(ldi ? ldi : m);
  libxs_dotrans(out, in, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}

#endif /*defined(LIBXS_BUILD)*/
