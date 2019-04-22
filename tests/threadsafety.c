/******************************************************************************
** Copyright (c) 2015-2019, Intel Corporation                                **
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
#include <libxs.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if !defined(MAX_NKERNELS)
# define MAX_NKERNELS 800
#endif
#if !defined(CHECK_PARALLEL_INIT)
# define CHECK_PARALLEL_INIT
#endif
#if !defined(CHECK_PARALLEL_JIT)
# define CHECK_PARALLEL_JIT
#endif
#if !defined(USE_VERBOSE)
# define USE_VERBOSE
#endif
#if !defined(ITYPE)
# define ITYPE float
#endif


int main(void)
{
  union { libxs_xmmfunction x; void* p; } f[MAX_NKERNELS];
  const ITYPE alpha = LIBXS_ALPHA, beta = LIBXS_BETA;
  const int prefetch = LIBXS_PREFETCH_AUTO;
  libxs_registry_info registry_info;
  const int max_shape = LIBXS_MAX_M;
  const int flags = LIBXS_FLAGS;
  int nkernels = MAX_NKERNELS;
  int result = EXIT_SUCCESS;
  int r[3*MAX_NKERNELS], i;
  int ndup = 0;

  /* generate set of random number for parallel region */
  for (i = 0; i < (3 * nkernels); i += 3) {
    r[i+0] = rand();
    r[i+1] = rand();
    r[i+2] = rand();
  }

#if defined(CHECK_PARALLEL_INIT)
# if defined(_OPENMP)
# pragma omp parallel for default(none) private(i) shared(nkernels)
# endif
  for (i = 0; i < nkernels; ++i) {
    if (0 == (i % 2)) {
      libxs_init();
    }
    else {
      libxs_finalize();
    }
  }
#endif
  libxs_init();

  result = libxs_get_registry_info(&registry_info);
  if (EXIT_SUCCESS == result) {
    nkernels = (int)LIBXS_MIN((size_t)nkernels, registry_info.capacity);
  }

#if defined(_OPENMP) && defined(CHECK_PARALLEL_JIT)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < nkernels; ++i) {
    const libxs_blasint m = r[3*i+0] % max_shape + 1;
    const libxs_blasint n = r[3*i+1] % max_shape + 1;
    const libxs_blasint k = r[3*i+2] % max_shape + 1;
    f[i].x.LIBXS_TPREFIX(ITYPE,mm) = LIBXS_MMDISPATCH_SYMBOL(ITYPE)(m, n, k,
      &m/*lda*/, &k/*ldb*/, &m/*ldc*/, &alpha, &beta, &flags, &prefetch);
  }

#if defined(_OPENMP) && !defined(CHECK_PARALLEL_JIT)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < nkernels; ++i) {
    if (EXIT_SUCCESS == result) {
      const libxs_blasint m = r[3*i+0] % max_shape + 1;
      const libxs_blasint n = r[3*i+1] % max_shape + 1;
      const libxs_blasint k = r[3*i+2] % max_shape + 1;
      union { libxs_xmmfunction x; void* p; } fi;
      libxs_descriptor_blob blob;
      const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init(&blob, LIBXS_GEMM_PRECISION(ITYPE),
        m, n, k, m/*lda*/, k/*ldb*/, m/*ldc*/, &alpha, &beta, flags, prefetch);

      fi.x = libxs_xmmdispatch(desc);
      if (NULL != fi.p && NULL != f[i].p) {
        if (fi.p != f[i].p) {
          libxs_mmkernel_info a_info, b_info;
          size_t a_size, b_size;
          const int ra = libxs_get_mmkernel_info(f[i].x, &a_info, &a_size);
          const int rb = libxs_get_mmkernel_info(fi.x, &b_info, &b_size);

          /* perform deeper check based on another code generation (used as reference) */
          if (EXIT_SUCCESS == ra && EXIT_SUCCESS == rb && (a_size != b_size ||
            0 != memcmp(f[i].p, fi.p, a_size)))
          {
#if defined(_DEBUG) || defined(USE_VERBOSE)
            fprintf(stderr, "Error: the %" PRIuPTR "x%" PRIuPTR "x%" PRIuPTR "-kernel does not match!\n",
              (uintptr_t)m, (uintptr_t)n, (uintptr_t)k);
#endif
#if defined(_OPENMP) && !defined(CHECK_PARALLEL_JIT)
# if (201107 <= _OPENMP)
#           pragma omp atomic write
# else
#           pragma omp critical
# endif
#endif
            result = EXIT_FAILURE;
          }
#if defined(_OPENMP) && !defined(CHECK_PARALLEL_JIT)
# if (201107 <= _OPENMP)
#         pragma omp atomic write
# else
#         pragma omp critical
# endif
#endif
          ++ndup;
        }
      }
#if (0 != LIBXS_JIT)
      else {
# if defined(_DEBUG) || defined(USE_VERBOSE)
        fprintf(stderr, "Error: no code generated for %" PRIuPTR "x%" PRIuPTR "x%" PRIuPTR "-kernel!\n",
          (uintptr_t)m, (uintptr_t)n, (uintptr_t)k);
# endif
# if defined(_OPENMP) && !defined(CHECK_PARALLEL_JIT)
#   if (201107 <= _OPENMP)
#       pragma omp atomic write
#   else
#       pragma omp critical
#   endif
# endif
        result = EXIT_FAILURE;
      }
#endif
    }
  }
#if defined(_DEBUG) || defined(USE_VERBOSE)
  if (0 != ndup) fprintf(stderr, "Info: %i kernel%s duplicated.\n", ndup, 1 != ndup ? "s" : "");
#endif

  /* test unregistering and freeing kernels */
  for (i = 0; i < nkernels; ++i) {
    int j = i + 1;
    /* avoid to double-release kernels */
    for (; j < nkernels; ++j) {
      if (f[i].p == f[j].p) f[j].p = NULL;
    }
    libxs_release_kernel(f[i].p);
  }

  libxs_finalize();

  return result;
}

