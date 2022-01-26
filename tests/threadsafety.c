/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if !defined(INCLUDE_LIBXS_LAST)
# include <libxs.h>
#endif
#include <inttypes.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(INCLUDE_LIBXS_LAST)
# include <libxs.h>
#endif

#if !defined(MAX_NKERNELS)
# define MAX_NKERNELS 800
#endif
#if !defined(CHECK_PARALLEL_INIT)
# define CHECK_PARALLEL_INIT
#endif
#if !defined(CHECK_PARALLEL_JIT)
# define CHECK_PARALLEL_JIT
#endif
#if !defined(CHECK_SEPARATE)
# define CHECK_SEPARATE
#endif
#if !defined(USE_VERBOSE)
# define USE_VERBOSE
#endif
#if !defined(ITYPE)
# define ITYPE float
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif


#if defined(CHECK_SEPARATE)
int test(libxs_blasint /*m*/, libxs_blasint /*n*/, libxs_blasint /*k*/);
int test(libxs_blasint m, libxs_blasint n, libxs_blasint k)
{
  const int flags = LIBXS_GEMM_FLAG_BETA_0;
  LIBXS_MMFUNCTION_TYPE2(ITYPE,OTYPE) kernel;
  int result = EXIT_FAILURE;
#if defined(_OPENMP) && !defined(CHECK_PARALLEL_JIT)
# pragma omp single
#endif
  kernel = LIBXS_MMDISPATCH_SYMBOL2(ITYPE,OTYPE)(m, n, k,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, &flags);
  if (NULL != kernel) {
    libxs_mmkernel_info info;
    libxs_xmmfunction xmm;
    xmm.LIBXS_TPREFIX2(ITYPE,OTYPE,mm) = kernel;
    result = libxs_get_mmkernel_info(xmm, &info);
    if (EXIT_SUCCESS == result) {
      const unsigned int um = (unsigned int)m, un = (unsigned int)n, uk = (unsigned int)k;
      if ( um != info.m || un != info.n || uk != info.k
        || um != info.lda || uk != info.ldb || um != info.ldc
        || LIBXS_DATATYPE(ITYPE) != info.iprecision
        || LIBXS_DATATYPE(OTYPE) != info.oprecision)
      {
#if defined(_DEBUG) || defined(USE_VERBOSE)
        fprintf(stderr, "Error: the %" PRIuPTR "x%" PRIuPTR "x%" PRIuPTR "-kernel does not match!\n",
          (uintptr_t)m, (uintptr_t)n, (uintptr_t)k);
#endif
        result = EXIT_FAILURE;
      }
    }
#if defined(_DEBUG) || defined(USE_VERBOSE)
    else {
      fprintf(stderr, "Error: the %" PRIuPTR "x%" PRIuPTR "x%" PRIuPTR "-kernel is corrupted!\n",
        (uintptr_t)m, (uintptr_t)n, (uintptr_t)k);
    }
#endif
  }
#if !defined(LIBXS_JIT) || (0 == LIBXS_JIT)
  else result = EXIT_SUCCESS;
#endif
  return result;
}
#endif /*defined(CHECK_SEPARATE)*/

int main(void)
{
  union { libxs_xmmfunction x; void* p; } f[MAX_NKERNELS];
  const OTYPE alpha = LIBXS_ALPHA, beta = LIBXS_BETA;
  const int prefetch = LIBXS_PREFETCH_NONE;
  libxs_registry_info registry_info;
  const int max_shape = LIBXS_MAX_M, flags = LIBXS_FLAGS;
  int result = EXIT_SUCCESS, nkernels = MAX_NKERNELS, ndup = 0, i;
#if defined(CHECK_SEPARATE)
  int mnk[3*MAX_NKERNELS] = { 8,8,8, 16,16,8 };
  const int shift = 1, nr = 2; /* nr: predefined triplets */
#endif
  int r[3*MAX_NKERNELS];
#if defined(_OPENMP)
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif

  /* generate set of random number for parallel region */
  for (i = 0; i < (3 * nkernels); i += 3) {
    r[i+0] = rand(); r[i+1] = rand(); r[i+2] = rand();
  }
#if defined(CHECK_SEPARATE)
  /* fill-up set of (m,n,k) for distinct test set */
  for (i = 3 * nr; i < (3 * nkernels); ++i) {
    mnk[i] = (r[i] + shift) % max_shape + 1;
  }
#endif

#if defined(CHECK_PARALLEL_INIT)
# if defined(_OPENMP)
# pragma omp parallel for num_threads(nthreads) private(i)
# endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
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

#if defined(CHECK_SEPARATE)
    for (i = 0; i < nkernels; i += nthreads) {
#if defined(_OPENMP) && defined(CHECK_PARALLEL_JIT)
#     pragma omp parallel num_threads(nthreads)
#endif
      {
#if defined(_OPENMP) && defined(CHECK_PARALLEL_JIT)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const int j = LIBXS_MIN(3 * (i + tid), nkernels - 3);
        const int ri = test(mnk[j+0], mnk[j+1], mnk[j+2]);
        if (EXIT_SUCCESS != ri) {
#if defined(_OPENMP) && defined(CHECK_PARALLEL_JIT)
# if (201107 <= _OPENMP)
#         pragma omp atomic write
# else
#         pragma omp critical
# endif
#endif
          result = ri;
        }
      }
    }
#endif
  }

  if (EXIT_SUCCESS == result) {
#if defined(_OPENMP) && defined(CHECK_PARALLEL_JIT)
#   pragma omp parallel for num_threads(nthreads) private(i)
#endif
    for (i = 0; i < nkernels; ++i) {
      const libxs_blasint m = r[3*i+0] % max_shape + 1;
      const libxs_blasint n = r[3*i+1] % max_shape + 1;
      const libxs_blasint k = r[3*i+2] % max_shape + 1;
      f[i].x.LIBXS_TPREFIX2(ITYPE,OTYPE,mm) = LIBXS_MMDISPATCH_SYMBOL2(ITYPE,OTYPE)(
        m, n, k, &m/*lda*/, &k/*ldb*/, &m/*ldc*/, &flags);
    }
  }

#if defined(_OPENMP) && !defined(CHECK_PARALLEL_JIT)
# pragma omp parallel for num_threads(nthreads) private(i)
#endif
  for (i = 0; i < nkernels; ++i) {
    if (EXIT_SUCCESS == result) {
      const libxs_blasint m = r[3*i+0] % max_shape + 1;
      const libxs_blasint n = r[3*i+1] % max_shape + 1;
      const libxs_blasint k = r[3*i+2] % max_shape + 1;
      union { libxs_xmmfunction x; void* p; } fi;
      libxs_descriptor_blob blob;
      const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init(&blob, LIBXS_DATATYPE2(ITYPE,OTYPE),
        m, n, k, m/*lda*/, k/*ldb*/, m/*ldc*/, &alpha, &beta, flags, prefetch);

      fi.x = libxs_xmmdispatch(desc);
      if (NULL != fi.p && NULL != f[i].p) {
        if (fi.p != f[i].p) {
          libxs_kernel_info a_info, b_info;
          const int ra = libxs_get_kernel_info(f[i].p, &a_info);
          const int rb = libxs_get_kernel_info(fi.p, &b_info);

          /* perform deeper check based on another code generation (used as reference) */
          if (EXIT_SUCCESS == ra && EXIT_SUCCESS == rb && (a_info.code_size != b_info.code_size ||
            0 != memcmp(f[i].p, fi.p, a_info.code_size)))
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
  if (EXIT_SUCCESS == result) {
    for (i = 0; i < nkernels; ++i) {
      int j = i + 1;
      /* avoid to double-release kernels */
      for (; j < nkernels; ++j) {
        if (f[i].p == f[j].p) f[j].p = NULL;
      }
      libxs_release_kernel(f[i].p);
    }
  }

  libxs_finalize();

  return result;
}

