/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_malloc.h>

#if !defined(LIBXS_MALLOC_UPSIZE)
# define LIBXS_MALLOC_UPSIZE (2 << 20)
#endif
#if !defined(LIBXS_MALLOC_EVICTSIZE)
# define LIBXS_MALLOC_EVICTSIZE (8 * LIBXS_MALLOC_UPSIZE)
#endif
#if !defined(LIBXS_MALLOC_EVICTWARMUP)
# define LIBXS_MALLOC_EVICTWARMUP 4
#endif

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(void)
{
  int nerrors = 0;
  void* pool[128];
  char storage[8*sizeof(pool)/sizeof(*pool)];
  void* backup[sizeof(pool)];
  const int npool = sizeof(pool) / sizeof(*pool), nrep = 1024;
  size_t num = npool;
  int i, j;

  libxs_pmalloc_init(sizeof(storage) / num, &num, pool, storage);
  memcpy(backup, pool, sizeof(pool));

  for (i = 0; i < nrep; ++i) {
# if defined(_OPENMP)
#   pragma omp parallel for private(j) schedule(dynamic,1)
# endif
    for (j = 0; j < npool; ++j) {
      void *const p = libxs_pmalloc(pool, &num);
      LIBXS_EXPECT(NULL != p);
    }
# if defined(_OPENMP)
#   pragma omp parallel for private(j) schedule(dynamic,1)
# endif
    for (j = 0; j < npool; ++j) {
      const int k = npool - j - 1;
      void *const p = backup[k];
      libxs_pfree(p, pool, &num);
    }
    if (npool != (int)num) break;
  }
  if (npool == (int)num) {
    for (i = 0, num = 0; i < npool; ++i) {
      const void *const p = backup[i];
      for (j = 0; j < npool; ++j) {
        if (p == pool[j]) {
          ++num; break;
        }
      }
    }
    nerrors += LIBXS_DELTA(npool, (int)num);
  }
  else ++nerrors;

  if (0 == nerrors) {
    const int max_nthreads = 1, max_nactive = 1;
    const int nrep_eviction = LIBXS_MALLOC_EVICTWARMUP + 4;
    const size_t nbytes = (size_t)LIBXS_MALLOC_EVICTSIZE + LIBXS_MALLOC_UPSIZE;
    size_t prev_nmallocs = 0;
    int saw_eviction = 0;

    libxs_malloc_pool(max_nthreads, max_nactive);
    for (i = 0; i < nrep_eviction; ++i) {
      libxs_malloc_pool_info_t pinfo;
      libxs_malloc_info_t minfo;
      void *p = NULL;

      p = libxs_malloc(0/*nbytes*/, 0/*auto*/);
      libxs_free(p);

      p = libxs_malloc(nbytes, 0/*auto*/);
      if (NULL != p) {
        memset(p, 0xA5, LIBXS_MIN(nbytes, (size_t)4096));
      }
      else {
        ++nerrors; break;
      }
      if (EXIT_SUCCESS != libxs_malloc_info(p, &minfo) || minfo.size < nbytes) {
        ++nerrors; break;
      }
      if (EXIT_SUCCESS != libxs_malloc_pool_info(&pinfo) || prev_nmallocs > pinfo.nmallocs) {
        ++nerrors; break;
      }
      prev_nmallocs = pinfo.nmallocs;
      libxs_free(p);
      if (EXIT_SUCCESS != libxs_malloc_pool_info(&pinfo)) {
        ++nerrors; break;
      }
      if ((size_t)(max_nthreads * max_nactive) * LIBXS_MALLOC_EVICTWARMUP <= pinfo.nmallocs
        && pinfo.size < nbytes)
      {
        saw_eviction = 1;
      }
    }
    nerrors += (0 == prev_nmallocs);
    LIBXS_UNUSED(saw_eviction);
    libxs_free_pool();
  }

  if (0 == nerrors) {
    return EXIT_SUCCESS;
  }
  else {
    FPRINTF(stderr, "Errors: %i\n", nerrors);
    return EXIT_FAILURE;
  }
}
