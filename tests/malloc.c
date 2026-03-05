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


static void* test_xmalloc(size_t size, const void* extra)
{
  LIBXS_UNUSED(extra);
  return malloc(size);
}

static void test_xfree(void* pointer, const void* extra)
{
  LIBXS_UNUSED(extra);
  free(pointer);
}


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
    libxs_malloc_pool_t *mpool = libxs_malloc_pool(NULL, NULL);

    nerrors += (NULL == mpool);
    for (i = 0; i < nrep_eviction && 0 == nerrors; ++i) {
      libxs_malloc_pool_info_t pinfo;
      libxs_malloc_info_t minfo;
      void *p = NULL;

      p = libxs_malloc(mpool, 0/*nbytes*/, 0/*auto*/);
      nerrors += (NULL != p); /* zero-size must return NULL */
      libxs_free(p);

      p = libxs_malloc(mpool, nbytes, 0/*auto*/);
      if (NULL != p) {
        memset(p, 0xA5, LIBXS_MIN(nbytes, (size_t)4096));
      }
      else {
        ++nerrors; break;
      }
      if (EXIT_SUCCESS != libxs_malloc_info(p, &minfo) || minfo.size < nbytes) {
        ++nerrors; break;
      }
      if (EXIT_SUCCESS != libxs_malloc_pool_info(mpool, &pinfo) || prev_nmallocs > pinfo.nmallocs) {
        ++nerrors; break;
      }
      prev_nmallocs = pinfo.nmallocs;
      libxs_free(p);
      if (EXIT_SUCCESS != libxs_malloc_pool_info(mpool, &pinfo)) {
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
    libxs_free_pool(mpool);
  }

  /* Test: multiple independent pools */
  if (0 == nerrors) {
    libxs_malloc_pool_t *pool_a = libxs_malloc_pool(NULL, NULL);
    libxs_malloc_pool_t *pool_b = libxs_malloc_pool(NULL, NULL);
    void *pa, *pb;
    libxs_malloc_pool_info_t info_a, info_b;

    nerrors += (NULL == pool_a || NULL == pool_b);
    pa = libxs_malloc(pool_a, 1024, 0);
    pb = libxs_malloc(pool_b, 2048, 0);
    nerrors += (NULL == pa || NULL == pb);
    if (0 == nerrors) {
      libxs_malloc_info_t mi;
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(pa, &mi) || mi.size < 1024);
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(pb, &mi) || mi.size < 2048);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(pool_a, &info_a));
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(pool_b, &info_b));
      nerrors += (1 != info_a.nactive || 1 != info_b.nactive);
    }
    libxs_free(pa);
    libxs_free(pb);
    /* after free, nactive should be 0 for both pools */
    if (0 == nerrors) {
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(pool_a, &info_a));
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(pool_b, &info_b));
      nerrors += (0 != info_a.nactive || 0 != info_b.nactive);
    }
    libxs_free_pool(pool_a);
    libxs_free_pool(pool_b);
  }

  /* Test: custom malloc/free function pointers */
  if (0 == nerrors) {
    libxs_malloc_pool_t *cpool = libxs_malloc_pool(malloc, free);
    void *p;
    libxs_malloc_info_t mi;
    libxs_malloc_pool_info_t pi;

    nerrors += (NULL == cpool);
    p = libxs_malloc(cpool, 4096, 64);
    nerrors += (NULL == p);
    if (NULL != p) {
      memset(p, 0xCC, 4096);
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(p, &mi) || mi.size < 4096);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(cpool, &pi) || 1 != pi.nactive);
      libxs_free(p);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(cpool, &pi) || 0 != pi.nactive);
    }
    libxs_free_pool(cpool);
  }

  /* Test: extended pool (libxs_malloc_xpool) with per-thread extra arg */
  if (0 == nerrors) {
    static int xmalloc_extra_ok = 0, xfree_extra_ok = 0;
    const int sentinel = 42;
    libxs_malloc_pool_t *xpool;
    void *p;
    libxs_malloc_info_t mi;
    libxs_malloc_pool_info_t pi;

    xpool = libxs_malloc_xpool(
      /* malloc_xfn */ test_xmalloc,
      /* free_xfn */   test_xfree,
      /* max_nthreads */ 4);
    nerrors += (NULL == xpool);

    /* set per-thread extra for this thread */
    libxs_malloc_arg(xpool, &sentinel);

    p = libxs_malloc(xpool, 2048, 0);
    nerrors += (NULL == p);
    if (NULL != p) {
      memset(p, 0xBB, 2048);
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(p, &mi) || mi.size < 2048);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(xpool, &pi) || 1 != pi.nactive);
      libxs_free(p);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(xpool, &pi) || 0 != pi.nactive);
    }
    libxs_free_pool(xpool);
  }

  /* Test: libxs_malloc_xpool rejects NULL fn or zero nthreads */
  if (0 == nerrors) {
    nerrors += (NULL != libxs_malloc_xpool(NULL, test_xfree, 4));
    nerrors += (NULL != libxs_malloc_xpool(test_xmalloc, NULL, 4));
    nerrors += (NULL != libxs_malloc_xpool(test_xmalloc, test_xfree, 0));
  }

  /* Test: libxs_malloc_arg on standard pool is a no-op */
  if (0 == nerrors) {
    libxs_malloc_pool_t *spool = libxs_malloc_pool(NULL, NULL);
    int dummy = 0;
    nerrors += (NULL == spool);
    libxs_malloc_arg(spool, &dummy); /* must not crash */
    libxs_free_pool(spool);
  }

  /* Test: NULL pool returns NULL */
  if (0 == nerrors) {
    nerrors += (NULL != libxs_malloc(NULL, 1024, 0));
  }

  /* Test: libxs_free(NULL) is safe */
  libxs_free(NULL);

  /* Test: libxs_free_pool(NULL) is safe */
  libxs_free_pool(NULL);

  if (0 == nerrors) {
    return EXIT_SUCCESS;
  }
  else {
    FPRINTF(stderr, "Errors: %i\n", nerrors);
    return EXIT_FAILURE;
  }
}
