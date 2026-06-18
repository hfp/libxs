/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_malloc.h>

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

static char test_malloc_storage[4 * 65536];
static size_t test_malloc_offset;


static void* test_storage_malloc(size_t size)
{
  void *result = NULL;
  if (test_malloc_offset + size <= sizeof(test_malloc_storage)) {
    result = test_malloc_storage + test_malloc_offset;
  }
  return result;
}


static void test_storage_free(void* pointer)
{
  LIBXS_UNUSED(pointer);
}


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
    static char evict_limit[] = "LIBXS_MALLOC_EVICT_LIMIT=1";
    const int max_nthreads = 1, max_nactive = 1;
    const int nrep_eviction = LIBXS_MALLOC_EVICTWARMUP + 4;
    const size_t nbytes = (size_t)LIBXS_MALLOC_EVICTSIZE + LIBXS_MALLOC_UPSIZE;
    size_t prev_nmallocs = 0;
    int saw_eviction = 0;
    libxs_malloc_pool_t *mpool = libxs_malloc_pool(NULL, NULL);

    nerrors += (EXIT_SUCCESS != LIBXS_PUTENV(evict_limit));
    nerrors += (NULL == mpool);
    for (i = 0; i < nrep_eviction && 0 == nerrors; ++i) {
      libxs_malloc_pool_info_t pinfo;
      libxs_malloc_info_t minfo;
      void *p = NULL;

      p = libxs_malloc(mpool, 0/*nbytes*/, LIBXS_MALLOC_AUTO);
      nerrors += (NULL != p); /* zero-size must return NULL */
      libxs_free(p);

      p = libxs_malloc(mpool, nbytes, LIBXS_MALLOC_AUTO);
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
    nerrors += (0 == saw_eviction);
    libxs_free_pool(mpool);
  }

  /* Test: multiple independent pools */
  if (0 == nerrors) {
    libxs_malloc_pool_t *pool_a = libxs_malloc_pool(NULL, NULL);
    libxs_malloc_pool_t *pool_b = libxs_malloc_pool(NULL, NULL);
    void *pa, *pb;
    libxs_malloc_pool_info_t info_a, info_b;

    nerrors += (NULL == pool_a || NULL == pool_b);
    pa = libxs_malloc(pool_a, 1024, LIBXS_MALLOC_AUTO);
    pb = libxs_malloc(pool_b, 2048, LIBXS_MALLOC_AUTO);
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
    p = libxs_malloc(cpool, 4096, LIBXS_MALLOC_AUTO);
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

  /* Test: standard pool rejects partial custom function pointer pairs */
  if (0 == nerrors) {
    nerrors += (NULL != libxs_malloc_pool(malloc, NULL));
    nerrors += (NULL != libxs_malloc_pool(NULL, free));
  }

  /* Test: explicit alignment (flags > 1) */
  if (0 == nerrors) {
    libxs_malloc_pool_t *apool = libxs_malloc_pool(NULL, NULL);
    void *p;
    libxs_malloc_info_t mi;

    nerrors += (NULL == apool);
    p = libxs_malloc(apool, 4096, 64);
    nerrors += (NULL == p);
    if (NULL != p) {
      nerrors += (0 != ((uintptr_t)p & 63)); /* must be 64-byte aligned */
      memset(p, 0xDD, 4096);
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(p, &mi) || mi.size < 4096);
      libxs_free(p);
    }
    /* larger alignment */
    p = libxs_malloc(apool, 8192, 4096);
    nerrors += (NULL == p);
    if (NULL != p) {
      nerrors += (0 != ((uintptr_t)p & 4095)); /* must be 4096-byte aligned */
      memset(p, 0xEE, 8192);
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(p, &mi) || mi.size < 8192);
      libxs_free(p);
    }
    libxs_free_pool(apool);
  }

  /* Test: inline allocation size overflow is rejected */
  if (0 == nerrors) {
    libxs_malloc_pool_t *opool = libxs_malloc_pool(NULL, NULL);
    const size_t huge = (size_t)-1 - 1024;
    void *p;

    nerrors += (NULL == opool);
    p = libxs_malloc(opool, huge, 4096);
    nerrors += (NULL != p);
    libxs_free(p);
    libxs_free_pool(opool);
  }

  /* Test: reuse refreshes inline metadata when alignment changes */
  if (0 == nerrors) {
    libxs_malloc_pool_t *rpool;
    const size_t big = (size_t)2 * 65536;
    void *p, *q;
    libxs_malloc_info_t mi;
    uintptr_t base, start, aligned64, aligned64k;
    size_t offset;

    test_malloc_offset = 0;
    base = (uintptr_t)test_malloc_storage;
    for (offset = 1; offset < 128 && 0 == test_malloc_offset; ++offset) {
      start = base + offset + sizeof(void*);
      aligned64 = (start + 63) & ~(uintptr_t)63;
      aligned64k = (start + 65535) & ~(uintptr_t)65535;
      if (aligned64 != aligned64k) test_malloc_offset = offset;
    }
    nerrors += (0 == test_malloc_offset);
    rpool = libxs_malloc_pool(test_storage_malloc, test_storage_free);
    nerrors += (NULL == rpool);
    p = libxs_malloc(rpool, big, 64);
    nerrors += (NULL == p);
    if (NULL != p) {
      memset(p, 0, big);
      libxs_free(p);
    }
    q = libxs_malloc(rpool, 1024, 65536);
    nerrors += (NULL == q);
    if (NULL != q) {
      memset(q, 0xAB, 1024);
      nerrors += (0 != ((uintptr_t)q & 65535));
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(q, &mi) || mi.size < 1024);
      libxs_free(q);
    }
    libxs_free_pool(rpool);
  }

  /* Test: reuse updates eviction accounting for larger requested sizes */
  if (0 == nerrors) {
    libxs_malloc_pool_t *ppool = libxs_malloc_pool(NULL, NULL);
    libxs_malloc_pool_info_t pi;
    void *p;

    nerrors += (NULL == ppool);
    p = libxs_malloc(ppool, 1024, 65536);
    nerrors += (NULL == p);
    if (NULL != p) libxs_free(p);
    p = libxs_malloc(ppool, 8192, 64);
    nerrors += (NULL == p);
    if (NULL != p) {
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(ppool, &pi) || pi.peak < 8192);
      libxs_free(p);
    }
    libxs_free_pool(ppool);
  }

  /* Test: LIBXS_MALLOC_NATIVE (registry-based, pointer preserved) */
  if (0 == nerrors) {
    libxs_malloc_pool_t *natpool = libxs_malloc_pool(malloc, free);
    void *p;
    libxs_malloc_info_t mi;
    libxs_malloc_pool_info_t pi;

    nerrors += (NULL == natpool);
    p = libxs_malloc(natpool, 4096, LIBXS_MALLOC_NATIVE);
    nerrors += (NULL == p);
    if (NULL != p) {
      memset(p, 0xAA, 4096);
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(p, &mi) || mi.size < 4096);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(natpool, &pi) || 1 != pi.nactive);
      libxs_free(p);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(natpool, &pi) || 0 != pi.nactive);
    }
    /* allocate, free, re-allocate (reuse path) */
    p = libxs_malloc(natpool, 2048, LIBXS_MALLOC_NATIVE);
    nerrors += (NULL == p);
    if (NULL != p) {
      memset(p, 0xBB, 2048);
      libxs_free(p);
    }
    p = libxs_malloc(natpool, 1024, LIBXS_MALLOC_NATIVE);
    nerrors += (NULL == p);
    if (NULL != p) {
      memset(p, 0xCC, 1024);
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(p, &mi) || mi.size < 1024);
      libxs_free(p);
    }
    /* grow: allocate larger than previous chunk */
    p = libxs_malloc(natpool, 8192, LIBXS_MALLOC_NATIVE);
    nerrors += (NULL == p);
    if (NULL != p) {
      memset(p, 0xDD, 8192);
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(p, &mi) || mi.size < 8192);
      libxs_free(p);
    }
    /* multiple concurrent native allocations */
    { void *a, *b, *c;
      a = libxs_malloc(natpool, 512, LIBXS_MALLOC_NATIVE);
      b = libxs_malloc(natpool, 1024, LIBXS_MALLOC_NATIVE);
      c = libxs_malloc(natpool, 2048, LIBXS_MALLOC_NATIVE);
      nerrors += (NULL == a || NULL == b || NULL == c);
      nerrors += (a == b || b == c || a == c); /* must be distinct */
      if (NULL != a) memset(a, 0x11, 512);
      if (NULL != b) memset(b, 0x22, 1024);
      if (NULL != c) memset(c, 0x33, 2048);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(natpool, &pi) || 3 != pi.nactive);
      libxs_free(a); libxs_free(b); libxs_free(c);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(natpool, &pi) || 0 != pi.nactive);
    }
    libxs_free_pool(natpool);
  }

  /* Test: extended pool (libxs_malloc_xpool) with per-thread extra arg */
  if (0 == nerrors) {
    const int sentinel = 42;
    libxs_malloc_pool_t *xpool;
    libxs_malloc_pool_info_t pi;
    libxs_malloc_info_t mi;
    void *p;

    xpool = libxs_malloc_xpool(
      /* malloc_xfn */ test_xmalloc,
      /* free_xfn */   test_xfree,
      /* max_nthreads */ 4);
    nerrors += (NULL == xpool);

    /* set per-thread extra for this thread */
    libxs_malloc_arg(xpool, &sentinel);

    p = libxs_malloc(xpool, 2048, LIBXS_MALLOC_AUTO);
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

  /* Test: NULL pool uses the default pool */
  if (0 == nerrors) {
    libxs_malloc_pool_info_t pinfo;
    libxs_malloc_info_t minfo;
    void *const p = libxs_malloc(NULL, 1024, LIBXS_MALLOC_AUTO);
    nerrors += (NULL == p);
    if (NULL != p) {
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(p, &minfo) || minfo.size < 1024);
      nerrors += (EXIT_SUCCESS != libxs_malloc_pool_info(NULL, &pinfo) || 1 > pinfo.nactive);
      libxs_free(p);
    }
  }

  /* Test: best-fit slot selection (smallest cached chunk that fits) */
  if (0 == nerrors) {
    libxs_malloc_pool_t *bpool = libxs_malloc_pool(malloc, free);
    void *a, *b, *c, *x, *y;
    libxs_malloc_info_t mi;

    nerrors += (NULL == bpool);
    a = libxs_malloc(bpool, 100 * 1024, LIBXS_MALLOC_NATIVE);
    b = libxs_malloc(bpool, 200 * 1024, LIBXS_MALLOC_NATIVE);
    c = libxs_malloc(bpool, 1000 * 1024, LIBXS_MALLOC_NATIVE);
    nerrors += (NULL == a || NULL == b || NULL == c);
    libxs_free(a);
    libxs_free(b);
    libxs_free(c);
    x = libxs_malloc(bpool, 150 * 1024, LIBXS_MALLOC_NATIVE);
    nerrors += (NULL == x);
    if (NULL != x) {
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(x, &mi));
      nerrors += (mi.size < 150 * 1024);
      nerrors += (mi.size > 250 * 1024);
    }
    y = libxs_malloc(bpool, 900 * 1024, LIBXS_MALLOC_NATIVE);
    nerrors += (NULL == y);
    if (NULL != y) {
      nerrors += (EXIT_SUCCESS != libxs_malloc_info(y, &mi));
      nerrors += (mi.size < 900 * 1024);
    }
    if (NULL != x) libxs_free(x);
    if (NULL != y) libxs_free(y);
    libxs_free_pool(bpool);
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
