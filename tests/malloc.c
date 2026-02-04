/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_malloc.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(void)
{
  const size_t size_malloc = 2507, alignment = (2U << 20);
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
#     pragma omp parallel for private(j) schedule(dynamic,1)
# endif
    for (j = 0; j < npool; ++j) {
      void *const p = libxs_pmalloc(pool, &num);
      LIBXS_EXPECT(NULL != p);
    }
# if defined(_OPENMP)
#     pragma omp parallel for private(j) schedule(dynamic,1)
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
    return EXIT_SUCCESS;
  }
  else {
    FPRINTF(stderr, "Errors: %i\n", nerrors);
    return EXIT_FAILURE;
  }
}
