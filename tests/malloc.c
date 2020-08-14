/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs.h>
#include <libxs_intrinsics_x86.h>

#if !defined(CHECK_SETUP) && 1
# define CHECK_SETUP
#endif
#if !defined(CHECK_REALLOC) && 1
# define CHECK_REALLOC
#endif
#if !defined(CHECK_SCRATCH) && 1
# define CHECK_SCRATCH
#endif


int main(void)
{
  const size_t size_malloc = 2507, alignment = (2U << 20);
  libxs_malloc_info malloc_info;
  int nerrors = 0;
  void *p;
#if defined(CHECK_SCRATCH)
  const size_t size_scratch = (24U << 20);
  void *q, *r;
#endif

#if defined(CHECK_SETUP)
  { /* check allocator setup */
    libxs_malloc_function malloc_fn;
    libxs_free_function free_fn;
    const void* context;
    malloc_fn.function = malloc; free_fn.function = free;
    libxs_set_default_allocator(NULL/*context*/, malloc_fn/*malloc*/, free_fn/*free*/);
    malloc_fn.function = NULL; free_fn.function = NULL;
    libxs_set_scratch_allocator(NULL/*context*/, malloc_fn/*NULL*/, free_fn/*NULL*/);

    /* check adoption of the default allocator */
    libxs_get_scratch_allocator(&context, &malloc_fn, &free_fn);
    if (NULL != context || malloc != malloc_fn.function || free != free_fn.function) {
      ++nerrors;
    }
  }
#endif

  /* allocate some amount of memory */
  p = libxs_malloc(size_malloc);

  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxs_get_malloc_info(p, &malloc_info) || malloc_info.size < size_malloc)) {
    ++nerrors;
  }

#if defined(CHECK_SCRATCH)
  q = libxs_aligned_scratch(size_scratch, 0/*auto*/);
  libxs_free(q);
  q = libxs_aligned_scratch(size_scratch / 3, 0/*auto*/);
  r = libxs_aligned_scratch(size_scratch / 3, 0/*auto*/);
  /* confirm malloc-info fails for an in-scratch buffer */
  if (NULL != r && EXIT_SUCCESS == libxs_get_malloc_info(r, &malloc_info)) {
    ++nerrors;
  }
#endif

#if defined(CHECK_REALLOC)
  if (NULL != p) { /* reallocate larger amount of memory */
    const int palign = 1 << LIBXS_INTRINSICS_BITSCANFWD64((uintptr_t)p);
    unsigned char* c = (unsigned char*)p;
    size_t i;
    for (i = 0; i < size_malloc; ++i) c[i] = (unsigned char)LIBXS_MOD2(i, 256);
    p = libxs_realloc(size_malloc * 2, p);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)p) % palign)) {
      ++nerrors;
    }
    c = (unsigned char*)p;
    for (i = size_malloc; i < (size_malloc * 2); ++i) c[i] = (unsigned char)LIBXS_MOD2(i, 256);
    /* reallocate again with same size */
    p = libxs_realloc(size_malloc * 2, p);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)p) % palign)) {
      ++nerrors;
    }
    c = (unsigned char*)p;
    for (i = 0; i < (size_malloc * 2); ++i) { /* check that content is preserved */
      nerrors += (c[i] == (unsigned char)LIBXS_MOD2(i, 256) ? 0 : 1);
    }
    /* reallocate with smaller size */
    p = libxs_realloc(size_malloc / 2, p);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)p) % palign)) {
      ++nerrors;
    }
    c = (unsigned char*)p;
    for (i = 0; i < size_malloc / 2; ++i) { /* check that content is preserved */
      nerrors += (c[i] == (unsigned char)LIBXS_MOD2(i, 256) ? 0 : 1);
    }
  }
  /* query and check the size_malloc of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxs_get_malloc_info(p, &malloc_info) || malloc_info.size < (size_malloc / 2))) {
    ++nerrors;
  }
  libxs_free(p); /* release buffer */
  /* check degenerated reallocation */
  p = libxs_realloc(size_malloc, NULL/*allocation*/);
  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxs_get_malloc_info(p, &malloc_info) || malloc_info.size < size_malloc)) {
    ++nerrors;
  }
#endif

  /* check that a NULL-pointer yields no size */
  if (EXIT_SUCCESS != libxs_get_malloc_info(NULL, &malloc_info) || 0 != malloc_info.size) {
    ++nerrors;
  }

  /* release NULL pointer */
  libxs_free(NULL);

  /* release buffer */
  libxs_free(p);

  /* allocate memory with specific alignment */
  p = libxs_aligned_malloc(size_malloc, alignment);

  /* check the alignment of the allocation */
  if (0 != (((uintptr_t)p) % alignment)) {
    ++nerrors;
  }

  /* release memory */
  libxs_free(p);
#if defined(CHECK_SCRATCH)
  libxs_free(q);
  libxs_free(r);
#endif

  /* check foreign memory */
  if (EXIT_SUCCESS == libxs_get_malloc_info(&size_malloc/*faulty pointer*/, &malloc_info)) {
    ++nerrors;
  }

  return 0 == nerrors ? EXIT_SUCCESS : EXIT_FAILURE;
}

