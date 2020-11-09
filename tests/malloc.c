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

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) fprintf(STREAM, __VA_ARGS__)
#else
# define FPRINTF(STREAM, ...)
#endif


int main(void)
{
  const size_t size_malloc = 2507, alignment = (2U << 20);
  libxs_malloc_info malloc_info;
  int avalue, nerrors = 0, n;
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
      FPRINTF(stderr, "Error: incorrect scratch allocator setup!\n");
      ++nerrors;
    }
  }
#endif

  /* allocate some amount of memory */
  p = libxs_malloc(size_malloc);

  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxs_get_malloc_info(p, &malloc_info) || malloc_info.size < size_malloc)) {
    FPRINTF(stderr, "Error: buffer info (1/4) failed!\n");
    ++nerrors;
  }

#if defined(CHECK_SCRATCH)
  q = libxs_aligned_scratch(size_scratch, 0/*auto*/);
  libxs_free(q);
  q = libxs_aligned_scratch(size_scratch / 3, 0/*auto*/);
  r = libxs_aligned_scratch(size_scratch / 3, 0/*auto*/);
  /* confirm malloc succeeds for an in-scratch buffer */
  if (NULL != q && NULL == r) {
    FPRINTF(stderr, "Error: in-scratch buffer allocation failed!\n");
    ++nerrors;
  }
#endif

#if defined(CHECK_REALLOC)
  if (NULL != p) { /* reallocate larger amount of memory */
    unsigned char* c = (unsigned char*)p;
    size_t i;
    avalue = 1 << LIBXS_INTRINSICS_BITSCANFWD64((uintptr_t)p);
    for (i = 0; i < size_malloc; ++i) c[i] = (unsigned char)LIBXS_MOD2(i, 256);
    p = libxs_realloc(size_malloc * 2, p);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)p) % avalue)) {
      FPRINTF(stderr, "Error: buffer alignment (1/3) not preserved!\n");
      ++nerrors;
    }
    c = (unsigned char*)p;
    for (i = size_malloc; i < (size_malloc * 2); ++i) c[i] = (unsigned char)LIBXS_MOD2(i, 256);
    /* reallocate again with same size */
    p = libxs_realloc(size_malloc * 2, p);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)p) % avalue)) {
      FPRINTF(stderr, "Error: buffer alignment (2/3) not preserved!\n");
      ++nerrors;
    }
    c = (unsigned char*)p;
    for (i = n = 0; i < (size_malloc * 2); ++i) { /* check that content is preserved */
      n += (c[i] == (unsigned char)LIBXS_MOD2(i, 256) ? 0 : 1);
    }
    if (0 < n) {
      FPRINTF(stderr, "Error: buffer content (1/2) not preserved!\n");
      nerrors += n;
    }
    /* reallocate with smaller size */
    p = libxs_realloc(size_malloc / 2, p);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)p) % avalue)) {
      FPRINTF(stderr, "Error: buffer alignment (3/3) not preserved!\n");
      ++nerrors;
    }
    c = (unsigned char*)p;
    for (i = n = 0; i < size_malloc / 2; ++i) { /* check that content is preserved */
      n += (c[i] == (unsigned char)LIBXS_MOD2(i, 256) ? 0 : 1);
    }
    if (0 < n) {
      FPRINTF(stderr, "Error: buffer content (2/2) not preserved!\n");
      nerrors += n;
    }
  }
  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxs_get_malloc_info(p, &malloc_info) || malloc_info.size < (size_malloc / 2))) {
    FPRINTF(stderr, "Error: buffer info (2/4) failed!\n");
    ++nerrors;
  }
  libxs_free(p); /* release buffer */
  /* check degenerated reallocation */
  p = libxs_realloc(size_malloc, NULL/*allocation*/);
  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxs_get_malloc_info(p, &malloc_info) || malloc_info.size < size_malloc)) {
    FPRINTF(stderr, "Error: buffer info (3/4) failed!\n");
    ++nerrors;
  }
#endif

  /* check that a NULL-pointer yields no size */
  if (EXIT_SUCCESS != libxs_get_malloc_info(NULL, &malloc_info) || 0 != malloc_info.size) {
    FPRINTF(stderr, "Error: buffer info (4/4) failed!\n");
    ++nerrors;
  }

  /* release NULL pointer */
  libxs_free(NULL);

  /* release buffer */
  libxs_free(p);

  /* allocate memory with specific alignment */
  p = libxs_aligned_malloc(size_malloc, alignment);
  /* check function that determines alignment */
  libxs_aligned(p, NULL/*inc*/, &avalue);

  /* check the alignment of the allocation */
  if (0 != (((uintptr_t)p) % alignment) || ((size_t)avalue) < alignment) {
    FPRINTF(stderr, "Error: buffer alignment (1/3) incorrect!\n");
    ++nerrors;
  }

  if (libxs_aligned(p, NULL/*inc*/, NULL/*alignment*/)) { /* pointer is SIMD-aligned */
    if (alignment < ((size_t)4 * libxs_cpuid_vlen32(libxs_get_target_archid()))) {
      FPRINTF(stderr, "Error: buffer alignment (2/3) incorrect!\n");
      ++nerrors;
    }
  }
  else { /* pointer is not SIMD-aligned */
    if (((size_t)4 * libxs_cpuid_vlen32(libxs_get_target_archid())) <= alignment) {
      FPRINTF(stderr, "Error: buffer alignment (3/3) incorrect!\n");
      ++nerrors;
    }
  }

  /* release memory */
  libxs_free(p);
#if defined(CHECK_SCRATCH)
  libxs_free(q);
  libxs_free(r);
#endif

  /* check foreign memory */
  if (EXIT_SUCCESS == libxs_get_malloc_info(&size_malloc/*faulty pointer*/, &malloc_info)) {
    FPRINTF(stderr, "Error: uncaught faulty pointer!\n");
    ++nerrors;
  }

  if (0 == nerrors) {
    return EXIT_SUCCESS;
  }
  else {
    FPRINTF(stderr, "Errors: %i\n", nerrors);
    return EXIT_FAILURE;
  }
}

