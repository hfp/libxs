/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
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
#ifndef LIBXS_MALLOC_H
#define LIBXS_MALLOC_H

#include "libxs_macros.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stddef.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/** Function type accepted for memory allocation (libxs_set_allocator). */
typedef LIBXS_RETARGETABLE void* (*libxs_malloc_function)(size_t size);
/** Function type accepted for memory release (libxs_set_allocator). */
typedef LIBXS_RETARGETABLE void (*libxs_free_function)(void* buffer);

/**
 * Setup the memory allocator, the malloc_fn and free_fn arguments are either
 * non-NULL functions (custom allocator), or NULL-pointers (reset to default).
 * It is supported to change the allocator with buffers still being allocated.
 */
LIBXS_API void libxs_set_allocator(/* malloc_fn/free_fn must correspond */
  libxs_malloc_function malloc_fn, libxs_free_function free_fn);

/** Allocate aligned memory (malloc/free interface). */
LIBXS_API void* libxs_aligned_malloc(size_t size,
  /**
   * =0: automatic alignment is requested based on size
   * 0>: uses the requested alignment
   */
  size_t alignment);

/** Allocate memory (malloc/free interface). */
LIBXS_API void* libxs_malloc(size_t size);

/** Deallocate memory (malloc/free interface). */
LIBXS_API void libxs_free(const void* memory);

/** Get the size of the allocated memory; zero in case of an error. */
LIBXS_API size_t libxs_malloc_size(const void* memory);

#endif /*LIBXS_MALLOC_H*/
