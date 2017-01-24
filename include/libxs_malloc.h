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


/** Function type accepted for memory allocation (see libxs_set_*_allocator). */
typedef union LIBXS_RETARGETABLE libxs_malloc_function {
  void* (*ctx_form)(void* context, size_t size);
  void* (*function)(size_t size);
} libxs_malloc_function;

/** Function type accepted for memory release (see libxs_set_*_allocator). */
typedef union LIBXS_RETARGETABLE libxs_free_function {
  void (*ctx_form)(void* context, void* buffer);
  void (*function)(void* buffer);
} libxs_free_function;

/**
 * To setup the custom default memory allocator, either a malloc_fn and a free_fn
 * are given, or two NULL-pointers designate to reset the default allocator to a
 * library-internal default. If a context is given (non-NULL ), the context-based
 * form of the memory allocation is used.
 * It is supported to change the allocator while buffers are pending.
 */
LIBXS_API void libxs_set_default_allocator(/* malloc_fn/free_fn must correspond */
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Retrieve the default memory allocator. */
LIBXS_API void libxs_get_default_allocator(void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/**
 * To setup the scratch memory allocator, a malloc_fn function and an optional free_fn
 * are given. A NULL-free acts as a "no-operation", and the deallocation is expected
 * to be controlled otherwise. If two NULL-pointers are given, the allocator is reset
 * to the currently active default memory allocator. If a context is given (non-NULL),
 * the context-based form of the memory allocation is used.
 * It is supported to change the allocator while buffers are pending.
 */
LIBXS_API void libxs_set_scratch_allocator(/* malloc_fn/free_fn must correspond */
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Retrieve the scratch memory allocator. */
LIBXS_API void libxs_get_scratch_allocator(void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/** Allocate aligned default memory. */
LIBXS_API void* libxs_aligned_malloc(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment);

/** Allocate aligned scratch memory. */
LIBXS_API void* libxs_aligned_scratch(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment);

/** Allocate memory (malloc/free interface). */
LIBXS_API void* libxs_malloc(size_t size);

/** Deallocate memory (malloc/free interface). */
LIBXS_API void libxs_free(const void* memory);

/** Get the size of the allocated memory; zero in case of an error. */
LIBXS_API size_t libxs_malloc_size(const void* memory);


#if defined(__cplusplus)

/** RAII idiom to temporarily setup an allocator for the lifetime of the scope. */
template<allocator_kind> class LIBXS_RETARGETABLE libxs_scoped_allocator {
public:
  /** Following the RAII idiom, the c'tor instantiates the new allocator. */
  libxs_scoped_allocator(void* context,
    libxs_malloc_function malloc_fn, libxs_free_function free_fn)
  :
    m_context(0), m_malloc(0), m_free(0)
  {
    allocator_kind::get(m_context, m_malloc, m_free);
    allocator_kind::set(context, malloc_fn, free_fn);
  }

  /** Following the RAII idiom, the d'tor restores the previous allocator. */
  ~libxs_scoped_allocator() {
    allocator_kind::set(m_context, m_malloc, m_free);
  }

private: /* no copy/assignment */
  explicit libxs_scoped_allocator(const libxs_scoped_allocator&);
  libxs_scoped_allocator& operator=(const libxs_scoped_allocator&);

private: /* saved/previous allocator */
  void* m_context;
  libxs_malloc_function m_malloc;
  libxs_free_function m_free;
};

/** Wrap default allocator to act as an allocator-kind (libxs_scoped_allocator). */
struct LIBXS_RETARGETABLE libxs_default_allocator {
  static void set(void* context,
    libxs_malloc_function malloc_fn, libxs_free_function free_fn)
  {
    libxs_set_default_allocator(context, malloc_fn, free_fn);
  }
  static void get(void*& context,
    libxs_malloc_function& malloc_fn, libxs_free_function& free_fn)
  {
    libxs_get_default_allocator(&context, &malloc_fn, &free_fn);
  }
};

/** Wrap scratch allocator to act as an allocator-kind (libxs_scoped_allocator). */
struct LIBXS_RETARGETABLE libxs_scratch_allocator {
  static void set(void* context,
    libxs_malloc_function malloc_fn, libxs_free_function free_fn)
  {
    libxs_set_scratch_allocator(context, malloc_fn, free_fn);
  }
  static void get(void*& context,
    libxs_malloc_function& malloc_fn, libxs_free_function& free_fn)
  {
    libxs_get_scratch_allocator(&context, &malloc_fn, &free_fn);
  }
};

#endif /*defined(__cplusplus)*/
#endif /*LIBXS_MALLOC_H*/
