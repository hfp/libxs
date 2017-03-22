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


/** Function types accepted for memory allocation (see libxs_*_allocator). */
typedef LIBXS_RETARGETABLE void* (*libxs_malloc_ctx)(void* /*context*/, size_t /*size*/);
typedef LIBXS_RETARGETABLE void* (*libxs_malloc_fun)(size_t /*size*/);
typedef union LIBXS_RETARGETABLE libxs_malloc_function {
  libxs_malloc_ctx ctx_form;
  libxs_malloc_fun function;
} libxs_malloc_function;

/** Function types accepted for releasing memory (see libxs_*_allocator). */
typedef LIBXS_RETARGETABLE void (*libxs_free_ctx)(void* /*context*/, void* /*buffer*/);
typedef LIBXS_RETARGETABLE void (*libxs_free_fun)(void* /*buffer*/);
typedef union LIBXS_RETARGETABLE libxs_free_function {
  libxs_free_ctx ctx_form;
  libxs_free_fun function;
} libxs_free_function;

LIBXS_INLINE LIBXS_RETARGETABLE libxs_malloc_function libxs_make_malloc_fun(libxs_malloc_fun malloc_fn) {
  libxs_malloc_function result; result.function = malloc_fn; return result;
}
LIBXS_INLINE LIBXS_RETARGETABLE libxs_free_function libxs_make_free_fun(libxs_free_fun free_fn) {
  libxs_free_function result; result.function = free_fn; return result;
}
LIBXS_INLINE LIBXS_RETARGETABLE libxs_malloc_function libxs_make_malloc_ctx(libxs_malloc_ctx malloc_fn) {
  libxs_malloc_function result; result.ctx_form = malloc_fn; return result;
}
LIBXS_INLINE LIBXS_RETARGETABLE libxs_free_function libxs_make_free_ctx(libxs_free_ctx free_fn) {
  libxs_free_function result; result.ctx_form = free_fn; return result;
}

/**
 * To setup the custom default memory allocator, either a malloc_fn and a free_fn
 * are given, or two NULL-pointers designate to reset the default allocator to a
 * library-internal default. If a context is given (non-NULL ), the context-based
 * form of the memory allocation is used.
 * It is supported to change the allocator while buffers are pending.
 */
LIBXS_API int libxs_set_default_allocator(/* malloc_fn/free_fn must correspond */
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Retrieve the default memory allocator. */
LIBXS_API int libxs_get_default_allocator(void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/**
 * To setup the scratch memory allocator, a malloc_fn function and an optional free_fn
 * are given. A NULL-free acts as a "no-operation", and the deallocation is expected
 * to be controlled otherwise. If two NULL-pointers are given, the allocator is reset
 * to the currently active default memory allocator. If a context is given (non-NULL),
 * the context-based form of the memory allocation is used.
 * It is supported to change the allocator while buffers are pending.
 */
LIBXS_API int libxs_set_scratch_allocator(/* malloc_fn/free_fn must correspond */
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Retrieve the scratch memory allocator. */
LIBXS_API int libxs_get_scratch_allocator(void** context,
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

/**
 * Release the scratch memory pool i.e., scratch memory
 * for which libxs_free has been called (non-pending).
 */
LIBXS_API void libxs_release_scratch(size_t* npending);

/** Get the size of the allocated memory; zero in case of an error. */
LIBXS_API size_t libxs_malloc_size(const void* memory);

/** Get the size of the allocated scratch memory. */
LIBXS_API size_t libxs_scratch_size(void);


#if defined(__cplusplus)

/** RAII idiom to temporarily setup an allocator for the lifetime of the scope. */
template<typename kind> class LIBXS_RETARGETABLE libxs_scoped_allocator {
public:
  /** C'tor, which instantiates the new allocator (plain form). */
  libxs_scoped_allocator(libxs_malloc_fun malloc_fn, libxs_free_fun free_fn) {
    kind::get(m_context, m_malloc, m_free);
    kind::set(0/*context*/, 0/*malloc_ctx*/, 0/*free_ctx*/, malloc_fn, free_fn);
  }

  /** C'tor, which instantiates the new allocator (context form). */
  libxs_scoped_allocator(void* context, libxs_malloc_ctx malloc_ctx, libxs_free_ctx free_ctx,
    libxs_malloc_fun malloc_fun = 0, libxs_free_fun free_fun = 0)
  {
    kind::get(m_context, m_malloc, m_free);
    kind::set(context, malloc_ctx, free_ctx, malloc_fun, free_fun);
  }

  /** Following the RAII idiom, the d'tor restores the previous allocator. */
  ~libxs_scoped_allocator() {
    kind::set(m_context,
      m_malloc.ctx_form, m_free.ctx_form,
      m_malloc.function, m_free.function);
  }

private: /* no copy/assignment */
  explicit libxs_scoped_allocator(const libxs_scoped_allocator&);
  libxs_scoped_allocator& operator=(const libxs_scoped_allocator&);

private: /* saved/previous allocator */
  void* m_context;
  libxs_malloc_function m_malloc;
  libxs_free_function m_free;
};

/** Allocator-kind to instantiate libxs_scoped_allocator<kind>. */
struct LIBXS_RETARGETABLE libxs_default_allocator {
  static void set(void* context,
    libxs_malloc_ctx malloc_ctx, libxs_free_ctx free_ctx,
    libxs_malloc_fun malloc_fun, libxs_free_fun free_fun)
  {
    if (0 == context) { /* use global form only when no context is given */
      libxs_set_default_allocator(0/*context*/,
        libxs_make_malloc_fun(malloc_fun),
        libxs_make_free_fun(free_fun));
    }
    else {
      libxs_set_default_allocator(context,
        libxs_make_malloc_ctx(malloc_ctx),
        libxs_make_free_ctx(free_ctx));
    }
  }
  static void get(void*& context,
    libxs_malloc_function& malloc_fn, libxs_free_function& free_fn)
  {
    libxs_get_default_allocator(&context, &malloc_fn, &free_fn);
  }
};

/** Allocator-kind to instantiate libxs_scoped_allocator<kind>. */
struct LIBXS_RETARGETABLE libxs_scratch_allocator {
  static void set(void* context,
    libxs_malloc_ctx malloc_ctx, libxs_free_ctx free_ctx,
    libxs_malloc_fun malloc_fun, libxs_free_fun free_fun)
  {
    if (0 != malloc_fun) { /* prefer/adopt global malloc/free functions */
      libxs_set_scratch_allocator(0/*context*/,
        libxs_make_malloc_fun(malloc_fun),
        libxs_make_free_fun(free_fun));
    }
    else {
      libxs_set_scratch_allocator(context,
        libxs_make_malloc_ctx(malloc_ctx),
        libxs_make_free_ctx(free_ctx));
    }
  }
  static void get(void*& context,
    libxs_malloc_function& malloc_fn, libxs_free_function& free_fn)
  {
    libxs_get_scratch_allocator(&context, &malloc_fn, &free_fn);
  }
};

/** Forward-declared types/functions used to implement libxs_tf_allocator. */
namespace tensorflow { class Allocator; Allocator* cpu_allocator(); }

/**
 * An object of this type adopts a memory allocator from TensorFlow.
 * All memory allocations of the requested kind within the current
 * scope (where the libxs_tf_allocator object lives) are subject
 * to TensorFlow's memory allocation scheme. The allocation kind
 * is usually "libxs_scratch_allocator"; using a second object
 * of kind "libxs_default_allocator" makes the default memory
 * allocation of LIBXS subject to TensorFlow as well.
 */
template<typename kind> class LIBXS_RETARGETABLE libxs_tf_allocator:
  public libxs_scoped_allocator<kind>
{
public:
  /** The TensorFlow allocator is adopted from the global CPU memory allocator. */
  explicit libxs_tf_allocator()
    : libxs_scoped_allocator<kind>(
      libxs_tf_allocator::malloc,
      libxs_tf_allocator::free)
  {}

  /** The TensorFlow allocator is adopted from the given OpKernelContext. */
  template<typename context_type>
  explicit libxs_tf_allocator(context_type& context)
    : libxs_scoped_allocator<kind>(&context,
      libxs_tf_allocator::malloc_ctx<context_type>,
      libxs_tf_allocator::free_ctx<context_type>,
      libxs_tf_allocator::malloc,
      libxs_tf_allocator::free)
  {}

private:
  /** Breaks the dependency with TensorFlow such that it is header-only. */
  template<typename type> static type& header_only(type& object) { return object; }

  static void* malloc(size_t size) {
    tensorflow::Allocator *const allocator = tensorflow::cpu_allocator();
    /* no waste with (useless) alignment; raw result is re-aligned anyways */
    return 0 != allocator ? header_only(*allocator).AllocateRaw(1/*alignment*/, size) : 0;
  }

  static void free(void* buffer) {
    tensorflow::Allocator *const allocator = tensorflow::cpu_allocator();
    if (0 != allocator) { header_only(*allocator).DeallocateRaw(buffer); }
  }

  template<typename context_type> static void* malloc_ctx(void* context, size_t size) {
    typedef typename context_type::WrappedAllocator::first_type allocator_ptr;
    context_type *const tf_context = static_cast<context_type*>(context);
    if (0 != tf_context && 0 != tf_context->device()) {
      allocator_ptr allocator = 0;
      if (0 < tf_context->num_outputs()) {
        allocator = tf_context->device()->GetStepAllocator(
          tf_context->output_alloc_attr(0),
          tf_context->resource_manager());
      }
      else if (0 < tf_context->num_inputs()) {
        allocator = tf_context->device()->GetStepAllocator(
          tf_context->input_alloc_attr(0),
          tf_context->resource_manager());
      }
      /* no waste with (useless) alignment; raw result is re-aligned anyways */
      return 0 != allocator ? allocator->AllocateRaw(1/*alignment*/, size) : 0;
    }
  }

  template<typename context_type> static void free_ctx(void* context, void* buffer) {
    typedef typename context_type::WrappedAllocator::first_type allocator_ptr;
    context_type *const tf_context = static_cast<context_type*>(context);
    if (0 != tf_context && 0 != tf_context->device()) {
      allocator_ptr allocator = 0;
      if (0 < tf_context->num_outputs()) {
        allocator = tf_context->device()->GetStepAllocator(
          tf_context->output_alloc_attr(0),
          tf_context->resource_manager());
      }
      else if (0 < tf_context->num_inputs()) {
        allocator = tf_context->device()->GetStepAllocator(
          tf_context->input_alloc_attr(0),
          tf_context->resource_manager());
      }
      if (0 != allocator) { allocator->DeallocateRaw(buffer); }
    }
  }
};

#endif /*defined(__cplusplus)*/
#endif /*LIBXS_MALLOC_H*/
