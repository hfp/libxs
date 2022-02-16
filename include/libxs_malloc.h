/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MALLOC_H
#define LIBXS_MALLOC_H

#include "libxs_mem.h"

/* include tensorflow/core/public/version.h prior to LIBXS otherwise the current TensorFlow API is assumed */
#if !defined(LIBXS_TF12) && (!defined(TF_VERSION_STRING) || \
  LIBXS_VERSION2(1, 12) <= LIBXS_VERSION2(TF_MAJOR_VERSION, TF_MINOR_VERSION))
# define LIBXS_TF12 /* TF_PATCH_VERSION does not matter */
#endif

/** Can be used with libxs_[get|set]_scratch_limit. */
#define LIBXS_SCRATCH_UNLIMITED ((size_t)LIBXS_UNLIMITED)
#define LIBXS_SCRATCH_DEFAULT 0


/** Function types accepted for memory allocation (see libxs_*_allocator). */
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void* (*libxs_malloc_ctx)(size_t /*size*/, const void* /*context*/);
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void* (*libxs_malloc_fun)(size_t /*size*/);
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_malloc_function {
  libxs_malloc_ctx ctx_form;
  libxs_malloc_fun function;
} libxs_malloc_function;

/** Function types accepted for releasing memory (see libxs_*_allocator). */
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_free_ctx)(void* /*buffer*/, const void* /*context*/);
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_free_fun)(void* /*buffer*/);
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_free_function {
  libxs_free_ctx ctx_form;
  libxs_free_fun function;
} libxs_free_function;

/**
 * To setup the custom default memory allocator, either a malloc_fn and a free_fn
 * are given, or two NULL-pointers designate to reset the default allocator to a
 * library-internal default. If a context is given (non-NULL), the context-based
 * form of the memory allocation is used.
 * Changing the allocator including the function for deallocation applies to
 * upcoming allocation/deallocation and works correctly for pending buffers.
 */
LIBXS_API int libxs_set_default_allocator(/* malloc_fn/free_fn must correspond */
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Retrieve the default memory allocator. */
LIBXS_API int libxs_get_default_allocator(const void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/**
 * To setup the scratch memory allocator, a malloc_fn function and an optional free_fn
 * are given. A NULL-free acts as a "no-operation", and the deallocation is expected
 * to be controlled otherwise. If two NULL-pointers are given, the allocator is reset
 * to the currently active default memory allocator. If a context is given (non-NULL),
 * the context-based form of the memory allocation is used.
 * Changing the allocator including the function for deallocation applies to
 * upcoming allocation/deallocation and works correctly for pending buffers.
 */
LIBXS_API int libxs_set_scratch_allocator(/* malloc_fn/free_fn must correspond */
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Retrieve the scratch memory allocator. */
LIBXS_API int libxs_get_scratch_allocator(const void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/** Allocate memory (malloc/free interface). */
LIBXS_API LIBXS_ATTRIBUTE_MALLOC void* libxs_malloc(size_t size);

/** Allocate aligned memory using the default allocator. */
LIBXS_API LIBXS_ATTRIBUTE_MALLOC void* libxs_aligned_malloc(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment);

/** Reallocate memory using the default allocator (alignment is preserved). */
LIBXS_API void* libxs_realloc(size_t size, void* ptr);

/**
 * Allocate aligned scratch memory. It is not supported
 * to query properties per libxs_get_malloc_info, but
 * libxs_get_scratch_info can used instead.
 */
LIBXS_API void* libxs_scratch_malloc(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment,
  /**
   * Identifies the call site, which is used
   * to determine the memory pool.
   */
  const void* caller);

/**
 * Binary form of libxs_scratch_malloc, which
 * expands the call-context automatically. This
 * macro is intentionally lower case.
 */
#define libxs_aligned_scratch(size, alignment) \
  libxs_scratch_malloc(size, alignment, \
    LIBXS_CALLER_ID)

/** Deallocate memory (malloc/free interface). */
LIBXS_API void libxs_free(const void* memory);

/**
 * Initialize the pool by drawing from the given storage a number of chunks of the given size.
 * If the capacity of the pool is num, the storage must be at least num x size.
 * The same num-counter must be used for pmalloc/pfree when referring to the same pool.
 */
LIBXS_API void libxs_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage);
/** Allocate from the given pool by using the original num-counter (libxs_pmalloc_init). */
LIBXS_API void* libxs_pmalloc(void* pool[], size_t* i);
/** Bring pointer back into the pool by using original num-counter (libxs_pmalloc_init). */
LIBXS_API void libxs_pfree(void* pointer, void* pool[], size_t* i);

/**
 * Release the entire scratch memory regardless
 * of whether it is still referenced or not.
 */
LIBXS_API void libxs_release_scratch(void);

/** Information about a buffer (default memory domain). */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_malloc_info {
  /** Size of the buffer. */
  size_t size;
} libxs_malloc_info;

/** Retrieve information about a buffer (default memory domain). */
LIBXS_API int libxs_get_malloc_info(const void* memory, libxs_malloc_info* info);

/** Information about the scratch memory domain. */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_scratch_info {
  /** Watermark memory across pools (size), unsatisfied (local), and library-internal memory. */
  size_t size, local, internal;
  /** Pending allocations (not released). */
  size_t npending;
  /** Number of allocations so far. */
  size_t nmallocs;
  /** Number of pools used. */
  unsigned int npools;
} libxs_scratch_info;

/** Retrieve information about the scratch memory domain. */
LIBXS_API int libxs_get_scratch_info(libxs_scratch_info* info);

/**
 * Limit the total size (Bytes) of the scratch memory.
 * LIBXS_SCRATCH_UNLIMITED removes any limit, and
 * LIBXS_SCRATCH_DEFAULT populates the default.
 * The related environment variable LIBXS_SCRATCH_LIMIT
 * allows units: <none>/b/B (Bytes), k/K, m/M, and g/G.
 */
LIBXS_API void libxs_set_scratch_limit(size_t nbytes);
/** Get the maximum size of the scratch memory domain. */
LIBXS_API size_t libxs_get_scratch_limit(void);

/**
 * Intercepts malloc/free to use scratch memory allocator.
 * (related environment variable LIBXS_MALLOC).
 * Optionally set the range of malloc-sizes to be intercepted.
 * The related environment variable LIBXS_MALLOC_LIMIT
 * allows units: <none>/b/B (Bytes), k/K, m/M, and g/G.
 */
LIBXS_API void libxs_set_malloc(int enabled, const size_t* lo, const size_t* hi);
/**
 * Determines if malloc/free are (and can be) intercepted.
 * Optionally gets the range of enabled malloc-sizes.
 */
LIBXS_API int libxs_get_malloc(size_t* lo, size_t* hi);

/**
 * Calculate the linear offset of the n-dimensional (ndims) offset (can be NULL),
 * and the (optional) linear size of the corresponding shape.
 */
LIBXS_API size_t libxs_offset(const size_t offset[], const size_t shape[], size_t ndims, size_t* size);


#if defined(__cplusplus)

/** RAII idiom to temporarily setup an allocator for the lifetime of the scope. */
template<typename kind> class LIBXS_RETARGETABLE libxs_scoped_allocator {
public:
  /** C'tor, which instantiates the new allocator (plain form). */
  libxs_scoped_allocator(libxs_malloc_fun malloc_fn, libxs_free_fun free_fn) {
    kind::get(m_context, m_malloc, m_free);
    kind::set(NULL/*context*/, NULL/*malloc_ctx*/, NULL/*free_ctx*/, malloc_fn, free_fn);
  }

  /** C'tor, which instantiates the new allocator (context form). */
  libxs_scoped_allocator(const void* context, libxs_malloc_ctx malloc_ctx, libxs_free_ctx free_ctx,
    libxs_malloc_fun malloc_fun = NULL, libxs_free_fun free_fun = NULL)
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

protected: /* saved/previous allocator */
  const void* m_context;
  libxs_malloc_function m_malloc;
  libxs_free_function m_free;
};

/** Allocator-kind to instantiate libxs_scoped_allocator<kind>. */
struct LIBXS_RETARGETABLE libxs_default_allocator {
  static void set(const void* context,
    libxs_malloc_ctx malloc_ctx, libxs_free_ctx free_ctx,
    libxs_malloc_fun malloc_fun, libxs_free_fun free_fun)
  {
    libxs_malloc_function malloc_fn;
    libxs_free_function free_fn;
    if (NULL == context) { /* use global form only when no context is given */
      malloc_fn.function = malloc_fun; free_fn.function = free_fun;
    }
    else {
      malloc_fn.ctx_form = malloc_ctx; free_fn.ctx_form = free_ctx;
    }
    libxs_set_default_allocator(context, malloc_fn, free_fn);
  }
  static void get(const void*& context,
    libxs_malloc_function& malloc_fn, libxs_free_function& free_fn)
  {
    libxs_get_default_allocator(&context, &malloc_fn, &free_fn);
  }
};

/** Allocator-kind to instantiate libxs_scoped_allocator<kind>. */
struct LIBXS_RETARGETABLE libxs_scratch_allocator {
  static void set(const void* context,
    libxs_malloc_ctx malloc_ctx, libxs_free_ctx free_ctx,
    libxs_malloc_fun malloc_fun, libxs_free_fun free_fun)
  {
    libxs_malloc_function malloc_fn;
    libxs_free_function free_fn;
    if (NULL != context) { /* adopt context form */
      malloc_fn.function = malloc_fun; free_fn.function = free_fun;
    }
    else { /* adopt global form */
      malloc_fn.ctx_form = malloc_ctx; free_fn.ctx_form = free_ctx;
    }
    libxs_set_scratch_allocator(context, malloc_fn, free_fn);
  }
  static void get(const void*& context,
    libxs_malloc_function& malloc_fn, libxs_free_function& free_fn)
  {
    libxs_get_scratch_allocator(&context, &malloc_fn, &free_fn);
  }
};

/** Forward-declared types/functions used to implement libxs_tf_allocator. */
namespace tensorflow {
  class Allocator;
#if defined(LIBXS_TF12)
  class DeviceBase; int DeviceNumaNode(const DeviceBase* /*device*/);
  Allocator* cpu_allocator(int /*numa_node*/);
#else
  Allocator* cpu_allocator();
#endif
}

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
      libxs_tf_allocator::template malloc_ctx<context_type>,
      libxs_tf_allocator::template free_ctx<context_type>,
      libxs_tf_allocator::malloc,
      libxs_tf_allocator::free)
  {}

  /** Global form of allocating memory (malloc signature). */
  static void* malloc(size_t size) {
#if defined(LIBXS_TF12)
    return libxs_tf_allocator::allocate(tensorflow::cpu_allocator(-1/*kNUMANoAffinity*/), size);
#else
    return libxs_tf_allocator::allocate(tensorflow::cpu_allocator(), size);
#endif
  }

  /** Global form of deallocating memory (free signature). */
  static void free(void* buffer) {
#if defined(LIBXS_TF12)
    libxs_tf_allocator::deallocate(tensorflow::cpu_allocator(-1/*kNUMANoAffinity*/), buffer);
#else
    libxs_tf_allocator::deallocate(tensorflow::cpu_allocator(), buffer);
#endif
  }

  /** Context based form of allocating memory. */
  template<typename context_type> static void* malloc_ctx(const void* context, size_t size) {
    typedef typename context_type::WrappedAllocator::first_type allocator_ptr;
    context_type *const tf_context = static_cast<context_type*>(context);
    allocator_ptr allocator = NULL;
    if (NULL != tf_context) {
#if !defined(LIBXS_TF12)
      if (NULL != tf_context->device()) {
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
      }
#else /* include tensorflow/core/public/version.h prior to LIBXS otherwise the current TensorFlow API is assumed */
      const int numa_node = DeviceNumaNode(tf_context->device());
      allocator = tensorflow::cpu_allocator(numa_node);
#endif
    }
    return libxs_tf_allocator::allocate(allocator, size);
  }

  /** Context based form of deallocating memory. */
  template<typename context_type> static void free_ctx(const void* context, void* buffer) {
    typedef typename context_type::WrappedAllocator::first_type allocator_ptr;
    context_type *const tf_context = static_cast<context_type*>(context);
    allocator_ptr allocator = NULL;
    if (NULL != tf_context) {
#if defined(LIBXS_TF12)
      const int numa_node = DeviceNumaNode(tf_context->device());
      allocator = tensorflow::cpu_allocator(numa_node);
#else
      if (NULL != tf_context->device()) {
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
      }
#endif
    }
    libxs_tf_allocator::deallocate(allocator, buffer);
  }

private:
  template<typename allocator_ptr> /* break interface dependency with TF */
  static void* allocate(allocator_ptr allocator, size_t size) {
    void* result;
    if (NULL != allocator) {
    /* no (useless) waste with alignment; raw result is re-aligned anyways */
      result = allocator->AllocateRaw(1/*alignment*/, size);
    }
    else {
      LIBXS_ASSERT_MSG(0/*false*/, "LIBXS ERROR: memory allocator is missing");
      result = NULL;
    }
    return result;
  }

  template<typename allocator_ptr> /* break interface dependency with TF */
  static void deallocate(allocator_ptr allocator, void* buffer) {
    LIBXS_ASSERT_MSG(NULL != allocator, "LIBXS ERROR: memory allocator is missing");
    if (NULL != allocator) allocator->DeallocateRaw(buffer);
  }
};

#endif /*defined(__cplusplus)*/
#endif /*LIBXS_MALLOC_H*/
