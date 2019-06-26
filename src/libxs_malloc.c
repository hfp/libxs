/******************************************************************************
** Copyright (c) 2014-2019, Intel Corporation                                **
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
#include "libxs_trace.h"
#include "libxs_main.h"
#include "libxs_hash.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if !defined(_LIBC) && 0
# define _LIBC
# include <malloc.h>
#endif
#if defined(__TBB)
# include <tbb/scalable_allocator.h>
#endif
#if defined(_WIN32)
# include <windows.h>
#else
# include <sys/mman.h>
# if defined(MAP_HUGETLB) && defined(MAP_POPULATE)
#   include <sys/utsname.h>
#   include <string.h>
# endif
# include <sys/types.h>
# include <unistd.h>
# include <errno.h>
# if defined(MAP_ANONYMOUS)
#   define LIBXS_MAP_ANONYMOUS MAP_ANONYMOUS
# else
#   define LIBXS_MAP_ANONYMOUS MAP_ANON
# endif
#endif
#if !defined(LIBXS_MALLOC_FALLBACK)
# define LIBXS_MALLOC_FINAL 3
#endif
#if defined(LIBXS_VTUNE)
# if (2 <= LIBXS_VTUNE) /* no header file required */
#   if !defined(LIBXS_VTUNE_JITVERSION)
#     define LIBXS_VTUNE_JITVERSION LIBXS_VTUNE
#   endif
#   define LIBXS_VTUNE_JIT_DESC_TYPE iJIT_Method_Load_V2
#   define LIBXS_VTUNE_JIT_LOAD 21
#   define LIBXS_VTUNE_JIT_UNLOAD 14
#   define iJIT_SAMPLING_ON 0x0001
LIBXS_EXTERN unsigned int iJIT_GetNewMethodID(void);
LIBXS_EXTERN /*iJIT_IsProfilingActiveFlags*/int iJIT_IsProfilingActive(void);
LIBXS_EXTERN int iJIT_NotifyEvent(/*iJIT_JVM_EVENT*/int event_type, void *EventSpecificData);
LIBXS_EXTERN_C typedef struct LineNumberInfo {
  unsigned int Offset;
  unsigned int LineNumber;
} LineNumberInfo;
LIBXS_EXTERN_C typedef struct iJIT_Method_Load_V2 {
  unsigned int method_id;
  char* method_name;
  void* method_load_address;
  unsigned int method_size;
  unsigned int line_number_size;
  LineNumberInfo* line_number_table;
  char* class_file_name;
  char* source_file_name;
  char* module_name;
} iJIT_Method_Load_V2;
# else /* more safe due to header dependency */
#   include <jitprofiling.h>
#   if !defined(LIBXS_VTUNE_JITVERSION)
#     define LIBXS_VTUNE_JITVERSION 2
#   endif
#   if (2 <= LIBXS_VTUNE_JITVERSION)
#     define LIBXS_VTUNE_JIT_DESC_TYPE iJIT_Method_Load_V2
#     define LIBXS_VTUNE_JIT_LOAD iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V2
#   else
#     define LIBXS_VTUNE_JIT_DESC_TYPE iJIT_Method_Load
#     define LIBXS_VTUNE_JIT_LOAD iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED
#   endif
#   define LIBXS_VTUNE_JIT_UNLOAD iJVM_EVENT_TYPE_METHOD_UNLOAD_START
# endif
# if !defined(LIBXS_MALLOC_FALLBACK)
#   define LIBXS_MALLOC_FALLBACK LIBXS_MALLOC_FINAL
# endif
#else
# if !defined(LIBXS_MALLOC_FALLBACK)
#   define LIBXS_MALLOC_FALLBACK 0
# endif
#endif /*defined(LIBXS_VTUNE)*/
#if !defined(LIBXS_MALLOC_XMAP_TEMPLATE)
# define LIBXS_MALLOC_XMAP_TEMPLATE ".libxs_jit." LIBXS_MKTEMP_PATTERN
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
#if defined(LIBXS_PERF)
# include "libxs_perf.h"
#endif

#if !defined(LIBXS_MALLOC_NOCRC)
# if defined(NDEBUG)
#   define LIBXS_MALLOC_NOCRC
# elif !defined(LIBXS_BUILD)
#   define LIBXS_MALLOC_NOCRC
# endif
#endif

#if !defined(LIBXS_MALLOC_ALIGNMAX)
# define LIBXS_MALLOC_ALIGNMAX (2 * 1024 * 1024)
#endif
#if !defined(LIBXS_MALLOC_ALIGNFCT)
# define LIBXS_MALLOC_ALIGNFCT 8
#endif
#if !defined(LIBXS_MALLOC_SEED)
# define LIBXS_MALLOC_SEED 1051981
#endif

#if (!defined(LIBXS_MALLOC_HOOK_GLIBC) && (defined(__MALLOC_HOOK_VOLATILE) || defined(LIBXS_GLIBC)))
# define LIBXS_MALLOC_HOOK_GLIBC 4 /* scratch threshold (prior-to-main allocations) */
#endif
#if !defined(LIBXS_MALLOC_CTXFORM) && !defined(NDEBUG) && 0
# define LIBXS_MALLOC_CTXFORM
#endif

/* allows to reclaim a pool for a different thread */
#if !defined(LIBXS_MALLOC_AFFINITY) && 1
# define LIBXS_MALLOC_AFFINITY
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_JOIN) && 0
# define LIBXS_MALLOC_SCRATCH_JOIN
#endif
/* protected against double-delete (if possible) */
#if !defined(LIBXS_MALLOC_DELETE_SAFE) && 0
# define LIBXS_MALLOC_DELETE_SAFE
#endif
/* map memory for scratch buffers */
#if !defined(LIBXS_MALLOC_MMAP_SCRATCH) && 0
# define LIBXS_MALLOC_MMAP_SCRATCH
#endif
/* map memory for hooked allocation */
#if !defined(LIBXS_MALLOC_MMAP_HOOK) && 1
# define LIBXS_MALLOC_MMAP_HOOK
#endif
/* map memory even for non-executable buffers */
#if !defined(LIBXS_MALLOC_MMAP) && 0
# define LIBXS_MALLOC_MMAP
#endif


LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE internal_malloc_info_type {
  libxs_free_function free;
  void *pointer, *reloc;
  const void* context;
  size_t size;
  int flags;
#if defined(LIBXS_VTUNE)
  unsigned int code_id;
#endif
#if !defined(LIBXS_MALLOC_NOCRC) /* hash *must* be the last entry */
  unsigned int hash;
#endif
} internal_malloc_info_type;

LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE internal_malloc_pool_type {
  char pad[LIBXS_CACHELINE];
  struct {
    char *buffer, *head;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    const void* site;
# if defined(LIBXS_MALLOC_AFFINITY) && (0 != LIBXS_SYNC)
    size_t tid;
# endif
#endif
    size_t minsize;
    size_t incsize;
    size_t counter;
  } instance;
} internal_malloc_pool_type;

/** Scratch pool, which supports up to MAX_NSCRATCH allocation sites. */
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
/* LIBXS_ALIGNED appears to contradict LIBXS_APIVAR, and causes multiple defined symbols (if below is seen in multiple translation units) */
LIBXS_APIVAR_ARRAY(char internal_malloc_pool_buffer, (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) * sizeof(internal_malloc_pool_type) + (LIBXS_CACHELINE) - 1);
#endif
LIBXS_APIVAR(size_t internal_malloc_scratch_nmallocs);
LIBXS_APIVAR(size_t internal_malloc_private_size);
LIBXS_APIVAR(size_t internal_malloc_scratch_size);

LIBXS_APIVAR(void* (*internal_memalign_hook)(size_t, size_t, const void*));
LIBXS_APIVAR(void* (*internal_malloc_hook)(size_t, const void*));
LIBXS_APIVAR(void* (*internal_realloc_hook)(void*, size_t, const void*));
LIBXS_APIVAR(void  (*internal_free_hook)(void*, const void*));


LIBXS_API_INTERN size_t libxs_alignment(size_t size, size_t alignment)
{
  size_t result = sizeof(void*);
  if ((LIBXS_MALLOC_ALIGNFCT * LIBXS_MALLOC_ALIGNMAX) <= size) {
    result = libxs_lcm(0 == alignment ? (LIBXS_ALIGNMENT) : libxs_lcm(alignment, LIBXS_ALIGNMENT), LIBXS_MALLOC_ALIGNMAX);
  }
  else {
    if ((LIBXS_MALLOC_ALIGNFCT * LIBXS_ALIGNMENT) <= size) {
      result = (0 == alignment ? (LIBXS_ALIGNMENT) : libxs_lcm(alignment, LIBXS_ALIGNMENT));
    }
    else if (0 != alignment) {
      result = libxs_lcm(alignment, result);
    }
  }
  return result;
}


LIBXS_API size_t libxs_offset(const size_t offset[], const size_t shape[], size_t ndims, size_t* size)
{
  size_t result = 0, size1 = 0;
  if (0 != ndims && NULL != shape) {
    size_t i;
    result = (NULL != offset ? offset[0] : 0);
    size1 = shape[0];
    for (i = 1; i < ndims; ++i) {
      result += (NULL != offset ? offset[i] : 0) * size1;
      size1 *= shape[i];
    }
  }
  if (NULL != size) *size = size1;
  return result;
}


LIBXS_API_INLINE const void* internal_malloc_site_auto(void)
{
  const void *stacktrace[4], *result;
  const unsigned int n = libxs_backtrace(stacktrace, sizeof(stacktrace) / sizeof(*stacktrace), 0/*skip*/);
#if defined(NDEBUG) /* internal_malloc_site is inlined */
  if (0 < n) result = stacktrace[0];
#else /* not inlined */
  if (3 < n) result = stacktrace[3];
#endif
  else result = NULL;
  return result;
}


LIBXS_API_INLINE const void* internal_malloc_site(const void* site)
{
  const void* result;
  if (NULL != site) {
#if !defined(LIBXS_STRING_POOLING)
    if ((LIBXS_MALLOC_INTERNAL_CALLER) != site) {
      const size_t length = strlen((const char*)site);
      const uintptr_t hash = (sizeof(void*) < length ? libxs_crc32(LIBXS_MALLOC_SEED, site, length) : ((uintptr_t)site));
      result = (const void*)((LIBXS_MALLOC_INTERNAL_CALLER_ID) != hash ? hash : (hash - 1));
      LIBXS_ASSERT((LIBXS_MALLOC_INTERNAL_CALLER) != result);
    }
    else
#endif
    {
      result = site;
    }
  }
  else {
    result = internal_malloc_site_auto();
  }
  return result;
}


LIBXS_API_INLINE internal_malloc_info_type* internal_malloc_info(const void* memory)
{
  internal_malloc_info_type *const result = (internal_malloc_info_type*)
    (NULL != memory ? (((const char*)memory) - sizeof(internal_malloc_info_type)) : NULL);
#if defined(LIBXS_MALLOC_NOCRC)
  return result;
#else /* calculate checksum over info */
  return (NULL != result && result->hash == libxs_crc32(LIBXS_MALLOC_SEED, result,
    ((const char*)&result->hash) - ((const char*)result)) ? result : NULL);
#endif
}


LIBXS_API_INLINE int internal_xfree(const void* memory, internal_malloc_info_type* info)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  void* buffer;
  LIBXS_ASSERT(NULL != memory && NULL != info);
  buffer = info->pointer;
#if !defined(LIBXS_BUILD) /* sanity check */
  if (NULL != buffer || 0 == info->size)
#endif
  {
    LIBXS_ASSERT(NULL != buffer || 0 == info->size);
    if (0 == (LIBXS_MALLOC_FLAG_MMAP & info->flags)) {
      if (NULL != info->free.function) {
#if defined(LIBXS_MALLOC_DELETE_SAFE)
        info->pointer = NULL; info->size = 0;
#endif
        if (NULL == info->context) {
          info->free.function(buffer);
        }
        else {
          info->free.ctx_form(buffer, info->context);
        }
      }
    }
    else {
#if defined(LIBXS_VTUNE)
      if (0 != (LIBXS_MALLOC_FLAG_X & info->flags) && 0 != info->code_id && iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
        iJIT_NotifyEvent(LIBXS_VTUNE_JIT_UNLOAD, &info->code_id);
      }
#endif
#if defined(_WIN32)
      result = (NULL == buffer || FALSE != VirtualFree(buffer, 0, MEM_RELEASE)) ? EXIT_SUCCESS : EXIT_FAILURE;
#else /* !_WIN32 */
      {
        const size_t alloc_size = info->size + (((const char*)memory) - ((const char*)buffer));
        void *const reloc = info->reloc;
        const int flags = info->flags;
        if (0 != munmap(buffer, alloc_size)) {
          if (0 != libxs_verbosity /* library code is expected to be mute */
            && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
          {
            const char *const error_message = strerror(errno);
            fprintf(stderr, "LIBXS ERROR: %s (munmap error #%i for range %p+%" PRIuPTR ")!\n",
              error_message, errno, buffer, (uintptr_t)alloc_size);
          }
          result = EXIT_FAILURE;
        }
        if (0 != (LIBXS_MALLOC_FLAG_X & flags) && EXIT_SUCCESS == result
          && NULL != reloc && MAP_FAILED != reloc && buffer != reloc
          && 0 != munmap(reloc, alloc_size))
        {
          if (0 != libxs_verbosity /* library code is expected to be mute */
            && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
          {
            const char *const error_message = strerror(errno);
            fprintf(stderr, "LIBXS ERROR: %s (munmap error #%i for range %p+%" PRIuPTR ")!\n",
              error_message, errno, reloc, (uintptr_t)alloc_size);
          }
          result = EXIT_FAILURE;
        }
      }
#endif
    }
  }
#if !defined(LIBXS_BUILD)
  else if ((LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: attempt to release memory from non-matching implementation!\n");
  }
#endif
  return result;
}


LIBXS_API_INLINE size_t internal_get_scratch_size(const internal_malloc_pool_type* exclude)
{
  size_t result = 0;
#if !defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) || (0 >= (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  LIBXS_UNUSED(exclude);
#else
  const internal_malloc_pool_type *const pools = (const internal_malloc_pool_type*)LIBXS_UP2(internal_malloc_pool_buffer, LIBXS_CACHELINE);
  const internal_malloc_pool_type* pool = pools;
  const internal_malloc_info_type* info = internal_malloc_info(pool->instance.buffer);
  unsigned int i;
  LIBXS_ASSERT(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
  if (NULL != info && pool != exclude && (LIBXS_MALLOC_INTERNAL_CALLER) != pool->instance.site) {
    result = info->size;
  }
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  LIBXS_ASSERT(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
  for (i = 1; i < libxs_scratch_pools; ++i) {
    pool = pools + i; info = internal_malloc_info(pool->instance.buffer);
    if (NULL != info && pool != exclude && (LIBXS_MALLOC_INTERNAL_CALLER) != pool->instance.site) {
      result += info->size;
    }
  }
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
  return result;
}


LIBXS_API_INLINE internal_malloc_pool_type* internal_scratch_malloc_pool(const void* memory)
{
  internal_malloc_pool_type* result = NULL;
  LIBXS_ASSERT(NULL != memory);
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  {
    internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)LIBXS_UP2(internal_malloc_pool_buffer, LIBXS_CACHELINE);
    const char *const buffer = (const char*)memory;
    const unsigned int npools = libxs_scratch_pools;
    unsigned int i = 0;
    LIBXS_ASSERT(npools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
    LIBXS_ASSERT(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
    for (; i < npools; ++i) {
      internal_malloc_pool_type *const pool = pools + i;
      const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer);
      /* check if memory belongs to scratch domain or local domain */
      if (NULL != info && pool->instance.buffer <= buffer && buffer < (pool->instance.buffer + info->size)) {
        result = pool;
        break;
      }
    }
  }
#endif
  return result;
}


LIBXS_API_INLINE void internal_scratch_malloc(void** memory, size_t size, size_t alignment, int flags, const void* caller)
{
  LIBXS_ASSERT(NULL != memory);
  if (0 == (LIBXS_MALLOC_FLAG_REALLOC & flags) || NULL == *memory) {
    static int error_once = 0;
    size_t local_size = 0;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    if (0 < libxs_scratch_pools && 0 < libxs_scratch_limit) {
      internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)LIBXS_UP2(internal_malloc_pool_buffer, LIBXS_CACHELINE);
      internal_malloc_pool_type *const end = pools + libxs_scratch_pools, *pool0 = end, *pool = pools;
      const void *const site = internal_malloc_site(caller);
      const size_t align_size = libxs_alignment(size, alignment);
      const size_t alloc_size = size + align_size - 1;
#if defined(LIBXS_MALLOC_AFFINITY) && (0 != LIBXS_SYNC)
      const unsigned int tid = libxs_get_tid();
#endif
      LIBXS_ASSERT(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
      for (; pool != end; ++pool) { /* find matching pool */
        if (site == pool->instance.site
# if defined(LIBXS_MALLOC_AFFINITY) && (0 != LIBXS_SYNC)
          && tid == pool->instance.tid
# elif 0
          && (NULL == pool->instance.buffer || size <= internal_malloc_info(pool->instance.buffer)->size)
# else
          && (NULL != pool->instance.buffer || 1 != pool->instance.counter)
# endif
        ) break;
        if (end == pool0 && NULL == pool->instance.site) pool0 = pool;
      }
#endif
      if (end == pool) pool = pool0; /* fall-back to new pool */
      if (end != pool) {
        const size_t counter = LIBXS_ATOMIC_ADD_FETCH(&pool->instance.counter, (size_t)1, LIBXS_ATOMIC_SEQ_CST);
        if (NULL != pool->instance.buffer || 1 != counter) {
          const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer);
          const size_t used_size = pool->instance.head - pool->instance.buffer;
          const size_t pool_size = (NULL != info ? info->size : 0);
          const size_t req_size = alloc_size + used_size;
          LIBXS_ASSERT(used_size <= pool_size);
          /* fast path: draw from pool-buffer */
          if (req_size <= pool_size) {
            void *const headaddr = &pool->instance.head;
            uintptr_t headptr = LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)((uintptr_t*)headaddr, alloc_size, LIBXS_ATOMIC_SEQ_CST);
            char *const head = (char*)headptr;
            *memory = LIBXS_ALIGN(head - alloc_size, align_size);
          }
          else { /* fall-back to local memory allocation */
            const size_t incsize = req_size - LIBXS_MIN(pool_size, req_size);
            pool->instance.incsize = LIBXS_MAX(pool->instance.incsize, incsize);
            LIBXS_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);
            local_size = size;
          }
        }
        else {
          const size_t scratch_size = internal_get_scratch_size(pool);
          const size_t limit_size = libxs_scratch_limit - LIBXS_MIN(scratch_size, libxs_scratch_limit);
          const size_t scale_size = (size_t)(libxs_scratch_scale * alloc_size);
          const size_t incsize = (size_t)(libxs_scratch_scale * pool->instance.incsize);
          const size_t maxsize = LIBXS_MAX(scale_size, pool->instance.minsize) + incsize;
          const size_t limsize = LIBXS_MIN(maxsize, limit_size);
#if defined(LIBXS_MALLOC_SCRATCH_JOIN)
          const size_t minsize = LIBXS_MAX(limsize, alloc_size);
#else
          const size_t minsize = limsize;
#endif
          LIBXS_ASSERT(1 <= libxs_scratch_scale);
          LIBXS_ASSERT(NULL == pool->instance.head);
          pool->instance.incsize = 0; /* reset */
          pool->instance.minsize = minsize;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
          pool->instance.site = site;
# if defined(LIBXS_MALLOC_AFFINITY) && (0 != LIBXS_SYNC)
          pool->instance.tid = tid;
# endif
#endif
          if ( /* allocate scratch pool */
#if !defined(LIBXS_MALLOC_SCRATCH_JOIN)
            alloc_size <= minsize &&
#endif
            EXIT_SUCCESS == libxs_xmalloc(memory, minsize, 0/*auto-align*/,
              (flags | LIBXS_MALLOC_FLAG_SCRATCH) & ~LIBXS_MALLOC_FLAG_REALLOC,
              NULL/*extra*/, 0/*extra_size*/))
          {
            pool->instance.buffer = (char*)*memory;
            pool->instance.head = pool->instance.buffer + alloc_size;
            *memory = LIBXS_ALIGN((char*)*memory, align_size);
            LIBXS_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXS_ATOMIC_RELAXED);
#if defined(LIBXS_MALLOC_SCRATCH_JOIN) /* library code is expected to be mute */
            if (limit_size < maxsize && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity)
              && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
            {
              fprintf(stderr, "LIBXS WARNING: scratch memory domain exhausted!\n");
            }
#endif
          }
          else { /* fall-back to local allocation */
            LIBXS_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);
            if (0 != libxs_verbosity /* library code is expected to be mute */
              && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
            {
#if !defined(LIBXS_MALLOC_SCRATCH_JOIN)
              if (alloc_size <= minsize)
#endif
              {
                fprintf(stderr, "LIBXS ERROR: failed to allocate scratch memory!\n");
              }
#if !defined(LIBXS_MALLOC_SCRATCH_JOIN)
              else if ((LIBXS_MALLOC_INTERNAL_CALLER) != caller
                && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity))
              {
                fprintf(stderr, "LIBXS WARNING: scratch memory domain exhausted!\n");
              }
#endif
            }
            local_size = size;
          }
        }
      }
      else { /* fall-back to local memory allocation */
        local_size = size;
      }
    }
    else { /* fall-back to local memory allocation */
      local_size = size;
    }
    if (0 != local_size)
#else
    local_size = size;
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
    { /* local memory allocation */
      if (EXIT_SUCCESS != libxs_xmalloc(memory, local_size, alignment,
          (flags | LIBXS_MALLOC_FLAG_SCRATCH) & ~LIBXS_MALLOC_FLAG_REALLOC, NULL/*extra*/, 0/*extra_size*/)
        && /* library code is expected to be mute */0 != libxs_verbosity
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: scratch memory fall-back failed!\n");
        LIBXS_ASSERT(NULL == *memory);
      }
      if ((LIBXS_MALLOC_INTERNAL_CALLER) != caller) {
        LIBXS_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXS_ATOMIC_RELAXED);
      }
    }
  }
  else { /* reallocate memory */
#if !defined(NDEBUG)
    int status = EXIT_SUCCESS;
#endif
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    internal_malloc_pool_type *const pool = internal_scratch_malloc_pool(*memory);
    if (NULL != pool) {
      const size_t counter = LIBXS_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);
      const void *const pool_buffer = pool->instance.buffer;
      internal_malloc_info_type *const info = internal_malloc_info(pool_buffer);
      LIBXS_ASSERT(pool->instance.buffer <= pool->instance.head);
      LIBXS_ASSERT(NULL != info);
      *memory = NULL; /* no reallocation */
      if (0 == counter) { /* in-use scratch is reported as dangling buffer at program termination */
        pool->instance.buffer = pool->instance.head = NULL;
# if defined(LIBXS_MALLOC_AFFINITY) && (0 != LIBXS_SYNC) && !defined(NDEBUG) /* library code is expected to be mute */
        if ((LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity) && libxs_get_tid() != pool->instance.tid) {
          static int error_once = 0;
          if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
            fprintf(stderr, "LIBXS WARNING: thread-id differs between allocation and deallocation!\n");
          }
        }
# endif
      }
      if (EXIT_SUCCESS == libxs_xmalloc(memory, size, alignment/* no need here to determine alignment of given buffer */,
        (flags | LIBXS_MALLOC_FLAG_REALLOC) & ~LIBXS_MALLOC_FLAG_SCRATCH,
        NULL/*extra*/, 0/*extra_size*/))
      {
        LIBXS_ASSERT(NULL != *memory);
        memcpy(*memory, pool_buffer, LIBXS_MIN(size, info->size));
      }
      else LIBXS_ASSERT(NULL == *memory);
      if (0 == counter) {
# if !defined(NDEBUG)
        status =
# endif
        internal_xfree(pool_buffer, info);
        assert(EXIT_SUCCESS == status); /* !LIBXS_ASSERT */
      }
    }
    else
#endif
    {
#if !defined(NDEBUG)
      status =
#endif
      libxs_xmalloc(memory, size, alignment/* no need here to determine alignment of given buffer */,
        (flags | LIBXS_MALLOC_FLAG_REALLOC) & ~LIBXS_MALLOC_FLAG_SCRATCH,
        NULL/*extra*/, 0/*extra_size*/);
      assert(EXIT_SUCCESS == status || NULL == *memory); /* !LIBXS_ASSERT */
    }
  }
}


#if defined(LIBXS_MALLOC_CTXFORM) || defined(LIBXS_MALLOC_HOOK_GLIBC)

LIBXS_API_INTERN void* internal_memalign(size_t /*alignment*/, size_t /*size*/, const void* /*caller*/);
LIBXS_API_INTERN void* internal_memalign(size_t alignment, size_t size, const void* caller)
{
  void* result;
  size_t scratch_nmallocs = 0;
#if defined(LIBXS_MALLOC_MMAP_HOOK)
  const int flags = LIBXS_MALLOC_FLAG_MMAP;
#else
  const int flags = LIBXS_MALLOC_FLAG_DEFAULT;
# if defined(LIBXS_MALLOC_HOOK_GLIBC)
  __memalign_hook = internal_memalign_hook;
  __malloc_hook = internal_malloc_hook;
  __realloc_hook = internal_realloc_hook;
  __free_hook = internal_free_hook;
# endif
#endif
  if (0 != (libxs_malloc_kind & 1) /* odd */
#if defined(LIBXS_MALLOC_HOOK_GLIBC) && (0 < LIBXS_MALLOC_HOOK_GLIBC)
    && (LIBXS_MALLOC_HOOK_GLIBC) < (scratch_nmallocs = LIBXS_ATOMIC_LOAD(&internal_malloc_scratch_nmallocs, LIBXS_ATOMIC_RELAXED))
#endif
    /* ensure allocations prior to main-function are non-scratch allocations */)
  {
    LIBXS_ASSERT(8 == sizeof(void*)); ((char*)&caller)[7] = 0;
    internal_scratch_malloc(&result, size, alignment, flags, caller);
  }
  else { /* even */
#if !defined(NDEBUG)
    int status =
#endif
    libxs_xmalloc(&result, size, alignment, flags, NULL/*extra*/, 0/*extra_size*/);
    assert(EXIT_SUCCESS == status || NULL == result); /* !LIBXS_ASSERT */
#if defined(LIBXS_MALLOC_HOOK_GLIBC) && (0 < LIBXS_MALLOC_HOOK_GLIBC)
    if ((LIBXS_MALLOC_HOOK_GLIBC) >= scratch_nmallocs) {
      LIBXS_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXS_ATOMIC_RELAXED);
    }
#endif
  }
#if !defined(LIBXS_MALLOC_MMAP_HOOK) && defined(LIBXS_MALLOC_HOOK_GLIBC)
  __memalign_hook = internal_memalign;
  __malloc_hook = internal_malloc;
  __realloc_hook = internal_realloc;
  __free_hook = internal_free;
#endif
  return result;
}

LIBXS_API_INTERN void* internal_malloc(size_t /*size*/, const void* /*caller*/);
LIBXS_API_INTERN void* internal_malloc(size_t size, const void* caller)
{
  return internal_memalign(0/*auto-align*/, size, caller);
}

LIBXS_API_INTERN void* internal_realloc(void* /*ptr*/, size_t /*size*/, const void* /*caller*/);
LIBXS_API_INTERN void* internal_realloc(void* ptr, size_t size, const void* caller)
{
  const int nzeros = LIBXS_INTRINSICS_BITSCANFWD64((uintptr_t)ptr), alignment = 1 << nzeros;
#if defined(LIBXS_MALLOC_MMAP_HOOK)
  const int flags = LIBXS_MALLOC_FLAG_MMAP;
#else
  const int flags = LIBXS_MALLOC_FLAG_DEFAULT;
# if defined(LIBXS_MALLOC_HOOK_GLIBC)
  __memalign_hook = internal_memalign_hook;
  __malloc_hook = internal_malloc_hook;
  __realloc_hook = internal_realloc_hook;
  __free_hook = internal_free_hook;
# endif
#endif
  LIBXS_ASSERT(0 == ((uintptr_t)ptr & ~(0xFFFFFFFFFFFFFFFF << nzeros)));
  if (0 == (libxs_malloc_kind & 1)) { /* even */
#if !defined(NDEBUG)
    int status =
#endif
    libxs_xmalloc(&ptr, size, alignment, flags | LIBXS_MALLOC_FLAG_REALLOC, NULL/*extra*/, 0/*extra_size*/);
    assert(EXIT_SUCCESS == status || NULL == ptr); /* !LIBXS_ASSERT */
  }
  else { /* odd */
    LIBXS_ASSERT(8 == sizeof(void*)); ((char*)&caller)[7] = 0;
    internal_scratch_malloc(&ptr, size, 0/*auto-align*/, flags | LIBXS_MALLOC_FLAG_REALLOC, caller);
  }
#if !defined(LIBXS_MALLOC_MMAP_HOOK) && defined(LIBXS_MALLOC_HOOK_GLIBC)
  __memalign_hook = internal_memalign;
  __malloc_hook = internal_malloc;
  __realloc_hook = internal_realloc;
  __free_hook = internal_free;
#endif
  return ptr;
}

LIBXS_API_INTERN void internal_free(void* /*ptr*/, const void* /*caller*/);
LIBXS_API_INTERN void internal_free(void* ptr, const void* caller)
{
  LIBXS_UNUSED(caller);
#if 0
  if (0 != libxs_malloc_kind)
#endif
  {
#if !defined(LIBXS_MALLOC_MMAP_HOOK) && defined(LIBXS_MALLOC_HOOK_GLIBC)
    __memalign_hook = internal_memalign_hook;
    __malloc_hook = internal_malloc_hook;
    __realloc_hook = internal_realloc_hook;
    __free_hook = internal_free_hook;
#endif
    libxs_free(ptr);
#if !defined(LIBXS_MALLOC_MMAP_HOOK) && defined(LIBXS_MALLOC_HOOK_GLIBC)
    __memalign_hook = internal_memalign;
    __malloc_hook = internal_malloc;
    __realloc_hook = internal_realloc;
    __free_hook = internal_free;
#endif
  }
}

#if defined(LIBXS_MALLOC_HOOK_GLIBC) && defined(LIBXS_GLIBC)
LIBXS_EXTERN_C void* (*volatile __memalign_hook)(size_t, size_t, const void*);
LIBXS_EXTERN_C void* (*volatile __malloc_hook)(size_t, const void*);
LIBXS_EXTERN_C void* (*volatile __realloc_hook)(void*, size_t, const void*);
LIBXS_EXTERN_C void  (*volatile __free_hook)(void*, const void*);
#endif

#endif /*defined(LIBXS_MALLOC_CTXFORM) || defined(LIBXS_MALLOC_HOOK_GLIBC)*/

LIBXS_API_INTERN int libxs_xset_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  int result = EXIT_SUCCESS;
  if (NULL != lock) {
    if (0 == libxs_ninit) libxs_init(); /* !LIBXS_INIT */
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
  }
  if (NULL != malloc_fn.function && NULL != free_fn.function) {
    libxs_default_allocator_context = context;
    libxs_default_malloc_fn = malloc_fn;
    libxs_default_free_fn = free_fn;
  }
  else {
    libxs_malloc_function internal_malloc_fn;
    libxs_free_function internal_free_fn;
    const void* internal_allocator = NULL;
#if defined(LIBXS_MALLOC_CTXFORM) /* debug purpose */
    internal_malloc_fn.ctx_form = internal_malloc;
    internal_free_fn.ctx_form = internal_free;
    internal_allocator = LIBXS_MALLOC_INTERNAL_CALLER;
#elif defined(__TBB)
    internal_malloc_fn.function = scalable_malloc;
    internal_free_fn.function = scalable_free;
    /*internal_allocator = NULL;*/
#else
# if defined(LIBXS_MALLOC_HOOK_GLIBC)
    if (1 < libxs_malloc_kind) { /* intercept */
      internal_memalign_hook = __memalign_hook;
      internal_malloc_hook = __malloc_hook;
      internal_realloc_hook = __realloc_hook;
      internal_free_hook = __free_hook;
      __memalign_hook = internal_memalign;
      __malloc_hook = internal_malloc;
      __realloc_hook = internal_realloc;
      __free_hook = internal_free;
    }
    else { /* reset hooks */
      __memalign_hook = internal_memalign_hook;
      __malloc_hook = internal_malloc_hook;
      __realloc_hook = internal_realloc_hook;
      __free_hook = internal_free_hook;
    }
# endif
    internal_malloc_fn.function = malloc;
    internal_free_fn.function = free;
    /*internal_allocator = NULL;*/
#endif
    if (NULL == malloc_fn.function && NULL == free_fn.function) {
      libxs_default_allocator_context = internal_allocator;
      libxs_default_malloc_fn = internal_malloc_fn;
      libxs_default_free_fn = internal_free_fn;
    }
    else { /* invalid allocator */
      static int error_once = 0;
      if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: allocator setup without malloc or free function!\n");
      }
      /* keep any valid (previously instantiated) default allocator */
      if (NULL == libxs_default_malloc_fn.function || NULL == libxs_default_free_fn.function) {
        libxs_default_allocator_context = internal_allocator;
        libxs_default_malloc_fn = internal_malloc_fn;
        libxs_default_free_fn = internal_free_fn;
      }
      result = EXIT_FAILURE;
    }
  }
  if (NULL != lock) {
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
  LIBXS_ASSERT(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_INTERN int libxs_xget_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn)
{
  int result = EXIT_SUCCESS;
  if (NULL != context || NULL != malloc_fn || NULL != free_fn) {
    if (NULL != lock) {
      if (0 == libxs_ninit) libxs_init(); /* !LIBXS_INIT */
      LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    }
    if (context) *context = libxs_default_allocator_context;
    if (NULL != malloc_fn) *malloc_fn = libxs_default_malloc_fn;
    if (NULL != free_fn) *free_fn = libxs_default_free_fn;
    if (NULL != lock) {
      LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
    }
  }
  else if (0 != libxs_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS ERROR: invalid signature used to get the default memory allocator!\n");
    }
    result = EXIT_FAILURE;
  }
  LIBXS_ASSERT(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_INTERN int libxs_xset_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (NULL != lock) {
    if (0 == libxs_ninit) libxs_init(); /* !LIBXS_INIT */
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
  }
  /* make sure the default allocator is setup before adopting it eventually */
  if (NULL == libxs_default_malloc_fn.function || NULL == libxs_default_free_fn.function) {
    const libxs_malloc_function null_malloc_fn = { NULL };
    const libxs_free_function null_free_fn = { NULL };
    libxs_xset_default_allocator(NULL/*already locked*/, NULL/*context*/, null_malloc_fn, null_free_fn);
  }
  if (NULL == malloc_fn.function && NULL == free_fn.function) { /* adopt default allocator */
    libxs_scratch_allocator_context = libxs_default_allocator_context;
    libxs_scratch_malloc_fn = libxs_default_malloc_fn;
    libxs_scratch_free_fn = libxs_default_free_fn;
  }
  else if (NULL != malloc_fn.function) {
    if (NULL == free_fn.function
      && /*warning*/(LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity)
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS WARNING: scratch allocator setup without free function!\n");
    }
    libxs_scratch_allocator_context = context;
    libxs_scratch_malloc_fn = malloc_fn;
    libxs_scratch_free_fn = free_fn; /* NULL allowed */
  }
  else { /* invalid scratch allocator */
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid scratch allocator (default used)!\n");
    }
    /* keep any valid (previously instantiated) scratch allocator */
    if (NULL == libxs_scratch_malloc_fn.function) {
      libxs_scratch_allocator_context = libxs_default_allocator_context;
      libxs_scratch_malloc_fn = libxs_default_malloc_fn;
      libxs_scratch_free_fn = libxs_default_free_fn;
    }
    result = EXIT_FAILURE;
  }
  if (NULL != lock) {
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
  LIBXS_ASSERT(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_INTERN int libxs_xget_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn)
{
  int result = EXIT_SUCCESS;
  if (NULL != context || NULL != malloc_fn || NULL != free_fn) {
    if (NULL != lock) {
      if (0 == libxs_ninit) libxs_init(); /* !LIBXS_INIT */
      LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    }
    if (context) *context = libxs_scratch_allocator_context;
    if (NULL != malloc_fn) *malloc_fn = libxs_scratch_malloc_fn;
    if (NULL != free_fn) *free_fn = libxs_scratch_free_fn;
    if (NULL != lock) {
      LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
    }
  }
  else if (0 != libxs_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS ERROR: invalid signature used to get the scratch memory allocator!\n");
    }
    result = EXIT_FAILURE;
  }
  LIBXS_ASSERT(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API int libxs_set_default_allocator(const void* context,
  libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  return libxs_xset_default_allocator(&libxs_lock_global, context, malloc_fn, free_fn);
}


LIBXS_API int libxs_get_default_allocator(const void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn)
{
  return libxs_xget_default_allocator(&libxs_lock_global, context, malloc_fn, free_fn);
}


LIBXS_API int libxs_set_scratch_allocator(const void* context,
  libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  return libxs_xset_scratch_allocator(&libxs_lock_global, context, malloc_fn, free_fn);
}


LIBXS_API int libxs_get_scratch_allocator(const void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn)
{
  return libxs_xget_scratch_allocator(&libxs_lock_global, context, malloc_fn, free_fn);
}


LIBXS_API int libxs_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG) || !defined(LIBXS_MALLOC_NOCRC)
  static int error_once = 0;
  if (NULL != size || NULL != extra)
#endif
  {
    const internal_malloc_info_type *const info = internal_malloc_info(memory);
    if (NULL != info) {
      if (size) *size = info->size;
      if (flags) *flags = info->flags;
      if (extra) *extra = info->pointer;
    }
    else {
#if !defined(LIBXS_MALLOC_NOCRC)
      if (NULL != memory /* library code is expected to be mute */
       && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity)
       && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS WARNING: checksum error for memory buffer %p!\n", memory);
      }
#endif
      if (NULL != size) *size = 0;
      if (NULL != flags) *flags = 0;
      if (NULL != extra) *extra = 0;
    }
  }
#if !defined(NDEBUG)
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: attachment error for memory buffer %p!\n", memory);
    }
    result = EXIT_FAILURE;
  }
  LIBXS_ASSERT(EXIT_SUCCESS == result);
#endif
  return result;
}


#if !defined(_WIN32)

LIBXS_API_INLINE void internal_xmalloc_mhint(void* buffer, size_t size)
{
  LIBXS_ASSERT((MAP_FAILED != buffer && NULL != buffer) || 0 == size);
#if defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE)
  /* proceed after failed madvise (even in case of an error; take what we got) */
  /* issue no warning as a failure seems to be related to the kernel version */
  madvise(buffer, size, MADV_NORMAL/*MADV_RANDOM*/
# if defined(MADV_NOHUGEPAGE) /* if not available, we then take what we got (THP) */
    | ((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) > size ? MADV_NOHUGEPAGE : 0)
# endif
# if defined(MADV_DONTDUMP)
    | ((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) > size ? 0 : MADV_DONTDUMP)
# endif
  );
#else
  LIBXS_UNUSED(buffer); LIBXS_UNUSED(size);
#endif
}


LIBXS_API_INLINE void* internal_xmalloc_xmap(const char* dir, size_t size, int flags, void** rx)
{
  void* result = MAP_FAILED;
  char filename[4096] = LIBXS_MALLOC_XMAP_TEMPLATE;
  int i = 0;
  LIBXS_ASSERT(NULL != rx);
  if (NULL != dir && 0 != *dir) {
    i = LIBXS_SNPRINTF(filename, sizeof(filename), "%s/" LIBXS_MALLOC_XMAP_TEMPLATE, dir);
  }
  if (0 <= i && i < (int)sizeof(filename)) {
    i = mkstemp(filename);
    if (0 <= i) {
      if (0 == unlink(filename) && 0 == ftruncate(i, size)) {
        void *const xmap = mmap(*rx, size, PROT_READ | PROT_EXEC, flags | MAP_SHARED /*| LIBXS_MAP_ANONYMOUS*/, i, 0/*offset*/);
        if (MAP_FAILED != xmap) {
          LIBXS_ASSERT(NULL != xmap);
          result = mmap(NULL, size, PROT_READ | PROT_WRITE, flags | MAP_SHARED /*| LIBXS_MAP_ANONYMOUS*/, i, 0/*offset*/);
          if (MAP_FAILED != result) {
            LIBXS_ASSERT(NULL != result);
            internal_xmalloc_mhint(xmap, size);
            *rx = xmap;
          }
          else {
            munmap(xmap, size);
          }
        }
      }
      close(i);
    }
  }
  return result;
}

#endif /*!defined(_WIN32)*/

LIBXS_API_INLINE void* internal_xmalloc_plain(
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn,
  size_t size, internal_malloc_info_type** info)
{
  void* result;
  LIBXS_ASSERT(NULL != info);
  if (NULL == *info || (
#if defined(LIBXS_MALLOC_CTXFORM)
    (free_fn.ctx_form != internal_free) &&
#elif defined(LIBXS_MALLOC_HOOK_GLIBC)
    (free_fn.ctx_form != internal_free || free_fn.ctx_form != __free_hook) &&
#endif
    free != free_fn.function))
  {
    if (NULL == *info || size != (*info)->size) {
      result = (NULL != malloc_fn.function
        ? (NULL == context ? malloc_fn.function(size) : malloc_fn.ctx_form(size, context))
        : (NULL));
    }
    else { /* no allocation, signal no-copy */
      result = (*info)->pointer;
      *info = NULL;
    }
  }
  else { /* reallocate */
#if !defined(LIBXS_MALLOC_DELETE_SAFE)
    LIBXS_ASSERT(NULL != (*info)->pointer);
#endif
#if defined(LIBXS_MALLOC_CTXFORM)
    if (internal_malloc == malloc_fn.ctx_form) {
      result = internal_realloc((*info)->pointer, size, context);
      *info = NULL; /* signal no-copy */
    }
    else if (malloc == malloc_fn.function) {
      result = realloc((*info)->pointer, size);
      *info = NULL; /* signal no-copy */
    }
    else /* error */
#elif defined(LIBXS_MALLOC_HOOK_GLIBC)
    if (internal_malloc == malloc_fn.ctx_form) {
      result = internal_realloc((*info)->pointer, size, context);
      *info = NULL; /* signal no-copy */
    }
    else if (__malloc_hook == malloc_fn.ctx_form) {
      result = __realloc_hook((*info)->pointer, size, context);
      *info = NULL; /* signal no-copy */
    }
    else /* error */
#else
    result = realloc((*info)->pointer, size);
    if (NULL != result) {
      *info = NULL; /* signal no-copy */
    }
    else /* error */
#endif
    { /* fall-back */
      result = (NULL != malloc_fn.function
        ? (NULL == context ? malloc_fn.function(size) : malloc_fn.ctx_form(size, context))
        : (NULL));
    }
  }
  return result;
}


LIBXS_API_INTERN int libxs_xmalloc(void** memory, size_t size, size_t alignment,
  int flags, const void* extra, size_t extra_size)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG)
  if (NULL != memory)
#endif
  {
    static int error_once = 0;
    if (0 != size) {
      /* ATOMIC BEGIN: this region should be atomic/locked */
        const void* context = libxs_default_allocator_context;
        libxs_malloc_function malloc_fn = libxs_default_malloc_fn;
        libxs_free_function free_fn = libxs_default_free_fn;
      /* ATOMIC END: this region should be atomic */
      internal_malloc_info_type* info = ((0 == (LIBXS_MALLOC_FLAG_REALLOC & flags) || NULL == *memory)
        ? NULL : internal_malloc_info(*memory));
      void *alloc_failed = NULL, *buffer = NULL, *reloc = (NULL == info ? NULL : info->pointer);
      size_t alloc_alignment = 0, alloc_size = 0;
#if defined(LIBXS_MALLOC_HOOK_GLIBC) && defined(LIBXS_MALLOC_MMAP_HOOK)
      if (0 != (LIBXS_MALLOC_FLAG_PRIVATE & flags) /*&& 1 < libxs_malloc_kind*/) {
        flags |= LIBXS_MALLOC_FLAG_MMAP;
      }
#endif
      flags |= LIBXS_MALLOC_FLAG_RW; /* normalize given flags since flags=0 is accepted as well */
      if (0 != (LIBXS_MALLOC_FLAG_SCRATCH & flags)) {
#if defined(LIBXS_MALLOC_MMAP_SCRATCH) /* try harder for uncommitted scratch memory */
        flags |= LIBXS_MALLOC_FLAG_MMAP;
#endif
        context = libxs_scratch_allocator_context;
        malloc_fn = libxs_scratch_malloc_fn;
        free_fn = libxs_scratch_free_fn;
      }
      if (NULL == malloc_fn.function || NULL == free_fn.function) {
        LIBXS_ASSERT(NULL == context);
        malloc_fn.function = malloc;
        free_fn.function = free;
      }
#if !defined(LIBXS_MALLOC_MMAP)
      if (0 == (LIBXS_MALLOC_FLAG_X & flags) && 0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) {
        alloc_alignment = (NULL == info ? libxs_alignment(size, alignment) : alignment);
        alloc_size = size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1;
        buffer = internal_xmalloc_plain(context, malloc_fn, free_fn, alloc_size, &info);
      }
      else
#endif
      if (NULL == info || size != info->size) {
#if defined(_WIN32)
        const int xflags = (0 != (LIBXS_MALLOC_FLAG_X & flags) ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
        static SIZE_T alloc_alignmax = 0, alloc_pagesize = 0;
        if (0 == alloc_alignmax) { /* first/one time */
          SYSTEM_INFO system_info;
          GetSystemInfo(&system_info);
          alloc_pagesize = system_info.dwPageSize;
          alloc_alignmax = GetLargePageMinimum();
        }
        if ((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) <= size) { /* attempt to use large pages */
          HANDLE process_token;
          alloc_alignment = (NULL == info
            ? (0 == alignment ? alloc_alignmax : libxs_lcm(alignment, alloc_alignmax))
            : libxs_lcm(alignment, alloc_alignmax));
          alloc_size = LIBXS_UP2(size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1, alloc_alignmax);
          if (TRUE == OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &process_token)) {
            TOKEN_PRIVILEGES tp;
            if (TRUE == LookupPrivilegeValue(NULL, TEXT("SeLockMemoryPrivilege"), &tp.Privileges[0].Luid)) {
              tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED; tp.PrivilegeCount = 1; /* enable privilege */
              if (TRUE == AdjustTokenPrivileges(process_token, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0)
                && ERROR_SUCCESS == GetLastError()/*may has failed (regardless of TRUE)*/)
              {
                /* VirtualAlloc cannot be used to reallocate memory */
                buffer = VirtualAlloc(NULL, alloc_size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, xflags);
              }
              tp.Privileges[0].Attributes = 0; /* disable privilege */
              AdjustTokenPrivileges(process_token, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
            }
            CloseHandle(process_token);
          }
        }
        else { /* small allocation using regular page-size */
          alloc_alignment = (NULL == info ? libxs_alignment(size, alignment) : alignment);
          alloc_size = LIBXS_UP2(size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1, alloc_pagesize);
        }
        if (alloc_failed == buffer) { /* small allocation or retry with regular page size */
          /* VirtualAlloc cannot be used to reallocate memory */
          buffer = VirtualAlloc(NULL, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
        }
        if (alloc_failed != buffer) {
          flags |= LIBXS_MALLOC_FLAG_MMAP; /* select the corresponding deallocation */
        }
        else if (0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) { /* fall-back allocation */
          buffer = internal_xmalloc_plain(context, malloc_fn, free_fn, alloc_size, &info);
        }
#else /* !defined(_WIN32) */
# if defined(MAP_HUGETLB)
        static int hugetlb = 1;
# endif
# if defined(MAP_32BIT)
        static int map32 = 1;
# endif
        int xflags = 0
# if defined(MAP_NORESERVE)
          | (((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) < size) ? 0 : MAP_NORESERVE)
# endif
# if defined(MAP_32BIT)
          | (((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) < size || 0 == map32) ? 0 : MAP_32BIT)
# endif
# if defined(MAP_HUGETLB) /* may fail depending on system settings */
          | (((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) < size && 0 != hugetlb) ? MAP_HUGETLB : 0)
# endif
# if defined(MAP_UNINITIALIZED) /* unlikely to be available */
          | MAP_UNINITIALIZED
# endif
# if defined(MAP_LOCKED) && /*disadvantage*/0
          | MAP_LOCKED
# endif
        ;
        /* prefault pages to avoid data race in Linux' page-fault handler pre-3.10.0-327 */
# if defined(MAP_HUGETLB) && defined(MAP_POPULATE)
        struct utsname osinfo;
        if (0 != (MAP_HUGETLB & xflags) && 0 <= uname(&osinfo) && 0 == strcmp("Linux", osinfo.sysname)) {
          unsigned int version_major = 3, version_minor = 10, version_update = 0, version_patch = 327;
          if (4 == sscanf(osinfo.release, "%u.%u.%u-%u", &version_major, &version_minor, &version_update, &version_patch) &&
            LIBXS_VERSION4(3, 10, 0, 327) > LIBXS_VERSION4(version_major, version_minor, version_update, version_patch))
          {
            /* TODO: lock across threads and processes */
            xflags |= MAP_POPULATE;
          }
        }
# endif
        alloc_alignment = (NULL == info ? libxs_alignment(size, alignment) : alignment);
        alloc_size = size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1;
        alloc_failed = MAP_FAILED;
        if (0 == (LIBXS_MALLOC_FLAG_X & flags)) {
          buffer = mmap(reloc, alloc_size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | xflags, -1, 0/*offset*/);
        }
        else { /* executable buffer requested */
          static /*LIBXS_TLS*/ int fallback = -1;
          if (0 > LIBXS_ATOMIC_LOAD(&fallback, LIBXS_ATOMIC_RELAXED)) { /* initialize fall-back allocation method */
            FILE *const selinux = fopen("/sys/fs/selinux/enforce", "rb");
            const char *const env = getenv("LIBXS_SE");
            if (NULL != selinux) {
              if (1 == fread(&libxs_se, 1/*sizeof(char)*/, 1/*count*/, selinux)) {
                libxs_se = ('0' != libxs_se ? 1 : 0);
              }
              else { /* conservative assumption in case of read-error */
                libxs_se = 1;
              }
              fclose(selinux);
            }
            LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&fallback, NULL == env
              /* libxs_se decides */
              ? (0 == libxs_se ? LIBXS_MALLOC_FINAL : LIBXS_MALLOC_FALLBACK)
              /* user's choice takes precedence */
              : ('0' != *env ? LIBXS_MALLOC_FALLBACK : LIBXS_MALLOC_FINAL),
              LIBXS_ATOMIC_SEQ_CST);
            LIBXS_ASSERT(0 <= fallback);
          }
          if (0 == fallback) {
            buffer = internal_xmalloc_xmap("/tmp", alloc_size, xflags, &reloc);
            if (alloc_failed == buffer) {
# if defined(MAP_32BIT)
              if (0 != (MAP_32BIT & xflags)) {
                buffer = internal_xmalloc_xmap("/tmp", alloc_size, xflags & ~MAP_32BIT, &reloc);
              }
              if (alloc_failed != buffer) map32 = 0; else
# endif
              fallback = 1;
            }
          }
          if (1 <= fallback) { /* continue with fall-back */
            if (1 == fallback) { /* 2nd try */
              static const char* envloc = NULL;
              if (NULL == envloc) {
                envloc = getenv("JITDUMPDIR");
                if (NULL == envloc) envloc = "";
              }
              buffer = internal_xmalloc_xmap(envloc, alloc_size, xflags, &reloc);
              if (alloc_failed == buffer) {
# if defined(MAP_32BIT)
                if (0 != (MAP_32BIT & xflags)) {
                  buffer = internal_xmalloc_xmap(envloc, alloc_size, xflags & ~MAP_32BIT, &reloc);
                }
                if (alloc_failed != buffer) map32 = 0; else
# endif
                fallback = 2;
              }
            }
            if (2 <= fallback) { /* continue with fall-back */
              if (2 == fallback) { /* 3rd try */
                static const char* envloc = NULL;
                if (NULL == envloc) {
                  envloc = getenv("HOME");
                  if (NULL == envloc) envloc = "";
                }
                buffer = internal_xmalloc_xmap(envloc, alloc_size, xflags, &reloc);
                if (alloc_failed == buffer) {
# if defined(MAP_32BIT)
                  if (0 != (MAP_32BIT & xflags)) {
                    buffer = internal_xmalloc_xmap(envloc, alloc_size, xflags & ~MAP_32BIT, &reloc);
                  }
                  if (alloc_failed != buffer) map32 = 0; else
# endif
                  fallback = 3;
                }
              }
              if (3 <= fallback) { /* continue with fall-back */
                if (3 == fallback) { /* 4th try */
                  buffer = mmap(reloc, alloc_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | xflags, -1, 0/*offset*/);
                  if (alloc_failed == buffer) {
# if defined(MAP_32BIT)
                    if (0 != (MAP_32BIT & xflags)) {
                      buffer = mmap(reloc, alloc_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | (xflags & ~MAP_32BIT), -1, 0/*offset*/);
                    }
                    if (alloc_failed != buffer) map32 = 0; else
# endif
                    fallback = 4;
                  }
                }
                if (4 == fallback && alloc_failed != buffer) { /* final */
                  LIBXS_ASSERT(fallback == LIBXS_MALLOC_FINAL + 1);
                  buffer = alloc_failed; /* trigger final fall-back */
                }
              }
            }
          }
        }
        if (alloc_failed != buffer) {
          LIBXS_ASSERT(NULL != buffer);
          flags |= LIBXS_MALLOC_FLAG_MMAP; /* select deallocation */
        }
        else { /* allocation failed */
# if defined(MAP_HUGETLB) /* no further attempts to rely on huge pages */
          if (0 != (xflags & MAP_HUGETLB)) {
            flags &= ~LIBXS_MALLOC_FLAG_MMAP; /* select deallocation */
            hugetlb = 0;
          }
# endif
# if defined(MAP_32BIT) /* no further attempts to map to 32-bit */
          if (0 != (xflags & MAP_32BIT)) {
            flags &= ~LIBXS_MALLOC_FLAG_MMAP; /* select deallocation */
            map32 = 0;
          }
# endif
          if (0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) { /* ultimate fall-back */
            buffer = (NULL != malloc_fn.function
              ? (NULL == context ? malloc_fn.function(alloc_size) : malloc_fn.ctx_form(alloc_size, context))
              : (NULL));
          }
          reloc = NULL;
        }
        if (MAP_FAILED != buffer && NULL != buffer) {
          internal_xmalloc_mhint(buffer, alloc_size);
        }
#endif
      }
      else {
        alloc_size = size + extra_size + sizeof(internal_malloc_info_type) + alignment - 1;
        alloc_alignment = alignment;
        buffer = info->pointer;
        flags |= info->flags;
        info = NULL;
      }
      if (alloc_failed != buffer && /*fall-back*/NULL != buffer) {
        char *const cbuffer = (char*)buffer, *const aligned = LIBXS_ALIGN(cbuffer + extra_size + sizeof(internal_malloc_info_type), alloc_alignment);
        internal_malloc_info_type *const buffer_info = (internal_malloc_info_type*)(aligned - sizeof(internal_malloc_info_type));
        LIBXS_ASSERT((aligned + size) <= (cbuffer + alloc_size));
        LIBXS_ASSERT(0 < alloc_alignment);
        if (NULL != extra || 0 == extra_size) {
          const char *const src = (const char*)extra;
          int i; for (i = 0; i < (int)extra_size; ++i) cbuffer[i] = src[i];
          if (0 == (LIBXS_MALLOC_FLAG_PRIVATE & flags)) { /* public */
            if (0 != (LIBXS_MALLOC_FLAG_SCRATCH & flags) && internal_malloc_scratch_size < alloc_size) {
              internal_malloc_scratch_size = alloc_size; /* accept data-race */
            }
          }
          else {
            if (0 == (LIBXS_MALLOC_FLAG_SCRATCH & flags)) {
              internal_malloc_private_size += alloc_size; /* accept data-race */
            }
            else if (internal_malloc_private_size < alloc_size) { /* scratch */
              internal_malloc_private_size = alloc_size; /* accept data-race */
            }
          }
          /* keep allocation function on record */
          if (0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) {
            buffer_info->context = context;
            buffer_info->free = free_fn;
          }
          else {
            buffer_info->free.function = NULL;
            buffer_info->context = NULL;
          }
          buffer_info->size = size; /* record user's size rather than allocated size */
          buffer_info->pointer = buffer;
          buffer_info->reloc = reloc;
          buffer_info->flags = flags;
#if !defined(LIBXS_MALLOC_NOCRC) /* calculate checksum over info */
          buffer_info->hash = libxs_crc32(LIBXS_MALLOC_SEED, buffer_info,
            /* info size minus actual hash value */
            (unsigned int)(((char*)&buffer_info->hash) - ((char*)buffer_info)));
#endif
          if (NULL != info) { /* copy previous content */
            memcpy(aligned, *memory, LIBXS_MIN(info->size, size));
            result = internal_xfree(*memory, info); /* !libxs_free */
            if (EXIT_SUCCESS == result) { /* finally commit/return allocated buffer */
              *memory = aligned;
            }
#if !defined(NDEBUG) /* display some extra context of the failure (reallocation) */
            else if (0 != libxs_verbosity /* library code is expected to be mute */
              && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
            {
              fprintf(stderr, "LIBXS ERROR: memory reallocation failed to release memory!\n");
            }
#endif
          }
          else { /* finally commit/return allocated buffer */
            *memory = aligned;
          }
        }
        else {
          if (0 != libxs_verbosity /* library code is expected to be mute */
            && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXS ERROR: incorrect extraneous data specification!\n");
          }
          result = EXIT_FAILURE;
        }
      }
      else {
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS ERROR: memory allocation error for size %" PRIuPTR " with flag=%i!\n", (uintptr_t)alloc_size, flags);
        }
        result = EXIT_FAILURE;
      }
    }
    else {
      if ((LIBXS_VERBOSITY_HIGH <= libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS WARNING: zero-sized memory allocation detected!\n");
      }
      *memory = NULL;
    }
  }
#if !defined(NDEBUG)
  else if (0 != size) {
    result = EXIT_FAILURE;
  }
#endif
  LIBXS_ASSERT(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_INTERN void libxs_xfree(const void* memory)
{
  /*const*/ internal_malloc_info_type *const info = internal_malloc_info(memory);
#if !defined(LIBXS_MALLOC_NOCRC)
  static int error_once = 0;
#endif
  if (NULL != info) {
#if !defined(NDEBUG)
    int status =
#endif
    internal_xfree(memory, info); /* !libxs_free */
    assert(EXIT_SUCCESS == status); /* !LIBXS_ASSERT */
  }
#if !defined(LIBXS_MALLOC_NOCRC)
  else if (NULL != memory /* library code is expected to be mute */
    && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity)
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: checksum error for memory buffer %p!\n", memory);
  }
#endif
}


#if defined(LIBXS_VTUNE)
LIBXS_API_INLINE void internal_get_vtune_jitdesc(const void* code,
  unsigned int code_id, size_t code_size, const char* code_name,
  LIBXS_VTUNE_JIT_DESC_TYPE* desc)
{
  LIBXS_ASSERT(NULL != code && 0 != code_id && 0 != code_size && NULL != desc);
  desc->method_id = code_id;
  /* incorrect constness (method_name) */
  desc->method_name = (char*)code_name;
  /* incorrect constness (method_load_address) */
  desc->method_load_address = (void*)code;
  desc->method_size = code_size;
  desc->line_number_size = 0;
  desc->line_number_table = NULL;
  desc->class_file_name = NULL;
  desc->source_file_name = NULL;
# if (2 <= LIBXS_VTUNE_JITVERSION)
  desc->module_name = "libxs.jit";
# endif
}
#endif


LIBXS_API_INTERN int libxs_malloc_attrib(void** memory, int flags, const char* name)
{
  internal_malloc_info_type *const info = (NULL != memory ? internal_malloc_info(*memory) : NULL);
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (NULL != info) {
    void *const buffer = info->pointer;
    const size_t size = info->size;
#if defined(_WIN32)
    LIBXS_ASSERT(NULL != buffer || 0 == size);
#else
    LIBXS_ASSERT((NULL != buffer && MAP_FAILED != buffer) || 0 == size);
#endif
    flags |= (info->flags & ~LIBXS_MALLOC_FLAG_RWX); /* merge with current flags */
    /* quietly keep the read permission, but eventually revoke write permissions */
    if (0 == (LIBXS_MALLOC_FLAG_W & flags) || 0 != (LIBXS_MALLOC_FLAG_X & flags)) {
      const size_t alignment = (size_t)(((const char*)(*memory)) - ((const char*)buffer));
      const size_t alloc_size = size + alignment;
      if (0 == (LIBXS_MALLOC_FLAG_X & flags)) { /* data-buffer; non-executable */
#if defined(_WIN32)
        /* TODO: implement memory protection under Microsoft Windows */
        LIBXS_UNUSED(alloc_size);
#else
        if (EXIT_SUCCESS != mprotect(buffer, alloc_size/*entire memory region*/, PROT_READ)
          && (LIBXS_VERBOSITY_HIGH <= libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS WARNING: read-only request for buffer failed!\n");
        }
#endif
      }
      else { /* executable buffer requested */
        void *const code_ptr = NULL != info->reloc ? ((void*)(((char*)info->reloc) + alignment)) : *memory;
        LIBXS_ASSERT(0 != (LIBXS_MALLOC_FLAG_X & flags));
        if (name && *name) { /* profiler support requested */
          if (0 > libxs_verbosity) { /* avoid dump when only the profiler is enabled */
            FILE* code_file = fopen(name, "rb");
            int diff = 0;
            if (NULL == code_file) { /* file does not exist */
              code_file = fopen(name, "wb");
              if (NULL != code_file) { /* dump byte-code into a file */
                fwrite(code_ptr, 1, size, code_file);
                fclose(code_file);
              }
            }
            else { /* check existing file */
              const char* check_a = (const char*)code_ptr;
              char check_b[4096];
              size_t rest = size;
              do {
                const size_t n = fread(check_b, 1, LIBXS_MIN(sizeof(check_b), rest), code_file);
                diff += memcmp(check_a, check_b, LIBXS_MIN(sizeof(check_b), n));
                check_a += n;
                rest -= n;
              } while (0 < rest && 0 == diff);
              fclose(code_file);
            }
            fprintf(stderr, "LIBXS-JIT-DUMP(ptr:file) %p : %s\n", code_ptr, name);
            if (0 != diff) { /* override existing dump and warn about erroneous condition */
              fprintf(stderr, "LIBXS ERROR: %s is shared by different code!\n", name);
              code_file = fopen(name, "wb");
              if (NULL != code_file) { /* dump byte-code into a file */
                fwrite(code_ptr, 1, size, code_file);
                fclose(code_file);
              }
            }
          }
#if defined(LIBXS_VTUNE)
          if (iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
            LIBXS_VTUNE_JIT_DESC_TYPE vtune_jit_desc;
            const unsigned int code_id = iJIT_GetNewMethodID();
            internal_get_vtune_jitdesc(code_ptr, code_id, size, name, &vtune_jit_desc);
            iJIT_NotifyEvent(LIBXS_VTUNE_JIT_LOAD, &vtune_jit_desc);
            info->code_id = code_id;
          }
          else {
            info->code_id = 0;
          }
#endif
#if defined(LIBXS_PERF)
          /* If JIT is enabled and a valid name is given, emit information for profiler
           * In jitdump case this needs to be done after mprotect as it gets overwritten
           * otherwise. */
          libxs_perf_dump_code(code_ptr, size, name);
#endif
        }
        if (NULL != info->reloc && info->pointer != info->reloc) {
#if defined(_WIN32)
          /* TODO: implement memory protection under Microsoft Windows */
#else
          /* memory is already protected at this point; relocate code */
          LIBXS_ASSERT(0 != (LIBXS_MALLOC_FLAG_MMAP & flags));
          *memory = code_ptr; /* relocate */
          info->pointer = info->reloc;
          info->reloc = NULL;
# if !defined(LIBXS_MALLOC_NOCRC) /* update checksum */
          info->hash = libxs_crc32(LIBXS_MALLOC_SEED, info,
            /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info)));
# endif   /* treat memory protection errors as soft error; ignore return value */
          munmap(buffer, alloc_size);
#endif
        }
#if !defined(_WIN32)
        else { /* malloc-based fall-back */
          int mprotect_result;
# if !defined(LIBXS_MALLOC_NOCRC) && defined(LIBXS_VTUNE) /* update checksum */
          info->hash = libxs_crc32(LIBXS_MALLOC_SEED, info,
            /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info)));
# endif   /* treat memory protection errors as soft error; ignore return value */
          mprotect_result = mprotect(buffer, alloc_size/*entire memory region*/, PROT_READ | PROT_EXEC);
          if (EXIT_SUCCESS != mprotect_result) {
            if (0 != libxs_se) { /* hard-error in case of SELinux */
              if (0 != libxs_verbosity /* library code is expected to be mute */
                && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
              {
                fprintf(stderr, "LIBXS ERROR: failed to allocate an executable buffer!\n");
              }
              result = mprotect_result;
            }
            else if ((LIBXS_VERBOSITY_HIGH <= libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
              && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
            {
              fprintf(stderr, "LIBXS WARNING: read-only request for JIT-buffer failed!\n");
            }
          }
        }
#endif
      }
    }
  }
  else if (NULL == memory || NULL == *memory) {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: libxs_malloc_attrib failed because NULL cannot be attributed!\n");
    }
    result = EXIT_FAILURE;
  }
#if !defined(LIBXS_MALLOC_NOCRC)
  else if (NULL != memory && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: checksum error for %s buffer %p!\n",
      0 != (LIBXS_MALLOC_FLAG_X & flags) ? "executable" : "memory", *memory);
  }
#endif
  return result;
}


LIBXS_API LIBXS_ATTRIBUTE_MALLOC void* libxs_aligned_malloc(size_t size, size_t alignment)
{
  void* result;
  LIBXS_INIT
  if (0 == (libxs_malloc_kind & 1)) { /* even */
#if !defined(NDEBUG)
    int status =
#endif
    libxs_xmalloc(&result, size, alignment, LIBXS_MALLOC_FLAG_DEFAULT, NULL/*extra*/, 0/*extra_size*/);
    assert(EXIT_SUCCESS == status || NULL == result); /* !LIBXS_ASSERT */
  }
  else { /* odd */
    const void *const caller = internal_malloc_site_auto();
    internal_scratch_malloc(&result, size, alignment,
#if defined(LIBXS_MALLOC_HOOK_GLIBC) && defined(LIBXS_MALLOC_MMAP_HOOK)
      LIBXS_MALLOC_FLAG_DEFAULT | LIBXS_MALLOC_FLAG_MMAP,
#else
      LIBXS_MALLOC_FLAG_DEFAULT,
#endif
      caller);
  }
  return result;
}


LIBXS_API void* libxs_realloc(size_t size, void* ptr)
{
  const int nzeros = LIBXS_INTRINSICS_BITSCANFWD64((uintptr_t)ptr), alignment = 1 << nzeros;
  LIBXS_ASSERT(0 == ((uintptr_t)ptr & ~(0xFFFFFFFFFFFFFFFF << nzeros)));
  LIBXS_INIT
  if (0 == (libxs_malloc_kind & 1)) { /* even */
#if !defined(NDEBUG)
    int status =
#endif
    libxs_xmalloc(&ptr, size, alignment, LIBXS_MALLOC_FLAG_REALLOC, NULL/*extra*/, 0/*extra_size*/);
    assert(EXIT_SUCCESS == status || NULL == ptr); /* !LIBXS_ASSERT */
  }
  else { /* odd */
    const void *const caller = internal_malloc_site_auto();
    internal_scratch_malloc(&ptr, size, alignment,
#if defined(LIBXS_MALLOC_HOOK_GLIBC) && defined(LIBXS_MALLOC_MMAP_HOOK)
      LIBXS_MALLOC_FLAG_REALLOC | LIBXS_MALLOC_FLAG_MMAP,
#else
      LIBXS_MALLOC_FLAG_REALLOC,
#endif
      caller);
  }
  return ptr;
}


LIBXS_API void* libxs_scratch_malloc(size_t size, size_t alignment, const void* caller)
{
  void* result;
  LIBXS_INIT
  internal_scratch_malloc(&result, size, alignment,
    LIBXS_MALLOC_INTERNAL_CALLER != caller ? LIBXS_MALLOC_FLAG_DEFAULT : LIBXS_MALLOC_FLAG_PRIVATE,
    caller);
  return result;
}


LIBXS_API LIBXS_ATTRIBUTE_MALLOC void* libxs_malloc(size_t size)
{
  return libxs_aligned_malloc(size, 0/*auto*/);
}


LIBXS_API void libxs_free(const void* memory)
{
  if (NULL != memory) {
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    internal_malloc_pool_type *const pool = internal_scratch_malloc_pool(memory);
    if (NULL != pool) { /* memory belongs to scratch domain */
      const size_t counter = LIBXS_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);
      LIBXS_ASSERT(pool->instance.buffer <= pool->instance.head);
      if (0 == counter) { /* reuse or reallocate scratch domain */
        const size_t scratch_size = internal_get_scratch_size(pool); /* exclude current pool */
        const size_t limit_size = libxs_scratch_limit - LIBXS_MIN(scratch_size, libxs_scratch_limit);
        const size_t maxsize = pool->instance.minsize + pool->instance.incsize;
        const size_t minsize = LIBXS_MIN(maxsize, limit_size);
        /* reuse scratch domain */
        if (minsize <= pool->instance.minsize) {
          pool->instance.head = LIBXS_MIN(pool->instance.head, (char*)memory);
        }
        else {
          const void *const pool_buffer = pool->instance.buffer;
          pool->instance.buffer = pool->instance.head = NULL;
# if defined(LIBXS_MALLOC_AFFINITY) && (0 != LIBXS_SYNC) && !defined(NDEBUG) /* library code is expected to be mute */
          if ((LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity) && libxs_get_tid() != pool->instance.tid) {
            static int error_once = 0;
            if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
              fprintf(stderr, "LIBXS WARNING: thread-id differs between allocation and deallocation!\n");
            }
          }
# endif
          libxs_xfree(pool_buffer);
        }
      }
      /* TODO: document/check that allocation/deallocation must follow the linear/scoped allocator policy */
      else { /* reuse scratch domain */
        pool->instance.head = LIBXS_MIN(pool->instance.head, (char*)memory);
      }
    }
    else
#endif
    { /* local */
      libxs_xfree(memory);
    }
  }
}


LIBXS_API void libxs_release_scratch(void)
{
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)LIBXS_UP2(internal_malloc_pool_buffer, LIBXS_CACHELINE);
  unsigned int i;
  LIBXS_ASSERT(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
  LIBXS_ASSERT(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
  /* acquire pending mallocs prior to cleanup (below libxs_xfree) */
  if (0 != libxs_verbosity) { /* library code is expected to be mute */
    libxs_scratch_info scratch_info;
    if (EXIT_SUCCESS == libxs_get_scratch_info(&scratch_info) && 0 < scratch_info.npending) {
      fprintf(stderr, "LIBXS ERROR: %lu pending scratch-memory allocations!\n",
        (unsigned long int)scratch_info.npending);
    }
  }
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &libxs_lock_global);
  for (i = 0; i < libxs_scratch_pools; ++i) libxs_xfree(pools[i].instance.buffer);
  memset(pools, 0, (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) * sizeof(internal_malloc_pool_type));
  internal_malloc_scratch_nmallocs = internal_malloc_scratch_size = 0; /* keep private watermark */
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxs_lock_global);
#endif
}


LIBXS_API int libxs_get_malloc_info(const void* memory, libxs_malloc_info* info)
{
  int result = EXIT_SUCCESS;
  if (NULL != info) {
    size_t size;
    result = libxs_get_malloc_xinfo(memory, &size, NULL/*flags*/, NULL/*extra*/);
    memset(info, 0, sizeof(libxs_malloc_info));
    if (EXIT_SUCCESS == result) {
      info->size = size;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API int libxs_get_scratch_info(libxs_scratch_info* info)
{
  int result = EXIT_SUCCESS;
  if (NULL != info) {
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    const internal_malloc_pool_type *const pools = (const internal_malloc_pool_type*)LIBXS_UP2(internal_malloc_pool_buffer, LIBXS_CACHELINE);
    unsigned int i;
    LIBXS_ASSERT(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
    memset(info, 0, sizeof(*info));
    info->npools = (unsigned int)LIBXS_MIN(internal_malloc_scratch_nmallocs, libxs_scratch_pools);
    info->npending = pools[0].instance.counter;
    info->nmallocs = internal_malloc_scratch_nmallocs;
    info->internal = internal_malloc_private_size;
    info->size = internal_malloc_scratch_size;
# if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    LIBXS_ASSERT(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
    for (i = 1; i < libxs_scratch_pools; ++i) {
      const internal_malloc_pool_type *const pool = pools + i;
      if ((LIBXS_MALLOC_INTERNAL_CALLER) != pool->instance.site) {
        info->npools += (unsigned int)LIBXS_MIN(pool->instance.minsize, 1);
        info->npending += pool->instance.counter;
      }
    }
# endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
#else
    memset(info, 0, sizeof(*info));
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API void libxs_set_scratch_limit(size_t nbytes)
{
  LIBXS_INIT
  libxs_scratch_limit = nbytes;
}


LIBXS_API size_t libxs_get_scratch_limit(void)
{
  LIBXS_INIT
  return libxs_scratch_limit;
}

