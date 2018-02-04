/******************************************************************************
** Copyright (c) 2014-2018, Intel Corporation                                **
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

/* must be defined *before* other files are included */
#if !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif
#include "libxs_trace.h"
#include "libxs_main.h"
#include "libxs_hash.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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
#   define LIBXS_MALLOC_FALLBACK 4
# endif
#else
# if !defined(LIBXS_MALLOC_FALLBACK)
#   define LIBXS_MALLOC_FALLBACK 0
# endif
#endif /*defined(LIBXS_VTUNE)*/
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
/* allows to reclaim a pool for a different thread */
#if !defined(LIBXS_MALLOC_NO_AFFINITY)
# define LIBXS_MALLOC_NO_AFFINITY ((unsigned int)-1)
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_JOIN) && 0
# define LIBXS_MALLOC_SCRATCH_JOIN
#endif
/* map memory for scratch buffers */
#if !defined(LIBXS_MALLOC_SCRATCH_MMAP) && 0
# define LIBXS_MALLOC_SCRATCH_MMAP
#endif
/* map memory even for non-executable buffers */
#if !defined(LIBXS_MALLOC_MMAP) && 0
# define LIBXS_MALLOC_MMAP
#endif


LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE internal_malloc_info_type {
  libxs_free_function free;
  void *context, *pointer, *reloc;
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
# if !defined(LIBXS_NO_SYNC)
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
/* LIBXS_ALIGNED appears to contradict LIBXS_API_VARIABLE, and causes multiple defined symbols (if below is seen in multiple translation units) */
LIBXS_API_VARIABLE(char internal_malloc_pool_buffer[(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS)*sizeof(internal_malloc_pool_type)+(LIBXS_CACHELINE)-1]);
#endif
LIBXS_API_VARIABLE(size_t internal_malloc_scratch_nmallocs);
LIBXS_API_VARIABLE(size_t internal_malloc_scratch_size);


LIBXS_API_DEFINITION size_t libxs_gcd(size_t a, size_t b)
{
  while (0 != b) {
    const size_t r = a % b;
    a = b;
    b = r;
  }
  return a;
}


LIBXS_API_DEFINITION size_t libxs_lcm(size_t a, size_t b)
{
  return (a * b) / libxs_gcd(a, b);
}


LIBXS_API_DEFINITION size_t libxs_alignment(size_t size, size_t alignment)
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


LIBXS_API_DEFINITION size_t libxs_offset(const size_t offset[], const size_t shape[], size_t ndims, size_t* size)
{
  size_t result = 0, size1 = 0;
  if (0 != ndims && 0 != shape) {
    size_t i;
    result = (0 != offset ? offset[0] : 0);
    size1 = shape[0];
    for (i = 1; i < ndims; ++i) {
      result += (0 != offset ? offset[i] : 0) * size1;
      size1 *= shape[i];
    }
  }
  if (0 != size) *size = size1;
  return result;
}


LIBXS_API_DEFINITION int libxs_xset_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK_DEFAULT)* lock,
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  int result = EXIT_SUCCESS;
  if (0 != lock) {
    LIBXS_INIT
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, lock);
  }
  if (0 != malloc_fn.function && 0 != free_fn.function) {
    libxs_default_allocator_context = context;
    libxs_default_malloc_fn = malloc_fn;
    libxs_default_free_fn = free_fn;
  }
  else {
    void* internal_allocator = 0;
    libxs_malloc_function internal_malloc_fn;
    libxs_free_function internal_free_fn;
#if defined(__TBB)
    internal_allocator = 0;
    internal_malloc_fn.function = scalable_malloc;
    internal_free_fn.function = scalable_free;
#else
    internal_allocator = 0;
    internal_malloc_fn.function = malloc;
    internal_free_fn.function = free;
#endif
    if (0 == malloc_fn.function && 0 == free_fn.function) {
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
      if (0 == libxs_default_malloc_fn.function || 0 == libxs_default_free_fn.function) {
        libxs_default_allocator_context = internal_allocator;
        libxs_default_malloc_fn = internal_malloc_fn;
        libxs_default_free_fn = internal_free_fn;
      }
      result = EXIT_FAILURE;
    }
  }
  if (0 != lock) {
    LIBXS_LOCK_RELEASE(LIBXS_LOCK_DEFAULT, lock);
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_DEFINITION int libxs_xget_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK_DEFAULT)* lock,
  void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn)
{
  int result = EXIT_SUCCESS;
  if (0 != context || 0 != malloc_fn || 0 != free_fn) {
    if (0 != lock) {
      LIBXS_INIT
      LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, lock);
    }
    if (context) *context = libxs_default_allocator_context;
    if (0 != malloc_fn) *malloc_fn = libxs_default_malloc_fn;
    if (0 != free_fn) *free_fn = libxs_default_free_fn;
    if (0 != lock) {
      LIBXS_LOCK_RELEASE(LIBXS_LOCK_DEFAULT, lock);
    }
  }
  else if (0 != libxs_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS ERROR: invalid signature used to get the default memory allocator!\n");
    }
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_DEFINITION int libxs_xset_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK_DEFAULT)* lock,
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (0 != lock) {
    LIBXS_INIT
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, lock);
  }
  /* make sure the default allocator is setup before adopting it eventually */
  if (0 == libxs_default_malloc_fn.function || 0 == libxs_default_free_fn.function) {
    const libxs_malloc_function null_malloc_fn = { 0 };
    const libxs_free_function null_free_fn = { 0 };
    libxs_xset_default_allocator(lock, 0/*context*/, null_malloc_fn, null_free_fn);
  }
  if (0 == malloc_fn.function && 0 == free_fn.function) { /* adopt default allocator */
    libxs_scratch_allocator_context = libxs_default_allocator_context;
    libxs_scratch_malloc_fn = libxs_default_malloc_fn;
    libxs_scratch_free_fn = libxs_default_free_fn;
  }
  else if (0 != malloc_fn.function) {
    if (0 == free_fn.function
      && /*warning*/(1 < libxs_verbosity || 0 > libxs_verbosity)
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
    if (0 == libxs_scratch_malloc_fn.function) {
      libxs_scratch_allocator_context = libxs_default_allocator_context;
      libxs_scratch_malloc_fn = libxs_default_malloc_fn;
      libxs_scratch_free_fn = libxs_default_free_fn;
    }
    result = EXIT_FAILURE;
  }
  if (0 != lock) {
    LIBXS_LOCK_RELEASE(LIBXS_LOCK_DEFAULT, lock);
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_DEFINITION int libxs_xget_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK_DEFAULT)* lock,
  void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn)
{
  int result = EXIT_SUCCESS;
  if (0 != context || 0 != malloc_fn || 0 != free_fn) {
    if (0 != lock) {
      LIBXS_INIT
      LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, lock);
    }
    if (context) *context = libxs_scratch_allocator_context;
    if (0 != malloc_fn) *malloc_fn = libxs_scratch_malloc_fn;
    if (0 != free_fn) *free_fn = libxs_scratch_free_fn;
    if (0 != lock) {
      LIBXS_LOCK_RELEASE(LIBXS_LOCK_DEFAULT, lock);
    }
  }
  else if (0 != libxs_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS ERROR: invalid signature used to get the scratch memory allocator!\n");
    }
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_DEFINITION int libxs_set_default_allocator(void* context,
  libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  return libxs_xset_default_allocator(&libxs_lock_global, context, malloc_fn, free_fn);
}


LIBXS_API_DEFINITION int libxs_get_default_allocator(void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn)
{
  return libxs_xget_default_allocator(&libxs_lock_global, context, malloc_fn, free_fn);
}


LIBXS_API_DEFINITION int libxs_set_scratch_allocator(void* context,
  libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  return libxs_xset_scratch_allocator(&libxs_lock_global, context, malloc_fn, free_fn);
}


LIBXS_API_DEFINITION int libxs_get_scratch_allocator(void** context,
  libxs_malloc_function* malloc_fn, libxs_free_function* free_fn)
{
  return libxs_xget_scratch_allocator(&libxs_lock_global, context, malloc_fn, free_fn);
}


LIBXS_API_INLINE internal_malloc_info_type* internal_malloc_info(const void* memory)
{
  internal_malloc_info_type *const result = (internal_malloc_info_type*)
    (0 != memory ? (((const char*)memory) - sizeof(internal_malloc_info_type)) : 0);
#if defined(LIBXS_MALLOC_NOCRC)
  return result;
#else /* calculate checksum over info */
  return (0 != result && result->hash == libxs_crc32(
    result, ((const char*)&result->hash) - ((const char*)result),
    LIBXS_MALLOC_SEED)) ? result : 0;
#endif
}


LIBXS_API_DEFINITION int libxs_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG) || !defined(LIBXS_MALLOC_NOCRC)
  static int error_once = 0;
  if (0 != size || 0 != extra)
#endif
  {
    const internal_malloc_info_type *const info = internal_malloc_info(memory);
    if (0 != info) {
      if (size) *size = info->size;
      if (flags) *flags = info->flags;
      if (extra) *extra = info->pointer;
    }
    else {
#if !defined(LIBXS_MALLOC_NOCRC)
      if (0 != memory && (1 < libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
       && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS WARNING: checksum error for memory buffer %p!\n", memory);
      }
#endif
      if (size) *size = 0;
      if (flags) *flags = 0;
      if (extra) *extra = 0;
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
  assert(EXIT_SUCCESS == result);
#endif
  return result;
}


#if !defined(_WIN32)

LIBXS_API_INLINE void internal_mhint(void* buffer, size_t size)
{
  assert((MAP_FAILED != buffer && 0 != buffer) || 0 == size);
  /* proceed after failed madvise (even in case of an error; take what we got) */
  /* issue no warning as a failure seems to be related to the kernel version */
  madvise(buffer, size, MADV_NORMAL/*MADV_RANDOM*/
#if defined(MADV_NOHUGEPAGE) /* if not available, we then take what we got (THP) */
    | ((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) > size ? MADV_NOHUGEPAGE : 0)
#endif
#if defined(MADV_DONTDUMP)
    | ((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) > size ? 0 : MADV_DONTDUMP)
#endif
  );
}


LIBXS_API_INLINE void* internal_xmap(const char* dir, size_t size, int flags, void** rx)
{
  void* result = MAP_FAILED;
  char filename[4096];
  int i = LIBXS_SNPRINTF(filename, sizeof(filename), "%s/.libxs_XXXXXX.jit", dir);
  assert(0 != rx);
  if (0 <= i && i < (int)sizeof(filename)) {
    i = mkstemps(filename, 4);
    if (-1 != i && 0 == unlink(filename) && 0 == ftruncate(i, size)) {
      void *const xmap = mmap(0, size, PROT_READ | PROT_EXEC, flags | MAP_SHARED /*| LIBXS_MAP_ANONYMOUS*/, i, 0);
      if (MAP_FAILED != xmap) {
        assert(0 != xmap);
        result = mmap(0, size, PROT_READ | PROT_WRITE, flags | MAP_SHARED /*| LIBXS_MAP_ANONYMOUS*/, i, 0);
        if (MAP_FAILED != result) {
          assert(0 != result);
          internal_mhint(xmap, size);
          *rx = xmap;
        }
        else {
          munmap(xmap, size);
        }
      }
    }
  }
  return result;
}

#endif /*!defined(_WIN32)*/

LIBXS_API_DEFINITION int libxs_xmalloc(void** memory, size_t size, size_t alignment,
  int flags, const void* extra, size_t extra_size)
{
  int result = EXIT_SUCCESS;
  if (memory) {
    static int error_once = 0;
    if (0 < size) {
      const size_t internal_size = size + extra_size + sizeof(internal_malloc_info_type);
      /* ATOMIC BEGIN: this region should be atomic/locked */
        void* context = libxs_default_allocator_context;
        libxs_malloc_function malloc_fn = libxs_default_malloc_fn;
        libxs_free_function free_fn = libxs_default_free_fn;
      /* ATOMIC END: this region should be atomic */
      size_t alloc_alignment = 0, alloc_size = 0;
      void *alloc_failed = 0, *buffer = 0, *reloc = 0;
      if (0 != (LIBXS_MALLOC_FLAG_SCRATCH & flags)) {
#if defined(LIBXS_MALLOC_SCRATCH_MMAP) /* try harder for uncommitted scratch memory */
        flags |= LIBXS_MALLOC_FLAG_MMAP;
#endif
        context = libxs_scratch_allocator_context;
        malloc_fn = libxs_scratch_malloc_fn;
        free_fn = libxs_scratch_free_fn;
      }
      flags |= LIBXS_MALLOC_FLAG_RW; /* normalize given flags since flags=0 is accepted as well */
#if !defined(LIBXS_MALLOC_MMAP)
      if (0 == (LIBXS_MALLOC_FLAG_X & flags) && 0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) {
        alloc_alignment = (0 == alignment ? libxs_alignment(size, alignment) : alignment);
        alloc_size = internal_size + alloc_alignment - 1;
        buffer = 0 != malloc_fn.function
          ? (0 == context ? malloc_fn.function(alloc_size) : malloc_fn.ctx_form(context, alloc_size))
          : 0;
      }
      else
#endif
      {
#if defined(_WIN32)
        const int xflags = (0 != (LIBXS_MALLOC_FLAG_X & flags) ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
        if ((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) > size) {
          alloc_alignment = (0 == alignment ? libxs_alignment(size, alignment) : alignment);
          alloc_size = internal_size + alloc_alignment - 1;
          buffer = VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
        }
        else {
          HANDLE process_token;
          const SIZE_T alloc_alignmax = GetLargePageMinimum();
          /* respect user-requested alignment */
          alloc_alignment = 0 == alignment ? alloc_alignmax : libxs_lcm(alignment, alloc_alignmax);
          alloc_size = LIBXS_UP2(internal_size, alloc_alignment); /* assume that alloc_alignment is POT */
          if (TRUE == OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &process_token)) {
            TOKEN_PRIVILEGES tp;
            if (TRUE == LookupPrivilegeValue(NULL, TEXT("SeLockMemoryPrivilege"), &tp.Privileges[0].Luid)) {
              tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED; tp.PrivilegeCount = 1; /* enable privilege */
              if ( TRUE == AdjustTokenPrivileges(process_token, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0)
                && ERROR_SUCCESS == GetLastError()/*may has failed (regardless of TRUE)*/)
              {
                buffer = VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, xflags);
              }
              tp.Privileges[0].Attributes = 0; /* disable privilege */
              AdjustTokenPrivileges(process_token, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
            }
            CloseHandle(process_token);
          }
          if (alloc_failed == buffer) { /* retry allocation with regular page size */
            alloc_alignment = (0 == alignment ? libxs_alignment(size, alignment) : alignment);
            alloc_size = internal_size + alloc_alignment - 1;
            buffer = VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
          }
        }
        if (alloc_failed != buffer) {
          flags |= LIBXS_MALLOC_FLAG_MMAP; /* select the corresponding deallocation */
        }
        else if (0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) { /* fall-back allocation */
          buffer = 0 != malloc_fn.function
            ? (0 == context ? malloc_fn.function(alloc_size) : malloc_fn.ctx_form(context, alloc_size))
            : 0;
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
        alloc_alignment = (0 == alignment ? libxs_alignment(size, alignment) : alignment);
        alloc_size = internal_size + alloc_alignment - 1;
        alloc_failed = MAP_FAILED;
        if (0 == (LIBXS_MALLOC_FLAG_X & flags)) {
          buffer = mmap(0, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | xflags, -1, 0);
        }
        else {
          static /*LIBXS_TLS*/ int fallback = -1;
          if (0 > fallback) { /* initialize fall-back allocation method */
            const char *const env = getenv("LIBXS_SE");
            int sevalue = 0;
            if (0 == env || 0 == *env) {
              FILE *const selinux = fopen("/sys/fs/selinux/enforce", "rb");
              if (0 != selinux) {
                if (1 != fread(&sevalue, sizeof(int), 1/*count*/, selinux)) {
                  sevalue = 1; /* conservative assumption in case of an error */
                }
                fclose(selinux);
              }
            }
            else { /* user's choice takes precedence */
              sevalue = atoi(env);
            }
            fallback = (0 == sevalue ? 4 : LIBXS_MALLOC_FALLBACK);
          }
          if (0 == fallback) {
            buffer = internal_xmap("/tmp", alloc_size, xflags, &reloc);
            if (alloc_failed == buffer) {
# if defined(MAP_32BIT)
              if (0 != (MAP_32BIT & xflags)) {
                buffer = internal_xmap("/tmp", alloc_size, xflags & ~MAP_32BIT, &reloc);
              }
              if (alloc_failed != buffer) map32 = 0; else
# endif
              fallback = 1;
            }
          }
          if (1 <= fallback) { /* continue with fall-back */
            if (1 == fallback) { /* 2nd try */
              buffer = internal_xmap(".", alloc_size, xflags, &reloc);
              if (alloc_failed == buffer) {
# if defined(MAP_32BIT)
                if (0 != (MAP_32BIT & xflags)) {
                  buffer = internal_xmap(".", alloc_size, xflags & ~MAP_32BIT, &reloc);
                }
                if (alloc_failed != buffer) map32 = 0; else
# endif
                fallback = 2;
              }
            }
            if (2 <= fallback) { /* continue with fall-back */
              if (2 == fallback) { /* 3rd try */
                const char *const envloc = getenv("HOME");
                buffer = internal_xmap(envloc, alloc_size, xflags, &reloc);
                if (alloc_failed == buffer) {
# if defined(MAP_32BIT)
                  if (0 != (MAP_32BIT & xflags)) {
                    buffer = internal_xmap(envloc, alloc_size, xflags & ~MAP_32BIT, &reloc);
                  }
                  if (alloc_failed != buffer) map32 = 0; else
# endif
                  fallback = 3;
                }
              }
              if (3 <= fallback) { /* continue with fall-back */
                if (3 == fallback) { /* 4th try */
                  const char *const envloc = getenv("JITDUMPDIR");
                  buffer = internal_xmap(envloc, alloc_size, xflags, &reloc);
                  if (alloc_failed == buffer) {
# if defined(MAP_32BIT)
                    if (0 != (MAP_32BIT & xflags)) {
                      buffer = internal_xmap(envloc, alloc_size, xflags & ~MAP_32BIT, &reloc);
                    }
                    if (alloc_failed != buffer) map32 = 0; else
# endif
                    fallback = 4;
                  }
                }
                if (4 <= fallback) { /* continue with fall-back */
                  if (4 == fallback) { /* 5th try */
                    buffer = mmap(0, alloc_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | xflags, -1, 0);
                    if (alloc_failed == buffer) {
# if defined(MAP_32BIT)
                      if (0 != (MAP_32BIT & xflags)) {
                        buffer = mmap(0, alloc_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                          MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | (xflags & ~MAP_32BIT), -1, 0);
                      }
                      if (alloc_failed != buffer) map32 = 0; else
# endif
                      fallback = 5;
                    }
                  }
                  if (5 == fallback && alloc_failed != buffer) { /* final */
                    buffer = alloc_failed; /* trigger final fall-back */
                  }
                }
              }
            }
          }
        }
        if (alloc_failed != buffer) {
          assert(0 != buffer);
          flags |= LIBXS_MALLOC_FLAG_MMAP; /* select deallocation */
        }
        else {
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
          if (0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) { /* fall-back allocation */
            buffer = 0 != malloc_fn.function
              ? (0 == context ? malloc_fn.function(alloc_size) : malloc_fn.ctx_form(context, alloc_size))
              : 0;
            reloc = buffer;
          }
          else {
            reloc = 0;
          }
        }
        if (MAP_FAILED != buffer && 0 != buffer) {
          internal_mhint(buffer, alloc_size);
        }
#endif
      }
      if (alloc_failed != buffer && /*fall-back*/0 != buffer) {
        char *const aligned = LIBXS_ALIGN(((char*)buffer) + extra_size + sizeof(internal_malloc_info_type), alloc_alignment);
        internal_malloc_info_type *const info = (internal_malloc_info_type*)(aligned - sizeof(internal_malloc_info_type));
        assert((aligned + size) <= (((char*)buffer) + alloc_size));
        if (0 != extra) memcpy(buffer, extra, extra_size);
#if !defined(NDEBUG)
        else if (0 == extra && 0 != extra_size) {
          result = EXIT_FAILURE;
        }
#endif
        if (0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) {
          info->context = context;
          info->free = free_fn;
        }
        else {
          info->free.function = 0;
          info->context = 0;
        }
        info->pointer = buffer;
        info->reloc = reloc;
        info->size = size;
        info->flags = flags;
#if !defined(LIBXS_MALLOC_NOCRC) /* calculate checksum over info */
        info->hash = libxs_crc32(info, /* info size minus actual hash value */
          (unsigned int)(((char*)&info->hash) - ((char*)info)), LIBXS_MALLOC_SEED);
#endif
        *memory = aligned;
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
      if ((1 < libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS WARNING: zero-sized memory allocation detected!\n");
      }
      *memory = 0;
    }
  }
  else if (0 != size) {
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_DEFINITION int libxs_xfree(const void* memory)
{
  /*const*/ internal_malloc_info_type *const info = internal_malloc_info(memory);
  int result = EXIT_SUCCESS;
#if !defined(_WIN32) || !defined(LIBXS_BUILD) || !defined(LIBXS_MALLOC_NOCRC)
  static int error_once = 0;
#endif
  if (0 != info) {
    void *const buffer = info->pointer;
#if !defined(LIBXS_BUILD) /* sanity check */
    if (0 != buffer || 0 == info->size)
#endif
    {
      assert(0 != buffer || 0 == info->size);
      if (0 == (LIBXS_MALLOC_FLAG_MMAP & info->flags)) {
        if (0 != info->free.function) {
#if 0 /* prevent double-delete */
          info->pointer = 0; info->size = 0;
#endif
          if (0 == info->context) {
            info->free.function(buffer);
          }
          else {
            info->free.ctx_form(info->context, buffer);
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
        result = (0 == buffer || FALSE != VirtualFree(buffer, 0, MEM_RELEASE)) ? EXIT_SUCCESS : EXIT_FAILURE;
#else /* defined(_WIN32) */
        {
          const size_t alloc_size = info->size + (((const char*)memory) - ((const char*)buffer));
          void *const reloc = info->reloc;
          const int flags = info->flags;
          if (0 != munmap(buffer, alloc_size)) {
            if (0 != libxs_verbosity /* library code is expected to be mute */
             && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
            {
              const char *const error_message = strerror(errno);
              fprintf(stderr, "LIBXS ERROR: %s (munmap error #%i for range %p+%llu)!\n",
                error_message, errno, buffer, (unsigned long long)alloc_size);
            }
            result = EXIT_FAILURE;
          }
          if (0 != (LIBXS_MALLOC_FLAG_X & flags) && EXIT_SUCCESS == result
           && 0 != reloc && MAP_FAILED != reloc && buffer != reloc
           && 0 != munmap(reloc, alloc_size))
          {
            if (0 != libxs_verbosity /* library code is expected to be mute */
             && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
            {
              const char *const error_message = strerror(errno);
              fprintf(stderr, "LIBXS ERROR: %s (munmap error #%i for range %p+%llu)!\n",
                error_message, errno, reloc, (unsigned long long)alloc_size);
            }
            result = EXIT_FAILURE;
          }
        }
#endif
      }
    }
#if !defined(LIBXS_BUILD)
    else if ((1 < libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS WARNING: attempt to release memory from non-matching implementation!\n");
    }
#endif
  }
#if !defined(LIBXS_MALLOC_NOCRC)
  else if (0 != memory && (1 < libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: checksum error for memory buffer %p!\n", memory);
  }
#endif
  assert(EXIT_SUCCESS == result);
  return result;
}


#if defined(LIBXS_VTUNE)
LIBXS_API_INLINE void internal_get_vtune_jitdesc(const void* code,
  unsigned int code_id, size_t code_size, const char* code_name,
  LIBXS_VTUNE_JIT_DESC_TYPE* desc)
{
  assert(0 != code && 0 != code_id && 0 != code_size && 0 != desc);
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


LIBXS_API_DEFINITION int libxs_malloc_attrib(void** memory, int flags, const char* name)
{
  internal_malloc_info_type *const info = 0 != memory ? internal_malloc_info(*memory) : 0;
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (0 != info) {
    void *const buffer = info->pointer;
    const size_t size = info->size;
#if defined(_WIN32)
    assert(0 != buffer || 0 == size);
#else
    assert((0 != buffer && MAP_FAILED != buffer) || 0 == size);
#endif
    /* quietly keep the read permission, but eventually revoke write permissions */
    if (0 == (LIBXS_MALLOC_FLAG_W & flags) || 0 != (LIBXS_MALLOC_FLAG_X & flags)) {
      const size_t alignment = (size_t)(((const char*)(*memory)) - ((const char*)buffer));
      const size_t alloc_size = size + alignment;
      if (0 == (LIBXS_MALLOC_FLAG_X & flags)) {
#if defined(_WIN32)
        /* TODO: implement memory protection under Microsoft Windows */
        LIBXS_UNUSED(alloc_size);
#else
        /* treat memory protection errors as soft error; ignore return value */
        mprotect(buffer, alloc_size/*entire memory region*/, PROT_READ);
#endif
      }
      else {
        void *const code_ptr =
#if !defined(_WIN32)
          0 != (LIBXS_MALLOC_FLAG_MMAP & flags) ? ((void*)(((char*)info->reloc) + alignment)) :
#endif
          *memory;
        assert(0 != (LIBXS_MALLOC_FLAG_X & flags));
        if (name && *name) { /* profiler support requested */
          if (0 > libxs_verbosity) { /* avoid dump when only the profiler is enabled */
            FILE *const code_file = fopen(name, "wb");
            if (0 != code_file) { /* dump byte-code into a file and print function pointer and filename */
              fprintf(stderr, "LIBXS-JIT-DUMP(ptr:file) %p : %s\n", code_ptr, name);
              fwrite(code_ptr, 1, size, code_file);
              fclose(code_file);
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
        if (0 != (LIBXS_MALLOC_FLAG_MMAP & flags)) {
#if defined(_WIN32)
          /* TODO: implement memory protection under Microsoft Windows */
#else
          /* memory is already protected at this point; relocate code */
          assert(info->pointer != info->reloc);
          *memory = code_ptr; /* relocate */
          info->pointer = info->reloc;
          info->reloc = 0;
# if !defined(LIBXS_MALLOC_NOCRC) /* update checksum */
          info->hash = libxs_crc32(info, /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info)), LIBXS_MALLOC_SEED);
# endif
          /* treat memory protection errors as soft error; ignore return value */
          munmap(buffer, alloc_size);
#endif
        }
#if !defined(_WIN32)
        else { /* malloc-based fall-back */
# if !defined(LIBXS_MALLOC_NOCRC) && defined(LIBXS_VTUNE) /* update checksum */
          info->hash = libxs_crc32(info, /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info)), LIBXS_MALLOC_SEED);
# endif
          /* treat memory protection errors as soft error; ignore return value */
          mprotect(buffer, alloc_size/*entire memory region*/, PROT_READ | PROT_EXEC);
        }
#endif
      }
    }
  }
  else if (0 == memory || 0 == *memory) {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: libxs_malloc_attrib failed because NULL cannot be attributed!\n");
    }
    result = EXIT_FAILURE;
  }
#if !defined(LIBXS_MALLOC_NOCRC)
  else if (0 != memory && (1 < libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: checksum error for %s buffer %p!\n",
      0 != (LIBXS_MALLOC_FLAG_X & flags) ? "executable" : "memory", *memory);
  }
#endif
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXS_API_DEFINITION void* libxs_aligned_malloc(size_t size, size_t alignment)
{
  void* result = 0;
  LIBXS_INIT
  return 0 == libxs_xmalloc(&result, size, alignment, LIBXS_MALLOC_FLAG_DEFAULT,
    0/*extra*/, 0/*extra_size*/) ? result : 0;
}


LIBXS_API_INLINE const void* internal_malloc_site(const char* site)
{
  const void* result;
  if (0 != site) {
#if !defined(LIBXS_STRING_POOLING)
    if ((LIBXS_MALLOC_SCRATCH_INTERNAL) != site) {
      const uintptr_t hash = libxs_hash(site, strlen(site), LIBXS_MALLOC_SEED);
      result = (const void*)((LIBXS_MALLOC_SCRATCH_INTERNAL_SITE) != hash ? hash : (hash - 1));
      assert((LIBXS_MALLOC_SCRATCH_INTERNAL) != result);
    }
    else
#endif
    {
      result = site;
    }
  }
  else {
#if defined(NDEBUG) /* internal_malloc_site is inlined */
# if defined(_WIN32) || defined(__CYGWIN__)
    void* stacktrace[] = { 0, 0, 0 };
# else
    void* stacktrace[] = { 0, 0 };
# endif
#else /* not inlined */
    void* stacktrace[] = { 0, 0, 0, 0 };
#endif
    const unsigned int size = sizeof(stacktrace) / sizeof(*stacktrace);
    result = (size == libxs_backtrace(stacktrace, size) ? stacktrace[size-1] : 0);
  }
  return result;
}


LIBXS_API_INLINE size_t internal_get_scratch_size(const internal_malloc_pool_type* exclude)
{
  size_t result = 0;
#if !defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) || (0 >= (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  LIBXS_UNUSED(exclude);
#else
  const internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)LIBXS_UP2(internal_malloc_pool_buffer, LIBXS_CACHELINE);
  const internal_malloc_pool_type* pool = pools;
  const internal_malloc_info_type* info = internal_malloc_info(pool->instance.buffer);
  unsigned int i;
  assert(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
  if (0 != info && pool != exclude && (LIBXS_MALLOC_SCRATCH_INTERNAL) != pool->instance.site) {
    result = info->size;
  }
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  assert(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
  for (i = 1; i < libxs_scratch_pools; ++i) {
    pool = pools + i; info = internal_malloc_info(pool->instance.buffer);
    if (0 != info && pool != exclude && (LIBXS_MALLOC_SCRATCH_INTERNAL) != pool->instance.site) {
      result += info->size;
    }
  }
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
  return result;
}


LIBXS_API_DEFINITION void* libxs_scratch_malloc(size_t size, size_t alignment, const char* caller)
{
  static int error_once = 0;
  size_t local_size = 0;
  void* result = 0;
  LIBXS_INIT
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  if (0 < libxs_scratch_pools && 0 < libxs_scratch_limit) {
    internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)((uintptr_t)(internal_malloc_pool_buffer + (LIBXS_CACHELINE)-1) & ~((LIBXS_CACHELINE)-1));
    internal_malloc_pool_type *const end = pools + libxs_scratch_pools, *pool0 = end, *pool = pools;
    const void *const site = internal_malloc_site(caller);
    const size_t align_size = (0 == alignment ? libxs_alignment(size, alignment) : alignment);
    const size_t alloc_size = size + align_size - 1;
#if !defined(LIBXS_NO_SYNC)
    const unsigned int tid = libxs_get_tid();
#endif
    assert(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    for (; pool != end; ++pool) { /* find exact matching pool */
# if !defined(LIBXS_NO_SYNC)
      if (site == pool->instance.site && (tid == pool->instance.tid
#   if defined(LIBXS_MALLOC_NO_AFFINITY)
        || (LIBXS_MALLOC_NO_AFFINITY) == pool->instance.tid
#   endif
      )) break;
# else
      if (site == pool->instance.site) break;
# endif
      if (end == pool0 && 0 == pool->instance.site) pool0 = pool;
    }
#endif
    if (end == pool) pool = pool0; /* fall-back to new pool */
    if (end != pool) {
      const internal_malloc_info_type* info = 0;
      const size_t counter = LIBXS_ATOMIC_ADD_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);
      info = internal_malloc_info(pool->instance.buffer);

      if (0 == pool->instance.buffer && 1 == counter) {
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
        assert(1 <= libxs_scratch_scale);
        assert(0 == pool->instance.head);
        pool->instance.incsize = 0; /* reset */
        pool->instance.minsize = minsize;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
        pool->instance.site = site;
# if !defined(LIBXS_NO_SYNC)
        pool->instance.tid = tid;
# endif
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_JOIN)
        if (alloc_size <= minsize && EXIT_SUCCESS == libxs_xmalloc(&result, minsize, 0/*auto*/,
#else
        if (EXIT_SUCCESS == libxs_xmalloc(&result, minsize, 0/*auto*/,
#endif
          LIBXS_MALLOC_FLAG_SCRATCH, 0/*extra*/, 0/*extra_size*/))
        {
          pool->instance.buffer = (char*)result;
          pool->instance.head = pool->instance.buffer + alloc_size;
          result = LIBXS_ALIGN((char*)result, align_size);
          if ((LIBXS_MALLOC_SCRATCH_INTERNAL) != caller) {
            if (internal_malloc_scratch_size < scratch_size) internal_malloc_scratch_size = scratch_size;
            LIBXS_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXS_ATOMIC_RELAXED);
#if defined(LIBXS_MALLOC_SCRATCH_JOIN) /* library code is expected to be mute */
            if (limit_size < maxsize && (1 < libxs_verbosity || 0 > libxs_verbosity)
              && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
            {
              fprintf(stderr, "LIBXS WARNING: scratch memory domain exhausted!\n");
            }
#endif
          }
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
            else if ((LIBXS_MALLOC_SCRATCH_INTERNAL) != caller
              && (1 < libxs_verbosity || 0 > libxs_verbosity))
            {
              fprintf(stderr, "LIBXS WARNING: scratch memory domain exhausted!\n");
            }
#endif
          }
          local_size = size;
        }
      }
      else {
        const size_t used_size = pool->instance.head - pool->instance.buffer;
        const size_t pool_size = (0 != info ? info->size : 0);
        const size_t req_size = alloc_size + used_size;
        assert(used_size <= pool_size);

        if (req_size <= pool_size) { /* fast path: draw from pool-buffer */
          void *const headaddr = &pool->instance.head;
          uintptr_t headptr = LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)((uintptr_t*)headaddr, alloc_size, LIBXS_ATOMIC_SEQ_CST);
          char *const head = (char*)headptr;
          result = LIBXS_ALIGN(head - alloc_size, align_size);
        }
        else { /* fall-back to local memory allocation */
          const size_t incsize = req_size - LIBXS_MIN(pool_size, req_size);
          pool->instance.incsize = LIBXS_MAX(pool->instance.incsize, incsize);
          LIBXS_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);
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
    if (EXIT_SUCCESS != libxs_xmalloc(&result, local_size, alignment,
      LIBXS_MALLOC_FLAG_SCRATCH, 0/*extra*/, 0/*extra_size*/) &&
      /* library code is expected to be mute */0 != libxs_verbosity &&
      1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: scratch memory fall-back failed!\n");
    }
    if ((LIBXS_MALLOC_SCRATCH_INTERNAL) != caller) {
      LIBXS_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXS_ATOMIC_RELAXED);
    }
  }
  return result;
}


LIBXS_API_DEFINITION void* libxs_malloc(size_t size)
{
  return libxs_aligned_malloc(size, 0/*auto*/);
}


LIBXS_API_DEFINITION void libxs_free(const void* memory)
{
  if (0 != memory) {
    unsigned int npools = 0, i = 0;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)((uintptr_t)(internal_malloc_pool_buffer + (LIBXS_CACHELINE)-1) & ~((LIBXS_CACHELINE)-1));
    const char *const buffer = (const char*)memory;

    assert(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
    assert(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
    npools = libxs_scratch_pools;

    for (; i < npools; ++i) {
      internal_malloc_pool_type *const pool = pools + i;
      const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer);

      /* check if memory belongs to scratch domain or local domain */
      if (0 != info && pool->instance.buffer <= buffer && buffer < (pool->instance.buffer + info->size)) {
        const size_t counter = LIBXS_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);

        assert(pool->instance.buffer <= pool->instance.head);
        if (0 == counter) { /* reallocate scratch domain */
          const size_t scratch_size = internal_get_scratch_size(pool); /* exclude current pool */
          const size_t limit_size = libxs_scratch_limit - LIBXS_MIN(scratch_size, libxs_scratch_limit);
          const size_t maxsize = pool->instance.minsize + pool->instance.incsize;
          const size_t minsize = LIBXS_MIN(maxsize, limit_size);

          if (pool->instance.minsize < minsize) {
            pool->instance.buffer = pool->instance.head = 0;
# if !defined(LIBXS_NO_SYNC)
#   if !defined(NDEBUG) /* library code is expected to be mute */
            if ((1 < libxs_verbosity || 0 > libxs_verbosity) && libxs_get_tid() != pool->instance.tid) {
              static int error_once = 0;
              if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
                fprintf(stderr, "LIBXS WARNING: thread-id differs between allocation and deallocation!\n");
              }
            }
#   endif
#   if defined(LIBXS_MALLOC_NO_AFFINITY) /* allow to reclaim the pool for any tid */
            if (limit_size < maxsize) pool->instance.tid = LIBXS_MALLOC_NO_AFFINITY;
#   endif
# endif
            libxs_xfree(pool->instance.buffer);
          }
          else { /* reuse scratch domain */
            pool->instance.head = (char*)LIBXS_MIN(pool->instance.head, buffer);
          }
          if ((LIBXS_MALLOC_SCRATCH_INTERNAL) != pool->instance.site) {
            const size_t watermark = scratch_size + info->size;
            if (internal_malloc_scratch_size < watermark) internal_malloc_scratch_size = watermark;
          }
        }
        /* TODO: document/check that allocation/deallocation must follow the linear/scoped allocator policy */
        else { /* reuse scratch domain */
          pool->instance.head = (char*)LIBXS_MIN(pool->instance.head, buffer);
        }
        i = npools + 1; /* break */
      }
    }
#endif
    if (i == npools) { /* local */
      libxs_xfree(memory);
    }
  }
}


LIBXS_API_DEFINITION void libxs_release_scratch(void)
{
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)((uintptr_t)(internal_malloc_pool_buffer + (LIBXS_CACHELINE)-1) & ~((LIBXS_CACHELINE)-1));
  unsigned int i;
  assert(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
  assert(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
  /* acquire pending mallocs prior to cleanup (below libxs_xfree) */
  if (0 != libxs_verbosity) { /* library code is expected to be mute */
    libxs_scratch_info scratch_info;
    if (EXIT_SUCCESS == libxs_get_scratch_info(&scratch_info) && 0 < scratch_info.npending) {
      fprintf(stderr, "LIBXS ERROR: %lu pending scratch-memory allocations!\n",
        (unsigned long int)scratch_info.npending);
    }
  }
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_DEFAULT, &libxs_lock_global);
  for (i = 0; i < libxs_scratch_pools; ++i) libxs_xfree(pools[i].instance.buffer);
  memset(pools, 0, (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) * sizeof(internal_malloc_pool_type));
  internal_malloc_scratch_nmallocs = internal_malloc_scratch_size = 0;
  LIBXS_LOCK_RELEASE(LIBXS_LOCK_DEFAULT, &libxs_lock_global);
#endif
}


LIBXS_API_DEFINITION int libxs_get_malloc_info(const void* memory, libxs_malloc_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != info) {
    size_t size;
    result = libxs_get_malloc_xinfo(memory, &size, 0/*flags*/, 0/*extra*/);
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


LIBXS_API_DEFINITION int libxs_get_scratch_info(libxs_scratch_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != info) {
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    const internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)((uintptr_t)(internal_malloc_pool_buffer + (LIBXS_CACHELINE)-1) & ~((LIBXS_CACHELINE)-1));
    unsigned int i;
    assert(sizeof(internal_malloc_pool_type) <= (LIBXS_CACHELINE));
    memset(info, 0, sizeof(libxs_scratch_info));
    info->npending = pools[0].instance.counter;
    info->nmallocs = internal_malloc_scratch_nmallocs;
    info->npools = LIBXS_MIN(1, libxs_scratch_pools);
    info->size = internal_malloc_scratch_size;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    assert(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
    for (i = 1; i < libxs_scratch_pools; ++i) {
      const internal_malloc_pool_type *const pool = pools + i;
      if ((LIBXS_MALLOC_SCRATCH_INTERNAL) != pool->instance.site) {
        info->npools += (unsigned int)LIBXS_MIN(pool->instance.minsize, 1);
        info->npending += pool->instance.counter;
      }
    }
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
#else
    memset(info, 0, sizeof(*info));
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_DEFINITION void libxs_set_scratch_limit(size_t nbytes)
{
  LIBXS_INIT
  libxs_scratch_limit = nbytes;
}


LIBXS_API_DEFINITION size_t libxs_get_scratch_limit(void)
{
  LIBXS_INIT
  return libxs_scratch_limit;
}


LIBXS_API_DEFINITION unsigned int libxs_hash(const void* data, size_t size, unsigned int seed)
{
  LIBXS_INIT
  return libxs_crc32(data, size, seed);
}

