/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_trace.h"
#include "libxs_main.h"
#include "libxs_hash.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD)))
# include <features.h>
# include <malloc.h>
#endif
#if !defined(LIBXS_MALLOC_GLIBC)
# if defined(__GLIBC__)
#   define LIBXS_MALLOC_GLIBC __GLIBC__
# else
#   define LIBXS_MALLOC_GLIBC 6
# endif
#endif
#if defined(_WIN32)
# include <windows.h>
# include <malloc.h>
# include <intrin.h>
#else
# include <sys/mman.h>
# if defined(__linux__)
#   include <linux/mman.h>
#   include <sys/syscall.h>
# endif
# if defined(MAP_POPULATE)
#   include <sys/utsname.h>
# endif
# include <sys/types.h>
# include <sys/stat.h>
# include <unistd.h>
# include <errno.h>
# if defined(__MAP_ANONYMOUS)
#   define LIBXS_MAP_ANONYMOUS __MAP_ANONYMOUS
# elif defined(MAP_ANONYMOUS)
#   define LIBXS_MAP_ANONYMOUS MAP_ANONYMOUS
# elif defined(MAP_ANON)
#   define LIBXS_MAP_ANONYMOUS MAP_ANON
# else
#  define LIBXS_MAP_ANONYMOUS 0x20
# endif
# if defined(MAP_SHARED)
#   define LIBXS_MAP_SHARED MAP_SHARED
# else
#   define LIBXS_MAP_SHARED 0
# endif
LIBXS_EXTERN int ftruncate(int, off_t) LIBXS_THROW;
LIBXS_EXTERN int mkstemp(char*) LIBXS_NOTHROW;
#endif
#if !defined(LIBXS_MALLOC_FINAL)
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

#if !defined(LIBXS_MALLOC_ALIGNMAX)
# define LIBXS_MALLOC_ALIGNMAX (2 << 20) /* 2 MB */
#endif
#if !defined(LIBXS_MALLOC_ALIGNFCT)
# define LIBXS_MALLOC_ALIGNFCT 16
#endif
#if !defined(LIBXS_MALLOC_SEED)
# define LIBXS_MALLOC_SEED 1051981
#endif

#if !defined(LIBXS_MALLOC_HOOK_KMP) && 0
# define LIBXS_MALLOC_HOOK_KMP
#endif
#if !defined(LIBXS_MALLOC_HOOK_QKMALLOC) && 0
# define LIBXS_MALLOC_HOOK_QKMALLOC
#endif
#if !defined(LIBXS_MALLOC_HOOK_IMALLOC) && 1
# define LIBXS_MALLOC_HOOK_IMALLOC
#endif
#if !defined(LIBXS_MALLOC_HOOK_CHECK) && 0
# define LIBXS_MALLOC_HOOK_CHECK 1
#endif

#if !defined(LIBXS_MALLOC_CRC_LIGHT) && !defined(_DEBUG) && 1
# define LIBXS_MALLOC_CRC_LIGHT
#endif
#if !defined(LIBXS_MALLOC_CRC_OFF)
# if defined(NDEBUG) && !defined(LIBXS_MALLOC_HOOK)
#   define LIBXS_MALLOC_CRC_OFF
# elif !defined(LIBXS_BUILD)
#   define LIBXS_MALLOC_CRC_OFF
# endif
#endif

#if !defined(LIBXS_MALLOC_SCRATCH_LIMIT)
# define LIBXS_MALLOC_SCRATCH_LIMIT 0xFFFFFFFF /* ~4 GB */
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_PADDING)
# define LIBXS_MALLOC_SCRATCH_PADDING LIBXS_CACHELINE
#endif
/* pointers are checked first if they belong to scratch */
#if !defined(LIBXS_MALLOC_SCRATCH_DELETE_FIRST) && 1
# define LIBXS_MALLOC_SCRATCH_DELETE_FIRST
#endif
/* can clobber memory if allocations are not exactly scoped */
#if !defined(LIBXS_MALLOC_SCRATCH_TRIM_HEAD) && 0
# define LIBXS_MALLOC_SCRATCH_TRIM_HEAD
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_JOIN) && 1
# define LIBXS_MALLOC_SCRATCH_JOIN
#endif
#if !defined(LIBXS_MALLOC_HUGE_PAGES) && 1
# define LIBXS_MALLOC_HUGE_PAGES
#endif
#if !defined(LIBXS_MALLOC_LOCK_PAGES) && 1
/* 0: on-map, 1: mlock, 2: mlock2/on-fault */
# define LIBXS_MALLOC_LOCK_PAGES 1
#endif
#if !defined(LIBXS_MALLOC_LOCK_ALL) && \
     defined(LIBXS_MALLOC_ALIGN_ALL) && 0
# define LIBXS_MALLOC_LOCK_ALL
#endif
/* protected against double-delete (if possible) */
#if !defined(LIBXS_MALLOC_DELETE_SAFE) && 0
# define LIBXS_MALLOC_DELETE_SAFE
#elif !defined(NDEBUG)
# define LIBXS_MALLOC_DELETE_SAFE
#endif
/* map memory for scratch buffers */
#if !defined(LIBXS_MALLOC_MMAP_SCRATCH) && 1
# define LIBXS_MALLOC_MMAP_SCRATCH
#endif
/* map memory for hooked allocation */
#if !defined(LIBXS_MALLOC_MMAP_HOOK) && 1
# define LIBXS_MALLOC_MMAP_HOOK
#endif
/* map memory also for non-executable buffers */
#if !defined(LIBXS_MALLOC_MMAP) && 1
# define LIBXS_MALLOC_MMAP
#endif

#define INTERNAL_MEMALIGN_REAL(RESULT, ALIGNMENT, SIZE) do { \
  const size_t internal_memalign_real_alignment_ = INTERNAL_MALLOC_AUTOALIGN(SIZE, ALIGNMENT); \
  (RESULT) = (0 != internal_memalign_real_alignment_ \
    ? __real_memalign(internal_memalign_real_alignment_, SIZE) \
    : __real_malloc(SIZE)); \
} while(0)
#define INTERNAL_REALLOC_REAL(RESULT, PTR, SIZE) (RESULT) = __real_realloc(PTR, SIZE)
#define INTERNAL_FREE_REAL(PTR) __real_free(PTR)

#if defined(LIBXS_MALLOC_LOCK_ALL) && defined(LIBXS_MALLOC_LOCK_PAGES) && 0 != (LIBXS_MALLOC_LOCK_PAGES)
# if 1 == (LIBXS_MALLOC_LOCK_PAGES) || !defined(MLOCK_ONFAULT) || !defined(SYS_mlock2)
#   define INTERNAL_MALLOC_LOCK_PAGES(BUFFER, SIZE) if ((LIBXS_MALLOC_ALIGNFCT * LIBXS_MALLOC_ALIGNMAX) <= (SIZE)) \
      mlock(BUFFER, SIZE)
# else
#   define INTERNAL_MALLOC_LOCK_PAGES(BUFFER, SIZE) if ((LIBXS_MALLOC_ALIGNFCT * LIBXS_MALLOC_ALIGNMAX) <= (SIZE)) \
      syscall(SYS_mlock2, BUFFER, SIZE, MLOCK_ONFAULT)
# endif
#else
# define INTERNAL_MALLOC_LOCK_PAGES(BUFFER, SIZE)
#endif

#if defined(LIBXS_MALLOC_ALIGN_ALL)
# define INTERNAL_MALLOC_AUTOALIGN(SIZE, ALIGNMENT) libxs_alignment(SIZE, ALIGNMENT)
#else
# define INTERNAL_MALLOC_AUTOALIGN(SIZE, ALIGNMENT) (ALIGNMENT)
#endif

#if defined(LIBXS_MALLOC_HOOK) && defined(LIBXS_MALLOC) && (0 != LIBXS_MALLOC)
# define INTERNAL_MEMALIGN_HOOK(RESULT, FLAGS, ALIGNMENT, SIZE, CALLER) { \
    const int internal_memalign_hook_recursive_ = LIBXS_ATOMIC_ADD_FETCH( \
      &internal_malloc_recursive, 1, LIBXS_ATOMIC_RELAXED); \
    if ( 1 < internal_memalign_hook_recursive_ /* protect against recursion */ \
      || 0 == (internal_malloc_kind & 1) || 0 >= internal_malloc_kind \
      || (internal_malloc_limit[0] > (SIZE)) \
      || (internal_malloc_limit[1] < (SIZE) && 0 != internal_malloc_limit[1])) \
    { \
      INTERNAL_MEMALIGN_REAL(RESULT, ALIGNMENT, SIZE); \
    } \
    else { /* redirect */ \
      LIBXS_INIT \
      if (NULL == (CALLER)) { /* libxs_trace_caller_id may allocate memory */ \
        internal_scratch_malloc(&(RESULT), SIZE, ALIGNMENT, FLAGS, \
          libxs_trace_caller_id(0/*level*/)); \
      } \
      else { \
        internal_scratch_malloc(&(RESULT), SIZE, ALIGNMENT, FLAGS, CALLER); \
      } \
    } \
    LIBXS_ATOMIC_SUB_FETCH(&internal_malloc_recursive, 1, LIBXS_ATOMIC_RELAXED); \
  }
# define INTERNAL_REALLOC_HOOK(RESULT, FLAGS, PTR, SIZE, CALLER) \
  if (0 == (internal_malloc_kind & 1) || 0 >= internal_malloc_kind \
    /*|| (0 != LIBXS_ATOMIC_LOAD(&internal_malloc_recursive, LIBXS_ATOMIC_RELAXED))*/ \
    || (internal_malloc_limit[0] > (SIZE)) \
    || (internal_malloc_limit[1] < (SIZE) && 0 != internal_malloc_limit[1])) \
  { \
    INTERNAL_REALLOC_REAL(RESULT, PTR, SIZE); \
  } \
  else { \
    const int nzeros = LIBXS_INTRINSICS_BITSCANFWD64((uintptr_t)(PTR)), alignment = 1 << nzeros; \
    LIBXS_ASSERT(0 == ((uintptr_t)(PTR) & ~(0xFFFFFFFFFFFFFFFF << nzeros))); \
    if (NULL == (CALLER)) { /* libxs_trace_caller_id may allocate memory */ \
      internal_scratch_malloc(&(PTR), SIZE, (size_t)alignment, FLAGS, \
        libxs_trace_caller_id(0/*level*/)); \
    } \
    else { \
      internal_scratch_malloc(&(PTR), SIZE, (size_t)alignment, FLAGS, CALLER); \
    } \
    (RESULT) = (PTR); \
  }
# define INTERNAL_FREE_HOOK(PTR, CALLER) { \
    LIBXS_UNUSED(CALLER); \
    if (0 == (internal_malloc_kind & 1) || 0 >= internal_malloc_kind \
      /*|| (0 != LIBXS_ATOMIC_LOAD(&internal_malloc_recursive, LIBXS_ATOMIC_RELAXED))*/ \
    ){ \
      INTERNAL_FREE_REAL(PTR); \
    } \
    else { /* recognize pointers not issued by LIBXS */ \
      libxs_free(PTR); \
    } \
  }
#elif defined(LIBXS_MALLOC_ALIGN_ALL)
# define INTERNAL_MEMALIGN_HOOK(RESULT, FLAGS, ALIGNMENT, SIZE, CALLER) do { \
    LIBXS_UNUSED(FLAGS); LIBXS_UNUSED(CALLER); \
    INTERNAL_MEMALIGN_REAL(RESULT, ALIGNMENT, SIZE); \
    INTERNAL_MALLOC_LOCK_PAGES(RESULT, SIZE); \
  } while(0)
# define INTERNAL_REALLOC_HOOK(RESULT, FLAGS, PTR, SIZE, CALLER) do { \
    LIBXS_UNUSED(FLAGS); LIBXS_UNUSED(CALLER); \
    INTERNAL_REALLOC_REAL(RESULT, PTR, SIZE); \
    INTERNAL_MALLOC_LOCK_PAGES(RESULT, SIZE); \
  } while(0)
# define INTERNAL_FREE_HOOK(PTR, CALLER) do { \
    LIBXS_UNUSED(CALLER); \
    INTERNAL_FREE_REAL(PTR); \
  } while(0)
#endif

#if !defined(WIN32)
# if defined(MAP_32BIT)
#   define INTERNAL_XMALLOC_MAP32(ENV, MAPSTATE, MFLAGS, SIZE, BUFFER, REPTR) \
    if (MAP_FAILED == (BUFFER) && 0 != (MAP_32BIT & (MFLAGS))) { \
      (BUFFER) = internal_xmalloc_xmap(ENV, SIZE, (MFLAGS) & ~MAP_32BIT, REPTR); \
      if (MAP_FAILED != (BUFFER)) (MAPSTATE) = 0; \
    }
# else
#   define INTERNAL_XMALLOC_MAP32(ENV, MAPSTATE, MFLAGS, SIZE, BUFFER, REPTR)
# endif

# define INTERNAL_XMALLOC(I, ENTRYPOINT, ENVVAR, ENVDEF, MAPSTATE, MFLAGS, SIZE, BUFFER, REPTR) \
  if ((ENTRYPOINT) <= (I) && (MAP_FAILED == (BUFFER) || NULL == (BUFFER))) { \
    static const char* internal_xmalloc_env_ = NULL; \
    if (NULL == internal_xmalloc_env_) { \
      internal_xmalloc_env_ = getenv(ENVVAR); \
      if (NULL == internal_xmalloc_env_) internal_xmalloc_env_ = ENVDEF; \
    } \
    (BUFFER) = internal_xmalloc_xmap(internal_xmalloc_env_, SIZE, MFLAGS, REPTR); \
    INTERNAL_XMALLOC_MAP32(internal_xmalloc_env_, MAPSTATE, MFLAGS, SIZE, BUFFER, REPTR); \
    if (MAP_FAILED != (BUFFER)) (ENTRYPOINT) = (I); \
  }

# define INTERNAL_XMALLOC_WATERMARK(NAME, WATERMARK, LIMIT, SIZE) { \
  const size_t internal_xmalloc_watermark_ = (WATERMARK) + (SIZE) / 2; /* accept data-race */ \
  if (internal_xmalloc_watermark_ < (LIMIT)) { \
    static size_t internal_xmalloc_watermark_verbose_ = 0; \
    (LIMIT) = internal_xmalloc_watermark_; /* accept data-race */ \
    if (internal_xmalloc_watermark_verbose_ < internal_xmalloc_watermark_ && \
      (LIBXS_VERBOSITY_HIGH <= libxs_verbosity || 0 > libxs_verbosity)) \
    { /* muted */ \
      char internal_xmalloc_watermark_buffer_[32]; \
      /* coverity[check_return] */ \
      libxs_format_size(internal_xmalloc_watermark_buffer_, sizeof(internal_xmalloc_watermark_buffer_), \
        internal_xmalloc_watermark_, "KM", "B", 10); \
      fprintf(stderr, "LIBXS WARNING: " NAME " watermark reached at %s!\n", internal_xmalloc_watermark_buffer_); \
      internal_xmalloc_watermark_verbose_ = internal_xmalloc_watermark_; \
    } \
  } \
}

# define INTERNAL_XMALLOC_KIND(KIND, NAME, FLAG, FLAGS, MFLAGS, WATERMARK, LIMIT, INFO, SIZE, BUFFER) \
  if (0 != ((KIND) & (MFLAGS))) { \
    if (MAP_FAILED != (BUFFER)) { \
      LIBXS_ASSERT(NULL != (BUFFER)); \
      LIBXS_ATOMIC_ADD_FETCH(&(WATERMARK), SIZE, LIBXS_ATOMIC_RELAXED); \
      (FLAGS) |= (FLAG); \
    } \
    else { /* retry */ \
      (BUFFER) = mmap(NULL == (INFO) ? NULL : (INFO)->pointer, SIZE, PROT_READ | PROT_WRITE, \
        MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | ((MFLAGS) & ~(KIND)), -1, 0/*offset*/); \
      if (MAP_FAILED != (BUFFER)) { /* successful retry */ \
        LIBXS_ASSERT(NULL != (BUFFER)); \
        INTERNAL_XMALLOC_WATERMARK(NAME, WATERMARK, LIMIT, SIZE); \
      } \
      (FLAGS) &= ~(FLAG); \
    } \
  } \
  else (FLAGS) &= ~(FLAG)
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
#if !defined(LIBXS_MALLOC_CRC_OFF) /* hash *must* be the last entry */
  unsigned int hash;
#endif
} internal_malloc_info_type;

LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE internal_malloc_pool_type {
  char pad[LIBXS_MALLOC_SCRATCH_PADDING];
  struct {
    size_t minsize, counter, incsize;
    char *buffer, *head;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    const void* site;
# if (0 != LIBXS_SYNC)
    unsigned int tid;
# endif
#endif
  } instance;
} internal_malloc_pool_type;

/* Scratch pool, which supports up to MAX_NSCRATCH allocation sites. */
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
/* LIBXS_ALIGNED appears to contradict LIBXS_APIVAR, and causes multiple defined symbols (if below is seen in multiple translation units) */
LIBXS_APIVAR_DEFINE(char internal_malloc_pool_buffer[(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS)*sizeof(internal_malloc_pool_type)+(LIBXS_MALLOC_SCRATCH_PADDING)-1]);
#endif
/* Maximum total size of the scratch memory domain. */
LIBXS_APIVAR_DEFINE(size_t internal_malloc_scratch_limit);
LIBXS_APIVAR_DEFINE(size_t internal_malloc_scratch_nmallocs);
LIBXS_APIVAR_DEFINE(size_t internal_malloc_private_max);
LIBXS_APIVAR_DEFINE(size_t internal_malloc_private_cur);
LIBXS_APIVAR_DEFINE(size_t internal_malloc_public_max);
LIBXS_APIVAR_DEFINE(size_t internal_malloc_public_cur);
LIBXS_APIVAR_DEFINE(size_t internal_malloc_local_max);
LIBXS_APIVAR_DEFINE(size_t internal_malloc_local_cur);
LIBXS_APIVAR_DEFINE(int internal_malloc_recursive);
/** 0: regular, 1/odd: intercept/scratch, otherwise: all/scratch */
LIBXS_APIVAR_DEFINE(int internal_malloc_kind);
#if defined(LIBXS_MALLOC_HOOK) && defined(LIBXS_MALLOC) && (0 != LIBXS_MALLOC)
/* Interval of bytes that permit interception (internal_malloc_kind) */
LIBXS_APIVAR_DEFINE(size_t internal_malloc_limit[2]);
#endif
#if (0 != LIBXS_SYNC) && defined(LIBXS_MALLOC_SCRATCH_JOIN)
LIBXS_APIVAR_DEFINE(int internal_malloc_join);
#endif
#if !defined(_WIN32)
# if defined(MAP_HUGETLB) && defined(LIBXS_MALLOC_HUGE_PAGES)
LIBXS_APIVAR_DEFINE(size_t internal_malloc_hugetlb);
# endif
# if defined(MAP_LOCKED) && defined(LIBXS_MALLOC_LOCK_PAGES)
LIBXS_APIVAR_DEFINE(size_t internal_malloc_plocked);
# endif
#endif


LIBXS_API_INTERN size_t libxs_alignment(size_t size, size_t alignment)
{
  size_t result;
  if ((LIBXS_MALLOC_ALIGNFCT * LIBXS_MALLOC_ALIGNMAX) <= size) {
    result = libxs_lcm(0 == alignment ? (LIBXS_ALIGNMENT) : libxs_lcm(alignment, LIBXS_ALIGNMENT), LIBXS_MALLOC_ALIGNMAX);
  }
  else { /* small-size request */
    if ((LIBXS_MALLOC_ALIGNFCT * LIBXS_ALIGNMENT) <= size) {
      result = (0 == alignment ? (LIBXS_ALIGNMENT) : libxs_lcm(alignment, LIBXS_ALIGNMENT));
    }
    else if (0 != alignment) { /* custom alignment */
      result = libxs_lcm(alignment, sizeof(void*));
    }
    else { /* tiny-size request */
      result = sizeof(void*);
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


LIBXS_API_INLINE
LIBXS_ATTRIBUTE_NO_SANITIZE(address)
internal_malloc_info_type* internal_malloc_info(const void* memory, int check)
{
  const char *const buffer = (const char*)memory;
  internal_malloc_info_type* result = (internal_malloc_info_type*)(NULL != memory
    ? (buffer - sizeof(internal_malloc_info_type)) : NULL);
#if defined(LIBXS_MALLOC_HOOK_CHECK)
  if ((LIBXS_MALLOC_HOOK_CHECK) < check) check = (LIBXS_MALLOC_HOOK_CHECK);
#endif
  if (0 != check && NULL != result) { /* check ownership */
#if !defined(_WIN32) /* mprotect: pass address rounded down to page/4k alignment */
    if (1 == check || 0 == mprotect((void*)(((uintptr_t)result) & 0xFFFFFFFFFFFFF000),
      sizeof(internal_malloc_info_type), PROT_READ | PROT_WRITE) || ENOMEM != errno)
#endif
    {
      const int flags_rs = LIBXS_MALLOC_FLAG_REALLOC | LIBXS_MALLOC_FLAG_SCRATCH;
      const int flags_px = LIBXS_MALLOC_FLAG_X | LIBXS_MALLOC_FLAG_PRIVATE;
      const int flags_mx = LIBXS_MALLOC_FLAG_X | LIBXS_MALLOC_FLAG_MMAP;
      const char *const pointer = (const char*)result->pointer;
      union { libxs_free_fun fun; const void* ptr; } convert;
      convert.fun = result->free.function;
      if (((flags_mx != (flags_mx & result->flags)) && NULL != result->reloc)
        || (0 == (LIBXS_MALLOC_FLAG_X & result->flags) ? 0 : (0 != (flags_rs & result->flags)))
        || (0 != (LIBXS_MALLOC_FLAG_X & result->flags) && NULL != result->context)
#if defined(LIBXS_VTUNE)
        || (0 == (LIBXS_MALLOC_FLAG_X & result->flags) && 0 != result->code_id)
#endif
        || (0 != (~LIBXS_MALLOC_FLAG_VALID & result->flags))
        || (0 == (LIBXS_MALLOC_FLAG_R & result->flags))
        || (pointer == convert.ptr || pointer == result->context || pointer >= buffer || NULL == pointer)
        || (LIBXS_MAX(LIBXS_MAX(internal_malloc_public_max, internal_malloc_local_max), internal_malloc_private_max) < result->size
            && 0 == (flags_px & result->flags)) || (0 == result->size)
        || (2 > libxs_ninit) /* before checksum calculation */
#if !defined(LIBXS_MALLOC_CRC_OFF) /* last check: checksum over info */
# if defined(LIBXS_MALLOC_CRC_LIGHT)
        || result->hash != LIBXS_CRC32U(LIBXS_BITS)(LIBXS_MALLOC_SEED, &result)
# else
        || result->hash != libxs_crc32(LIBXS_MALLOC_SEED, result,
            (const char*)&result->hash - (const char*)result)
# endif
#endif
      ) { /* mismatch */
        result = NULL;
      }
    }
#if !defined(_WIN32)
    else { /* mismatch */
      result = NULL;
    }
#endif
  }
  return result;
}


LIBXS_API_INTERN int internal_xfree(const void* /*memory*/, internal_malloc_info_type* /*info*/);
LIBXS_API_INTERN int internal_xfree(const void* memory, internal_malloc_info_type* info)
{
#if !defined(LIBXS_BUILD) || !defined(_WIN32)
  static int error_once = 0;
#endif
  int result = EXIT_SUCCESS, flags;
  void* buffer;
  size_t size;
  LIBXS_ASSERT(NULL != memory && NULL != info);
  buffer = info->pointer;
  flags = info->flags;
  size = info->size;
#if !defined(LIBXS_BUILD) /* sanity check */
  if (NULL != buffer || 0 == size)
#endif
  {
    const size_t alloc_size = size + (((const char*)memory) - ((const char*)buffer));
    LIBXS_ASSERT(NULL != buffer || 0 == size);
    if (0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) {
      if (NULL != info->free.function) {
#if defined(LIBXS_MALLOC_DELETE_SAFE)
        info->pointer = NULL; info->size = 0;
#endif
        if (NULL == info->context) {
#if defined(LIBXS_MALLOC_HOOK) && 0
          if (free == info->free.function) {
            __real_free(buffer);
          }
          else
#endif
          if (NULL != info->free.function) {
            info->free.function(buffer);
          }
        }
        else {
          LIBXS_ASSERT(NULL != info->free.ctx_form);
          info->free.ctx_form(buffer, info->context);
        }
      }
    }
    else {
#if defined(LIBXS_VTUNE)
      if (0 != (LIBXS_MALLOC_FLAG_X & flags) && 0 != info->code_id && iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
        iJIT_NotifyEvent(LIBXS_VTUNE_JIT_UNLOAD, &info->code_id);
      }
#endif
#if defined(_WIN32)
      result = (NULL == buffer || FALSE != VirtualFree(buffer, 0, MEM_RELEASE)) ? EXIT_SUCCESS : EXIT_FAILURE;
#else /* !_WIN32 */
      {
        const size_t unmap_size = LIBXS_UP2(alloc_size, LIBXS_PAGE_MINSIZE);
        void *const reloc = info->reloc;
        if (0 != munmap(buffer, unmap_size)) {
          if (0 != libxs_verbosity /* library code is expected to be mute */
            && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXS ERROR: %s (attempted to unmap buffer %p+%" PRIuPTR ")!\n",
              strerror(errno), buffer, (uintptr_t)unmap_size);
          }
          result = EXIT_FAILURE;
        }
        if (0 != (LIBXS_MALLOC_FLAG_X & flags) && EXIT_SUCCESS == result
          && NULL != reloc && MAP_FAILED != reloc && buffer != reloc
          && 0 != munmap(reloc, unmap_size))
        {
          if (0 != libxs_verbosity /* library code is expected to be mute */
            && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXS ERROR: %s (attempted to unmap code %p+%" PRIuPTR ")!\n",
              strerror(errno), reloc, (uintptr_t)unmap_size);
          }
          result = EXIT_FAILURE;
        }
      }
#endif
    }
    if (0 == (LIBXS_MALLOC_FLAG_X & flags)) { /* update statistics */
#if !defined(_WIN32)
# if defined(MAP_HUGETLB) && defined(LIBXS_MALLOC_HUGE_PAGES)
      if (0 != (LIBXS_MALLOC_FLAG_PHUGE & flags)) { /* huge pages */
        LIBXS_ASSERT(0 != (LIBXS_MALLOC_FLAG_MMAP & flags));
        LIBXS_ATOMIC_SUB_FETCH(&internal_malloc_hugetlb, alloc_size, LIBXS_ATOMIC_RELAXED);
      }
# endif
# if defined(MAP_LOCKED) && defined(LIBXS_MALLOC_LOCK_PAGES)
      if (0 != (LIBXS_MALLOC_FLAG_PLOCK & flags)) { /* page-locked */
        LIBXS_ASSERT(0 != (LIBXS_MALLOC_FLAG_MMAP & flags));
        LIBXS_ATOMIC_SUB_FETCH(&internal_malloc_plocked, alloc_size, LIBXS_ATOMIC_RELAXED);
      }
# endif
#endif
      if (0 == (LIBXS_MALLOC_FLAG_PRIVATE & flags)) { /* public */
        if (0 != (LIBXS_MALLOC_FLAG_SCRATCH & flags)) { /* scratch */
#if 1
          const size_t current = (size_t)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(
            &internal_malloc_public_cur, LIBXS_ATOMIC_RELAXED);
          LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&internal_malloc_public_cur,
            alloc_size <= current ? (current - alloc_size) : 0, LIBXS_ATOMIC_RELAXED);
#else
          LIBXS_ATOMIC(LIBXS_ATOMIC_SUB_FETCH, LIBXS_BITS)(
            &internal_malloc_public_cur, alloc_size, LIBXS_ATOMIC_RELAXED);
#endif
        }
        else { /* local */
#if 1
          const size_t current = (size_t)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(
            &internal_malloc_local_cur, LIBXS_ATOMIC_RELAXED);
          LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&internal_malloc_local_cur,
            alloc_size <= current ? (current - alloc_size) : 0, LIBXS_ATOMIC_RELAXED);
#else
          LIBXS_ATOMIC(LIBXS_ATOMIC_SUB_FETCH, LIBXS_BITS)(
            &internal_malloc_local_cur, alloc_size, LIBXS_ATOMIC_RELAXED);
#endif
        }
      }
      else { /* private */
#if 1
        const size_t current = (size_t)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(
          &internal_malloc_private_cur, LIBXS_ATOMIC_RELAXED);
        LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&internal_malloc_private_cur,
          alloc_size <= current ? (current - alloc_size) : 0, LIBXS_ATOMIC_RELAXED);
#else
        LIBXS_ATOMIC(LIBXS_ATOMIC_SUB_FETCH, LIBXS_BITS)(
          &internal_malloc_private_cur, alloc_size, LIBXS_ATOMIC_RELAXED);
#endif
      }
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
#if !defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) || (1 >= (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  LIBXS_UNUSED(exclude);
#else
  const internal_malloc_pool_type* pool = (const internal_malloc_pool_type*)LIBXS_UP2(
    (uintptr_t)internal_malloc_pool_buffer, LIBXS_MALLOC_SCRATCH_PADDING);
# if (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  const internal_malloc_pool_type *const end = pool + libxs_scratch_pools;
  LIBXS_ASSERT(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
  for (; pool != end; ++pool)
# endif /*(1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
  {
    if (0 != pool->instance.minsize) {
# if 1 /* memory info is not used */
      if (pool != exclude && (LIBXS_MALLOC_INTERNAL_CALLER) != pool->instance.site) {
        result += pool->instance.minsize;
      }
# else
      const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer, 0/*no check*/);
      if (NULL != info && pool != exclude && (LIBXS_MALLOC_INTERNAL_CALLER) != pool->instance.site) {
        result += info->size;
      }
# endif
    }
    else break; /* early exit */
  }
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
  return result;
}


LIBXS_API_INLINE internal_malloc_pool_type* internal_scratch_malloc_pool(const void* memory)
{
  internal_malloc_pool_type* result = NULL;
  internal_malloc_pool_type* pool = (internal_malloc_pool_type*)LIBXS_UP2(
    (uintptr_t)internal_malloc_pool_buffer, LIBXS_MALLOC_SCRATCH_PADDING);
  const char *const buffer = (const char*)memory;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  const unsigned int npools = libxs_scratch_pools;
#else
  const unsigned int npools = 1;
#endif
  internal_malloc_pool_type *const end = pool + npools;
  LIBXS_ASSERT(npools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
  LIBXS_ASSERT(NULL != memory);
  for (; pool != end; ++pool) {
    if (0 != pool->instance.minsize) {
      if (0 != pool->instance.counter
#if 1 /* should be implied by non-zero counter */
        && NULL != pool->instance.buffer
#endif
      ){/* check if memory belongs to scratch domain or local domain */
#if 1
        const size_t size = pool->instance.minsize;
#else
        const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer, 0/*no check*/);
        const size_t size = info->size;
#endif
        if (pool->instance.buffer == buffer /* fast path */ ||
           (pool->instance.buffer < buffer && buffer < (pool->instance.buffer + size)))
        {
          result = pool;
          break;
        }
      }
    }
    else break; /* early exit */
  }
  return result;
}


LIBXS_API_INTERN void internal_scratch_free(const void* /*memory*/, internal_malloc_pool_type* /*pool*/);
LIBXS_API_INTERN void internal_scratch_free(const void* memory, internal_malloc_pool_type* pool)
{
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  const size_t counter = LIBXS_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);
  char *const pool_buffer = pool->instance.buffer;
# if (!defined(NDEBUG) || defined(LIBXS_MALLOC_SCRATCH_TRIM_HEAD))
  char *const buffer = (char*)memory; /* non-const */
  LIBXS_ASSERT(pool_buffer <= buffer && buffer < pool_buffer + pool->instance.minsize);
# endif
  LIBXS_ASSERT(pool_buffer <= pool->instance.head);
  if (0 == counter) { /* reuse or reallocate scratch domain */
    internal_malloc_info_type *const info = internal_malloc_info(pool_buffer, 0/*no check*/);
    const size_t scale_size = (size_t)(1 != libxs_scratch_scale ? (libxs_scratch_scale * info->size) : info->size); /* hysteresis */
    const size_t size = pool->instance.minsize + pool->instance.incsize;
    LIBXS_ASSERT(0 == (LIBXS_MALLOC_FLAG_X & info->flags)); /* scratch memory is not executable */
    if (size <= scale_size) { /* reuse scratch domain */
      pool->instance.head = pool_buffer; /* reuse scratch domain */
    }
    else { /* release buffer */
# if !defined(NDEBUG)
      static int error_once = 0;
# endif
      pool->instance.buffer = pool->instance.head = NULL;
# if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
      pool->instance.site = NULL; /* clear affinity */
# endif
# if !defined(NDEBUG)
      if (EXIT_SUCCESS != internal_xfree(pool_buffer, info)
        && 0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: memory deallocation failed!\n");
      }
# else
      internal_xfree(pool_buffer, info); /* !libxs_free */
# endif
    }
  }
# if defined(LIBXS_MALLOC_SCRATCH_TRIM_HEAD) /* TODO: document linear/scoped allocator policy */
  else if (buffer < pool->instance.head) { /* reuse scratch domain */
    pool->instance.head = buffer;
  }
# else
  LIBXS_UNUSED(memory);
# endif
#else
  LIBXS_UNUSED(memory); LIBXS_UNUSED(pool);
#endif
}


LIBXS_API_INTERN void internal_scratch_malloc(void** /*memory*/, size_t /*size*/, size_t /*alignment*/, int /*flags*/, const void* /*caller*/);
LIBXS_API_INTERN void internal_scratch_malloc(void** memory, size_t size, size_t alignment, int flags, const void* caller)
{
  LIBXS_ASSERT(NULL != memory && 0 == (LIBXS_MALLOC_FLAG_X & flags));
  if (0 == (LIBXS_MALLOC_FLAG_REALLOC & flags) || NULL == *memory) {
    static int error_once = 0;
    size_t local_size = 0;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    if (0 < libxs_scratch_pools) {
      internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)LIBXS_UP2(
        (uintptr_t)internal_malloc_pool_buffer, LIBXS_MALLOC_SCRATCH_PADDING);
      internal_malloc_pool_type *const end = pools + libxs_scratch_pools, *pool = pools;
      const size_t align_size = libxs_alignment(size, alignment), alloc_size = size + align_size - 1;
# if (0 != LIBXS_SYNC)
      const unsigned int tid = libxs_get_tid();
# endif
      unsigned int npools = 1;
# if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
      const void *const site = caller; /* no further attempt in case of NULL */
      internal_malloc_pool_type *pool0 = end;
      for (; pool != end; ++pool) { /* counter: memory info is not employed as pools are still manipulated */
        if (NULL != pool->instance.buffer) {
          if ((LIBXS_MALLOC_INTERNAL_CALLER) != pool->instance.site) ++npools; /* count number of occupied pools */
          if ( /* find matching pool and enter fast path (draw from pool-buffer) */
#   if (0 != LIBXS_SYNC) && !defined(LIBXS_MALLOC_SCRATCH_JOIN)
            (site == pool->instance.site && tid == pool->instance.tid))
#   elif (0 != LIBXS_SYNC)
            (site == pool->instance.site && (0 != internal_malloc_join || tid == pool->instance.tid)))
#   else
            (site == pool->instance.site))
#   endif
          {
            break;
          }
        }
        else {
          if (end == pool0) pool0 = pool; /* first available pool*/
          if (0 == pool->instance.minsize) { /* early exit */
            pool = pool0; break;
          }
        }
      }
# endif
      LIBXS_ASSERT(NULL != pool);
      if (end != pool && 0 <= internal_malloc_kind) {
        const size_t counter = LIBXS_ATOMIC_ADD_FETCH(&pool->instance.counter, (size_t)1, LIBXS_ATOMIC_SEQ_CST);
        if (NULL != pool->instance.buffer || 1 != counter) { /* attempt to (re-)use existing pool */
          const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer, 1/*check*/);
          const size_t pool_size = ((NULL != info && 0 != counter) ? info->size : 0);
          const size_t used_size = pool->instance.head - pool->instance.buffer;
          const size_t req_size = alloc_size + used_size;
          if (req_size <= pool_size) { /* fast path: draw from pool-buffer */
# if (0 != LIBXS_SYNC) && defined(LIBXS_MALLOC_SCRATCH_JOIN)
            void *const headaddr = &pool->instance.head;
            char *const head = (0 == internal_malloc_join
              ? (pool->instance.head += alloc_size)
              : ((char*)LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
                (uintptr_t*)headaddr, alloc_size, LIBXS_ATOMIC_SEQ_CST)));
# else
            char *const head = (char*)(pool->instance.head += alloc_size);
# endif
            *memory = LIBXS_ALIGN(head - alloc_size, align_size);
          }
          else { /* fall-back to local memory allocation */
            const size_t incsize = req_size - LIBXS_MIN(pool_size, req_size);
            pool->instance.incsize = LIBXS_MAX(pool->instance.incsize, incsize);
# if (0 != LIBXS_SYNC) && defined(LIBXS_MALLOC_SCRATCH_JOIN)
            if (0 == internal_malloc_join) {
              --pool->instance.counter;
            }
            else {
              LIBXS_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);
            }
# else
            --pool->instance.counter;
# endif
            if (
# if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
              (LIBXS_MALLOC_INTERNAL_CALLER) != pool->instance.site &&
# endif
              0 == (LIBXS_MALLOC_FLAG_PRIVATE & flags))
            {
              const size_t watermark = LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
                &internal_malloc_local_cur, alloc_size, LIBXS_ATOMIC_RELAXED);
              if (internal_malloc_local_max < watermark) internal_malloc_local_max = watermark; /* accept data-race */
            }
            else {
              const size_t watermark = LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
                &internal_malloc_private_cur, alloc_size, LIBXS_ATOMIC_RELAXED);
              if (internal_malloc_private_max < watermark) internal_malloc_private_max = watermark; /* accept data-race */
            }
            local_size = size;
          }
        }
        else { /* fresh pool */
          const size_t scratch_limit = libxs_get_scratch_limit();
          const size_t scratch_size = internal_get_scratch_size(pool); /* exclude current pool */
          const size_t limit_size = (1 < npools ? (scratch_limit - LIBXS_MIN(scratch_size, scratch_limit)) : LIBXS_SCRATCH_UNLIMITED);
          const size_t scale_size = (size_t)(1 != libxs_scratch_scale ? (libxs_scratch_scale * alloc_size) : alloc_size); /* hysteresis */
          const size_t incsize = (size_t)(libxs_scratch_scale * pool->instance.incsize);
          const size_t maxsize = LIBXS_MAX(scale_size, pool->instance.minsize) + incsize;
          const size_t limsize = LIBXS_MIN(maxsize, limit_size);
          const size_t minsize = limsize;
          LIBXS_ASSERT(1 <= libxs_scratch_scale);
          LIBXS_ASSERT(1 == counter);
          pool->instance.incsize = 0; /* reset */
          pool->instance.minsize = minsize;
# if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
          pool->instance.site = site;
#   if (0 != LIBXS_SYNC)
          pool->instance.tid = tid;
#   endif
# endif
          if (alloc_size <= minsize && /* allocate scratch pool */
            EXIT_SUCCESS == libxs_xmalloc(memory, minsize, 0/*auto-align*/,
              (flags | LIBXS_MALLOC_FLAG_SCRATCH) & ~LIBXS_MALLOC_FLAG_REALLOC,
              NULL/*extra*/, 0/*extra_size*/))
          {
            pool->instance.buffer = (char*)*memory;
            pool->instance.head = pool->instance.buffer + alloc_size;
            *memory = LIBXS_ALIGN((char*)*memory, align_size);
# if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
            if ((LIBXS_MALLOC_INTERNAL_CALLER) != pool->instance.site)
# endif
            {
              LIBXS_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXS_ATOMIC_RELAXED);
            }
          }
          else { /* fall-back to local allocation */
            LIBXS_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXS_ATOMIC_SEQ_CST);
            if (0 != libxs_verbosity /* library code is expected to be mute */
              && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
            {
              if (alloc_size <= minsize) {
                fprintf(stderr, "LIBXS ERROR: failed to allocate scratch memory!\n");
              }
              else if ((LIBXS_MALLOC_INTERNAL_CALLER) != caller
                && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity))
              {
                fprintf(stderr, "LIBXS WARNING: scratch memory domain exhausted!\n");
              }
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
          flags & ~(LIBXS_MALLOC_FLAG_SCRATCH | LIBXS_MALLOC_FLAG_REALLOC), NULL/*extra*/, 0/*extra_size*/)
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
    const void *const preserve = *memory;
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    internal_malloc_pool_type *const pool = internal_scratch_malloc_pool(preserve);
    if (NULL != pool) {
      const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer, 0/*no check*/);
      void* buffer;
      LIBXS_ASSERT(pool->instance.buffer <= pool->instance.head && NULL != info);
      internal_scratch_malloc(&buffer, size, alignment,
        ~LIBXS_MALLOC_FLAG_REALLOC & (LIBXS_MALLOC_FLAG_SCRATCH | flags), caller);
      if (NULL != buffer) {
        memcpy(buffer, preserve, LIBXS_MIN(size, info->size)); /* TODO: memmove? */
        *memory = buffer;
      }
      internal_scratch_free(memory, pool);
    }
    else
#endif
    { /* non-pooled (potentially foreign pointer) */
#if !defined(NDEBUG)
      const int status =
#endif
      libxs_xmalloc(memory, size, alignment/* no need here to determine alignment of given buffer */,
        ~LIBXS_MALLOC_FLAG_SCRATCH & flags, NULL/*extra*/, 0/*extra_size*/);
      assert(EXIT_SUCCESS == status || NULL == *memory); /* !LIBXS_ASSERT */
    }
  }
}


#if defined(LIBXS_MALLOC_HOOK_DYNAMIC)
LIBXS_APIVAR_PRIVATE_DEF(libxs_malloc_fntype libxs_malloc_fn);

#if defined(LIBXS_MALLOC_HOOK_QKMALLOC)
LIBXS_API_INTERN void* internal_memalign_malloc(size_t /*alignment*/, size_t /*size*/);
LIBXS_API_INTERN void* internal_memalign_malloc(size_t alignment, size_t size)
{
  LIBXS_UNUSED(alignment);
  LIBXS_ASSERT(NULL != libxs_malloc_fn.malloc.dlsym);
  return libxs_malloc_fn.malloc.ptr(size);
}
#elif defined(LIBXS_MALLOC_HOOK_KMP)
LIBXS_API_INTERN void* internal_memalign_twiddle(size_t /*alignment*/, size_t /*size*/);
LIBXS_API_INTERN void* internal_memalign_twiddle(size_t alignment, size_t size)
{
  LIBXS_ASSERT(NULL != libxs_malloc_fn.alignmem.dlsym);
  return libxs_malloc_fn.alignmem.ptr(size, alignment);
}
#endif
#endif /*defined(LIBXS_MALLOC_HOOK_DYNAMIC)*/

#if (defined(LIBXS_MALLOC_HOOK) && defined(LIBXS_MALLOC) && (0 != LIBXS_MALLOC)) || defined(LIBXS_MALLOC_ALIGN_ALL)
LIBXS_API_INTERN void* internal_memalign_hook(size_t /*alignment*/, size_t /*size*/, const void* /*caller*/);
LIBXS_API_INTERN void* internal_memalign_hook(size_t alignment, size_t size, const void* caller)
{
  void* result;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_MMAP, alignment, size, caller);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_DEFAULT, alignment, size, caller);
# endif
  return result;
}

LIBXS_API void* __wrap_memalign(size_t /*alignment*/, size_t /*size*/);
LIBXS_API void* __wrap_memalign(size_t alignment, size_t size)
{
  void* result;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_MMAP, alignment, size, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_DEFAULT, alignment, size, NULL/*caller*/);
# endif
  return result;
}

LIBXS_API_INTERN void* internal_malloc_hook(size_t /*size*/, const void* /*caller*/);
LIBXS_API_INTERN void* internal_malloc_hook(size_t size, const void* caller)
{
  return internal_memalign_hook(0/*auto-alignment*/, size, caller);
}

LIBXS_API void* __wrap_malloc(size_t /*size*/);
LIBXS_API void* __wrap_malloc(size_t size)
{
  void* result;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_MMAP, 0/*auto-alignment*/, size, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_DEFAULT, 0/*auto-alignment*/, size, NULL/*caller*/);
# endif
  return result;
}

#if defined(LIBXS_MALLOC_HOOK_CALLOC)
LIBXS_API void* __wrap_calloc(size_t /*num*/, size_t /*size*/);
LIBXS_API void* __wrap_calloc(size_t num, size_t size)
{
  void* result;
  const size_t nbytes = num * size;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_MMAP, 0/*auto-alignment*/, nbytes, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_DEFAULT, 0/*auto-alignment*/, nbytes, NULL/*caller*/);
# endif
  /* TODO: signal anonymous/zeroed pages */
  if (NULL != result) memset(result, 0, nbytes);
  return result;
}
#endif

#if defined(LIBXS_MALLOC_HOOK_REALLOC)
LIBXS_API_INTERN void* internal_realloc_hook(void* /*ptr*/, size_t /*size*/, const void* /*caller*/);
LIBXS_API_INTERN void* internal_realloc_hook(void* ptr, size_t size, const void* caller)
{
  void* result;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_REALLOC_HOOK(result, LIBXS_MALLOC_FLAG_REALLOC | LIBXS_MALLOC_FLAG_MMAP, ptr, size, caller);
# else
  INTERNAL_REALLOC_HOOK(result, LIBXS_MALLOC_FLAG_REALLOC | LIBXS_MALLOC_FLAG_DEFAULT, ptr, size, caller);
# endif
  return result;
}

LIBXS_API void* __wrap_realloc(void* /*ptr*/, size_t /*size*/);
LIBXS_API void* __wrap_realloc(void* ptr, size_t size)
{
  void* result;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_REALLOC_HOOK(result, LIBXS_MALLOC_FLAG_REALLOC | LIBXS_MALLOC_FLAG_MMAP, ptr, size, NULL/*caller*/);
# else
  INTERNAL_REALLOC_HOOK(result, LIBXS_MALLOC_FLAG_REALLOC | LIBXS_MALLOC_FLAG_DEFAULT, ptr, size, NULL/*caller*/);
# endif
  return result;
}
#endif

LIBXS_API_INTERN void internal_free_hook(void* /*ptr*/, const void* /*caller*/);
LIBXS_API_INTERN void internal_free_hook(void* ptr, const void* caller)
{
  INTERNAL_FREE_HOOK(ptr, caller);
}

LIBXS_API void __wrap_free(void* /*ptr*/);
LIBXS_API void __wrap_free(void* ptr)
{
  INTERNAL_FREE_HOOK(ptr, NULL/*caller*/);
}
#endif

#if defined(LIBXS_MALLOC_HOOK_DYNAMIC) && ((defined(LIBXS_MALLOC) && (0 != LIBXS_MALLOC)) || defined(LIBXS_MALLOC_ALIGN_ALL))
LIBXS_API LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE_MALLOC void* memalign(size_t /*alignment*/, size_t /*size*/) LIBXS_THROW;
LIBXS_API LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE_MALLOC void* memalign(size_t alignment, size_t size) LIBXS_THROW
{
  void* result;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_MMAP, alignment, size, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_DEFAULT, alignment, size, NULL/*caller*/);
# endif
  return result;
}

LIBXS_API LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE_MALLOC void* malloc(size_t /*size*/) LIBXS_THROW;
LIBXS_API LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE_MALLOC void* malloc(size_t size) LIBXS_THROW
{
  void* result;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_MMAP, 0/*auto-alignment*/, size, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_DEFAULT, 0/*auto-alignment*/, size, NULL/*caller*/);
# endif
  return result;
}

#if defined(LIBXS_MALLOC_HOOK_CALLOC)
LIBXS_API LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE_MALLOC void* calloc(size_t /*num*/, size_t /*size*/) LIBXS_THROW;
LIBXS_API LIBXS_ATTRIBUTE_WEAK LIBXS_ATTRIBUTE_MALLOC void* calloc(size_t num, size_t size) LIBXS_THROW
{
  void* result;
  const size_t nbytes = num * size;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_MMAP, 0/*auto-alignment*/, nbytes, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXS_MALLOC_FLAG_DEFAULT, 0/*auto-alignment*/, nbytes, NULL/*caller*/);
# endif
  /* TODO: signal anonymous/zeroed pages */
  if (NULL != result) memset(result, 0, nbytes);
  return result;
}
#endif

#if defined(LIBXS_MALLOC_HOOK_REALLOC)
LIBXS_API LIBXS_ATTRIBUTE_WEAK void* realloc(void* /*ptr*/, size_t /*size*/) LIBXS_THROW;
LIBXS_API LIBXS_ATTRIBUTE_WEAK void* realloc(void* ptr, size_t size) LIBXS_THROW
{
  void* result;
# if defined(LIBXS_MALLOC_MMAP_HOOK)
  INTERNAL_REALLOC_HOOK(result, LIBXS_MALLOC_FLAG_REALLOC | LIBXS_MALLOC_FLAG_MMAP, ptr, size, NULL/*caller*/);
# else
  INTERNAL_REALLOC_HOOK(result, LIBXS_MALLOC_FLAG_REALLOC | LIBXS_MALLOC_FLAG_DEFAULT, ptr, size, NULL/*caller*/);
# endif
  return result;
}
#endif

LIBXS_API LIBXS_ATTRIBUTE_WEAK void free(void* /*ptr*/) LIBXS_THROW;
LIBXS_API LIBXS_ATTRIBUTE_WEAK void free(void* ptr) LIBXS_THROW
{
  INTERNAL_FREE_HOOK(ptr, NULL/*caller*/);
}
#endif


LIBXS_API_INTERN void libxs_malloc_init(void)
{
#if (0 != LIBXS_SYNC) && defined(LIBXS_MALLOC_SCRATCH_JOIN)
  const char *const env = getenv("LIBXS_MALLOC_JOIN");
  if (NULL != env && 0 != *env) internal_malloc_join = atoi(env);
#endif
#if defined(LIBXS_MALLOC_HOOK_DYNAMIC)
# if defined(LIBXS_MALLOC_HOOK_QKMALLOC)
  void* handle_qkmalloc = NULL;
  dlerror(); /* clear an eventual error status */
  handle_qkmalloc = dlopen("libqkmalloc.so", RTLD_LAZY);
  if (NULL != handle_qkmalloc) {
    libxs_malloc_fn.memalign.ptr = internal_memalign_malloc;
    libxs_malloc_fn.malloc.dlsym = dlsym(handle_qkmalloc, "malloc");
    if (NULL == dlerror() && NULL != libxs_malloc_fn.malloc.dlsym) {
#   if defined(LIBXS_MALLOC_HOOK_CALLOC)
      libxs_malloc_fn.calloc.dlsym = dlsym(handle_qkmalloc, "calloc");
      if (NULL == dlerror() && NULL != libxs_malloc_fn.calloc.dlsym)
#   endif
      {
#   if defined(LIBXS_MALLOC_HOOK_REALLOC)
        libxs_malloc_fn.realloc.dlsym = dlsym(handle_qkmalloc, "realloc");
        if (NULL == dlerror() && NULL != libxs_malloc_fn.realloc.dlsym)
#   endif
        {
          libxs_malloc_fn.free.dlsym = dlsym(handle_qkmalloc, "free");
        }
      }
    }
    dlclose(handle_qkmalloc);
  }
  if (NULL == libxs_malloc_fn.free.ptr)
# elif defined(LIBXS_MALLOC_HOOK_KMP)
  dlerror(); /* clear an eventual error status */
  libxs_malloc_fn.alignmem.dlsym = dlsym(LIBXS_RTLD_NEXT, "kmp_aligned_malloc");
  if (NULL == dlerror() && NULL != libxs_malloc_fn.alignmem.dlsym) {
    libxs_malloc_fn.memalign.ptr = internal_memalign_twiddle;
    libxs_malloc_fn.malloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "kmp_malloc");
    if (NULL == dlerror() && NULL != libxs_malloc_fn.malloc.dlsym) {
# if defined(LIBXS_MALLOC_HOOK_CALLOC)
      libxs_malloc_fn.calloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "kmp_calloc");
      if (NULL == dlerror() && NULL != libxs_malloc_fn.calloc.dlsym)
# endif
      {
# if defined(LIBXS_MALLOC_HOOK_REALLOC)
        libxs_malloc_fn.realloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "kmp_realloc");
        if (NULL == dlerror() && NULL != libxs_malloc_fn.realloc.dlsym)
# endif
        {
          libxs_malloc_fn.free.dlsym = dlsym(LIBXS_RTLD_NEXT, "kmp_free");
        }
      }
    }
  }
  if (NULL == libxs_malloc_fn.free.ptr)
# endif /*defined(LIBXS_MALLOC_HOOK_QKMALLOC)*/
  {
    dlerror(); /* clear an eventual error status */
# if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD)))
    libxs_malloc_fn.memalign.dlsym = dlsym(LIBXS_RTLD_NEXT, "__libc_memalign");
    if (NULL == dlerror() && NULL != libxs_malloc_fn.memalign.dlsym) {
      libxs_malloc_fn.malloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "__libc_malloc");
      if (NULL == dlerror() && NULL != libxs_malloc_fn.malloc.dlsym) {
#   if defined(LIBXS_MALLOC_HOOK_CALLOC)
        libxs_malloc_fn.calloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "__libc_calloc");
        if (NULL == dlerror() && NULL != libxs_malloc_fn.calloc.dlsym)
#   endif
        {
#   if defined(LIBXS_MALLOC_HOOK_REALLOC)
          libxs_malloc_fn.realloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "__libc_realloc");
          if (NULL == dlerror() && NULL != libxs_malloc_fn.realloc.dlsym)
#   endif
          {
            libxs_malloc_fn.free.dlsym = dlsym(LIBXS_RTLD_NEXT, "__libc_free");
          }
        }
      }
    }
    if (NULL == libxs_malloc_fn.free.ptr) {
      void* handle_libc = NULL;
      dlerror(); /* clear an eventual error status */
      handle_libc = dlopen("libc.so." LIBXS_STRINGIFY(LIBXS_MALLOC_GLIBC), RTLD_LAZY);
      if (NULL != handle_libc) {
        libxs_malloc_fn.memalign.dlsym = dlsym(handle_libc, "__libc_memalign");
        if (NULL == dlerror() && NULL != libxs_malloc_fn.memalign.dlsym) {
          libxs_malloc_fn.malloc.dlsym = dlsym(handle_libc, "__libc_malloc");
          if (NULL == dlerror() && NULL != libxs_malloc_fn.malloc.dlsym) {
#   if defined(LIBXS_MALLOC_HOOK_CALLOC)
            libxs_malloc_fn.calloc.dlsym = dlsym(handle_libc, "__libc_calloc");
            if (NULL == dlerror() && NULL != libxs_malloc_fn.calloc.dlsym)
#   endif
            {
#   if defined(LIBXS_MALLOC_HOOK_REALLOC)
              libxs_malloc_fn.realloc.dlsym = dlsym(handle_libc, "__libc_realloc");
              if (NULL == dlerror() && NULL != libxs_malloc_fn.realloc.dlsym)
#   endif
              {
                libxs_malloc_fn.free.dlsym = dlsym(handle_libc, "__libc_free");
              }
            }
          }
        }
        dlclose(handle_libc);
      }
    }
#   if 0
    { /* attempt to setup deprecated GLIBC hooks */
      union { const void* dlsym; void* (**ptr)(size_t, size_t, const void*); } hook_memalign;
      dlerror(); /* clear an eventual error status */
      hook_memalign.dlsym = dlsym(LIBXS_RTLD_NEXT, "__memalign_hook");
      if (NULL == dlerror() && NULL != hook_memalign.dlsym) {
        union { const void* dlsym; void* (**ptr)(size_t, const void*); } hook_malloc;
        hook_malloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "__malloc_hook");
        if (NULL == dlerror() && NULL != hook_malloc.dlsym) {
#   if defined(LIBXS_MALLOC_HOOK_REALLOC)
          union { const void* dlsym; void* (**ptr)(void*, size_t, const void*); } hook_realloc;
          hook_realloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "__realloc_hook");
          if (NULL == dlerror() && NULL != hook_realloc.dlsym)
#   endif
          {
            union { const void* dlsym; void (**ptr)(void*, const void*); } hook_free;
            hook_free.dlsym = dlsym(LIBXS_RTLD_NEXT, "__free_hook");
            if (NULL == dlerror() && NULL != hook_free.dlsym) {
              *hook_memalign.ptr = internal_memalign_hook;
              *hook_malloc.ptr = internal_malloc_hook;
#   if defined(LIBXS_MALLOC_HOOK_REALLOC)
              *hook_realloc.ptr = internal_realloc_hook;
#   endif
              *hook_free.ptr = internal_free_hook;
            }
          }
        }
      }
    }
#   endif
# else /* TODO */
# endif /*(defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD)))*/
  }
  if (NULL != libxs_malloc_fn.free.ptr) {
# if defined(LIBXS_MALLOC_HOOK_IMALLOC)
    union { const void* dlsym; libxs_malloc_fun* ptr; } i_malloc;
    i_malloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "i_malloc");
    if (NULL == dlerror() && NULL != i_malloc.dlsym) {
#   if defined(LIBXS_MALLOC_HOOK_CALLOC)
      union { const void* dlsym; void* (**ptr)(size_t, size_t); } i_calloc;
      i_calloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "i_calloc");
      if (NULL == dlerror() && NULL != i_calloc.dlsym)
#   endif
      {
#   if defined(LIBXS_MALLOC_HOOK_REALLOC)
        union { const void* dlsym; libxs_realloc_fun* ptr; } i_realloc;
        i_realloc.dlsym = dlsym(LIBXS_RTLD_NEXT, "i_realloc");
        if (NULL == dlerror() && NULL != i_realloc.dlsym)
#   endif
        {
          union { const void* dlsym; libxs_free_fun* ptr; } i_free;
          i_free.dlsym = dlsym(LIBXS_RTLD_NEXT, "i_free");
          if (NULL == dlerror() && NULL != i_free.dlsym) {
            *i_malloc.ptr = libxs_malloc_fn.malloc.ptr;
#   if defined(LIBXS_MALLOC_HOOK_CALLOC)
            *i_calloc.ptr = libxs_malloc_fn.calloc.ptr;
#   endif
#   if defined(LIBXS_MALLOC_HOOK_REALLOC)
            *i_realloc.ptr = libxs_malloc_fn.realloc.ptr;
#   endif
            *i_free.ptr = libxs_malloc_fn.free.ptr;
          }
        }
      }
    }
# endif /*defined(LIBXS_MALLOC_HOOK_IMALLOC)*/
  }
  else { /* fall-back: potentially recursive */
# if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD)))
    libxs_malloc_fn.memalign.ptr = __libc_memalign;
    libxs_malloc_fn.malloc.ptr = __libc_malloc;
#   if defined(LIBXS_MALLOC_HOOK_CALLOC)
    libxs_malloc_fn.calloc.ptr = __libc_calloc;
#   endif
#   if defined(LIBXS_MALLOC_HOOK_REALLOC)
    libxs_malloc_fn.realloc.ptr = __libc_realloc;
#   endif
    libxs_malloc_fn.free.ptr = __libc_free;
# else
    libxs_malloc_fn.memalign.ptr = libxs_memalign_internal;
    libxs_malloc_fn.malloc.ptr = malloc;
#   if defined(LIBXS_MALLOC_HOOK_CALLOC)
    libxs_malloc_fn.calloc.ptr = calloc;
#   endif
#   if defined(LIBXS_MALLOC_HOOK_REALLOC)
    libxs_malloc_fn.realloc.ptr = realloc;
#   endif
    libxs_malloc_fn.free.ptr = free;
# endif
  }
#endif
}


LIBXS_API_INTERN void libxs_malloc_finalize(void)
{
}


LIBXS_API_INTERN int libxs_xset_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  int result = EXIT_SUCCESS;
  if (NULL != lock) {
    LIBXS_INIT
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
    internal_malloc_fn.function = __real_malloc;
    internal_free_fn.function = __real_free;
    /*internal_allocator = NULL;*/
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
      LIBXS_INIT
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
    LIBXS_INIT
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
      LIBXS_INIT
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
  int result;
#if !defined(NDEBUG)
  if (NULL != size || NULL != extra)
#endif
  {
    const int check = ((NULL == flags || 0 == (LIBXS_MALLOC_FLAG_X & *flags)) ? 2 : 1);
    const internal_malloc_info_type *const info = internal_malloc_info(memory, check);
    if (NULL != info) {
      if (NULL != size) *size = info->size;
      if (NULL != flags) *flags = info->flags;
      if (NULL != extra) *extra = info->pointer;
      result = EXIT_SUCCESS;
    }
    else { /* potentially foreign buffer */
      result = (NULL != memory ? EXIT_FAILURE : EXIT_SUCCESS);
      if (NULL != size) *size = 0;
      if (NULL != flags) *flags = 0;
      if (NULL != extra) *extra = 0;
    }
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: attachment error for memory buffer %p!\n", memory);
    }
    LIBXS_ASSERT_MSG(0/*false*/, "LIBXS ERROR: attachment error");
    result = EXIT_FAILURE;
  }
#endif
  return result;
}


#if !defined(_WIN32)

LIBXS_API_INLINE void internal_xmalloc_mhint(void* buffer, size_t size)
{
  LIBXS_ASSERT((MAP_FAILED != buffer && NULL != buffer) || 0 == size);
#if (defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE))
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
  LIBXS_ASSERT(NULL != rx && MAP_FAILED != *rx);
  if (NULL != dir && 0 != *dir) {
    i = LIBXS_SNPRINTF(filename, sizeof(filename), "%s/" LIBXS_MALLOC_XMAP_TEMPLATE, dir);
  }
  if (0 <= i && i < (int)sizeof(filename)) {
    /* coverity[secure_temp] */
    i = mkstemp(filename);
    if (0 <= i) {
      if (0 == unlink(filename) && 0 == ftruncate(i, size) /*&& 0 == chmod(filename, S_IRWXU)*/) {
        const int mflags = (flags | LIBXS_MAP_SHARED);
        void *const xmap = mmap(*rx, size, PROT_READ | PROT_EXEC, mflags, i, 0/*offset*/);
        if (MAP_FAILED != xmap) {
          LIBXS_ASSERT(NULL != xmap);
#if defined(MAP_32BIT)
          result = mmap(NULL, size, PROT_READ | PROT_WRITE, mflags & ~MAP_32BIT, i, 0/*offset*/);
#else
          result = mmap(NULL, size, PROT_READ | PROT_WRITE, mflags, i, 0/*offset*/);
#endif
          if (MAP_FAILED != result) {
            LIBXS_ASSERT(NULL != result);
            internal_xmalloc_mhint(xmap, size);
            *rx = xmap;
          }
          else {
            munmap(xmap, size);
            *rx = NULL;
          }
        }
      }
      close(i);
    }
  }
  return result;
}

#endif /*!defined(_WIN32)*/

LIBXS_API_INLINE void* internal_xrealloc(void** ptr, internal_malloc_info_type** info, size_t size,
  libxs_realloc_fun realloc_fn, libxs_free_fun free_fn)
{
  char *const base = (char*)(NULL != *info ? (*info)->pointer : *ptr), *result;
  LIBXS_ASSERT(NULL != *ptr);
  /* may implicitly invalidate info */
  result = (char*)realloc_fn(base, size);
  if (result == base) { /* signal no-copy */
    LIBXS_ASSERT(NULL != result);
    *info = NULL; /* no delete */
    *ptr = NULL; /* no copy */
  }
  else if (NULL != result) { /* copy */
    const size_t offset_src = (const char*)*ptr - base;
    *ptr = result + offset_src; /* copy */
    *info = NULL; /* no delete */
  }
#if !defined(NDEBUG) && 0
  else { /* failed */
    if (NULL != *info) {
      /* implicitly invalidates info */
      internal_xfree(*ptr, *info);
    }
    else { /* foreign pointer */
      free_fn(*ptr);
    }
    *info = NULL; /* no delete */
    *ptr = NULL; /* no copy */
  }
#else
  LIBXS_UNUSED(free_fn);
#endif
  return result;
}


LIBXS_API_INTERN void* internal_xmalloc(void** /*ptr*/, internal_malloc_info_type** /*info*/, size_t /*size*/,
  const void* /*context*/, libxs_malloc_function /*malloc_fn*/, libxs_free_function /*free_fn*/);
LIBXS_API_INTERN void* internal_xmalloc(void** ptr, internal_malloc_info_type** info, size_t size,
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn)
{
  void* result;
  LIBXS_ASSERT(NULL != ptr && NULL != info && NULL != malloc_fn.function);
  if (NULL == *ptr) {
    result = (NULL == context
      ? malloc_fn.function(size)
      : malloc_fn.ctx_form(size, context));
  }
  else { /* reallocate */
    if (NULL != free_fn.function /* prefer free_fn since it is part of pointer-info */
      ? (__real_free == free_fn.function || free == free_fn.function)
      : (__real_malloc == malloc_fn.function || malloc == malloc_fn.function))
    {
#if defined(LIBXS_MALLOC_HOOK_REALLOC)
      result = internal_xrealloc(ptr, info, size, __real_realloc, __real_free);
#else
      result = internal_xrealloc(ptr, info, size, realloc, __real_free);
#endif
    }
    else { /* fall-back with regular allocation */
      result = (NULL == context
        ? malloc_fn.function(size)
        : malloc_fn.ctx_form(size, context));
      if (NULL == result) { /* failed */
        if (NULL != *info) {
          internal_xfree(*ptr, *info);
        }
        else { /* foreign pointer */
          (NULL != free_fn.function ? free_fn.function : __real_free)(*ptr);
        }
        *ptr = NULL; /* safe delete */
      }
    }
  }
  return result;
}


LIBXS_API int libxs_xmalloc(void** memory, size_t size, size_t alignment,
  int flags, const void* extra, size_t extra_size)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG)
  if (NULL != memory)
#endif
  {
    static int error_once = 0;
    if (0 != size) {
      size_t alloc_alignment = 0, alloc_size = 0, max_preserve = 0;
      internal_malloc_info_type* info = NULL;
      void* buffer = NULL, * reloc = NULL;
      /* ATOMIC BEGIN: this region should be atomic/locked */
      const void* context = libxs_default_allocator_context;
      libxs_malloc_function malloc_fn = libxs_default_malloc_fn;
      libxs_free_function free_fn = libxs_default_free_fn;
      if (0 != (LIBXS_MALLOC_FLAG_SCRATCH & flags)) {
        context = libxs_scratch_allocator_context;
        malloc_fn = libxs_scratch_malloc_fn;
        free_fn = libxs_scratch_free_fn;
#if defined(LIBXS_MALLOC_MMAP_SCRATCH)
        flags |= LIBXS_MALLOC_FLAG_MMAP;
#endif
      }
      if ((0 != (internal_malloc_kind & 1) && 0 < internal_malloc_kind)
        || NULL == malloc_fn.function || NULL == free_fn.function)
      {
        malloc_fn.function = __real_malloc;
        free_fn.function = __real_free;
        context = NULL;
      }
      /* ATOMIC END: this region should be atomic */
      flags |= LIBXS_MALLOC_FLAG_RW; /* normalize given flags since flags=0 is accepted as well */
      if (0 != (LIBXS_MALLOC_FLAG_REALLOC & flags) && NULL != *memory) {
        info = internal_malloc_info(*memory, 2/*check*/);
        if (NULL != info) {
          max_preserve = info->size;
        }
        else { /* reallocation of unknown allocation */
          flags &= ~LIBXS_MALLOC_FLAG_MMAP;
        }
      }
      else *memory = NULL;
#if !defined(LIBXS_MALLOC_MMAP)
      if (0 == (LIBXS_MALLOC_FLAG_X & flags) && 0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) {
        alloc_alignment = (0 == (LIBXS_MALLOC_FLAG_REALLOC & flags) ? libxs_alignment(size, alignment) : alignment);
        alloc_size = size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1;
        buffer = internal_xmalloc(memory, &info, alloc_size, context, malloc_fn, free_fn);
      }
      else
#endif
      if (NULL == info || size != info->size) {
#if defined(_WIN32) ||defined(__CYGWIN__)
        const int mflags = (0 != (LIBXS_MALLOC_FLAG_X & flags) ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
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
                buffer = VirtualAlloc(NULL, alloc_size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, mflags);
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
        if (NULL == buffer) { /* small allocation or retry with regular page size */
          /* VirtualAlloc cannot be used to reallocate memory */
          buffer = VirtualAlloc(NULL, alloc_size, MEM_RESERVE | MEM_COMMIT, mflags);
        }
        if (NULL != buffer) {
          flags |= LIBXS_MALLOC_FLAG_MMAP; /* select the corresponding deallocation */
        }
        else if (0 == (LIBXS_MALLOC_FLAG_MMAP & flags)) { /* fall-back allocation */
          buffer = internal_xmalloc(memory, &info, alloc_size, context, malloc_fn, free_fn);
        }
#else /* !defined(_WIN32) */
# if defined(MAP_HUGETLB) && defined(LIBXS_MALLOC_HUGE_PAGES)
        static size_t limit_hugetlb = LIBXS_SCRATCH_UNLIMITED;
# endif
# if defined(MAP_LOCKED) && defined(LIBXS_MALLOC_LOCK_PAGES)
        static size_t limit_plocked = LIBXS_SCRATCH_UNLIMITED;
# endif
# if defined(MAP_32BIT)
        static int map32 = 1;
# endif
        int mflags = 0
# if defined(MAP_UNINITIALIZED) && 0/*fails with WSL*/
          | MAP_UNINITIALIZED /* unlikely available */
# endif
# if defined(MAP_NORESERVE)
          | (LIBXS_MALLOC_ALIGNMAX < size ? 0 : MAP_NORESERVE)
# endif
# if defined(MAP_32BIT)
          | ((0 != (LIBXS_MALLOC_FLAG_X & flags) && 0 != map32
            && (LIBXS_X86_AVX512_CORE > libxs_target_archid)
            && (LIBXS_X86_AVX512 < libxs_target_archid ||
                LIBXS_X86_AVX > libxs_target_archid)) ? MAP_32BIT : 0)
# endif
# if defined(MAP_HUGETLB) && defined(LIBXS_MALLOC_HUGE_PAGES)
          | ((0 == (LIBXS_MALLOC_FLAG_X & flags)
            && ((LIBXS_MALLOC_ALIGNMAX * LIBXS_MALLOC_ALIGNFCT) <= size ||
              0 != (LIBXS_MALLOC_FLAG_PHUGE & flags))
            && (internal_malloc_hugetlb + size) < limit_hugetlb) ? MAP_HUGETLB : 0)
# endif
# if defined(MAP_LOCKED) && defined(LIBXS_MALLOC_LOCK_PAGES) && 0 == (LIBXS_MALLOC_LOCK_PAGES)
          | (((0 != (LIBXS_MALLOC_FLAG_PLOCK & flags) || 0 == (LIBXS_MALLOC_FLAG_X & flags))
            && (internal_malloc_plocked + size) < limit_plocked) ? MAP_LOCKED : 0)
# endif
        ; /* mflags */
# if defined(MAP_POPULATE)
        { static int prefault = 0;
          if (0 == prefault) { /* prefault only on Linux 3.10.0-327 (and later) to avoid data race in page-fault handler */
            struct utsname osinfo; unsigned int version_major = 3, version_minor = 10, version_update = 0, version_patch = 327;
            if (0 <= uname(&osinfo) && 0 == strcmp("Linux", osinfo.sysname)
              && 4 == sscanf(osinfo.release, "%u.%u.%u-%u", &version_major, &version_minor, &version_update, &version_patch)
              && LIBXS_VERSION4(3, 10, 0, 327) > LIBXS_VERSION4(version_major, version_minor, version_update, version_patch))
            {
              mflags |= MAP_POPULATE; prefault = 1;
            }
            else prefault = -1;
          }
          else if (1 == prefault) mflags |= MAP_POPULATE;
        }
# endif
        /* make allocated size at least a multiple of the smallest page-size to avoid split-pages (unmap!) */
        alloc_alignment = libxs_lcm(0 == alignment ? libxs_alignment(size, alignment) : alignment, LIBXS_PAGE_MINSIZE);
        alloc_size = LIBXS_UP2(size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1, alloc_alignment);
        if (0 == (LIBXS_MALLOC_FLAG_X & flags)) { /* anonymous and non-executable */
# if defined(MAP_32BIT)
          LIBXS_ASSERT(0 == (MAP_32BIT & mflags));
# endif
# if 0
          LIBXS_ASSERT(NULL != info || NULL == *memory); /* no memory mapping of foreign pointer */
# endif
          buffer = mmap(NULL == info ? NULL : info->pointer, alloc_size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | mflags, -1, 0/*offset*/);
# if defined(MAP_HUGETLB) && defined(LIBXS_MALLOC_HUGE_PAGES)
          INTERNAL_XMALLOC_KIND(MAP_HUGETLB, "huge-page", LIBXS_MALLOC_FLAG_PHUGE, flags, mflags,
            internal_malloc_hugetlb, limit_hugetlb, info, alloc_size, buffer);
# endif
# if defined(MAP_LOCKED) && defined(LIBXS_MALLOC_LOCK_PAGES)
#   if 0 == (LIBXS_MALLOC_LOCK_PAGES)
          INTERNAL_XMALLOC_KIND(MAP_LOCKED, "locked-page", LIBXS_MALLOC_FLAG_PLOCK, flags, mflags,
            internal_malloc_plocked, limit_plocked, info, alloc_size, buffer);
#   else
          if (0 != (MAP_LOCKED & mflags) && MAP_FAILED != buffer) {
            LIBXS_ASSERT(NULL != buffer);
#     if 1 == (LIBXS_MALLOC_LOCK_PAGES) || !defined(MLOCK_ONFAULT) || !defined(SYS_mlock2)
            if (0 == mlock(buffer, alloc_size))
#     elif 0 /* mlock2 is potentially not exposed */
            if (0 == mlock2(buffer, alloc_size, MLOCK_ONFAULT))
#     else
            if (0 == syscall(SYS_mlock2, buffer, alloc_size, MLOCK_ONFAULT))
#     endif
            {
              LIBXS_ATOMIC_ADD_FETCH(&internal_malloc_plocked, alloc_size, LIBXS_ATOMIC_RELAXED);
              flags |= LIBXS_MALLOC_FLAG_PLOCK;
            }
            else { /* update watermark */
              INTERNAL_XMALLOC_WATERMARK("locked-page", internal_malloc_plocked, limit_plocked, alloc_size);
              flags &= ~LIBXS_MALLOC_FLAG_PLOCK;
            }
          }
#   endif
# endif
        }
        else { /* executable buffer requested */
          static /*LIBXS_TLS*/ int entrypoint = -1; /* fall-back allocation method */
# if defined(MAP_HUGETLB) && defined(LIBXS_MALLOC_HUGE_PAGES)
          LIBXS_ASSERT(0 == (MAP_HUGETLB & mflags));
# endif
# if defined(MAP_LOCKED) && defined(LIBXS_MALLOC_LOCK_PAGES)
          LIBXS_ASSERT(0 == (MAP_LOCKED & mflags));
# endif
          if (0 > (int)LIBXS_ATOMIC_LOAD(&entrypoint, LIBXS_ATOMIC_RELAXED)) {
            const char *const env = getenv("LIBXS_SE");
            LIBXS_ATOMIC_STORE(&entrypoint, NULL == env
              /* libxs_se decides */
              ? (0 == libxs_se ? LIBXS_MALLOC_FINAL : LIBXS_MALLOC_FALLBACK)
              /* user's choice takes precedence */
              : ('0' != *env ? LIBXS_MALLOC_FALLBACK : LIBXS_MALLOC_FINAL),
              LIBXS_ATOMIC_SEQ_CST);
            LIBXS_ASSERT(0 <= entrypoint);
          }
          INTERNAL_XMALLOC(0, entrypoint, "JITDUMPDIR", "", map32, mflags, alloc_size, buffer, &reloc); /* 1st try */
          INTERNAL_XMALLOC(1, entrypoint, "TMPDIR", "/tmp", map32, mflags, alloc_size, buffer, &reloc); /* 2nd try */
          INTERNAL_XMALLOC(2, entrypoint, "HOME", "", map32, mflags, alloc_size, buffer, &reloc); /* 3rd try */
          if (3 >= entrypoint && (MAP_FAILED == buffer || NULL == buffer)) { /* 4th try */
            buffer = mmap(reloc, alloc_size, PROT_READ | PROT_WRITE | PROT_EXEC,
# if defined(MAP_32BIT)
              MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | (0 == map32 ? (mflags & ~MAP_32BIT) : mflags),
# else
              MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | mflags,
# endif
              -1, 0/*offset*/);
            if (MAP_FAILED != buffer) entrypoint = 3;
# if defined(MAP_32BIT)
            else if (0 != (MAP_32BIT & mflags) && 0 != map32) {
              buffer = mmap(reloc, alloc_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | LIBXS_MAP_ANONYMOUS | (mflags & ~MAP_32BIT),
                - 1, 0/*offset*/);
              if (MAP_FAILED != buffer) {
                entrypoint = 3;
                map32 = 0;
              }
            }
# endif
          }
          /* upgrade to SE-mode and retry lower entry-points */
          if (MAP_FAILED == buffer && 0 == libxs_se) {
            libxs_se = 1; entrypoint = 0;
            INTERNAL_XMALLOC(0, entrypoint, "JITDUMPDIR", "", map32, mflags, alloc_size, buffer, &reloc); /* 1st try */
            INTERNAL_XMALLOC(1, entrypoint, "TMPDIR", "/tmp", map32, mflags, alloc_size, buffer, &reloc); /* 2nd try */
            INTERNAL_XMALLOC(2, entrypoint, "HOME", "", map32, mflags, alloc_size, buffer, &reloc); /* 3rd try */
          }
        }
        if (MAP_FAILED != buffer && NULL != buffer) {
          flags |= LIBXS_MALLOC_FLAG_MMAP; /* select deallocation */
        }
        else { /* allocation failed */
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
#endif /* !defined(_WIN32) */
      }
      else { /* reallocation of the same pointer and size */
        alloc_size = size + extra_size + sizeof(internal_malloc_info_type) + alignment - 1;
        if (NULL != info) {
          buffer = info->pointer;
          flags |= info->flags;
        }
        else {
          flags |= LIBXS_MALLOC_FLAG_MMAP;
          buffer = *memory;
        }
        alloc_alignment = alignment;
        *memory = NULL; /* signal no-copy */
      }
      if (
#if !defined(_WIN32) && !defined(__clang_analyzer__)
        MAP_FAILED != buffer &&
#endif
        NULL != buffer)
      {
        char *const cbuffer = (char*)buffer, *const aligned = LIBXS_ALIGN(
          cbuffer + extra_size + sizeof(internal_malloc_info_type), alloc_alignment);
        internal_malloc_info_type *const buffer_info = (internal_malloc_info_type*)(
          aligned - sizeof(internal_malloc_info_type));
        LIBXS_ASSERT((aligned + size) <= (cbuffer + alloc_size));
        LIBXS_ASSERT(0 < alloc_alignment);
        /* former content must be preserved prior to setup of buffer_info */
        if (NULL != *memory) { /* preserve/copy previous content */
#if 0
          LIBXS_ASSERT(0 != (LIBXS_MALLOC_FLAG_REALLOC & flags));
#endif
          /* content behind foreign pointers is not explicitly preserved; buffers may overlap */
          memmove(aligned, *memory, LIBXS_MIN(max_preserve, size));
          if (NULL != info /* known allocation (non-foreign pointer) */
            && EXIT_SUCCESS != internal_xfree(*memory, info) /* !libxs_free */
            && 0 != libxs_verbosity /* library code is expected to be mute */
            && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
          { /* display some extra context of the failure (reallocation) */
            fprintf(stderr, "LIBXS ERROR: memory reallocation failed to release memory!\n");
          }
        }
        if (NULL != extra || 0 == extra_size) {
          const char *const src = (const char*)extra;
          int i; for (i = 0; i < (int)extra_size; ++i) cbuffer[i] = src[i];
        }
        else if (0 != libxs_verbosity /* library code is expected to be mute */
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS ERROR: incorrect extraneous data specification!\n");
          /* no EXIT_FAILURE because valid buffer is returned */
        }
        if (0 == (LIBXS_MALLOC_FLAG_X & flags)) { /* update statistics */
          if (0 == (LIBXS_MALLOC_FLAG_PRIVATE & flags)) { /* public */
            if (0 != (LIBXS_MALLOC_FLAG_SCRATCH & flags)) { /* scratch */
              const size_t watermark = LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
                &internal_malloc_public_cur, alloc_size, LIBXS_ATOMIC_RELAXED);
              if (internal_malloc_public_max < watermark) internal_malloc_public_max = watermark; /* accept data-race */
            }
            else { /* local */
              const size_t watermark = LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
                &internal_malloc_local_cur, alloc_size, LIBXS_ATOMIC_RELAXED);
              if (internal_malloc_local_max < watermark) internal_malloc_local_max = watermark; /* accept data-race */
            }
          }
          else if (0 != (LIBXS_MALLOC_FLAG_SCRATCH & flags)) { /* private scratch */
            const size_t watermark = LIBXS_ATOMIC(LIBXS_ATOMIC_ADD_FETCH, LIBXS_BITS)(
              &internal_malloc_private_cur, alloc_size, LIBXS_ATOMIC_RELAXED);
            if (internal_malloc_private_max < watermark) internal_malloc_private_max = watermark; /* accept data-race */
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
#if defined(LIBXS_VTUNE)
        buffer_info->code_id = 0;
#endif /* info must be initialized to calculate correct checksum */
#if !defined(LIBXS_MALLOC_CRC_OFF)
# if defined(LIBXS_MALLOC_CRC_LIGHT)
        buffer_info->hash = LIBXS_CRC32U(LIBXS_BITS)(LIBXS_MALLOC_SEED, &buffer_info);
# else
        buffer_info->hash = libxs_crc32(LIBXS_MALLOC_SEED, buffer_info,
          (unsigned int)(((char*)&buffer_info->hash) - ((char*)buffer_info)));
# endif
#endif  /* finally commit/return allocated buffer */
        *memory = aligned;
      }
      else {
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          char alloc_size_buffer[32];
          libxs_format_size(alloc_size_buffer, sizeof(alloc_size_buffer), alloc_size, "KM", "B", 10);
          fprintf(stderr, "LIBXS ERROR: failed to allocate %s with flag=%i!\n", alloc_size_buffer, flags);
        }
        result = EXIT_FAILURE;
        *memory = NULL;
      }
    }
    else {
      if ((LIBXS_VERBOSITY_HIGH <= libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS WARNING: zero-sized memory allocation detected!\n");
      }
      *memory = NULL; /* no EXIT_FAILURE */
    }
  }
#if !defined(NDEBUG)
  else if (0 != size) {
    result = EXIT_FAILURE;
  }
#endif
  return result;
}


LIBXS_API void libxs_xfree(const void* memory, int check)
{
#if (!defined(LIBXS_MALLOC_HOOK) || defined(_DEBUG))
  static int error_once = 0;
#endif
  /*const*/ internal_malloc_info_type *const info = internal_malloc_info(memory, check);
  if (NULL != info) { /* !libxs_free */
#if (!defined(LIBXS_MALLOC_HOOK) || defined(_DEBUG))
    if (EXIT_SUCCESS != internal_xfree(memory, info)) {
      if ( 0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: memory deallocation failed!\n");
      }
    }
#else
    internal_xfree(memory, info);
#endif
  }
  else if (NULL != memory) {
#if 1
    union { const void* const_ptr; void* ptr; } cast;
    cast.const_ptr = memory; /* C-cast still warns */
    __real_free(cast.ptr);
#endif
#if (!defined(LIBXS_MALLOC_HOOK) || defined(_DEBUG))
    if ( 0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: deallocation does not match allocation!\n");
    }
#endif
  }
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
  internal_malloc_info_type *const info = (NULL != memory ? internal_malloc_info(*memory, 0/*no check*/) : NULL);
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
        void *const code_ptr = (NULL != info->reloc ? ((void*)(((char*)info->reloc) + alignment)) : *memory);
        LIBXS_ASSERT(0 != (LIBXS_MALLOC_FLAG_X & flags));
        if (name && *name) { /* profiler support requested */
          if (0 > libxs_verbosity) { /* avoid dump if just the profiler is enabled */
            LIBXS_EXPECT(EXIT_SUCCESS, libxs_dump("LIBXS-JIT-DUMP", name, code_ptr, size, 1/*unique*/));
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
# if !defined(LIBXS_MALLOC_CRC_OFF) /* update checksum */
#   if defined(LIBXS_MALLOC_CRC_LIGHT)
          { const internal_malloc_info_type *const code_info = internal_malloc_info(code_ptr, 0/*no check*/);
            info->hash = LIBXS_CRC32U(LIBXS_BITS)(LIBXS_MALLOC_SEED, &code_info);
          }
#   else
          info->hash = libxs_crc32(LIBXS_MALLOC_SEED, info,
            /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info)));
#   endif
# endif   /* treat memory protection errors as soft error; ignore return value */
          munmap(buffer, alloc_size);
#endif
        }
#if !defined(_WIN32)
        else { /* malloc-based fall-back */
          int mprotect_result;
# if !defined(LIBXS_MALLOC_CRC_OFF) && defined(LIBXS_VTUNE) /* check checksum */
#   if defined(LIBXS_MALLOC_CRC_LIGHT)
          assert(info->hash == LIBXS_CRC32U(LIBXS_BITS)(LIBXS_MALLOC_SEED, &info)); /* !LIBXS_ASSERT */
#   else
          assert(info->hash == libxs_crc32(LIBXS_MALLOC_SEED, info, /* !LIBXS_ASSERT */
            /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info))));
#   endif
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
  else if ((LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity)
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: %s buffer %p does not match!\n",
      0 != (LIBXS_MALLOC_FLAG_X & flags) ? "executable" : "memory", *memory);
  }
  return result;
}


LIBXS_API LIBXS_ATTRIBUTE_MALLOC void* libxs_aligned_malloc(size_t size, size_t alignment)
{
  void* result = NULL;
  LIBXS_INIT
  if (2 > internal_malloc_kind) {
#if !defined(NDEBUG)
    int status =
#endif
    libxs_xmalloc(&result, size, alignment, LIBXS_MALLOC_FLAG_DEFAULT, NULL/*extra*/, 0/*extra_size*/);
    assert(EXIT_SUCCESS == status || NULL == result); /* !LIBXS_ASSERT */
  }
  else { /* scratch */
    const void *const caller = libxs_trace_caller_id(0/*level*/);
    internal_scratch_malloc(&result, size, alignment, LIBXS_MALLOC_FLAG_DEFAULT, caller);
  }
  return result;
}


LIBXS_API void* libxs_realloc(size_t size, void* ptr)
{
  const int nzeros = LIBXS_INTRINSICS_BITSCANFWD64((uintptr_t)ptr), alignment = 1 << nzeros;
  LIBXS_ASSERT(0 == ((uintptr_t)ptr & ~(0xFFFFFFFFFFFFFFFF << nzeros)));
  LIBXS_INIT
  if (2 > internal_malloc_kind) {
#if !defined(NDEBUG)
    int status =
#endif
    libxs_xmalloc(&ptr, size, alignment, LIBXS_MALLOC_FLAG_REALLOC, NULL/*extra*/, 0/*extra_size*/);
    assert(EXIT_SUCCESS == status || NULL == ptr); /* !LIBXS_ASSERT */
  }
  else { /* scratch */
    const void *const caller = libxs_trace_caller_id(0/*level*/);
    internal_scratch_malloc(&ptr, size, alignment, LIBXS_MALLOC_FLAG_REALLOC, caller);
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
#if defined(LIBXS_MALLOC_SCRATCH_DELETE_FIRST) || /* prefer safe method if possible */ \
  !defined(LIBXS_MALLOC_HOOK)
# if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
    internal_malloc_pool_type *const pool = internal_scratch_malloc_pool(memory);
    if (NULL != pool) { /* memory belongs to scratch domain */
      internal_scratch_free(memory, pool);
    }
    else
# endif
    { /* local */
      libxs_xfree(memory, 2/*check*/);
    }
#else /* lookup matching pool */
    internal_malloc_info_type *const info = internal_malloc_info(memory, 2/*check*/);
    static int error_once = 0;
    if (NULL != info && 0 == (LIBXS_MALLOC_FLAG_SCRATCH & info->flags)) { /* !libxs_free */
# if !defined(NDEBUG)
      if (EXIT_SUCCESS != internal_xfree(memory, info)
        && 0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: memory deallocation failed!\n");
      }
# else
      internal_xfree(memory, info); /* !libxs_free */
# endif
    }
    else {
# if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
      internal_malloc_pool_type *const pool = internal_scratch_malloc_pool(memory);
      if (NULL != pool) { /* memory belongs to scratch domain */
        internal_scratch_free(memory, pool);
      }
      else
# endif
      {
# if defined(NDEBUG) && defined(LIBXS_MALLOC_HOOK)
        __real_free((void*)memory);
# else
#   if defined(LIBXS_MALLOC_HOOK)
        __real_free((void*)memory);
#   endif
        if (0 != libxs_verbosity && /* library code is expected to be mute */
            1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS ERROR: deallocation does not match allocation!\n");
        }
# endif
      }
    }
#endif
  }
}


LIBXS_API_INTERN void libxs_xrelease_scratch(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock)
{
#if defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
  internal_malloc_pool_type* pools = NULL;
  libxs_scratch_info scratch_info;
  LIBXS_ASSERT(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
  if (NULL != lock) {
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
  }
# if defined(LIBXS_MALLOC_DELETE_SAFE)
  if (0 == (internal_malloc_kind & 1) || 0 >= internal_malloc_kind)
# endif
  {
    unsigned int i;
    pools = (internal_malloc_pool_type*)LIBXS_UP2(
      (uintptr_t)internal_malloc_pool_buffer, LIBXS_MALLOC_SCRATCH_PADDING);
    for (i = 0; i < libxs_scratch_pools; ++i) {
      if (0 != pools[i].instance.minsize) {
        if (
# if !defined(LIBXS_MALLOC_SCRATCH_DELETE_FIRST)
          1 < pools[i].instance.counter &&
# endif
          NULL != pools[i].instance.buffer)
        {
          internal_malloc_info_type *const info = internal_malloc_info(pools[i].instance.buffer, 2/*check*/);
          if (NULL != info) internal_xfree(info->pointer, info);
        }
      }
      else break; /* early exit */
    }
  }
  LIBXS_EXPECT(EXIT_SUCCESS, libxs_get_scratch_info(&scratch_info));
  if (0 != scratch_info.npending && /* library code is expected to be mute */
    (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity))
  {
    char pending_size_buffer[32];
    libxs_format_size(pending_size_buffer, sizeof(pending_size_buffer),
      internal_malloc_public_cur + internal_malloc_local_cur, "KM", "B", 10);
    fprintf(stderr, "LIBXS WARNING: %s pending scratch-memory by %" PRIuPTR " allocation%s!\n",
      pending_size_buffer, (uintptr_t)scratch_info.npending, 1 < scratch_info.npending ? "s" : "");
  }
  if (NULL != pools) {
    memset(pools, 0, (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) * sizeof(internal_malloc_pool_type));
    /* no reset: keep private watermark (internal_malloc_private_max, internal_malloc_private_cur) */
    internal_malloc_public_max = internal_malloc_public_cur = 0;
    internal_malloc_local_max = internal_malloc_local_cur = 0;
    internal_malloc_scratch_nmallocs = 0;
  }
  if (NULL != lock) {
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
#endif
}


LIBXS_API void libxs_release_scratch(void)
{
  libxs_xrelease_scratch(&libxs_lock_global);
}


LIBXS_API int libxs_get_malloc_info(const void* memory, libxs_malloc_info* info)
{
  int result = EXIT_SUCCESS;
  if (NULL != info) {
    size_t size;
    result = libxs_get_malloc_xinfo(memory, &size, NULL/*flags*/, NULL/*extra*/);
    LIBXS_MEMZERO127(info);
    if (EXIT_SUCCESS == result) {
      info->size = size;
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else if (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity) {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS WARNING: foreign memory buffer %p discovered!\n", memory);
      }
    }
#endif
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
    LIBXS_MEMZERO127(info);
    info->nmallocs = internal_malloc_scratch_nmallocs;
    info->internal = internal_malloc_private_max;
    info->local = internal_malloc_local_max;
    info->size = internal_malloc_public_max;
    { const internal_malloc_pool_type* pool = (const internal_malloc_pool_type*)LIBXS_UP2(
        (uintptr_t)internal_malloc_pool_buffer, LIBXS_MALLOC_SCRATCH_PADDING);
# if (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
      const internal_malloc_pool_type *const end = pool + libxs_scratch_pools;
      LIBXS_ASSERT(libxs_scratch_pools <= LIBXS_MALLOC_SCRATCH_MAX_NPOOLS);
      for (; pool != end; ++pool) if ((LIBXS_MALLOC_INTERNAL_CALLER) != pool->instance.site) {
# endif
        if (0 != pool->instance.minsize) {
          const size_t npending = pool->instance.counter;
# if defined(LIBXS_MALLOC_SCRATCH_DELETE_FIRST)
          info->npending += npending;
# else
          info->npending += 1 < npending ? (npending - 1) : 0;
# endif
          ++info->npools;
        }
# if (1 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))
        else break; /* early exit */
      }
# endif
    }
#else
    LIBXS_MEMZERO127(info);
#endif /*defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXS_MALLOC_SCRATCH_MAX_NPOOLS))*/
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API void libxs_set_scratch_limit(size_t nbytes)
{
  /* !LIBXS_INIT */
  internal_malloc_scratch_limit = nbytes;
}


LIBXS_API size_t libxs_get_scratch_limit(void)
{
  size_t result;
  /* !LIBXS_INIT */
  if (LIBXS_SCRATCH_DEFAULT != internal_malloc_scratch_limit) {
    result = internal_malloc_scratch_limit;
  }
  else if (0 == internal_malloc_kind) {
    result = LIBXS_MALLOC_SCRATCH_LIMIT;
  }
  else {
    result = LIBXS_SCRATCH_UNLIMITED;
  }
  return result;
}


LIBXS_API void libxs_set_malloc(int enabled, const size_t* lo, const size_t* hi)
{
  /* !LIBXS_INIT */
#if defined(LIBXS_MALLOC_HOOK) && defined(LIBXS_MALLOC) && (0 != LIBXS_MALLOC)
# if (0 < LIBXS_MALLOC)
  LIBXS_UNUSED(enabled);
  internal_malloc_kind = LIBXS_MALLOC;
# else
  internal_malloc_kind = enabled;
# endif
  /* setup lo/hi after internal_malloc_kind! */
  if (NULL != lo) internal_malloc_limit[0] = *lo;
  if (NULL != hi) {
    const size_t scratch_limit = libxs_get_scratch_limit();
    const size_t malloc_upper = LIBXS_MIN(*hi, scratch_limit);
    internal_malloc_limit[1] = LIBXS_MAX(malloc_upper, internal_malloc_limit[0]);
  }
#else
  LIBXS_UNUSED(lo); LIBXS_UNUSED(hi);
  internal_malloc_kind = enabled;
#endif
  libxs_malloc_init();
}


LIBXS_API int libxs_get_malloc(size_t* lo, size_t* hi)
{
  LIBXS_INIT
#if defined(LIBXS_MALLOC_HOOK) && defined(LIBXS_MALLOC) && (0 != LIBXS_MALLOC)
  if (NULL != lo) *lo = internal_malloc_limit[0];
  if (NULL != hi) *hi = internal_malloc_limit[1];
#else
  if (NULL != lo) *lo = 0;
  if (NULL != hi) *hi = 0;
#endif
  return internal_malloc_kind;
}

