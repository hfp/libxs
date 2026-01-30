/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_mem.h>
#include "libxs_hash.h"

#if 0

#include <signal.h>
#if !defined(NDEBUG)
# include <errno.h>
#endif
#if !defined(_WIN32)
# if defined(__GNUC__) || defined(__PGI) || defined(_CRAYC)
#   include <sys/time.h>
#   include <time.h>
# endif
# include <sys/types.h>
# include <sys/mman.h>
# include <sys/stat.h>
# include <fcntl.h>
#endif
#if defined(__APPLE__)
# include <libkern/OSCacheControl.h>
/*# include <mach/mach_time.h>*/
# include <pthread.h>
#endif
#if defined(__powerpc64__)
# include <sys/platform/ppc.h>
#endif

/* used internally to re-implement certain exit-handler */
#if !defined(LIBXS_EXIT_SUCCESS)
# define LIBXS_EXIT_SUCCESS() exit(EXIT_SUCCESS)
#endif
#if !defined(LIBXS_CODE_MAXSIZE)
# define LIBXS_CODE_MAXSIZE 131072
#endif
#if !defined(LIBXS_DIFF_SIZE)
# define LIBXS_DIFF_SIZE LIBXS_DESCRIPTOR_SIGSIZE
#endif
#if !defined(LIBXS_HASH_SIZE)
/* can be smaller than MAXSIZE/SIGSIZE at the expense of collisions */
# define LIBXS_HASH_SIZE 32
#endif
#if !defined(LIBXS_HASH_SEED)
# define LIBXS_HASH_SEED 25071975
#endif
#if !defined(LIBXS_ENABLE_DEREG) && 0
# define LIBXS_ENABLE_DEREG
#endif
#if !defined(LIBXS_REGUSER_HASH) && 1
# define LIBXS_REGUSER_HASH
#endif
#if !defined(LIBXS_REGUSER_ALIGN) && 1
# define LIBXS_REGUSER_ALIGN
#endif
#if !defined(LIBXS_REGLOCK_TRY) && 0
# define LIBXS_REGLOCK_TRY
#endif
#if !defined(LIBXS_UNIFY_LOCKS) && 1
# define LIBXS_UNIFY_LOCKS
#endif
#if !defined(LIBXS_REGKEY_PAD) && 0
# define LIBXS_REGKEY_PAD
#endif
#if !defined(LIBXS_CACHE_PAD) && 1
# define LIBXS_CACHE_PAD
#endif
#if !defined(LIBXS_AUTOPIN) && 0
# define LIBXS_AUTOPIN
#endif
#if !defined(LIBXS_MAIN_DELIMS)
# define LIBXS_MAIN_DELIMS ";,:"
#endif

#if !defined(_WIN32) && !defined(__CYGWIN__)
LIBXS_EXTERN int posix_memalign(void**, size_t, size_t) LIBXS_NOTHROW;
#endif

/* flag fused into the memory address of a code version in case of non-JIT */
#define LIBXS_CODE_STATIC (1ULL << (8 * sizeof(void*) - 1))
/* flag fused into the memory address of a code version in case of collision */
#if 1 /* beneficial when registry approaches capacity (collisions) */
# define LIBXS_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 2))
#endif
#if !defined(LIBXS_COLLISION_COUNT_STATIC) && 0
# define LIBXS_COLLISION_COUNT_STATIC
#endif

#if (0 != LIBXS_SYNC)
# if !defined(INTERNAL_REGLOCK_MAXN)
#   if defined(_MSC_VER)
#     define INTERNAL_REGLOCK_MAXN 0
#   else
#     define INTERNAL_REGLOCK_MAXN 0
#   endif
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
#   if !defined(LIBXS_CACHE_MAXSIZE) && (8 > INTERNAL_REGLOCK_MAXN)
#     define LIBXS_CACHE_MAXSIZE LIBXS_CAPACITY_CACHE
#   endif
#   if !defined(LIBXS_REGLOCK)
#     define LIBXS_REGLOCK LIBXS_LOCK_DEFAULT
#   endif
#   if !defined(LIBXS_CLEANUP_NTRY)
#     define LIBXS_CLEANUP_NTRY 7
#   endif
#   if LIBXS_LOCK_TYPE_ISPOD(LIBXS_REGLOCK)
LIBXS_EXTERN_C typedef union internal_reglocktype {
  char pad[LIBXS_CACHELINE];
  LIBXS_LOCK_TYPE(LIBXS_REGLOCK) state;
} internal_reglocktype;
#   else
LIBXS_EXTERN_C typedef union internal_reglocktype {
  LIBXS_LOCK_TYPE(LIBXS_REGLOCK) state;
} internal_reglocktype;
#   endif
LIBXS_APIVAR_DEFINE(internal_reglocktype internal_reglock[INTERNAL_REGLOCK_MAXN]);
# else /* RW-lock */
#   if !defined(LIBXS_CACHE_MAXSIZE)
#     define LIBXS_CACHE_MAXSIZE LIBXS_CAPACITY_CACHE
#   endif
#   if !defined(LIBXS_REGLOCK)
#     if defined(LIBXS_UNIFY_LOCKS)
#       define LIBXS_REGLOCK LIBXS_LOCK
#     elif defined(_MSC_VER)
#       define LIBXS_REGLOCK LIBXS_LOCK_MUTEX
#     elif 0
#       define LIBXS_REGLOCK LIBXS_LOCK_RWLOCK
#     else
#       define LIBXS_REGLOCK LIBXS_LOCK_DEFAULT
#     endif
#   endif
LIBXS_APIVAR_DEFINE(LIBXS_LOCK_TYPE(LIBXS_REGLOCK)* internal_reglock_ptr);
# endif
#elif !defined(LIBXS_CACHE_MAXSIZE)
# define LIBXS_CACHE_MAXSIZE LIBXS_CAPACITY_CACHE
#endif
#if defined(LIBXS_UNPACKED) /* CCE/Classic */
# define LIBXS_CACHE_STRIDE LIBXS_MAX(sizeof(libxs_descriptor), LIBXS_DESCRIPTOR_MAXSIZE)
#else
# define LIBXS_CACHE_STRIDE LIBXS_DESCRIPTOR_MAXSIZE
#endif

#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
# define INTERNAL_FIND_CODE_CACHE_GROW(RESULT_INDEX, CACHE_SIZE) \
    RESULT_INDEX = CACHE_SIZE; CACHE_SIZE = (unsigned char)(0 != (CACHE_SIZE) ? ((CACHE_SIZE) << 1) : 1)
# define INTERNAL_FIND_CODE_CACHE_EVICT(RESULT_INDEX, CACHE_SIZE, CACHE_HIT) \
    RESULT_INDEX = (unsigned char)LIBXS_MOD2((CACHE_HIT) + ((CACHE_SIZE) - 1), CACHE_SIZE)
#endif

#if (0 == LIBXS_SYNC)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) {
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) }
#else
# if defined(LIBXS_REGLOCK_TRY)
#   define INTERNAL_REGLOCK_TRY(DIFF, CODE) \
    if (1 != internal_reglock_count) { /* (re-)try and get (meanwhile) generated code */ \
      LIBXS_ASSERT(NULL != internal_registry); /* engine is not shut down */ \
      continue; \
    } \
    else { /* exit dispatch and let client fall back */ \
      DIFF = 0; CODE = 0; break; \
    }
# else
#   define INTERNAL_REGLOCK_TRY(DIFF, CODE) \
      LIBXS_ASSERT(NULL != internal_registry); /* engine is not shut down */ \
      continue
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) /*do*/ { \
      const unsigned int LOCKINDEX = (0 != internal_reglock_count ? LIBXS_MOD2(INDEX, internal_reglock_count) : 0); \
      if (LIBXS_LOCK_ACQUIRED(LIBXS_REGLOCK) != LIBXS_LOCK_TRYLOCK(LIBXS_REGLOCK, &internal_reglock[LOCKINDEX].state)) { \
        INTERNAL_REGLOCK_TRY(DIFF, CODE); \
      }
#   define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, &internal_reglock[LOCKINDEX].state); } while(0)
# else /* RW-lock */
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) /*do*/ { \
      if (LIBXS_LOCK_ACQUIRED(LIBXS_REGLOCK) != LIBXS_LOCK_TRYLOCK(LIBXS_REGLOCK, internal_reglock_ptr)) { \
        INTERNAL_REGLOCK_TRY(DIFF, CODE); \
      }
#   define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, internal_reglock_ptr); } /*while(0)*/
# endif
#endif

#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
LIBXS_EXTERN_C typedef struct internal_cache_entry_type {
  libxs_descriptor keys[LIBXS_CACHE_MAXSIZE];
  libxs_code_pointer code[LIBXS_CACHE_MAXSIZE];
  unsigned int id; /* to invalidate */
  unsigned char size, hit;
} internal_cache_entry_type;

LIBXS_EXTERN_C typedef union internal_cache_type {
# if defined(LIBXS_CACHE_PAD)
  char pad[LIBXS_UP2(sizeof(internal_cache_entry_type),LIBXS_CACHELINE)];
# endif
  internal_cache_entry_type entry;
} internal_cache_type;

# if defined(LIBXS_NTHREADS_USE)
LIBXS_APIVAR_DEFINE(internal_cache_type* internal_cache_buffer);
# endif
LIBXS_APIVAR_DEFINE(int internal_cache_size);
#endif /*defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))*/
LIBXS_EXTERN_C typedef union internal_regkey_type {
#if defined(LIBXS_REGKEY_PAD)
  char pad[LIBXS_UP2(sizeof(libxs_descriptor), LIBXS_CACHELINE)];
#endif
  libxs_descriptor entry;
} internal_regkey_type;

/** Determines the try-lock property (1<N: disabled, N=1: enabled [N=0: disabled in case of RW-lock]). */
LIBXS_APIVAR_DEFINE(int internal_reglock_count);
LIBXS_APIVAR_DEFINE(size_t internal_registry_nbytes);
LIBXS_APIVAR_DEFINE(unsigned int internal_registry_nleaks);
LIBXS_APIVAR_DEFINE(internal_regkey_type* internal_registry_keys);
LIBXS_APIVAR_DEFINE(libxs_code_pointer* internal_registry);
LIBXS_APIVAR_DEFINE(internal_statistic_type internal_statistic[2/*DP/SP*/][4/*sml/med/big/xxx*/]);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_sml);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_med);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_mnk);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_gemv);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_meltw);
LIBXS_APIVAR_DEFINE(unsigned int internal_statistic_num_user);
LIBXS_APIVAR_DEFINE(const char* internal_build_state);
/** Time stamp (startup time of library). */
LIBXS_APIVAR_DEFINE(libxs_timer_tickint internal_timer_start);
LIBXS_APIVAR_DEFINE(libxs_cpuid_info internal_cpuid_info);

#define LIBXS_TIMER_DURATION_FDIV(A, B) ((double)(A) / (B))
#define LIBXS_TIMER_DURATION_IDIV(A, B) ((A) <= (B) \
  ? LIBXS_TIMER_DURATION_FDIV(A, B) \
  : ((A) / (B) + LIBXS_TIMER_DURATION_FDIV((A) % (B), B)))

#if defined(_WIN32)
# define INTERNAL_SINGLETON_HANDLE HANDLE
# define INTERNAL_SINGLETON(HANDLE) (NULL != (HANDLE))
#else
# define INTERNAL_SINGLETON_HANDLE int
# define INTERNAL_SINGLETON(HANDLE) (0 <= (HANDLE) && '\0' != *internal_singleton_fname)
LIBXS_APIVAR_DEFINE(char internal_singleton_fname[64]);
#endif
LIBXS_APIVAR_DEFINE(INTERNAL_SINGLETON_HANDLE internal_singleton_handle);
LIBXS_APIVAR_DEFINE(char internal_stdio_fname[64]);

LIBXS_EXTERN_C typedef struct internal_sigentry_type {
  int signum; void (*signal)(int);
} internal_sigentry_type;
LIBXS_APIVAR_DEFINE(internal_sigentry_type internal_sigentries[4]);

/* definition of corresponding variables */
LIBXS_APIVAR_PRIVATE_DEF(unsigned int libxs_scratch_pools);
LIBXS_APIVAR_PRIVATE_DEF(double libxs_scratch_scale);
LIBXS_APIVAR_PRIVATE_DEF(double libxs_timer_scale);
LIBXS_APIVAR_PRIVATE_DEF(unsigned int libxs_statistic_num_spmdm);
LIBXS_APIVAR_PRIVATE_DEF(unsigned int libxs_thread_count);
/* definition of corresponding variables */
LIBXS_APIVAR_PUBLIC_DEF(LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_lock_global);
LIBXS_APIVAR_PUBLIC_DEF(unsigned int libxs_ninit);
LIBXS_APIVAR_PUBLIC_DEF(int libxs_stdio_handle);
LIBXS_APIVAR_PUBLIC_DEF(int libxs_nosync);


LIBXS_API_INLINE unsigned int internal_print_number(unsigned int n, char default_unit, char* unit)
{
  unsigned int number = n;
  LIBXS_ASSERT(NULL != unit);
  *unit = default_unit;
  if ((1000000) <= n) {
    number = (n + 500000) / 1000000;
    *unit = 'm';
  }
  else if (9999 < n) {
    number = (n + 500) / 1000;
    *unit = 'k';
  }
  return number;
}


#if !defined(_WIN32)
LIBXS_API_INLINE void internal_register_static_code(
  libxs_datatype precision, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  libxs_xmmfunction xgemm, libxs_code_pointer* registry)
{
  const libxs_blasint lda = m, ldb = k, ldc = m;
  /*const*/ int precondition = LIBXS_GEMM_NO_BYPASS_DIMS(m, n, k) && LIBXS_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc);
  if (precondition) {
    const size_t size = (LIBXS_HASH_SIZE) - sizeof(libxs_descriptor_kind);
    const size_t size_desc = sizeof(libxs_gemm_descriptor);
    libxs_descriptor_blob blob;
    const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init(&blob, precision, precision, precision, precision,
      m, n, k, lda, ldb, ldc, LIBXS_FLAGS | ((LIBXS_BETA == 0) ? (LIBXS_GEMM_FLAG_BETA_0) : 0), LIBXS_GEMM_PREFETCH_NONE);
    unsigned int i = LIBXS_MOD2(
      libxs_crc32(LIBXS_HASH_SEED, desc, LIBXS_MIN(size_desc, size)),
      LIBXS_CAPACITY_REGISTRY);
    libxs_code_pointer* dst_entry = registry + i;
#if !defined(NDEBUG)
    libxs_code_pointer code = { 0 }; code.xgemm = xgemm;
    LIBXS_ASSERT(NULL != code.ptr_const && NULL != registry);
    LIBXS_ASSERT(0 == (LIBXS_CODE_STATIC & code.uval));
#endif
    if (NULL != dst_entry->ptr_const) { /* collision */
      const unsigned int i0 = i;
      do { /* continue to linearly search for an available slot */
        i = LIBXS_MOD2(i + 1, LIBXS_CAPACITY_REGISTRY);
        if (NULL == registry[i].ptr_const) break;
      } while (i != i0);
      /* out of capacity (no registry slot available) */
      LIBXS_ASSERT(NULL == registry[i].ptr_const || i == i0);
      if (NULL == registry[i].ptr_const) { /* registry not exhausted */
        internal_update_mmstatistic(desc, 0, 1/*collision*/, 0, 0);
#if defined(LIBXS_HASH_COLLISION) /* mark entry as a collision */
        dst_entry->uval |= LIBXS_HASH_COLLISION;
#endif
        dst_entry = registry + i; /* update destination */
      }
    }
    if (NULL == dst_entry->ptr_const) { /* registry not exhausted */
      internal_registry_keys[i].entry.kind = LIBXS_KERNEL_KIND_MATMUL;
      LIBXS_ASSIGN127(&internal_registry_keys[i].entry.gemm.desc, desc);
      dst_entry->xgemm = xgemm;
      /* mark current entry as static code (non-JIT) */
      dst_entry->uval |= LIBXS_CODE_STATIC;
    }
    internal_update_mmstatistic(desc, 1/*try*/, 0, 0, 0);
  }
}
#endif


LIBXS_API_INTERN LIBXS_ATTRIBUTE_NO_TRACE void internal_dump(FILE* ostream, int urgent);
LIBXS_API_INTERN void internal_dump(FILE* ostream, int urgent)
{
  char *const env_dump_build = getenv("LIBXS_DUMP_BUILD");
  char *const env_dump_files = (NULL != getenv("LIBXS_DUMP_FILES")
    ? getenv("LIBXS_DUMP_FILES")
    : getenv("LIBXS_DUMP_FILE"));
  LIBXS_ASSERT_MSG(INTERNAL_SINGLETON(internal_singleton_handle), "Invalid handle");
  /* determine whether this instance is unique or not */
  if (NULL != env_dump_files && '\0' != *env_dump_files && 0 == urgent) { /* dump per-node info */
    const char* filename = strtok(env_dump_files, LIBXS_MAIN_DELIMS);
    char buffer[1024] = "";
    for (; NULL != filename; filename = strtok(NULL, LIBXS_MAIN_DELIMS)) {
      FILE* file = fopen(filename, "r");
      if (NULL != file) buffer[0] = '\0';
      else { /* parse keywords */
        const int seconds = atoi(filename);
        if (0 == seconds) {
          const char *const pid = strstr(filename, "PID");
          if (NULL != pid) { /* PID-keyword is present */
            int n = (int)(pid - filename);
            n = LIBXS_SNPRINTF(buffer, sizeof(buffer), "%.*s%u%s", n, filename, libxs_get_pid(), filename + n + 3);
            if (0 < n && (int)sizeof(buffer) > n) {
              file = fopen(buffer, "r");
              filename = buffer;
            }
          }
        }
        else {
          fprintf(stderr, "LIBXS INFO: PID=%u\n", libxs_get_pid());
          if (0 < seconds) {
#if defined(_WIN32)
            Sleep((DWORD)(1000 * seconds));
#else
            LIBXS_EXPECT(EXIT_SUCCESS == sleep(seconds));
#endif
          }
          else for (;;) LIBXS_SYNC_YIELD;
        }
      }
      if (NULL != file) {
        int c = fgetc(file);
        fprintf(ostream, "\n\nLIBXS_DUMP_FILE: %s\n", filename);
        /* coverity[tainted_data] */
        while (EOF != c) {
          fputc(c, stdout);
          c = fgetc(file);
        }
        fputc('\n', stdout);
        fclose(file);
      }
    }
  }
  if  (NULL != internal_build_state /* dump build state */
    && NULL != env_dump_build && '\0' != *env_dump_build)
  {
    const int dump_build = atoi(env_dump_build);
    if (0 == urgent ? (0 < dump_build) : (0 > dump_build)) {
      fprintf(ostream, "\n\nBUILD_DATE=%i\n", LIBXS_CONFIG_BUILD_DATE);
      fprintf(ostream, "%s\n", internal_build_state);
    }
  }
}


LIBXS_API double libxs_timer_duration_rtc(libxs_timer_tickint tick0, libxs_timer_tickint tick1)
{
  const libxs_timer_tickint delta = LIBXS_DELTA(tick0, tick1);
#if defined(_WIN32)
  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  return LIBXS_TIMER_DURATION_IDIV(delta, (libxs_timer_tickint)frequency.QuadPart);
#elif defined(CLOCK_MONOTONIC)
# if defined(__APPLE__) && 0
  mach_timebase_info_data_t frequency;
  mach_timebase_info(&frequency);
  return LIBXS_TIMER_DURATION_IDIV(delta * frequency.numer, 1000000000ULL * frequency.denom);
# else
  return LIBXS_TIMER_DURATION_IDIV(delta, 1000000000ULL);
# endif
#else
  return LIBXS_TIMER_DURATION_IDIV(delta, 1000000ULL);
#endif
}


LIBXS_API libxs_timer_tickint libxs_timer_tick_rtc(void)
{
#if defined(_WIN32)
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return (libxs_timer_tickint)t.QuadPart;
#elif defined(CLOCK_MONOTONIC)
# if defined(__APPLE__) && 0
  return mach_absolute_time();
# else
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return 1000000000ULL * t.tv_sec + t.tv_nsec;
# endif
#else
  struct timeval t;
  gettimeofday(&t, 0);
  return 1000000ULL * t.tv_sec + t.tv_usec;
#endif
}


LIBXS_API LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
libxs_timer_tickint libxs_timer_tick_tsc(void)
{
  libxs_timer_tickint result;
#if defined(LIBXS_TIMER_RDTSC)
  LIBXS_TIMER_RDTSC(result);
#else
  result = libxs_timer_tick_rtc();
#endif
  return result;
}


LIBXS_API_INTERN void internal_finalize(void);
LIBXS_API_INTERN void internal_finalize(void)
{
  libxs_finalize();
  /* release scratch memory pool */
  if (EXIT_SUCCESS != atexit(internal_release_scratch) && 0 != libxs_verbosity) {
    fprintf(stderr, "LIBXS ERROR: failed to perform final cleanup!\n");
  }
#if (0 != LIBXS_SYNC)
  /* determine whether this instance is unique or not */
  if (INTERNAL_SINGLETON(internal_singleton_handle)) {
    internal_dump(stdout, 0/*urgent*/);
    /* cleanup singleton */
# if defined(_WIN32)
    ReleaseMutex(internal_singleton_handle);
    CloseHandle(internal_singleton_handle);
# else
    unlink(internal_singleton_fname);
    close(internal_singleton_handle);
# endif
  }
#endif
  if (0 != libxs_verbosity) LIBXS_STDIO_RELEASE(); /* synchronize I/O */
#if (0 != LIBXS_SYNC)
# if !defined(_WIN32)
  if (0 < libxs_stdio_handle) {
    LIBXS_ASSERT('\0' != *internal_stdio_fname);
    unlink(internal_stdio_fname);
    close(libxs_stdio_handle - 1);
  }
# endif
  { /* release locks */
# if (1 < INTERNAL_REGLOCK_MAXN)
    int i; for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_DESTROY(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
    LIBXS_LOCK_DESTROY(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
    LIBXS_LOCK_DESTROY(LIBXS_LOCK, &libxs_lock_global);
  }
#endif
}


LIBXS_API_INTERN void internal_libxs_signal(int /*signum*/);
LIBXS_API_INTERN void internal_libxs_signal(int signum) {
  int n = (int)(sizeof(internal_sigentries) / sizeof(*internal_sigentries)), i = 0;
  for (; i < n; ++i) {
    if (signum == internal_sigentries[i].signum) {
      if (0 == libxs_get_tid()) {
        libxs_verbosity = LIBXS_MAX(LIBXS_VERBOSITY_HIGH + 1, libxs_verbosity);
        internal_finalize();
        signal(signum,
          (NULL == internal_sigentries[i].signal || SIG_ERR == internal_sigentries[i].signal)
            ? SIG_DFL : internal_sigentries[i].signal); /* restore */
        raise(signum);
      }
    }
  }
}


LIBXS_API void __wrap__gfortran_runtime_warning_at(const char* /*where*/, const char* /*message*/, ...);
LIBXS_API void __wrap__gfortran_runtime_warning_at(const char* where, const char* message, ...)
{ /* link application with "-Wl,--wrap=_gfortran_runtime_warning_at" */
  LIBXS_UNUSED(message);
  LIBXS_UNUSED(where);
}


#if defined(LIBXS_INTERCEPT_DYNAMIC)
LIBXS_API LIBXS_ATTRIBUTE_WEAK void _gfortran_stop_string(const char* /*message*/, int /*len*/, int /*quiet*/);
LIBXS_API LIBXS_ATTRIBUTE_WEAK void _gfortran_stop_string(const char* message, int len, int quiet)
{ /* STOP termination handler for GNU Fortran runtime */
  static int once = 0;
  if (1 == LIBXS_ATOMIC_ADD_FETCH(&once, 1, LIBXS_ATOMIC_SEQ_CST)) {
    union { const void* dlsym; void (*ptr)(const char*, int, int); } stop;
    dlerror(); /* clear an eventual error status */
    stop.dlsym = dlsym(LIBXS_RTLD_NEXT, "_gfortran_stop_string");
    if (NULL != stop.dlsym) {
      stop.ptr(message, len, quiet);
    }
    else LIBXS_EXIT_SUCCESS(); /* statically linked runtime */
  }
}

LIBXS_API LIBXS_ATTRIBUTE_WEAK void for_stop_core(const char* /*message*/, int /*len*/);
LIBXS_API LIBXS_ATTRIBUTE_WEAK void for_stop_core(const char* message, int len)
{ /* STOP termination handler for Intel Fortran runtime */
  static int once = 0;
  if (1 == LIBXS_ATOMIC_ADD_FETCH(&once, 1, LIBXS_ATOMIC_SEQ_CST)) {
    union { const void* dlsym; void (*ptr)(const char*, int); } stop;
    dlerror(); /* clear an eventual error status */
    stop.dlsym = dlsym(LIBXS_RTLD_NEXT, "for_stop_core");
    if (NULL != stop.dlsym) {
      stop.ptr(message, len);
    }
    else LIBXS_EXIT_SUCCESS(); /* statically linked runtime */
  }
}

LIBXS_API LIBXS_ATTRIBUTE_WEAK void for_stop_core_quiet(void);
LIBXS_API LIBXS_ATTRIBUTE_WEAK void for_stop_core_quiet(void)
{ /* STOP termination handler for Intel Fortran runtime */
  static int once = 0;
  if (1 == LIBXS_ATOMIC_ADD_FETCH(&once, 1, LIBXS_ATOMIC_SEQ_CST)) {
    union { const void* dlsym; void (*ptr)(void); } stop;
    dlerror(); /* clear an eventual error status */
    stop.dlsym = dlsym(LIBXS_RTLD_NEXT, "for_stop_core_quiet");
    if (NULL != stop.dlsym) {
      stop.ptr();
    }
    else LIBXS_EXIT_SUCCESS(); /* statically linked runtime */
  }
}
#endif


LIBXS_API_INTERN size_t internal_strlen(const char* /*cstr*/, size_t /*maxlen*/);
LIBXS_API_INTERN size_t internal_strlen(const char* cstr, size_t maxlen)
{
  size_t result = 0;
  if (NULL != cstr) {
    while ('\0' != cstr[result] && result < maxlen) ++result;
  }
  return result;
}


LIBXS_API_INTERN size_t internal_parse_nbytes(const char* /*nbytes*/, size_t /*ndefault*/, int* /*valid*/);
LIBXS_API_INTERN size_t internal_parse_nbytes(const char* nbytes, size_t ndefault, int* valid)
{
  size_t result = ndefault;
  if (NULL != nbytes && '\0' != *nbytes) {
    size_t u = internal_strlen(nbytes, 32) - 1;
    const char units[] = "kmgKMG", *const unit = strchr(units, nbytes[u]);
    char* end = NULL;
    /* take parsed value with increased type-width */
    const long long int ibytes = strtol(nbytes, &end, 10);
    if (NULL != end && ( /* no obvious error */
      /* must match allowed set of units */
      (NULL != unit && *unit == *end) ||
      /* value is given without unit */
      (NULL == unit && '\0' == *end)))
    {
      result = (size_t)ibytes;
      if ((size_t)LIBXS_UNLIMITED != result) {
        u = (NULL != unit ? ((unit - units) % 3) : 3);
        if (u < 3) result <<= (u + 1) * 10;
      }
      if (NULL != valid) *valid = 1;
    }
    else if (NULL != valid) *valid = 0;
  }
  else if (NULL != valid) {
    *valid = 0;
  }
  return result;
}


LIBXS_API_INTERN LIBXS_ATTRIBUTE_NO_TRACE void internal_init(void);
LIBXS_API_INTERN void internal_init(void)
{
  int i;
#if (0 != LIBXS_SYNC) /* setup the locks in a thread-safe fashion */
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &libxs_lock_global);
# if (1 < INTERNAL_REGLOCK_MAXN)
  for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_ACQUIRE(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
  LIBXS_LOCK_ACQUIRE(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
#endif
  if (NULL == internal_registry) { /* double-check after acquiring the lock(s) */
#if defined(LIBXS_INTERCEPT_DYNAMIC) && defined(LIBXS_AUTOPIN)
    /* clear error status (dummy condition: it does not matter if MPI_Init or MPI_Abort) */
    const char *const dlsymname = (NULL == dlerror() ? "MPI_Init" : "MPI_Abort");
    const void *const dlsymbol = dlsym(LIBXS_RTLD_NEXT, dlsymname);
    const void *const dlmpi = (NULL == dlerror() ? dlsymbol : NULL);
#endif
    const char *const env_verbose = getenv("LIBXS_VERBOSE");
    void* new_registry = NULL, * new_keys = NULL;
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
# if defined(LIBXS_NTHREADS_USE)
    void* new_cache = NULL;
# endif
    const char *const env_cache = getenv("LIBXS_CACHE");
    if (NULL != env_cache && '\0' != *env_cache) {
      const int cache_size = atoi(env_cache), cache_size2 = (int)LIBXS_UP2POT(cache_size);
      internal_cache_size = LIBXS_MIN(cache_size2, LIBXS_CACHE_MAXSIZE);
    }
    else {
      internal_cache_size = LIBXS_CACHE_MAXSIZE;
    }
#endif
    /* setup verbosity as early as possible since below code may rely on verbose output */
    if (NULL != env_verbose) {
      libxs_verbosity = ('\0' != *env_verbose ? atoi(env_verbose) : 1);
    }
#if defined(_DEBUG)
    else {
      libxs_verbosity = INT_MAX; /* quiet -> verbose */
    }
#endif
#if (0 == LIBXS_JIT)
    if (2 > libxs_ninit && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity)) {
      fprintf(stderr, "LIBXS: JIT-code generation was disabled at compile-time.\n");
    }
#endif
#if defined(LIBXS_AUTOPIN)
# if defined(LIBXS_INTERCEPT_DYNAMIC)
    /* MPI: unwanted affinity can slow-down unrelated jobs (over-subscription), e.g., CP2K regtests */
    if (NULL == dlmpi)
# endif
    { /* setup some viable affinity if nothing else is present */
      const char *const gomp_cpu_affinity = getenv("GOMP_CPU_AFFINITY");
      const char *const kmp_affinity = getenv("KMP_AFFINITY");
      const char *const omp_proc_bind = getenv("OMP_PROC_BIND");
      if  ((NULL == gomp_cpu_affinity || 0 == *gomp_cpu_affinity)
        && (NULL == kmp_affinity || 0 == *kmp_affinity)
        && (NULL == omp_proc_bind || 0 == *omp_proc_bind))
      {
        static char affinity[] = "OMP_PROC_BIND=TRUE";
        LIBXS_EXPECT(EXIT_SUCCESS == LIBXS_PUTENV(affinity));
        if (LIBXS_VERBOSITY_HIGH < libxs_verbosity || 0 > libxs_verbosity) { /* library code is expected to be mute */
          fprintf(stderr, "LIBXS: prepared to pin threads.\n");
        }
      }
    }
# if defined(LIBXS_INTERCEPT_DYNAMIC) && 1
    else if (NULL == getenv("I_MPI_SHM_HEAP")) {
      static char shmheap[] = "I_MPI_SHM_HEAP=1";
      LIBXS_EXPECT(EXIT_SUCCESS == LIBXS_PUTENV(shmheap));
    }
# endif
#endif
#if !defined(_WIN32) && 0
    umask(S_IRUSR | S_IWUSR); /* setup default/secure file mask */
#endif
    { const char *const env = getenv("LIBXS_SYNC");
      libxs_nosync = (NULL == env || 0 == *env) ? 0/*default*/ : atoi(env);
    }
    /* clear internal counters/statistic */
    for (i = 0; i < 4/*sml/med/big/xxx*/; ++i) {
      LIBXS_MEMZERO127(&internal_statistic[0/*DP*/][i]);
      LIBXS_MEMZERO127(&internal_statistic[1/*SP*/][i]);
    }
    internal_statistic_mnk = LIBXS_MAX_DIM;
    internal_statistic_sml = 13;
    internal_statistic_med = 23;
    LIBXS_ASSERT(LIBXS_ISPOT(LIBXS_CAPACITY_REGISTRY));
    libxs_hash_init(libxs_cpuid(NULL)); /* used by debug memory allocation (checksum) */
    libxs_memory_init(libxs_cpuid(NULL));
    if (
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      (EXIT_SUCCESS == libxs_xmalloc(&new_cache, /* if internal_cache_size is zero, allocation must still happen (later control-flow too expensive) */
        sizeof(internal_cache_type) * (LIBXS_NTHREADS_MAX), LIBXS_CACHELINE/*alignment*/,
        LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_cache) &&
#endif
      (EXIT_SUCCESS == libxs_xmalloc(&new_keys, (LIBXS_CAPACITY_REGISTRY) * sizeof(internal_regkey_type), 0/*auto-align*/,
        LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_keys) &&
      (EXIT_SUCCESS == libxs_xmalloc(&new_registry, (LIBXS_CAPACITY_REGISTRY) * sizeof(libxs_code_pointer), 0/*auto-align*/,
        LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_registry))
    {
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      LIBXS_ASSERT(NULL != new_cache); /* SA: suppress false positive */
      memset(new_cache, 0, (LIBXS_NTHREADS_MAX) * sizeof(internal_cache_type));
#endif
      libxs_xcopy_init(libxs_cpuid(NULL));
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) ((libxs_code_pointer*)new_registry)[i].ptr = NULL;
      /*LIBXS_ASSERT(NULL == internal_registry && NULL == internal_registry_keys);*/
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      LIBXS_ASSERT(NULL == internal_cache_buffer);
      internal_cache_buffer = (internal_cache_type*)new_cache;
#endif
      internal_registry_keys = (internal_regkey_type*)new_keys; /* prior to registering static kernels */
#if defined(LIBXS_BUILD) && !defined(LIBXS_DEFAULT_CONFIG)
#     include <libxs_dispatch.h>
#endif
      libxs_gemm_init();
#if defined(LIBXS_TRACE)
      { int filter_threadid = 0/*only main-thread*/, filter_mindepth = 0, filter_maxnsyms = 0;
        const int init_code = libxs_trace_init(filter_threadid, filter_mindepth, filter_maxnsyms);
        if (EXIT_SUCCESS != init_code && 0 != libxs_verbosity) { /* library code is expected to be mute */
          fprintf(stderr, "LIBXS ERROR: failed to initialize TRACE (error #%i)!\n", init_code);
        }
      }
#endif
      { /* commit the registry buffer and enable global visibility */
        void *const pv_registry = &internal_registry;
        LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)((void**)pv_registry, (void*)new_registry, LIBXS_ATOMIC_SEQ_CST);
      }
    }
    else {
      if (0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS ERROR: failed to allocate internal buffers!\n");
      }
      libxs_xfree(new_registry, 0/*no check*/);
      libxs_xfree(new_keys, 0/*no check*/);
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      libxs_xfree(new_cache, 0/*no check*/);
#endif
    }
  }
#if (0 != LIBXS_SYNC) /* release locks */
# if (1 < INTERNAL_REGLOCK_MAXN)
  for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
  LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxs_lock_global);
#endif
}


LIBXS_API_CTOR void libxs_init(void)
{
  if (NULL == (const void*)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&internal_registry, LIBXS_ATOMIC_SEQ_CST)) {
    static unsigned int counter = 0, gid = 0;
    const unsigned int tid = LIBXS_ATOMIC_ADD_FETCH(&counter, 1, LIBXS_ATOMIC_SEQ_CST);
    LIBXS_ASSERT(0 < tid);
    /* libxs_ninit (1: initialization started, 2: library initialized, higher: to invalidate code-TLS) */
    if (1 == tid) {
      libxs_timer_tickint s0 = libxs_timer_tick_rtc(); /* warm-up */
      libxs_timer_tickint t0 = libxs_timer_tick_tsc(); /* warm-up */
      s0 = libxs_timer_tick_rtc(); t0 = libxs_timer_tick_tsc(); /* start timing */
      { const unsigned int ninit = LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_SEQ_CST);
        LIBXS_UNUSED_NDEBUG(ninit);
        assert(1 == ninit); /* !LIBXS_ASSERT */
      }
      gid = tid; /* protect initialization */
      LIBXS_UNUSED_NDEBUG(gid);
#if (0 != LIBXS_SYNC)
      { /* construct and initialize locks */
# if defined(LIBXS_REGLOCK_TRY)
        const char *const env_trylock = getenv("LIBXS_TRYLOCK");
# endif
        LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK) attr_global = { 0 };
# if (1 < INTERNAL_REGLOCK_MAXN)
        int i;
        LIBXS_LOCK_ATTR_TYPE(LIBXS_REGLOCK) attr;
        LIBXS_LOCK_ATTR_INIT(LIBXS_REGLOCK, &attr);
# elif defined(LIBXS_UNIFY_LOCKS)
        internal_reglock_ptr = &libxs_lock_global;
# else
        static LIBXS_LOCK_TYPE(LIBXS_REGLOCK) internal_reglock;
        internal_reglock_ptr = &internal_reglock;
        LIBXS_LOCK_ATTR_TYPE(LIBXS_REGLOCK) attr;
        LIBXS_LOCK_ATTR_INIT(LIBXS_REGLOCK, &attr);
        LIBXS_LOCK_INIT(LIBXS_REGLOCK, internal_reglock_ptr, &attr);
        LIBXS_LOCK_ATTR_DESTROY(LIBXS_REGLOCK, &attr);
# endif
        LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &attr_global);
        LIBXS_LOCK_INIT(LIBXS_LOCK, &libxs_lock_global, &attr_global);
        LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &attr_global);
        /* control number of locks needed; LIBXS_TRYLOCK implies only 1 lock */
# if defined(LIBXS_REGLOCK_TRY)
        if (NULL == env_trylock || 0 == *env_trylock)
# endif
        { /* no LIBXS_TRYLOCK */
# if defined(LIBXS_VTUNE)
          internal_reglock_count = 1; /* avoid duplicated kernels */
# elif (1 < INTERNAL_REGLOCK_MAXN)
          const char *const env_nlocks = getenv("LIBXS_NLOCKS");
          const int reglock_count = (NULL == env_nlocks || 0 == *env_nlocks || 1 > atoi(env_nlocks))
            ? (INTERNAL_REGLOCK_MAXN) : LIBXS_MIN(atoi(env_nlocks), INTERNAL_REGLOCK_MAXN);
          internal_reglock_count = (int)LIBXS_LO2POT(reglock_count);
# else
          internal_reglock_count = 0;
# endif
        }
# if defined(LIBXS_REGLOCK_TRY)
        else { /* LIBXS_TRYLOCK environment variable specified */
          internal_reglock_count = (0 != atoi(env_trylock) ? 1
#   if (1 < INTERNAL_REGLOCK_MAXN)
            : INTERNAL_REGLOCK_MAXN);
#   else
            : 0);
#   endif
        }
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
        LIBXS_ASSERT(1 <= internal_reglock_count);
        for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_INIT(LIBXS_REGLOCK, &internal_reglock[i].state, &attr);
        LIBXS_LOCK_ATTR_DESTROY(LIBXS_REGLOCK, &attr);
# endif
      }
#endif
      { /* determine whether this instance is unique or not */
#if (0 != LIBXS_SYNC)
# if defined(_WIN32)
        internal_singleton_handle = CreateMutex(NULL, TRUE, "GlobalLIBXS");
# else
        const unsigned int userid = (unsigned int)getuid();
        const int result_sgltn = LIBXS_SNPRINTF(internal_singleton_fname, sizeof(internal_singleton_fname), "/tmp/.libxs.%u",
          /*rely on user id to avoid permission issues in case of left-over files*/userid);
        const int result_stdio = LIBXS_SNPRINTF(internal_stdio_fname, sizeof(internal_stdio_fname), "/tmp/.libxs.stdio.%u",
          /*rely on user id to avoid permission issues in case of left-over files*/userid);
        struct flock singleton_flock;
        int file_handle;
        singleton_flock.l_start = 0;
        singleton_flock.l_len = 0; /* entire file */
        singleton_flock.l_type = F_WRLCK; /* exclusive across PIDs */
        singleton_flock.l_whence = SEEK_SET;
        file_handle = ((0 < result_sgltn && (int)sizeof(internal_singleton_fname) > result_sgltn)
          ? open(internal_singleton_fname, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR) : -1);
        internal_singleton_handle = fcntl(file_handle, F_SETLK, &singleton_flock);
        if (0 <= file_handle && 0 > internal_singleton_handle) close(file_handle);
        libxs_stdio_handle = ((0 < result_stdio && (int)sizeof(internal_stdio_fname) > result_stdio)
          ? (open(internal_stdio_fname, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR) + 1) : 0);
# endif  /* coverity[leaked_handle] */
#endif
      }
      { /* calibrate timer */
        int result_atexit = EXIT_SUCCESS;
        libxs_timer_tickint s1, t1;
        internal_init(); /* must be first to initialize verbosity, etc. */
        if (INTERNAL_SINGLETON(internal_singleton_handle)) { /* after internal_init */
          internal_dump(stdout, 1/*urgent*/);
        }
        s1 = libxs_timer_tick_rtc(); t1 = libxs_timer_tick_tsc(); /* mid-timing */
#if defined(NDEBUG)
        libxs_cpuid(&internal_cpuid_info);
        if (0 != internal_cpuid_info.constant_tsc && t0 < t1) {
          libxs_timer_scale = libxs_timer_duration_rtc(s0, s1) / (t1 - t0);
        }
#endif
        internal_sigentries[0].signal = signal(SIGABRT, internal_libxs_signal);
        internal_sigentries[0].signum = SIGABRT;
        internal_sigentries[1].signal = signal(SIGSEGV, internal_libxs_signal);
        internal_sigentries[1].signum = SIGSEGV;
        result_atexit = atexit(internal_finalize);
        s1 = libxs_timer_tick_rtc(); t1 = libxs_timer_tick_tsc(); /* final timing */
        /* set timer-scale and determine start of the "uptime" (shown at termination) */
        if (t0 < t1 && 0.0 < libxs_timer_scale) {
          const double scale = libxs_timer_duration_rtc(s0, s1) / (t1 - t0);
          const double diff = LIBXS_DELTA(libxs_timer_scale, scale) / scale;
          if (5E-4 > diff) {
            libxs_timer_scale = scale;
            internal_timer_start = t0;
          }
          else {
            libxs_timer_scale = 0;
            internal_timer_start = s0;
#if defined(_DEBUG)
            libxs_se = 1;
#endif
          }
        }
        else {
          internal_timer_start = s0;
          libxs_timer_scale = 0;
        }
        if (0 != libxs_verbosity) { /* library code is expected to be mute */
          if (EXIT_SUCCESS != result_atexit) {
            fprintf(stderr, "LIBXS ERROR: failed to register termination procedure!\n");
          }
#if defined(NDEBUG)
          if (0 == libxs_timer_scale && 0 == internal_cpuid_info.constant_tsc
            && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity))
          {
            fprintf(stderr, "LIBXS WARNING: timer is maybe not cycle-accurate!\n");
          }
#endif
        }
      }
      LIBXS_EXPECT(0 < LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_SEQ_CST));
    }
    else /*if (gid != tid)*/ { /* avoid recursion */
      LIBXS_ASSERT(gid != tid);
      while (2 > LIBXS_ATOMIC_LOAD(&libxs_ninit, LIBXS_ATOMIC_SEQ_CST)) LIBXS_SYNC_YIELD;
      internal_init();
    }
#if defined(LIBXS_PERF)
    libxs_perf_init(libxs_timer_tick_rtc);
#endif
  }
}


LIBXS_API LIBXS_ATTRIBUTE_NO_TRACE void libxs_finalize(void);
LIBXS_API_DTOR void libxs_finalize(void)
{
  libxs_code_pointer* registry = (libxs_code_pointer*)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(
    &internal_registry, LIBXS_ATOMIC_SEQ_CST);
  if (NULL != registry) {
    int i;
#if (0 != LIBXS_SYNC)
    if (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK) == LIBXS_LOCK_TRYLOCK(LIBXS_LOCK, &libxs_lock_global)) {
# if (1 < INTERNAL_REGLOCK_MAXN)
    { /* acquire locks and thereby shortcut lazy initialization later on */
      int ntry = 0, n;
      do {
        for (i = 0, n = 0; i < internal_reglock_count; ++i) {
          if (LIBXS_LOCK_ACQUIRED(LIBXS_REGLOCK) == LIBXS_LOCK_TRYLOCK(LIBXS_REGLOCK, &internal_reglock[i].state)) ++n;
        }
        ntry += (0 == n ? 1 : 0);
      } while (n < internal_reglock_count && ntry < LIBXS_CLEANUP_NTRY);
    }
# elif !defined(LIBXS_UNIFY_LOCKS)
    LIBXS_LOCK_ACQUIRE(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
#endif
    registry = (libxs_code_pointer*)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(
      &internal_registry, LIBXS_ATOMIC_SEQ_CST);
    if (NULL != registry) {
      internal_regkey_type *const registry_keys = internal_registry_keys;
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      internal_cache_type *const cache_buffer = internal_cache_buffer;
#endif
      unsigned int rest = 0, errors = 0;
#if defined(LIBXS_TRACE)
      i = libxs_trace_finalize();
      if (EXIT_SUCCESS != i && 0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS ERROR: failed to finalize trace (error #%i)!\n", i);
      }
#endif
#if defined(LIBXS_PERF)
      libxs_perf_finalize();
#endif
      libxs_xcopy_finalize();
      libxs_gemm_finalize();
      /* coverity[check_return] */
      LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_SEQ_CST); /* invalidate code cache (TLS) */
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      internal_cache_buffer = NULL;
#endif
      internal_registry_keys = NULL; /* make registry keys unavailable */
      LIBXS_ATOMIC(LIBXS_ATOMIC_STORE_ZERO, LIBXS_BITS)((uintptr_t*)&internal_registry, LIBXS_ATOMIC_SEQ_CST);
      internal_registry_nbytes = 0; internal_registry_nleaks = 0;
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
        /*const*/ libxs_code_pointer code = registry[i];
        if (NULL != code.ptr_const) {
          const libxs_descriptor_kind kind = LIBXS_DESCRIPTOR_KIND(registry_keys[i].entry.kind);
          /* check if the registered entity is a GEMM kernel */
          switch (kind) {
            case LIBXS_KERNEL_KIND_MATMUL: {
              const libxs_gemm_descriptor *const desc = &registry_keys[i].entry.gemm.desc;
              if (1 < desc->m && 1 < desc->n) {
                const unsigned int njit = (0 == (LIBXS_CODE_STATIC & code.uval) ? 1 : 0);
                const unsigned int nsta = (0 != (LIBXS_CODE_STATIC & code.uval) ? 1 : 0);
                /* count whether kernel is static or JIT-code */
                internal_update_mmstatistic(desc, 0, 0, njit, nsta);
                ++rest;
              }
              else {
                ++internal_statistic_num_gemv;
              }
            } break;
            case LIBXS_KERNEL_KIND_MELTW: {
              ++internal_statistic_num_meltw;
            } break;
            case LIBXS_KERNEL_KIND_USER: {
              ++internal_statistic_num_user;
            } break;
            default: if (LIBXS_KERNEL_UNREGISTERED <= kind) {
              ++errors;
            }
            else {
              ++rest;
            }
          }
          if (0 != libxs_verbosity) { /* library code is expected to be mute */
            if (0 != errors) {
              fprintf(stderr, "LIBXS ERROR: code registry is corrupted!\n");
            }
            if (LIBXS_CAPACITY_REGISTRY == (rest + errors + internal_statistic_num_gemv +
              internal_statistic_num_user + internal_statistic_num_meltw))
            {
              fprintf(stderr, "LIBXS WARNING: code registry was exhausted!\n");
            }
          }
          if (0 == (LIBXS_CODE_STATIC & code.uval)) { /* check for allocated/generated JIT-code */
            void* buffer = NULL;
            size_t size = 0;
#if defined(LIBXS_HASH_COLLISION)
            code.uval &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
#endif
            if (EXIT_SUCCESS == libxs_get_malloc_xinfo(code.ptr_const, &size, NULL/*flags*/, &buffer)) {
              if (LIBXS_KERNEL_KIND_USER == kind && 0 > libxs_verbosity) { /* dump user-data just like JIT'ted code */
                char name[16] = "";
                int nchar;
#if defined(LIBXS_REGUSER_HASH)
                const size_t descsize = LIBXS_DESCRIPTOR_ISBIG(registry_keys[i].entry.kind)
                  ? LIBXS_DESCRIPTOR_MAXSIZE : LIBXS_DESCRIPTOR_SIGSIZE;
                const unsigned int id = libxs_crc32(LIBXS_HASH_SEED, registry_keys[i].entry.user.desc,
                  descsize - sizeof(libxs_descriptor_kind));
                LIBXS_ASSERT(descsize > sizeof(libxs_descriptor_kind));
#else
                const unsigned int id = internal_statistic_num_user;
#endif
                nchar = LIBXS_SNPRINTF(name, sizeof(name), "%010u.user", id);
                if (0 < nchar && (int)sizeof(name) > nchar) {
                  LIBXS_EXPECT(EXIT_SUCCESS == libxs_dump("LIBXS-USER-DUMP",
                    name, code.ptr_const, size, 0/*unique*/, 0/*overwrite*/));
                }
              }
#if !defined(NDEBUG)
              registry[i].ptr = NULL;
#endif
              libxs_xfree(code.ptr_const, 0/*no check*/);
              /* round-up size (it is fine to assume 4 KB pages since it is likely more accurate than not rounding up) */
              internal_registry_nbytes += LIBXS_UP2(size + (((const char*)code.ptr_const) - (char*)buffer), LIBXS_PAGE_MINSIZE);
            }
            else ++internal_registry_nleaks;
          }
        }
      }
      /* release buffers (registry, keys, cache) */
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      libxs_xfree(cache_buffer, 0/*no check*/);
#endif
      libxs_xfree(registry_keys, 0/*no check*/);
      libxs_xfree(registry, 0/*no check*/);
    }
#if (0 != LIBXS_SYNC) /* LIBXS_LOCK_RELEASE, but no LIBXS_LOCK_DESTROY */
# if (1 < INTERNAL_REGLOCK_MAXN)
    for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
    LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, internal_reglock_ptr);
# endif
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxs_lock_global); }
#endif
  }
}


LIBXS_API int libxs_get_verbosity(void)
{
  /*LIBXS_INIT*/
  return libxs_verbosity;
}


LIBXS_API void libxs_set_verbosity(int level)
{
  /*LIBXS_INIT*/
  LIBXS_ATOMIC_STORE(&libxs_verbosity, level, LIBXS_ATOMIC_RELAXED);
}


LIBXS_API int libxs_typesize(libxs_datatype datatype)
{
  int result = 0;
  switch (datatype) {
    case LIBXS_DATATYPE_F64: result = 8; break;
    case LIBXS_DATATYPE_F32: result = 4; break;
    case LIBXS_DATATYPE_I64: result = 4; break;
    case LIBXS_DATATYPE_I32: result = 4; break;
    case LIBXS_DATATYPE_U32: result = 4; break;
    case LIBXS_DATATYPE_I16: result = 2; break;
    case LIBXS_DATATYPE_U16: result = 2; break;
    case LIBXS_DATATYPE_I8:  result = 1; break;
    default: {
      static int error_once = 0;
      LIBXS_ASSERT_MSG(0, "unsupported data type");
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: unsupported data type!\n");
      }
    }
  }
  return result;
}


LIBXS_API const char* libxs_get_typename(libxs_datatype datatype)
{
  switch (datatype) {
    case LIBXS_DATATYPE_F64:  return "f64";
    case LIBXS_DATATYPE_F32:  return "f32";
    case LIBXS_DATATYPE_I64:  return "i64";
    case LIBXS_DATATYPE_I32:  return "i32";
    case LIBXS_DATATYPE_U32:  return "u32";
    case LIBXS_DATATYPE_I16:  return "i16";
    case LIBXS_DATATYPE_U16:  return "u16";
    case LIBXS_DATATYPE_I8:   return "i8";
    default: return "void";
  }
}


LIBXS_API int libxs_dvalue(libxs_datatype datatype, const void* value, double* dvalue)
{
  int result = EXIT_SUCCESS;
  if (NULL != value) {
    LIBXS_ASSERT(NULL != dvalue);
    switch (datatype) {
      case LIBXS_DATATYPE_F64: *dvalue =         (*(const double   *)value); break;
      case LIBXS_DATATYPE_F32: *dvalue = (double)(*(const float    *)value); break;
      case LIBXS_DATATYPE_I64: *dvalue = (double)(*(const long long*)value); break;
      case LIBXS_DATATYPE_I32: *dvalue = (double)(*(const int      *)value); break;
      case LIBXS_DATATYPE_I16: *dvalue = (double)(*(const short    *)value); break;
      case LIBXS_DATATYPE_I8:  *dvalue = (double)(*(const char     *)value); break;
      default: result = EXIT_FAILURE;
    }
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXS_API_INTERN int libxs_dump(const char* title, const char* name, const void* data, size_t size, int unique, int overwrite)
{
  int result;
  if (NULL != name && '\0' != *name && NULL != data && 0 != size) {
    FILE* data_file = ((0 != unique || 0 == overwrite) ? fopen(name, "rb") : NULL);
    int diff = 0, result_close;
    if (NULL == data_file) { /* file does not exist */
      data_file = fopen(name, "wb");
      if (NULL != data_file) { /* dump data into a file */
        result = ((size == fwrite(data, 1, size, data_file)) ? EXIT_SUCCESS : EXIT_FAILURE);
        result_close = fclose(data_file);
        if (EXIT_SUCCESS == result) result = result_close;
      }
      else result = EXIT_FAILURE;
    }
    else if (0 != unique) { /* check existing file */
      const char* check_a = (const char*)data;
      char check_b[4096];
      size_t rest = size;
      do {
        const size_t n = fread(check_b, 1, LIBXS_MIN(sizeof(check_b), rest), data_file);
        diff += memcmp(check_a, check_b, LIBXS_MIN(sizeof(check_b), n));
        check_a += n;
        rest -= n;
      } while (0 < rest && 0 == diff);
      result = fclose(data_file);
    }
    else {
      result = fclose(data_file);
    }
    if (EXIT_SUCCESS == result && NULL != title && '\0' != *title) {
      fprintf(stderr, "%s(ptr:file) %p : %s\n", title, data, name);
    }
    if (0 != diff) { /* overwrite existing dump and warn about erroneous condition */
      fprintf(stderr, "LIBXS DUMP: %s is not a unique filename!\n", name);
      data_file = (0 != overwrite ? fopen(name, "wb") : NULL);
      if (NULL != data_file) { /* dump data into a file */
        if (size != fwrite(data, 1, size, data_file)) result = EXIT_FAILURE;
        result_close = fclose(data_file);
        if (EXIT_SUCCESS == result) result = result_close;
      }
      if (EXIT_SUCCESS == result) result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INLINE void internal_pad_descriptor(libxs_descriptor* desc, signed char size)
{
  LIBXS_ASSERT(LIBXS_DESCRIPTOR_MAXSIZE < 128 && NULL != desc);
  LIBXS_ASSERT(LIBXS_DIFF_SIZE <= LIBXS_DESCRIPTOR_MAXSIZE);
  LIBXS_ASSERT(LIBXS_HASH_SIZE <= LIBXS_DIFF_SIZE);
  for (; size < LIBXS_DIFF_SIZE; ++size) desc->data[size] = 0;
}


LIBXS_API_INLINE libxs_code_pointer internal_find_code(libxs_descriptor* desc, size_t desc_size, size_t user_size)
{
  libxs_code_pointer flux_entry = { 0 };
  const int is_big_desc = LIBXS_DESCRIPTOR_ISBIG(desc->kind);
  const signed char size = (signed char)(sizeof(libxs_descriptor_kind) + desc_size);
  LIBXS_DIFF_DECL(LIBXS_DIFF_SIZE, xdesc);
#if !defined(NDEBUG) && (0 != LIBXS_JIT)
  int build = EXIT_SUCCESS;
#endif
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
# if defined(LIBXS_NTHREADS_USE)
  const unsigned int tid = libxs_get_tid();
  internal_cache_type *const cache = internal_cache_buffer + tid;
# else
  static LIBXS_TLS internal_cache_type internal_cache_buffer /*= { 0 }*/;
  internal_cache_type *const cache = &internal_cache_buffer;
# endif
  unsigned char cache_index;
  const unsigned int ninit = LIBXS_ATOMIC_LOAD(&libxs_ninit, LIBXS_ATOMIC_SEQ_CST);
  internal_pad_descriptor(desc, size);
  if (0 == is_big_desc) {
    LIBXS_DIFF_LOAD(LIBXS_DIFF_SIZE, xdesc, desc);
    LIBXS_DIFF_N(unsigned char, cache_index, LIBXS_DIFF(LIBXS_DIFF_SIZE), xdesc, cache->entry.keys,
      LIBXS_DIFF_SIZE, LIBXS_CACHE_STRIDE, cache->entry.hit, cache->entry.size);
  }
  else {
    cache_index = (unsigned char)libxs_diff_n(desc, cache->entry.keys,
      size, LIBXS_CACHE_STRIDE, cache->entry.hit, cache->entry.size);
  }
  if (ninit == cache->entry.id && cache_index < cache->entry.size) { /* valid hit */
    flux_entry = cache->entry.code[cache_index];
    cache->entry.hit = cache_index;
  }
  else
#else
  internal_pad_descriptor(desc, size);
#endif
  {
    unsigned int i, i0, mode = 0, diff = 1;
    unsigned int hash = LIBXS_CRC32(LIBXS_HASH_SIZE)(LIBXS_HASH_SEED, desc);
    i0 = i = LIBXS_MOD2(hash, LIBXS_CAPACITY_REGISTRY);
    LIBXS_ASSERT(&desc->kind == &desc->gemm.pad && desc->kind == desc->gemm.pad);
    LIBXS_ASSERT(NULL != internal_registry);
    do { /* use calculated location and check if the requested code is already JITted */
#if (1 < INTERNAL_REGLOCK_MAXN) || !LIBXS_LOCK_TYPE_ISRW(LIBXS_REGLOCK) /* read registered code */
# if 1 /* omitting an atomic load is safe but avoids race-detectors to highlight this location */
      uintptr_t *const fluxaddr = &internal_registry[i].uval;
      flux_entry.uval = LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(fluxaddr, LIBXS_ATOMIC_RELAXED);
# else
      flux_entry = internal_registry[i];
# endif
#else
      LIBXS_LOCK_ACQREAD(LIBXS_REGLOCK, internal_reglock_ptr);
      flux_entry = internal_registry[i]; /* read registered code */
      LIBXS_LOCK_RELREAD(LIBXS_REGLOCK, internal_reglock_ptr);
#endif
      if ((NULL != flux_entry.ptr_const || 1 == mode) && 2 > mode) { /* confirm entry */
        if (NULL != flux_entry.ptr_const) {
          if (0 == is_big_desc) {
#if !defined(LIBXS_CACHE_MAXSIZE) || (0 == (LIBXS_CACHE_MAXSIZE))
            LIBXS_DIFF_LOAD(LIBXS_DIFF_SIZE, xdesc, desc);
#endif
            diff = LIBXS_DIFF(LIBXS_DIFF_SIZE)(xdesc, internal_registry_keys + i, 0/*dummy*/);
          }
          else {
            diff = libxs_diff(desc, internal_registry_keys + i, size);
          }
        }
#if !defined(NDEBUG)
        else LIBXS_ASSERT(0 != diff);
#endif
        if (0 != diff) { /* search for code version */
          if (0 == mode) { /* transition to higher mode */
            i0 = i; /* keep current position on record */
#if defined(LIBXS_HASH_COLLISION)
            /* enter code generation, and collision fix-up */
            if (0 == (LIBXS_HASH_COLLISION & flux_entry.uval)) {
              LIBXS_ASSERT(NULL != flux_entry.ptr_const); /* collision */
              mode = 3;
            }
            else
#endif      /* search for an existing code version */
            mode = 1; /* else */
          }
          i = LIBXS_MOD2(i + 1, LIBXS_CAPACITY_REGISTRY);
          if (i == i0) { /* search finished, no code version exists */
#if defined(LIBXS_HASH_COLLISION)
            mode = 3; /* enter code generation, and collision fix-up */
#else
            mode = 2; /* enter code generation */
#endif
#if defined(LIBXS_COLLISION_COUNT_STATIC)
            if (LIBXS_KERNEL_KIND_MATMUL == LIBXS_DESCRIPTOR_KIND(desc->kind)) {
              internal_update_mmstatistic(&desc->gemm.desc, 0, 1/*collision*/, 0, 0);
            }
#endif
          }
          LIBXS_ASSERT(0 != diff); /* continue */
        }
      }
      else { /* enter code generation (there is no code version yet) */
        LIBXS_ASSERT(0 == mode || 1 < mode);
#if (0 == LIBXS_JIT)
        LIBXS_UNUSED(user_size);
#else
        if (LIBXS_X86_GENERIC <= libxs_target_archid || /* check if JIT is supported (CPUID) */
           (LIBXS_KERNEL_KIND_USER == LIBXS_DESCRIPTOR_KIND(desc->kind)))
        {
          LIBXS_ASSERT(0 != mode || NULL == flux_entry.ptr_const/*code version does not exist*/);
          INTERNAL_FIND_CODE_LOCK(lock, i, diff, flux_entry.ptr); /* lock the registry entry */
          if (NULL == internal_registry[i].ptr_const) { /* double-check registry after acquiring the lock */
            libxs_build_request request /*= { 0 }*/; /* setup the code build request */
            LIBXS_ASSERT(LIBXS_KERNEL_UNREGISTERED > LIBXS_DESCRIPTOR_KIND(desc->kind));
            request.kind = (libxs_build_kind)LIBXS_DESCRIPTOR_KIND(desc->kind);
            request.descriptor.ptr = &desc->gemm.desc;
            request.user_size = user_size;
# if defined(NDEBUG)
            if (EXIT_SUCCESS == libxs_build(&request, i, &flux_entry) && NULL != flux_entry.ptr_const)
# else
            build = libxs_build(&request, i, &flux_entry);
            if (EXIT_SUCCESS == build && NULL != flux_entry.ptr_const)
# endif
            {
              LIBXS_ASSIGN127(internal_registry_keys + i, desc);
# if (1 < INTERNAL_REGLOCK_MAXN)
              LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&internal_registry[i].ptr, flux_entry.ptr, LIBXS_ATOMIC_SEQ_CST);
# else
              internal_registry[i] = flux_entry;
# endif
# if defined(LIBXS_HASH_COLLISION)
              if (2 < mode) { /* arrived from collision state; now mark as collision */
                libxs_code_pointer fix_entry;
#   if (1 < INTERNAL_REGLOCK_MAXN)
                fix_entry.ptr = (void*)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(
                  &internal_registry[i0].ptr, LIBXS_ATOMIC_RELAXED);
#   else
                fix_entry = internal_registry[i0];
#   endif
                LIBXS_ASSERT(NULL != fix_entry.ptr_const);
                if (0 == (LIBXS_HASH_COLLISION & fix_entry.uval)) {
                  fix_entry.uval |= LIBXS_HASH_COLLISION; /* mark current entry as collision */
#   if (1 < INTERNAL_REGLOCK_MAXN)
                  LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, LIBXS_BITS)(&internal_registry[i0].ptr,
                    fix_entry.ptr, LIBXS_ATOMIC_RELAXED);
#   else
                  internal_registry[i0] = fix_entry;
#   endif
                }
#   if !defined(LIBXS_COLLISION_COUNT_STATIC)
                if (LIBXS_KERNEL_KIND_MATMUL == LIBXS_DESCRIPTOR_KIND(desc->kind)) {
                  internal_update_mmstatistic(&desc->gemm.desc, 0, 1/*collision*/, 0, 0);
                }
#   endif
              }
# endif
            }
            if (LIBXS_KERNEL_KIND_MATMUL == LIBXS_DESCRIPTOR_KIND(desc->kind)) {
              internal_update_mmstatistic(&desc->gemm.desc, 1/*try*/, 0, 0, 0);
            }
            /* leave here even in case of a build-error; do not use break (inside of locked region) */
            diff = 0;
          }
          INTERNAL_FIND_CODE_UNLOCK(lock);
          if (0 != diff) { /* acquire registry slot */
            if (0 == mode) { /* initial condition */
              mode = 2; /* continue to linearly search for an empty slot */
              i0 = i; /* keep current position on record */
            }
            do { /* continue to linearly search for an available slot */
              i = LIBXS_MOD2(i + 1, LIBXS_CAPACITY_REGISTRY);
              if (NULL == internal_registry[i].ptr_const) break;
            } while (i != i0);
            if (i == i0) { /* out of capacity (no registry slot available) */
              diff = 0; /* do not use break if inside of locked region */
#if !defined(NDEBUG) && (0 != LIBXS_JIT)
              build = EXIT_FAILURE;
#endif
            }
            flux_entry.ptr = NULL; /* no result */
          }
        }
        else /* JIT-code generation not available */
#endif
        { /* leave the dispatch loop */
          if (LIBXS_KERNEL_KIND_MATMUL == LIBXS_DESCRIPTOR_KIND(desc->kind)) {
            internal_update_mmstatistic(&desc->gemm.desc, 1/*try*/, 0, 0, 0);
          }
#if !defined(NDEBUG) && (0 != LIBXS_JIT)
          build = EXIT_FAILURE;
#endif
          flux_entry.ptr = NULL;
          diff = 0;
        }
      }
    } while (0 != diff);
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
    if (NULL != flux_entry.ptr_const) { /* keep code version on record (cache) */
      LIBXS_ASSERT(0 == diff);
      if (ninit == cache->entry.id) { /* maintain cache */
        if (cache->entry.size < internal_cache_size) { /* grow */
          INTERNAL_FIND_CODE_CACHE_GROW(cache_index, cache->entry.size);
          LIBXS_ASSERT(cache->entry.size <= internal_cache_size);
        }
        else { /* evict */
          LIBXS_ASSERT(cache->entry.hit < cache->entry.size);
          INTERNAL_FIND_CODE_CACHE_EVICT(cache_index, cache->entry.size, cache->entry.hit);
        }
      }
      else if (0 != internal_cache_size) { /* reset cache */
        /* INTERNAL_FIND_CODE_CACHE_GROW doubles size (and would expose invalid entries) */
        memset(cache->entry.keys, 0, LIBXS_CACHE_MAXSIZE * sizeof(*cache->entry.keys));
        cache->entry.id = ninit;
        cache->entry.size = 1;
        cache_index = 0;
      }
      LIBXS_MEMCPY127(cache->entry.keys + cache_index, desc, 0 == is_big_desc ? LIBXS_DIFF_SIZE : size);
      cache->entry.code[cache_index] = flux_entry;
      cache->entry.hit = cache_index;
    }
# if !defined(NDEBUG)
    else {
      memset(cache, 0, sizeof(*cache));
    }
# endif
#endif
  }
#if defined(LIBXS_HASH_COLLISION)
  flux_entry.uval &= ~(LIBXS_CODE_STATIC | LIBXS_HASH_COLLISION); /* clear non-JIT and collision flag */
#else
  flux_entry.uval &= ~LIBXS_CODE_STATIC; /* clear non-JIT flag */
#endif
#if (0 != LIBXS_JIT)
  assert( /*!LIBXS_ASSERT*/
    LIBXS_KERNEL_KIND_MATMUL != LIBXS_DESCRIPTOR_KIND(desc->kind)
    || NULL != flux_entry.ptr_const
    || 1 == internal_reglock_count
    || EXIT_SUCCESS != build);
#endif
  return flux_entry;
}


LIBXS_API_INTERN const libxs_kernel_xinfo* libxs_get_kernel_xinfo(libxs_code_pointer code,
  const libxs_descriptor** desc, size_t* code_size)
{
  libxs_kernel_xinfo* result = NULL;
  void *const result_address = &result;
  int flags = LIBXS_MALLOC_FLAG_X;
  if (NULL != code.ptr_const && EXIT_SUCCESS == libxs_get_malloc_xinfo(
    code.ptr_const, code_size, &flags, (void**)result_address) && NULL != result)
  {
    if (NULL != desc) {
      if (NULL != internal_registry && NULL != internal_registry_keys && result->registered < (LIBXS_CAPACITY_REGISTRY)
#if defined(LIBXS_HASH_COLLISION)
        && code.uval == (~LIBXS_HASH_COLLISION & internal_registry[result->registered].uval)
#else
        && code.ptr_const == internal_registry[result->registered].ptr_const
#endif
        && LIBXS_KERNEL_UNREGISTERED > LIBXS_DESCRIPTOR_KIND(internal_registry_keys[result->registered].entry.kind))
      {
        *desc = &internal_registry_keys[result->registered].entry;
      }
      else *desc = NULL;
    }
  }
  else {
    LIBXS_ASSERT(NULL == result);
    if (NULL != code_size) *code_size = 0;
    if (NULL != desc) *desc = NULL;
  }
  return result;
}


LIBXS_API int libxs_get_kernel_info(const void* kernel, libxs_kernel_info* info)
{
  int result;
  const libxs_kernel_xinfo* xinfo;
  libxs_kernel_info result_info /*= { 0 }*/;
  const libxs_descriptor* desc;
  libxs_code_pointer code = { 0 };
  code.ptr_const = kernel;
  LIBXS_MEMZERO127(&result_info);
  xinfo = libxs_get_kernel_xinfo(code, &desc, &result_info.code_size);
  result_info.is_reference_kernel = xinfo->is_reference_kernel;
  if (NULL != xinfo) {
    if (NULL != desc) {
      const libxs_kernel_kind kind = (libxs_kernel_kind)LIBXS_DESCRIPTOR_KIND(desc->kind);
      result_info.kind = kind;
      if (LIBXS_KERNEL_KIND_USER == kind) {
        result_info.code_size = 0; /* invalid */
      }
    }
    else {
      result_info.kind = LIBXS_KERNEL_UNREGISTERED;
    }
    result_info.nflops = xinfo->nflops;
    LIBXS_ASSIGN127(info, &result_info);
    result = EXIT_SUCCESS;
  }
  else {
    LIBXS_ASSERT(NULL == desc);
    if (NULL != info) {
      LIBXS_ASSIGN127(info, &result_info);
      result = EXIT_FAILURE;
    }
    else {
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


LIBXS_API int libxs_get_registry_info(libxs_registry_info* info)
{
  int result = EXIT_SUCCESS;
  /*LIBXS_INIT*/ /* verbosity */
  if (0 != info && 0 != internal_registry) {
    size_t i;
    LIBXS_MEMZERO127(info); /* info->nstatic = 0; info->size = 0; */
    info->nbytes = (LIBXS_CAPACITY_REGISTRY) * (sizeof(libxs_code_pointer) + sizeof(libxs_descriptor));
    info->capacity = LIBXS_CAPACITY_REGISTRY;
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
    info->ncache = internal_cache_size;
#else
    info->ncache = 0;
#endif
    for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
      libxs_code_pointer code = internal_registry[i];
      if (0 != code.ptr_const && EXIT_SUCCESS == result) {
        if (0 == (LIBXS_CODE_STATIC & code.uval)) { /* check for allocated/generated JIT-code */
          size_t buffer_size = 0;
          void* buffer = 0;
#if defined(LIBXS_HASH_COLLISION)
          code.uval &= ~LIBXS_HASH_COLLISION; /* clear collision flag */
#endif
          result = libxs_get_malloc_xinfo(code.ptr_const, &buffer_size, NULL/*flags*/, &buffer);
          if (EXIT_SUCCESS == result) {
            info->nbytes += LIBXS_UP2(buffer_size + (((const char*)code.ptr_const) - (char*)buffer), LIBXS_PAGE_MINSIZE);
          }
        }
        else {
          ++info->nstatic;
        }
        ++info->size;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INLINE void* internal_get_registry_entry(int i, libxs_kernel_kind kind, const void** key)
{
  void* result = NULL;
  LIBXS_ASSERT(kind < LIBXS_KERNEL_UNREGISTERED && NULL != internal_registry);
  for (; i < (LIBXS_CAPACITY_REGISTRY); ++i) {
    const libxs_code_pointer regentry = internal_registry[i];
    if (EXIT_SUCCESS == libxs_get_malloc_xinfo(regentry.ptr_const,
      NULL/*code_size*/, NULL/*flags*/, &result) && NULL != result)
    {
      const libxs_kernel_xinfo info = *(const libxs_kernel_xinfo*)result;
      const libxs_descriptor *const desc = &internal_registry_keys[info.registered].entry;
      if (LIBXS_DESCRIPTOR_KIND(desc->kind) == (int)kind) {
        if (NULL != key) {
#if defined(LIBXS_REGUSER_ALIGN)
          if (LIBXS_KERNEL_KIND_USER == kind) {
            const size_t offset = LIBXS_UP2(desc->user.desc - desc->data, 4 < desc->user.size ? 8 : 4);
            *key = desc->data + offset;
          }
          else
#endif
          *key = desc->user.desc;
        }
        result = regentry.ptr;
        break;
      }
    }
  }
  return result;
}


LIBXS_API void* libxs_get_registry_begin(const void** key)
{
  void* result = NULL;
  if (NULL != internal_registry) {
    result = internal_get_registry_entry(0, key);
  }
  return result;
}


LIBXS_API void* libxs_get_registry_next(const void* regentry, const void** key)
{
  void* result = NULL;
  const libxs_descriptor* desc;
  libxs_code_pointer entry = { 0 };
  entry.ptr_const = regentry;
  if (NULL != libxs_get_kernel_xinfo(entry, &desc, NULL/*code_size*/)
    /* given regentry is indeed a registered kernel */
    && NULL != desc)
  {
    result = internal_get_registry_entry(
      (int)(desc - &internal_registry_keys->entry + 1),
      (libxs_kernel_kind)LIBXS_DESCRIPTOR_KIND(desc->kind), key);
  }
  return result;
}


LIBXS_API void* libxs_xregister(const void* key, size_t key_size,
  size_t value_size, const void* value_init)
{
  libxs_descriptor wrap /*= { 0 }*/;
#if defined(LIBXS_REGUSER_ALIGN)
  const size_t offset = LIBXS_UP2(wrap.user.desc - wrap.data, 4 < key_size ? 8 : 4);
#else
  const size_t offset = wrap.user.desc - wrap.data;
#endif
  static int error_once = 0;
  void* result;
  /*LIBXS_INIT*/ /* verbosity */
  if (NULL != key && 0 < key_size && LIBXS_DESCRIPTOR_MAXSIZE >= (offset + key_size)) {
    void* dst;
#if defined(LIBXS_UNPACKED) || defined(LIBXS_REGUSER_ALIGN)
    LIBXS_MEMSET127(&wrap, 0, offset);
#endif
    LIBXS_MEMCPY127(wrap.data + offset, key, key_size);
    wrap.user.size = LIBXS_CAST_UCHAR(key_size);
    wrap.kind = (libxs_descriptor_kind)(LIBXS_DESCRIPTOR_SIGSIZE >= (offset + key_size)
      ? ((libxs_descriptor_kind)LIBXS_KERNEL_KIND_USER)
      : LIBXS_DESCRIPTOR_BIG(LIBXS_KERNEL_KIND_USER));
    dst = internal_find_code(&wrap, offset + key_size - sizeof(libxs_descriptor_kind), value_size).ptr;
    if (NULL != dst) {
      size_t size;
      if (EXIT_SUCCESS == libxs_get_malloc_xinfo(dst, &size, NULL/*flags*/, NULL/*extra*/)
        && value_size <= size)
      {
        if (NULL != value_init) memcpy(dst, value_init, value_size);
        result = dst;
      }
      else {
        if (0 != libxs_verbosity /* library code is expected to be mute */
          /*&& 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)*/)
        {
          fprintf(stderr, "LIBXS ERROR: value too large for previously registered key!\n");
        }
        result = NULL;
      }
    }
    else result = NULL;
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      if (LIBXS_DESCRIPTOR_MAXSIZE >= key_size) {
        fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xregister specified!\n");
      }
      else {
        fprintf(stderr, "LIBXS ERROR: libxs_xregister has maximum key-size of %i Byte!\n",
          LIBXS_DESCRIPTOR_MAXSIZE);
      }
    }
    result = NULL;
  }
  return result;
}


LIBXS_API void* libxs_xdispatch(const void* key, size_t key_size)
{
  libxs_descriptor wrap /*= { 0 }*/;
#if defined(LIBXS_REGUSER_ALIGN)
  const size_t offset = LIBXS_UP2(wrap.user.desc - wrap.data, 4 < key_size ? 8 : 4);
#else
  const size_t offset = wrap.user.desc - wrap.data;
#endif
  void* result;
  /*LIBXS_INIT*/ /* verbosity */
#if !defined(NDEBUG)
  if (NULL != key && 0 < key_size && LIBXS_DESCRIPTOR_MAXSIZE >= (offset + key_size))
#endif
  {
#if defined(LIBXS_UNPACKED) || defined(LIBXS_REGUSER_ALIGN)
    LIBXS_MEMSET127(&wrap, 0, offset);
#endif
    LIBXS_MEMCPY127(wrap.data + offset, key, key_size);
    wrap.user.size = LIBXS_CAST_UCHAR(key_size);
    wrap.kind = (libxs_descriptor_kind)(LIBXS_DESCRIPTOR_SIGSIZE >= (offset + key_size)
      ? ((libxs_descriptor_kind)LIBXS_KERNEL_KIND_USER)
      : LIBXS_DESCRIPTOR_BIG(LIBXS_KERNEL_KIND_USER));
    result = internal_find_code(&wrap, offset + key_size - sizeof(libxs_descriptor_kind), 0/*user_size*/).ptr;
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xdispatch specified!\n");
    }
    result = NULL;
  }
#endif
  return result;
}


LIBXS_API void libxs_xrelease(const void* key, size_t key_size)
{
  libxs_release_kernel(libxs_xdispatch(key, key_size));
}


LIBXS_API void libxs_release_kernel(const void* kernel)
{
  if (NULL != kernel) {
    static int error_once = 0;
    libxs_kernel_xinfo* extra = NULL;
    void *const extra_address = &extra;
    /*LIBXS_INIT*/
    if (EXIT_SUCCESS == libxs_get_malloc_xinfo(
      kernel, NULL/*size*/, NULL/*flags*/, (void**)extra_address) && NULL != extra)
    {
      const unsigned int regindex = extra->registered;
      if ((LIBXS_CAPACITY_REGISTRY) <= regindex) {
        libxs_xfree(kernel, 0/*no check*/);
      }
      else { /* attempt to unregister kernel */
        libxs_kernel_info info /*= { 0 }*/;
#if !defined(LIBXS_ENABLE_DEREG)
        if (EXIT_SUCCESS == libxs_get_kernel_info(kernel, &info)
          && LIBXS_KERNEL_KIND_USER == info.kind)
#endif
        {
          LIBXS_ASSERT(LIBXS_KERNEL_UNREGISTERED > info.kind);
          /* coverity[check_return] */
          LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_SEQ_CST); /* invalidate code cache (TLS) */
          internal_registry[regindex].ptr = NULL;
#if !defined(NDEBUG)
          memset(internal_registry_keys + regindex, 0, sizeof(*internal_registry_keys));
#endif
          libxs_xfree(kernel, 0/*no check*/);
        }
#if !defined(LIBXS_ENABLE_DEREG)
        else if (0 != libxs_verbosity /* library code is expected to be mute */
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS WARNING: attempt to unregister JIT-kernel!\n");
        }
#endif
      }
    }
    else if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: failed to release kernel!\n");
    }
  }
}


#if !defined(__linux__) && defined(__APPLE__)
LIBXS_EXTERN char*** _NSGetArgv(void);
LIBXS_EXTERN int* _NSGetArgc(void);
#endif


LIBXS_API_INTERN int libxs_print_cmdline(void* buffer, size_t buffer_size, const char* prefix, const char* postfix)
{
  int result = 0;
#if defined(__linux__)
  FILE *const cmdline = fopen("/proc/self/cmdline", "r");
  if (NULL != cmdline) {
    char a, b;
    if (1 == fread(&a, 1, 1, cmdline) && '\0' != a) {
      if (NULL != prefix && '\0' != *prefix) {
        result = (0 == buffer_size ? fprintf((FILE*)buffer, "%s", prefix)
          : LIBXS_SNPRINTF((char*)buffer, buffer_size, "%s", prefix));
      }
      while (1 == fread(&b, 1, 1, cmdline)) {
        result += (0 == buffer_size ? fprintf((FILE*)buffer, "%c", a)
          : LIBXS_SNPRINTF((char*)buffer + result, buffer_size - result, "%c", a));
        a = ('\0' != b ? b : ' ');
      };
    }
    fclose(cmdline);
  }
#else
  char** argv = NULL;
  int argc = 0;
# if defined(_WIN32)
  argv = __argv;
  argc = __argc;
# elif defined(__APPLE__)
  argv = (NULL != _NSGetArgv() ? *_NSGetArgv() : NULL);
  argc = (NULL != _NSGetArgc() ? *_NSGetArgc() : 0);
# endif
  if (0 < argc) {
    int i = 1;
    if (NULL != prefix && '\0' != *prefix) {
# if defined(_WIN32)
      const char *const cmd = strrchr(argv[0], '\\');
      const char *const exe = (NULL != cmd ? (cmd + 1) : argv[0]);
      result += (0 == buffer_size ? fprintf((FILE*)buffer, "%s%s", prefix, exe)
        : LIBXS_SNPRINTF((char*)buffer + result, buffer_size - result, "%s%s", prefix, exe));
# else
      result += (0 == buffer_size ? fprintf((FILE*)buffer, "%s%s", prefix, argv[0])
        : LIBXS_SNPRINTF((char*)buffer + result, buffer_size - result, "%s%s", prefix, argv[0]));
# endif
    }
    for (; i < argc; ++i) {
      result += (0 == buffer_size ? fprintf((FILE*)buffer, " %s", argv[i])
        : LIBXS_SNPRINTF((char*)buffer + result, buffer_size - result, " %s", argv[i]));
    }
  }
#endif
  if (0 < result && NULL != postfix && '\0' != *postfix) {
    result += (0 == buffer_size ? fprintf((FILE*)buffer, "%s", postfix)
      : LIBXS_SNPRINTF((char*)buffer + result, buffer_size - result, "%s", postfix));
  }
  return result;
}


#if defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_init)(void);
LIBXS_API void LIBXS_FSYMBOL(libxs_init)(void)
{
  libxs_init();
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_finalize)(void);
LIBXS_API void LIBXS_FSYMBOL(libxs_finalize)(void)
{
  libxs_finalize();
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_release_kernel)(const void** /*kernel*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_release_kernel)(const void** kernel)
{
#if !defined(NDEBUG)
  if (NULL != kernel)
#endif
  {
    libxs_release_kernel(*kernel);
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: invalid argument passed into libxs_release_kernel!\n");
    }
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xregister)(void** /*regval*/, const void* /*key*/, const int* /*keysize*/,
  const int* /*valsize*/, const void* /*valinit*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xregister)(void** regval, const void* key, const int* keysize,
  const int* valsize, const void* valinit)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != regval && NULL != key && NULL != keysize && NULL != valsize)
#endif
  {
    *regval = libxs_xregister(key, *keysize, *valsize, valinit);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xregister specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xdispatch)(void** /*regval*/, const void* /*key*/, const int* /*keysize*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xdispatch)(void** regval, const void* key, const int* keysize)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != regval && NULL != key && NULL != keysize)
#endif
  {
    *regval = libxs_xdispatch(key, *keysize);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xdispatch specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_xrelease)(const void* /*key*/, const int* /*keysize*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_xrelease)(const void* key, const int* keysize)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != key && NULL != keysize)
#endif
  {
    libxs_xrelease(key, *keysize);
  }
#if !defined(NDEBUG)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_xrelease specified!\n");
  }
#endif
}

#endif /*defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))*/

#endif
