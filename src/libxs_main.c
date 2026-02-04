/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_timer.h>
#include <libxs_cpuid.h>
#include <libxs_sync.h>
#include "libxs_main.h"

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
#if !defined(LIBXS_HASH_SEED)
# define LIBXS_HASH_SEED 25071975
#endif
#if !defined(LIBXS_REGLOCK_TRY) && 0
# define LIBXS_REGLOCK_TRY
#endif
#if !defined(LIBXS_UNIFY_LOCKS) && 1
# define LIBXS_UNIFY_LOCKS
#endif
#if !defined(LIBXS_AUTOPIN) && 0
# define LIBXS_AUTOPIN
#endif
#if !defined(LIBXS_MAIN_DELIMS)
# define LIBXS_MAIN_DELIMS ";,:"
#endif

#if !defined(LIBXS_CACHE_MAXSIZE)
# define LIBXS_CACHE_MAXSIZE LIBXS_CAPACITY_CACHE
#endif


extern libxs_registry_t* internal_registry_state;

/** Time stamp (startup time of library). */
LIBXS_APIVAR_DEFINE(libxs_timer_tick_t internal_timer_start);
LIBXS_APIVAR_DEFINE(libxs_cpuid_info_t internal_cpuid_info);
LIBXS_APIVAR_DEFINE(const char* internal_build_state);

/* definition of corresponding variables */
LIBXS_APIVAR_PUBLIC_DEF(int libxs_verbosity);

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
LIBXS_APIVAR_PRIVATE_DEF(double libxs_timer_scale);
LIBXS_APIVAR_PRIVATE_DEF(unsigned int libxs_thread_count);
/* definition of corresponding variables */
LIBXS_APIVAR_PUBLIC_DEF(LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_lock_global);
LIBXS_APIVAR_PUBLIC_DEF(unsigned int libxs_ninit);
LIBXS_APIVAR_PUBLIC_DEF(int libxs_stdio_handle);
LIBXS_APIVAR_PUBLIC_DEF(int libxs_nosync);


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
            n = LIBXS_SNPRINTF(buffer, sizeof(buffer), "%.*s%u%s", n, filename, libxs_pid(), filename + n + 3);
            if (0 < n && (int)sizeof(buffer) > n) {
              file = fopen(buffer, "r");
              filename = buffer;
            }
          }
        }
        else {
          fprintf(stderr, "LIBXS INFO: PID=%u\n", libxs_pid());
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
      fprintf(ostream, "%s\n", internal_build_state);
    }
  }
}


LIBXS_API double libxs_timer_duration_rtc(libxs_timer_tick_t tick0, libxs_timer_tick_t tick1)
{
  const libxs_timer_tick_t delta = LIBXS_DELTA(tick0, tick1);
#if defined(_WIN32)
  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  return LIBXS_TIMER_DURATION_IDIV(delta, (libxs_timer_tick_t)frequency.QuadPart);
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


LIBXS_API libxs_timer_tick_t libxs_timer_tick_rtc(void)
{
#if defined(_WIN32)
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return (libxs_timer_tick_t)t.QuadPart;
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
libxs_timer_tick_t libxs_timer_tick_tsc(void)
{
  libxs_timer_tick_t result;
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
    /*LIBXS_LOCK_DESTROY(LIBXS_REGLOCK, internal_reglock_ptr);*/
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
      if (0 == libxs_tid()) {
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
  /*LIBXS_LOCK_ACQUIRE(LIBXS_REGLOCK, internal_reglock_ptr);*/
# endif
#endif
#if 0
  if (NULL == internal_registry_state || NULL == internal_registry_state->registry) { /* double-check after acquiring the lock(s) */
#if defined(LIBXS_INTERCEPT_DYNAMIC) && defined(LIBXS_AUTOPIN)
    /* clear error status (dummy condition: it does not matter if MPI_Init or MPI_Abort) */
    const char *const dlsymname = (NULL == dlerror() ? "MPI_Init" : "MPI_Abort");
    const void *const dlsymbol = dlsym(LIBXS_RTLD_NEXT, dlsymname);
    const void *const dlmpi = (NULL == dlerror() ? dlsymbol : NULL);
#endif
    const char *const env_verbose = getenv("LIBXS_VERBOSE");
    int cache_size = 0;
    void* new_state = NULL;
    void* new_registry = NULL, * new_keys = NULL;
#if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
# if defined(LIBXS_NTHREADS_USE)
    void* new_cache = NULL;
# endif
    const char *const env_cache = getenv("LIBXS_CACHE");
    if (NULL != env_cache && '\0' != *env_cache) {
      const int env_cache_size = atoi(env_cache), cache_size2 = (int)LIBXS_UP2POT(env_cache_size);
      cache_size = LIBXS_MIN(cache_size2, LIBXS_CACHE_MAXSIZE);
    }
    else {
      cache_size = LIBXS_CACHE_MAXSIZE;
    }
#endif
    LIBXS_UNUSED(cache_size);
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
    LIBXS_ASSERT(LIBXS_ISPOT(LIBXS_CAPACITY_REGISTRY));
#if 0
    /*libxs_hash_init(libxs_cpuid(NULL));*/
    /*libxs_memory_init(libxs_cpuid(NULL));*/
    if ((EXIT_SUCCESS == libxs_xmalloc(&new_state, sizeof(libxs_registry_t), 0/*auto-align*/,
            LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_state)
    #if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
          && (EXIT_SUCCESS == libxs_xmalloc(&new_cache, /* if internal_cache_size is zero, allocation must still happen (later control-flow too expensive) */
            sizeof(internal_cache_type) * (LIBXS_NTHREADS_MAX), LIBXS_CACHELINE/*alignment*/,
            LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_cache)
    #endif
          && (EXIT_SUCCESS == libxs_xmalloc(&new_keys, (LIBXS_CAPACITY_REGISTRY) * sizeof(internal_regkey_type), 0/*auto-align*/,
            LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_keys)
          && (EXIT_SUCCESS == libxs_xmalloc(&new_registry, (LIBXS_CAPACITY_REGISTRY) * sizeof(libxs_code_pointer_t), 0/*auto-align*/,
            LIBXS_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_registry))
    {
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      LIBXS_ASSERT(NULL != new_cache); /* SA: suppress false positive */
      memset(new_cache, 0, (LIBXS_NTHREADS_MAX) * sizeof(internal_cache_type));
#endif
      libxs_xcopy_init(libxs_cpuid(NULL));
      for (i = 0; i < (LIBXS_CAPACITY_REGISTRY); ++i) ((libxs_code_pointer_t*)new_registry)[i].ptr = NULL;
      /*LIBXS_ASSERT(NULL == internal_registry && NULL == internal_registry_keys);*/
    #if defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      LIBXS_ASSERT(NULL != new_state);
    # if defined(LIBXS_NTHREADS_USE)
      ((libxs_registry_t*)new_state)->cache_buffer = (internal_cache_type*)new_cache;
    # else
      ((libxs_registry_t*)new_state)->cache_buffer = NULL;
    # endif
      ((libxs_registry_t*)new_state)->cache_size = cache_size;
    #endif
      ((libxs_registry_t*)new_state)->keys = (internal_regkey_type*)new_keys; /* prior to registering static kernels */
      ((libxs_registry_t*)new_state)->registry = (libxs_code_pointer_t*)new_registry;
      ((libxs_registry_t*)new_state)->capacity = LIBXS_CAPACITY_REGISTRY;
      ((libxs_registry_t*)new_state)->size = 0;
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
      internal_registry_state = (libxs_registry_t*)new_state; /* commit the registry buffer and enable global visibility */
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
      libxs_xfree(new_state, 0/*no check*/);
    }
#endif
  }
#endif
#if (0 != LIBXS_SYNC) /* release locks */
# if (1 < INTERNAL_REGLOCK_MAXN)
  for (i = 0; i < internal_reglock_count; ++i) LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXS_UNIFY_LOCKS)
  /*LIBXS_LOCK_RELEASE(LIBXS_REGLOCK, internal_reglock_ptr);*/
# endif
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &libxs_lock_global);
#endif
}


LIBXS_API_CTOR void libxs_init(void)
{
  if (NULL == (const void*)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(&internal_registry_state, LIBXS_ATOMIC_SEQ_CST)) {
    static unsigned int counter = 0, gid = 0;
    const unsigned int tid = LIBXS_ATOMIC_ADD_FETCH(&counter, 1, LIBXS_ATOMIC_SEQ_CST);
    LIBXS_ASSERT(0 < tid);
    /* libxs_ninit (1: initialization started, 2: library initialized, higher: to invalidate code-TLS) */
    if (1 == tid) {
      libxs_timer_tick_t s0 = libxs_timer_tick_rtc(); /* warm-up */
      libxs_timer_tick_t t0 = libxs_timer_tick_tsc(); /* warm-up */
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
#if 0
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
#endif
        LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &attr_global);
        LIBXS_LOCK_INIT(LIBXS_LOCK, &libxs_lock_global, &attr_global);
        LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &attr_global);
#if 0
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
#endif
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
        libxs_timer_tick_t s1, t1;
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
  }
}


LIBXS_API LIBXS_ATTRIBUTE_NO_TRACE void libxs_finalize(void);
LIBXS_API_DTOR void libxs_finalize(void)
{
#if 0
  libxs_registry_t* registry_state = (libxs_registry_t*)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(
    &internal_registry_state, LIBXS_ATOMIC_SEQ_CST);
  libxs_code_pointer_t* registry = (NULL != registry_state ? registry_state->registry : NULL);
  if (NULL != registry_state && NULL != registry) {
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
    registry_state = (libxs_registry_t*)LIBXS_ATOMIC(LIBXS_ATOMIC_LOAD, LIBXS_BITS)(
      &internal_registry_state, LIBXS_ATOMIC_SEQ_CST);
    registry = (NULL != registry_state ? registry_state->registry : NULL);
    if (NULL != registry_state && NULL != registry) {
      internal_regkey_type *const registry_keys = registry_state->keys;
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      internal_cache_type *const cache_buffer = registry_state->cache_buffer;
#endif
      unsigned int rest = 0, errors = 0;
#if defined(LIBXS_TRACE)
      i = libxs_trace_finalize();
      if (EXIT_SUCCESS != i && 0 != libxs_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXS ERROR: failed to finalize trace (error #%i)!\n", i);
      }
#endif
      /* coverity[check_return] */
      LIBXS_ATOMIC_ADD_FETCH(&libxs_ninit, 1, LIBXS_ATOMIC_SEQ_CST); /* invalidate code cache (TLS) */
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      registry_state->cache_buffer = NULL;
#endif
      registry_state->keys = NULL; /* make registry keys unavailable */
      registry_state->registry = NULL;
      registry_state->size = 0;
      LIBXS_ATOMIC(LIBXS_ATOMIC_STORE_ZERO, LIBXS_BITS)((uintptr_t*)&internal_registry_state, LIBXS_ATOMIC_SEQ_CST);
      /* release buffers (registry, keys, cache) */
#if 0
#if defined(LIBXS_NTHREADS_USE) && defined(LIBXS_CACHE_MAXSIZE) && (0 < (LIBXS_CACHE_MAXSIZE))
      libxs_xfree(cache_buffer, 0/*no check*/);
#endif
      libxs_xfree(registry_keys, 0/*no check*/);
      libxs_xfree(registry, 0/*no check*/);
      libxs_xfree(registry_state, 0/*no check*/);
#endif
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
#endif
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


LIBXS_API const char* libxs_typename(libxs_datatype datatype)
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

#endif /*defined(LIBXS_BUILD) && (!defined(LIBXS_NOFORTRAN) || defined(__clang_analyzer__))*/
