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
#ifndef LIBXS_SYNC_H
#define LIBXS_SYNC_H

#include "libxs_intrinsics_x86.h"

#if defined(LIBXS_NO_SYNC)
# undef _REENTRANT
#elif !defined(_REENTRANT)
# define _REENTRANT
#endif

#if !defined(LIBXS_TLS)
# if defined(_REENTRANT) && !defined(LIBXS_NO_TLS)
#   if defined(__CYGWIN__) && defined(__clang__)
#     define LIBXS_NO_TLS
#     define LIBXS_TLS
#   else
#     if (defined(_WIN32) && !defined(__GNUC__)) || defined(__PGI)
#       define LIBXS_TLS LIBXS_ATTRIBUTE(thread)
#     elif defined(__GNUC__) || defined(_CRAYC)
#       define LIBXS_TLS __thread
#     elif defined(__cplusplus)
#       define LIBXS_TLS thread_local
#     else
#       error Missing TLS support!
#     endif
#   endif
# else
#   if !defined(LIBXS_NO_TLS)
#     define LIBXS_NO_TLS
#   endif
#   define LIBXS_TLS
# endif
#endif

#if defined(__MIC__)
# define LIBXS_SYNC_PAUSE _mm_delay_32(8/*delay*/)
#elif !defined(LIBXS_INTRINSICS_NONE) && !defined(LIBXS_INTRINSICS_LEGACY)
# define LIBXS_SYNC_PAUSE _mm_pause()
#else
# define LIBXS_SYNC_PAUSE LIBXS_FLOCK(stdout); LIBXS_FUNLOCK(stdout)
#endif

#if defined(__GNUC__)
# if !defined(LIBXS_GCCATOMICS)
    /* note: the following version check does *not* prevent non-GNU compilers to adopt GCC's atomics */
#   if (LIBXS_VERSION3(4, 7, 4) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#     define LIBXS_GCCATOMICS 1
#   else
#     define LIBXS_GCCATOMICS 0
#   endif
# endif
#endif

#define LIBXS_ATOMIC_RELAXED __ATOMIC_RELAXED
#define LIBXS_ATOMIC_SEQ_CST __ATOMIC_SEQ_CST

#define LIBXS_NONATOMIC_LOAD(SRC_PTR, KIND) (*(SRC_PTR))
#define LIBXS_NONATOMIC_STORE(DST_PTR, VALUE, KIND) (*(DST_PTR) = VALUE)
#define LIBXS_NONATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXS_NONATOMIC_STORE(DST_PTR, 0, KIND)
#define LIBXS_NONATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) += VALUE)
#define LIBXS_NONATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) -= VALUE)
#define LIBXS_NONATOMIC_SET(DST, VALUE) ((DST) = (VALUE))

#if defined(_REENTRANT) && defined(LIBXS_GCCATOMICS)
# if (0 != LIBXS_GCCATOMICS)
#   define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) __atomic_load_n(SRC_PTR, KIND)
#   define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) __atomic_store_n(DST_PTR, VALUE, KIND)
#   define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) =*/ __atomic_add_fetch(DST_PTR, VALUE, KIND)
#   define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) =*/ __atomic_sub_fetch(DST_PTR, VALUE, KIND)
# else
#   define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) __sync_or_and_fetch(SRC_PTR, 0)
#   define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) while (*(DST_PTR) != (VALUE)) \
      if (0/*false*/ != __sync_bool_compare_and_swap(DST_PTR, *(DST_PTR), VALUE)) break
    /* use store side-effect of built-in (dummy assignment to mute warning) */
#   if 0 /* disabled as it appears to hang on some systems; fall-back is below */
#   define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) { \
      const int libxs_store_zero_ = (0 != __sync_and_and_fetch(DST_PTR, 0)) ? 1 : 0; \
      LIBXS_UNUSED(libxs_store_zero_); \
    }
#   endif
#   define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) = */__sync_add_and_fetch(DST_PTR, VALUE)
#   define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) = */__sync_sub_and_fetch(DST_PTR, VALUE)
# endif
# define LIBXS_SYNCHRONIZE __sync_synchronize()
/* TODO: distinct implementation of LIBXS_ATOMIC_SYNC_* wrt LIBXS_GCCATOMICS */
# define LIBXS_ATOMIC_SYNC_CHECK(LOCK, VALUE) while ((VALUE) == (LOCK)); LIBXS_SYNC_PAUSE
# define LIBXS_ATOMIC_SYNC_SET(LOCK) do { LIBXS_ATOMIC_SYNC_CHECK(LOCK, 1); } while(0 != __sync_lock_test_and_set(&(LOCK), 1))
# define LIBXS_ATOMIC_SYNC_UNSET(LOCK) __sync_lock_release(&(LOCK))
#elif defined(_REENTRANT) && defined(_WIN32) /* TODO: atomics */
# define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) (*((SRC_PTR) /*+ InterlockedOr((LONG volatile*)(SRC_PTR), 0) * 0*/))
# define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) (*(DST_PTR) = VALUE)
# define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) += VALUE)
# define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) -= VALUE)
# define LIBXS_ATOMIC_SYNC_CHECK(LOCK, VALUE) while ((VALUE) == (LOCK)); LIBXS_SYNC_PAUSE
# define LIBXS_ATOMIC_SYNC_SET(LOCK) { int libxs_sync_set_i_; \
    do { LIBXS_ATOMIC_SYNC_CHECK(LOCK, 1); \
      libxs_sync_set_i_ = LOCK; LOCK = 1; \
    } while(0 != libxs_sync_set_i_); \
  }
# define LIBXS_ATOMIC_SYNC_UNSET(LOCK) (LOCK) = 0
# define LIBXS_SYNCHRONIZE /* TODO */
#else
# define LIBXS_ATOMIC_LOAD LIBXS_NONATOMIC_LOAD
# define LIBXS_ATOMIC_STORE LIBXS_NONATOMIC_STORE
# define LIBXS_ATOMIC_ADD_FETCH LIBXS_NONATOMIC_ADD_FETCH
# define LIBXS_ATOMIC_SUB_FETCH LIBXS_NONATOMIC_SUB_FETCH
# define LIBXS_ATOMIC_SYNC_CHECK(LOCK, VALUE) LIBXS_UNUSED(LOCK)
# define LIBXS_ATOMIC_SYNC_SET(LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_ATOMIC_SYNC_UNSET(LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_SYNCHRONIZE
#endif
#if !defined(LIBXS_ATOMIC_STORE_ZERO)
# define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXS_ATOMIC_STORE(DST_PTR, 0, KIND)
#endif
#if !defined(LIBXS_ATOMIC_SET) /* TODO */
# define LIBXS_ATOMIC_SET(DST, VALUE) ((DST) = (VALUE))
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if defined(_REENTRANT)
  /* OpenMP based locks need to stay disabled unless both
   * libxs and libxsext are built with OpenMP support.
   */
# if defined(_OPENMP) && defined(LIBXS_OMP)
#   include <omp.h>
#   define LIBXS_LOCK_ACQUIRED 1
#   define LIBXS_LOCK_ATTR_TYPE const void*
#   define LIBXS_LOCK_ATTR_INIT(ATTR) LIBXS_UNUSED(ATTR)
#   define LIBXS_LOCK_ATTR_DESTROY(ATTR) LIBXS_UNUSED(ATTR)
#   define LIBXS_LOCK_TYPE omp_lock_t
#   define LIBXS_LOCK_CONSTRUCT LIBXS_LOCK_TYPE()
#   define LIBXS_LOCK_INIT(LOCK, ATTR) omp_init_lock(LOCK)
#   define LIBXS_LOCK_DESTROY(LOCK) omp_destroy_lock(LOCK)
#   define LIBXS_LOCK_ACQUIRE(LOCK) omp_set_lock(LOCK)
#   define LIBXS_LOCK_TRYLOCK(LOCK) omp_test_lock(LOCK)
#   define LIBXS_LOCK_RELEASE(LOCK) omp_unset_lock(LOCK)
# elif defined(_WIN32)
#   include <windows.h>
#   if defined(LIBXS_LOCK_MUTEX)
#     define LIBXS_LOCK_ACQUIRED WAIT_OBJECT_0
#     define LIBXS_LOCK_ATTR_TYPE LPSECURITY_ATTRIBUTES
#     define LIBXS_LOCK_ATTR_INIT(ATTR) *(ATTR) = NULL
#     define LIBXS_LOCK_ATTR_DESTROY(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_TYPE HANDLE
#     define LIBXS_LOCK_CONSTRUCT 0
#     define LIBXS_LOCK_INIT(LOCK, ATTR) *(LOCK) = CreateMutex(*(ATTR), FALSE, NULL)
#     define LIBXS_LOCK_DESTROY(LOCK) CloseHandle(*(LOCK))
#     define LIBXS_LOCK_ACQUIRE(LOCK) WaitForSingleObject(*(LOCK), INFINITE)
#     define LIBXS_LOCK_TRYLOCK(LOCK) WaitForSingleObject(*(LOCK), 0)
#     define LIBXS_LOCK_RELEASE(LOCK) ReleaseMutex(*(LOCK))
#   else
#     define LIBXS_LOCK_ACQUIRED TRUE
#     define LIBXS_LOCK_ATTR_TYPE const void*
#     define LIBXS_LOCK_ATTR_INIT(ATTR) *(ATTR) = NULL
#     define LIBXS_LOCK_ATTR_DESTROY(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_TYPE CRITICAL_SECTION
#     define LIBXS_LOCK_CONSTRUCT LIBXS_LOCK_TYPE()
#     define LIBXS_LOCK_INIT(LOCK, ATTR) InitializeCriticalSection(LOCK)
#     define LIBXS_LOCK_DESTROY(LOCK) DeleteCriticalSection(LOCK)
#     define LIBXS_LOCK_ACQUIRE(LOCK) EnterCriticalSection(LOCK)
#     define LIBXS_LOCK_TRYLOCK(LOCK) TryEnterCriticalSection(LOCK)
#     define LIBXS_LOCK_RELEASE(LOCK) LeaveCriticalSection(LOCK)
#   endif
# else
#   include <pthread.h>
#   if defined(LIBXS_LOCK_MUTEX) || (defined(__APPLE__) && defined(__MACH__))
#     define LIBXS_LOCK_ACQUIRED 0
#     define LIBXS_LOCK_ATTR_TYPE pthread_mutexattr_t
#     if defined(NDEBUG)
#       define LIBXS_LOCK_ATTR_INIT(ATTR) pthread_mutexattr_init(ATTR); \
                pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_NORMAL)
#     else
#       define LIBXS_LOCK_ATTR_INIT(ATTR) pthread_mutexattr_init(ATTR); \
                pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_ERRORCHECK)
#     endif
#     define LIBXS_LOCK_ATTR_DESTROY(ATTR) pthread_mutexattr_destroy(ATTR)
#     define LIBXS_LOCK_TYPE pthread_mutex_t
#     define LIBXS_LOCK_CONSTRUCT PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP
#     define LIBXS_LOCK_INIT(LOCK, ATTR) pthread_mutex_init(LOCK, ATTR)
#     define LIBXS_LOCK_DESTROY(LOCK) pthread_mutex_destroy(LOCK)
#     define LIBXS_LOCK_ACQUIRE(LOCK) pthread_mutex_lock(LOCK)
#     define LIBXS_LOCK_TRYLOCK(LOCK) pthread_mutex_trylock(LOCK)
#     define LIBXS_LOCK_RELEASE(LOCK) pthread_mutex_unlock(LOCK)
#   else
#     define LIBXS_LOCK_ACQUIRED 0
#     define LIBXS_LOCK_ATTR_TYPE int
#     define LIBXS_LOCK_ATTR_INIT(ATTR) *(ATTR) = 0
#     define LIBXS_LOCK_ATTR_DESTROY(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_TYPE pthread_spinlock_t
#     define LIBXS_LOCK_CONSTRUCT PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP
#     define LIBXS_LOCK_INIT(LOCK, ATTR) pthread_spin_init(LOCK, *(ATTR))
#     define LIBXS_LOCK_DESTROY(LOCK) pthread_spin_destroy(LOCK)
#     define LIBXS_LOCK_ACQUIRE(LOCK) pthread_spin_lock(LOCK)
#     define LIBXS_LOCK_TRYLOCK(LOCK) pthread_spin_trylock(LOCK)
#     define LIBXS_LOCK_RELEASE(LOCK) pthread_spin_unlock(LOCK)
#   endif
# endif
#else
# define LIBXS_LOCK_ACQUIRED 0
# define LIBXS_LOCK_ATTR_TYPE const void*
# define LIBXS_LOCK_ATTR_INIT(ATTR) *(ATTR) = NULL
# define LIBXS_LOCK_ATTR_DESTROY(ATTR) LIBXS_UNUSED(ATTR)
# define LIBXS_LOCK_TYPE const void*
# define LIBXS_LOCK_CONSTRUCT 0
# define LIBXS_LOCK_INIT(LOCK, ATTR) *(LOCK) = NULL; LIBXS_UNUSED(ATTR)
# define LIBXS_LOCK_DESTROY(LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_ACQUIRE(LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_TRYLOCK(LOCK) LIBXS_LOCK_ACQUIRED
# define LIBXS_LOCK_RELEASE(LOCK) LIBXS_UNUSED(LOCK)
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/** Opaque type which represents a barrier. */
typedef struct LIBXS_RETARGETABLE libxs_barrier libxs_barrier;

/** Create barrier from one of the threads. */
LIBXS_API libxs_barrier* libxs_barrier_create(int ncores, int nthreads_per_core);
/** Initialize the barrier from each thread of the team. */
LIBXS_API void libxs_barrier_init(libxs_barrier* barrier, int tid);
/** Wait for the entire team to arrive. */
LIBXS_API void libxs_barrier_wait(libxs_barrier* barrier, int tid);
/** Release the resources associated with this barrier. */
LIBXS_API void libxs_barrier_release(const libxs_barrier* barrier);

/** Utility function to receive the process ID of the calling process. */
LIBXS_API unsigned int libxs_get_pid(void);
/**
 * Utility function to receive a Thread-ID (TID) for the calling thread.
 * The TID is not related to a specific threading runtime. TID=0 may not
 * represent the main thread. TIDs are zero-based and consecutive numbers.
 */
LIBXS_API unsigned int libxs_get_tid(void);

#endif /*LIBXS_SYNC_H*/
