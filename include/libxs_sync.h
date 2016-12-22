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

#include "libxs_macros.h"

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

#if (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(LIBXS_GCCATOMICS)
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
#   if 0 /* disabled as it appears to hang on some systems; fallback impl. is below */
#   define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) { \
      const int libxs_store_zero_ = (0 != __sync_and_and_fetch(DST_PTR, 0)) ? 1 : 0; \
      LIBXS_UNUSED(libxs_store_zero_); \
    }
#   endif
#   define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) = */__sync_add_and_fetch(DST_PTR, VALUE)
#   define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) /**(DST_PTR) = */__sync_sub_and_fetch(DST_PTR, VALUE)
# endif
#elif (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(_WIN32) /*TODO*/
#   define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) (*(SRC_PTR))
#   define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) (*(DST_PTR) = VALUE)
#   define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) += VALUE)
#   define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) -= VALUE)
#else
#   define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) (*(SRC_PTR))
#   define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) (*(DST_PTR) = VALUE)
#   define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) += VALUE)
#   define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) -= VALUE)
#endif
#if !defined(LIBXS_ATOMIC_STORE_ZERO)
# define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXS_ATOMIC_STORE(DST_PTR, 0, KIND)
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if defined(_REENTRANT)
# if defined(_WIN32) /*TODO*/
#   define LIBXS_LOCK_ACQUIRED WAIT_OBJECT_0
#   define LIBXS_LOCK_TYPE HANDLE
#   define LIBXS_LOCK_CONSTRUCT 0
#   define LIBXS_LOCK_INIT(LOCK) /*TODO*/
#   define LIBXS_LOCK_DESTROY(LOCK) CloseHandle(LOCK)
#   define LIBXS_LOCK_ACQUIRE(LOCK) WaitForSingleObject(LOCK, INFINITE)
#   define LIBXS_LOCK_TRYLOCK(LOCK) WaitForSingleObject(LOCK, 0)
#   define LIBXS_LOCK_RELEASE(LOCK) ReleaseMutex(LOCK)
# else
#   include <pthread.h>
#   define LIBXS_LOCK_ACQUIRED 0
#   define LIBXS_LOCK_TYPE pthread_mutex_t
#   define LIBXS_LOCK_CONSTRUCT PTHREAD_MUTEX_INITIALIZER
#   define LIBXS_LOCK_INIT(LOCK) pthread_mutex_init(LOCK, 0)
#   define LIBXS_LOCK_DESTROY(LOCK) pthread_mutex_destroy(LOCK)
#   define LIBXS_LOCK_ACQUIRE(LOCK) pthread_mutex_lock(LOCK)
#   define LIBXS_LOCK_TRYLOCK(LOCK) pthread_mutex_trylock(LOCK)
#   define LIBXS_LOCK_RELEASE(LOCK) pthread_mutex_unlock(LOCK)
# endif
#else
# define LIBXS_LOCK_ACQUIRED 0
# define LIBXS_LOCK_TYPE const void*
# define LIBXS_LOCK_CONSTRUCT 0
# define LIBXS_LOCK_INIT(LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_DESTROY(LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_ACQUIRE(LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_TRYLOCK(LOCK) LIBXS_UNUSED(LOCK)
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
/** Utility function to receive the thread ID of the calling thread. */
LIBXS_API unsigned int libxs_get_tid(void);

#endif /*LIBXS_SYNC_H*/
