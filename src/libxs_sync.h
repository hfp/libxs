/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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

#if defined(LIBXS_NOSYNC)
# undef _REENTRANT
#elif !defined(_REENTRANT)
# define _REENTRANT
#endif

#if defined(_REENTRANT)
# if (defined(_WIN32) && !defined(__GNUC__))
#   define LIBXS_TLS LIBXS_ATTRIBUTE(thread)
# elif defined(__GNUC__)
#   define LIBXS_TLS __thread
# elif defined(__cplusplus)
#   define LIBXS_TLS thread_local
# else
#   error Missing TLS support!
# endif
#else
# define LIBXS_TLS
#endif

#if defined(_WIN32) /*TODO*/
# define LIBXS_LOCK_TYPE HANDLE
# define LIBXS_LOCK_CONSTRUCT 0
# define LIBXS_LOCK_DESTROY(LOCK) CloseHandle(LOCK)
# define LIBXS_LOCK_ACQUIRE(LOCK) WaitForSingleObject(LOCK, INFINITE)
# define LIBXS_LOCK_TRYLOCK(LOCK) WaitForSingleObject(LOCK, 0)
# define LIBXS_LOCK_RELEASE(LOCK) ReleaseMutex(LOCK)
#else /* PThreads: include <pthread.h> */
# define LIBXS_LOCK_TYPE pthread_mutex_t
# define LIBXS_LOCK_CONSTRUCT PTHREAD_MUTEX_INITIALIZER
# define LIBXS_LOCK_DESTROY(LOCK) do { LIBXS_LOCK_TYPE libxs_lock_aquire_ = LOCK; pthread_mutex_destroy(&libxs_lock_aquire_); } while(0)
# define LIBXS_LOCK_ACQUIRE(LOCK) do { LIBXS_LOCK_TYPE libxs_lock_aquire_ = LOCK; pthread_mutex_lock   (&libxs_lock_aquire_); } while(0)
# define LIBXS_LOCK_TRYLOCK(LOCK) do { LIBXS_LOCK_TYPE libxs_lock_aquire_ = LOCK; pthread_mutex_trylock(&libxs_lock_aquire_); } while(0)
# define LIBXS_LOCK_RELEASE(LOCK) do { LIBXS_LOCK_TYPE libxs_lock_aquire_ = LOCK; pthread_mutex_unlock (&libxs_lock_aquire_); } while(0)
#endif

#if defined(__GNUC__)
# if !defined(LIBXS_GCCATOMICS)
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
#   define LIBXS_ATOMIC_LOAD(SRC, KIND) __atomic_load_n(&(SRC), KIND)
#   define LIBXS_ATOMIC_STORE(DST, VALUE, KIND) __atomic_store_n(&(DST), VALUE, KIND)
#   define LIBXS_ATOMIC_ADD_FETCH(DST, VALUE, KIND) /*DST = */__atomic_add_fetch(&(DST), VALUE, KIND)
# else
#   define LIBXS_ATOMIC_LOAD(SRC, KIND) __sync_or_and_fetch(&(SRC), 0)
#   define LIBXS_ATOMIC_STORE(DST, VALUE, KIND) { \
      /*const*/void* old = DST; \
      while (!__sync_bool_compare_and_swap(&(DST), old, VALUE)) old = DST; \
    }
#   define LIBXS_ATOMIC_STORE_ZERO(DST, KIND) { \
      /* use store side-effect of built-in (dummy assignment to mute warning) */ \
      void *const dummy = __sync_and_and_fetch(&(DST), 0); \
      LIBXS_UNUSED(dummy); \
    }
#   define LIBXS_ATOMIC_ADD_FETCH(DST, VALUE, KIND) /*DST = */__sync_add_and_fetch(&(DST), VALUE)
# endif
#elif (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(_WIN32) /*TODO*/
#   define LIBXS_ATOMIC_LOAD(SRC, KIND) SRC
#   define LIBXS_ATOMIC_STORE(DST, VALUE, KIND) DST = VALUE
#   define LIBXS_ATOMIC_ADD_FETCH(DST, VALUE, KIND) DST += VALUE
#else
#   define LIBXS_ATOMIC_LOAD(SRC, KIND) SRC
#   define LIBXS_ATOMIC_STORE(DST, VALUE, KIND) DST = VALUE
#   define LIBXS_ATOMIC_ADD_FETCH(DST, VALUE, KIND) DST += VALUE
#endif

#if !defined(LIBXS_ATOMIC_STORE_ZERO)
# define LIBXS_ATOMIC_STORE_ZERO(DST, KIND) LIBXS_ATOMIC_STORE(DST, 0, KIND)
#endif

#endif /*LIBXS_SYNC_H*/
