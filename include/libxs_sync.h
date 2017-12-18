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

#if !defined(LIBXS_TLS)
# if !defined(LIBXS_NO_SYNC) && !defined(LIBXS_NO_TLS)
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

#if !defined(LIBXS_NO_SYNC) && defined(LIBXS_GCCATOMICS)
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
# define LIBXS_SYNC_BARRIER __asm__ __volatile__ ("" ::: "memory")
# define LIBXS_SYNCHRONIZE __sync_synchronize()
    /* TODO: distinct implementation of LIBXS_ATOMIC_SYNC_* wrt LIBXS_GCCATOMICS */
# define LIBXS_ATOMIC_SYNC_CHECK(LOCK, VALUE) while ((VALUE) == (LOCK)); LIBXS_SYNC_PAUSE
# define LIBXS_ATOMIC_SYNC_SET(LOCK) do { LIBXS_ATOMIC_SYNC_CHECK(LOCK, 1); } while(0 != __sync_lock_test_and_set(&(LOCK), 1))
# define LIBXS_ATOMIC_SYNC_UNSET(LOCK) __sync_lock_release(&(LOCK))
#elif !defined(LIBXS_NO_SYNC) && defined(_WIN32) /* TODO: atomics */
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
# define LIBXS_SYNC_BARRIER _ReadWriteBarrier()
# define LIBXS_SYNCHRONIZE /* TODO */
#else
# define LIBXS_ATOMIC_LOAD LIBXS_NONATOMIC_LOAD
# define LIBXS_ATOMIC_STORE LIBXS_NONATOMIC_STORE
# define LIBXS_ATOMIC_ADD_FETCH LIBXS_NONATOMIC_ADD_FETCH
# define LIBXS_ATOMIC_SUB_FETCH LIBXS_NONATOMIC_SUB_FETCH
# define LIBXS_ATOMIC_SYNC_CHECK(LOCK, VALUE) LIBXS_UNUSED(LOCK)
# define LIBXS_ATOMIC_SYNC_SET(LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_ATOMIC_SYNC_UNSET(LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_SYNC_BARRIER
# define LIBXS_SYNCHRONIZE
#endif
#if !defined(LIBXS_ATOMIC_STORE_ZERO)
# define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXS_ATOMIC_STORE(DST_PTR, 0, KIND)
#endif
#if !defined(LIBXS_ATOMIC_SET) /* TODO */
# define LIBXS_ATOMIC_SET(DST, VALUE) (*(DST) = (VALUE))
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if !defined(LIBXS_NO_SYNC)
  /** Default lock-kind */
# define LIBXS_LOCK_DEFAULT LIBXS_LOCK_SPINLOCK
# if !defined(LIBXS_LOCK_SYSTEM)
#   define LIBXS_LOCK_SYSTEM
# endif
  /* OpenMP based locks need to stay disabled unless both
   * libxs and libxsext are built with OpenMP support.
   */
# if defined(_OPENMP) && defined(LIBXS_OMP)
#   include <omp.h>
#   define LIBXS_LOCK_SPINLOCK
#   define LIBXS_LOCK_MUTEX
#   define LIBXS_LOCK_RWLOCK
#   define LIBXS_LOCK_ACQUIRED(KIND) 1
#   define LIBXS_LOCK_ATTR_TYPE(KIND) const void*
#   define LIBXS_LOCK_ATTR_INIT(KIND, ATTR) LIBXS_UNUSED(ATTR)
#   define LIBXS_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXS_UNUSED(ATTR)
#   define LIBXS_LOCK_TYPE(KIND) omp_lock_t
#   define LIBXS_LOCK_INIT(KIND, LOCK, ATTR) omp_init_lock(LOCK)
#   define LIBXS_LOCK_DESTROY(KIND, LOCK) omp_destroy_lock(LOCK)
#   define LIBXS_LOCK_TRYLOCK(KIND, LOCK) omp_test_lock(LOCK)
#   define LIBXS_LOCK_ACQUIRE(KIND, LOCK) omp_set_lock(LOCK)
#   define LIBXS_LOCK_RELEASE(KIND, LOCK) omp_unset_lock(LOCK)
#   define LIBXS_LOCK_TRYREAD(KIND, LOCK) LIBXS_LOCK_TRYLOCK(KIND, LOCK)
#   define LIBXS_LOCK_ACQREAD(KIND, LOCK) LIBXS_LOCK_ACQUIRE(KIND, LOCK)
#   define LIBXS_LOCK_RELREAD(KIND, LOCK) LIBXS_LOCK_RELEASE(KIND, LOCK)
# else
    /* Lock type, initialization, destruction, (try-)lock, unlock, etc */
#   define LIBXS_LOCK_TYPE(KIND) LIBXS_CONCATENATE(LIBXS_LOCK_TYPE_, KIND)
#   define LIBXS_LOCK_INIT(KIND, LOCK, ATTR) LIBXS_CONCATENATE(LIBXS_LOCK_INIT_, KIND)(LOCK, ATTR)
#   define LIBXS_LOCK_DESTROY(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_DESTROY_, KIND)(LOCK)
#   define LIBXS_LOCK_TRYLOCK(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_TRYLOCK_, KIND)(LOCK)
#   define LIBXS_LOCK_ACQUIRE(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_ACQUIRE_, KIND)(LOCK)
#   define LIBXS_LOCK_RELEASE(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_RELEASE_, KIND)(LOCK)
#   define LIBXS_LOCK_TRYREAD(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_TRYREAD_, KIND)(LOCK)
#   define LIBXS_LOCK_ACQREAD(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_ACQREAD_, KIND)(LOCK)
#   define LIBXS_LOCK_RELREAD(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_RELREAD_, KIND)(LOCK)
    /* Attribute type, initialization, destruction */
#   define LIBXS_LOCK_ATTR_TYPE(KIND) LIBXS_CONCATENATE(LIBXS_LOCK_ATTR_TYPE_, KIND)
#   define LIBXS_LOCK_ATTR_INIT(KIND, ATTR) LIBXS_CONCATENATE(LIBXS_LOCK_ATTR_INIT_, KIND)(ATTR)
#   define LIBXS_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXS_CONCATENATE(LIBXS_LOCK_ATTR_DESTROY_, KIND)(ATTR)
    /* implementation */
#   if defined(_WIN32) \
    /* Cygwin's Pthread implementation appears to be broken; use Win32 */ \
    || defined(__CYGWIN__)
#     if defined(__CYGWIN__) /* hack: make SRW-locks available */
#       if defined(_WIN32_WINNT)
#         define LIBXS_WIN32_WINNT _WIN32_WINNT
#         undef _WIN32_WINNT
#         define _WIN32_WINNT (LIBXS_WIN32_WINNT | 0x0600)
#       else
#         define _WIN32_WINNT 0x0600
#       endif
#     endif
#     include <windows.h>
#     define LIBXS_LOCK_SPINLOCK spin
#     define LIBXS_LOCK_MUTEX mutex
#     define LIBXS_LOCK_RWLOCK rwlock
#     define LIBXS_LOCK_ACQUIRED(KIND) LIBXS_CONCATENATE(LIBXS_LOCK_ACQUIRED_, KIND)
      /* implementation spinlock */
#     define LIBXS_LOCK_ACQUIRED_spin TRUE
#     define LIBXS_LOCK_TYPE_spin CRITICAL_SECTION
#     define LIBXS_LOCK_INIT_spin(LOCK, ATTR) InitializeCriticalSection(LOCK)
#     define LIBXS_LOCK_DESTROY_spin(LOCK) DeleteCriticalSection((LIBXS_LOCK_TYPE_spin*)(LOCK))
#     define LIBXS_LOCK_TRYLOCK_spin(LOCK) TryEnterCriticalSection(LOCK)
#     define LIBXS_LOCK_ACQUIRE_spin(LOCK) EnterCriticalSection(LOCK)
#     define LIBXS_LOCK_RELEASE_spin(LOCK) LeaveCriticalSection(LOCK)
#     define LIBXS_LOCK_TRYREAD_spin(LOCK) LIBXS_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXS_LOCK_ACQREAD_spin(LOCK) LIBXS_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXS_LOCK_RELREAD_spin(LOCK) LIBXS_LOCK_RELEASE_spin(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_spin int
#     define LIBXS_LOCK_ATTR_INIT_spin(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_ATTR_DESTROY_spin(ATTR) LIBXS_UNUSED(ATTR)
      /* implementation mutex */
#     define LIBXS_LOCK_ACQUIRED_mutex WAIT_OBJECT_0
#     if defined(LIBXS_LOCK_SYSTEM)
#       define LIBXS_LOCK_TYPE_mutex HANDLE
#       define LIBXS_LOCK_INIT_mutex(LOCK, ATTR) (*(LOCK) = CreateMutex(*(ATTR), FALSE, NULL))
#       define LIBXS_LOCK_DESTROY_mutex(LOCK) CloseHandle(*(LOCK))
#       define LIBXS_LOCK_TRYLOCK_mutex(LOCK) WaitForSingleObject(*(LOCK), 0)
#       define LIBXS_LOCK_ACQUIRE_mutex(LOCK) WaitForSingleObject(*(LOCK), INFINITE)
#       define LIBXS_LOCK_RELEASE_mutex(LOCK) ReleaseMutex(*(LOCK))
#       define LIBXS_LOCK_TRYREAD_mutex(LOCK) LIBXS_LOCK_TRYLOCK_mutex(LOCK)
#       define LIBXS_LOCK_ACQREAD_mutex(LOCK) LIBXS_LOCK_ACQUIRE_mutex(LOCK)
#       define LIBXS_LOCK_RELREAD_mutex(LOCK) LIBXS_LOCK_RELEASE_mutex(LOCK)
#       define LIBXS_LOCK_ATTR_TYPE_mutex LPSECURITY_ATTRIBUTES
#       define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) (*(ATTR) = NULL)
#       define LIBXS_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXS_UNUSED(ATTR)
#     endif
      /* implementation rwlock */
#     define LIBXS_LOCK_ACQUIRED_rwlock TRUE
#     if defined(LIBXS_LOCK_SYSTEM)
#       define LIBXS_LOCK_TYPE_rwlock SRWLOCK
#       define LIBXS_LOCK_INIT_rwlock(LOCK, ATTR) InitializeSRWLock(LOCK)
#       define LIBXS_LOCK_DESTROY_rwlock(LOCK) LIBXS_UNUSED(LOCK)
#       define LIBXS_LOCK_TRYLOCK_rwlock(LOCK) TryAcquireSRWLockExclusive(LOCK)
#       define LIBXS_LOCK_ACQUIRE_rwlock(LOCK) AcquireSRWLockExclusive(LOCK)
#       define LIBXS_LOCK_RELEASE_rwlock(LOCK) ReleaseSRWLockExclusive(LOCK)
#       define LIBXS_LOCK_TRYREAD_rwlock(LOCK) TryAcquireSRWLockShared(LOCK)
#       define LIBXS_LOCK_ACQREAD_rwlock(LOCK) AcquireSRWLockShared(LOCK)
#       define LIBXS_LOCK_RELREAD_rwlock(LOCK) ReleaseSRWLockShared(LOCK)
#       define LIBXS_LOCK_ATTR_TYPE_rwlock int
#       define LIBXS_LOCK_ATTR_INIT_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#       define LIBXS_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#     endif
#   else
#     include <pthread.h>
#     if defined(__APPLE__) && defined(__MACH__)
#       define LIBXS_LOCK_SPINLOCK mutex
#     else
#       define LIBXS_LOCK_SPINLOCK spin
#     endif
#     define LIBXS_LOCK_MUTEX mutex
#     define LIBXS_LOCK_RWLOCK rwlock
#     define LIBXS_LOCK_ACQUIRED(KIND) 0
      /* implementation spinlock */
#     define LIBXS_LOCK_TYPE_spin pthread_spinlock_t
#     define LIBXS_LOCK_INIT_spin(LOCK, ATTR) LIBXS_EXPECT(0, pthread_spin_init(LOCK, *(ATTR)))
#     define LIBXS_LOCK_DESTROY_spin(LOCK) LIBXS_EXPECT(0, pthread_spin_destroy(LOCK))
#     define LIBXS_LOCK_TRYLOCK_spin(LOCK) pthread_spin_trylock(LOCK)
#     define LIBXS_LOCK_ACQUIRE_spin(LOCK) LIBXS_EXPECT(0, pthread_spin_lock(LOCK))
#     define LIBXS_LOCK_RELEASE_spin(LOCK) LIBXS_EXPECT(0, pthread_spin_unlock(LOCK))
#     define LIBXS_LOCK_TRYREAD_spin(LOCK) LIBXS_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXS_LOCK_ACQREAD_spin(LOCK) LIBXS_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXS_LOCK_RELREAD_spin(LOCK) LIBXS_LOCK_RELEASE_spin(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_spin int
#     define LIBXS_LOCK_ATTR_INIT_spin(ATTR) (*(ATTR) = 0)
#     define LIBXS_LOCK_ATTR_DESTROY_spin(ATTR) LIBXS_UNUSED(ATTR)
#     if defined(LIBXS_LOCK_SYSTEM)
        /* implementation mutex */
#       define LIBXS_LOCK_TYPE_mutex pthread_mutex_t
#       define LIBXS_LOCK_INIT_mutex(LOCK, ATTR) LIBXS_EXPECT(0, pthread_mutex_init(LOCK, ATTR))
#       define LIBXS_LOCK_DESTROY_mutex(LOCK) LIBXS_EXPECT(0, pthread_mutex_destroy(LOCK))
#       define LIBXS_LOCK_TRYLOCK_mutex(LOCK) pthread_mutex_trylock(LOCK)
#       define LIBXS_LOCK_ACQUIRE_mutex(LOCK) LIBXS_EXPECT(0, pthread_mutex_lock(LOCK))
#       define LIBXS_LOCK_RELEASE_mutex(LOCK) LIBXS_EXPECT(0, pthread_mutex_unlock(LOCK))
#       define LIBXS_LOCK_TRYREAD_mutex(LOCK) LIBXS_LOCK_TRYLOCK_mutex(LOCK)
#       define LIBXS_LOCK_ACQREAD_mutex(LOCK) LIBXS_LOCK_ACQUIRE_mutex(LOCK)
#       define LIBXS_LOCK_RELREAD_mutex(LOCK) LIBXS_LOCK_RELEASE_mutex(LOCK)
#       define LIBXS_LOCK_ATTR_TYPE_mutex pthread_mutexattr_t
#       if defined(NDEBUG)
#         define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) pthread_mutexattr_init(ATTR); \
                            pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_NORMAL)
#       else
#         define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) LIBXS_EXPECT(0, pthread_mutexattr_init(ATTR)); \
                        LIBXS_EXPECT(0, pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_ERRORCHECK))
#       endif
#       define LIBXS_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXS_EXPECT(0, pthread_mutexattr_destroy(ATTR))
        /* implementation rwlock */
#       define LIBXS_LOCK_TYPE_rwlock pthread_rwlock_t
#       define LIBXS_LOCK_INIT_rwlock(LOCK, ATTR) LIBXS_EXPECT(0, pthread_rwlock_init(LOCK, ATTR))
#       define LIBXS_LOCK_DESTROY_rwlock(LOCK) LIBXS_EXPECT(0, pthread_rwlock_destroy(LOCK))
#       define LIBXS_LOCK_TRYLOCK_rwlock(LOCK) pthread_rwlock_trywrlock(LOCK)
#       define LIBXS_LOCK_ACQUIRE_rwlock(LOCK) LIBXS_EXPECT(0, pthread_rwlock_wrlock(LOCK))
#       define LIBXS_LOCK_RELEASE_rwlock(LOCK) LIBXS_EXPECT(0, pthread_rwlock_unlock(LOCK))
#       define LIBXS_LOCK_TRYREAD_rwlock(LOCK) pthread_rwlock_tryrdlock(LOCK)
#       define LIBXS_LOCK_ACQREAD_rwlock(LOCK) LIBXS_EXPECT(0, pthread_rwlock_rdlock(LOCK))
#       define LIBXS_LOCK_RELREAD_rwlock(LOCK) LIBXS_LOCK_RELEASE_rwlock(LOCK)
#       define LIBXS_LOCK_ATTR_TYPE_rwlock pthread_rwlockattr_t
#       define LIBXS_LOCK_ATTR_INIT_rwlock(ATTR) LIBXS_EXPECT(0, pthread_rwlockattr_init(ATTR))
#       define LIBXS_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXS_EXPECT(0, pthread_rwlockattr_destroy(ATTR))
#     endif
#   endif
# endif
# if !defined(LIBXS_LOCK_SYSTEM)
#   define LIBXS_LOCK_TYPE_mutex libxs_mutex*
#   define LIBXS_LOCK_INIT_mutex(LOCK, ATTR) (*(LOCK) = libxs_mutex_create())
#   define LIBXS_LOCK_DESTROY_mutex(LOCK) libxs_mutex_destroy(*(LOCK))
#   define LIBXS_LOCK_TRYLOCK_mutex(LOCK) libxs_mutex_trylock(*(LOCK))
#   define LIBXS_LOCK_ACQUIRE_mutex(LOCK) libxs_mutex_acquire(*(LOCK))
#   define LIBXS_LOCK_RELEASE_mutex(LOCK) libxs_mutex_release(*(LOCK))
#   define LIBXS_LOCK_TRYREAD_mutex(LOCK) LIBXS_LOCK_TRYLOCK_mutex(LOCK)
#   define LIBXS_LOCK_ACQREAD_mutex(LOCK) LIBXS_LOCK_ACQUIRE_mutex(LOCK)
#   define LIBXS_LOCK_RELREAD_mutex(LOCK) LIBXS_LOCK_RELEASE_mutex(LOCK)
#   define LIBXS_LOCK_ATTR_TYPE_mutex int
#   define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) LIBXS_UNUSED(ATTR)
#   define LIBXS_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXS_UNUSED(ATTR)

#   define LIBXS_LOCK_TYPE_rwlock libxs_rwlock*
#   define LIBXS_LOCK_INIT_rwlock(LOCK, ATTR) (*(LOCK) = libxs_rwlock_create())
#   define LIBXS_LOCK_DESTROY_rwlock(LOCK) libxs_rwlock_destroy(*(LOCK))
#   define LIBXS_LOCK_TRYLOCK_rwlock(LOCK) libxs_rwlock_trylock(*(LOCK))
#   define LIBXS_LOCK_ACQUIRE_rwlock(LOCK) libxs_rwlock_acquire(*(LOCK))
#   define LIBXS_LOCK_RELEASE_rwlock(LOCK) libxs_rwlock_release(*(LOCK))
#   define LIBXS_LOCK_TRYREAD_rwlock(LOCK) libxs_rwlock_tryread(*(LOCK))
#   define LIBXS_LOCK_ACQREAD_rwlock(LOCK) libxs_rwlock_acqread(*(LOCK))
#   define LIBXS_LOCK_RELREAD_rwlock(LOCK) libxs_rwlock_relread(*(LOCK))
#   define LIBXS_LOCK_ATTR_TYPE_rwlock int
#   define LIBXS_LOCK_ATTR_INIT_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#   define LIBXS_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXS_UNUSED(ATTR)
# endif
#else
# define LIBXS_LOCK_SPINLOCK spin
# define LIBXS_LOCK_MUTEX mutex
# define LIBXS_LOCK_RWLOCK rwlock
# define LIBXS_LOCK_ACQUIRED(KIND) 0
# define LIBXS_LOCK_ATTR_TYPE(KIND) int
# define LIBXS_LOCK_ATTR_INIT(KIND, ATTR) LIBXS_UNUSED(ATTR)
# define LIBXS_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXS_UNUSED(ATTR)
# define LIBXS_LOCK_TYPE(KIND) int
# define LIBXS_LOCK_INIT(KIND, LOCK, ATTR) LIBXS_UNUSED(LOCK); LIBXS_UNUSED(ATTR)
# define LIBXS_LOCK_DESTROY(KIND, LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_TRYLOCK(KIND, LOCK) LIBXS_LOCK_ACQUIRED(KIND)
# define LIBXS_LOCK_ACQUIRE(KIND, LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_RELEASE(KIND, LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_TRYREAD(KIND, LOCK) LIBXS_LOCK_TRYLOCK(KIND, LOCK)
# define LIBXS_LOCK_ACQREAD(KIND, LOCK) LIBXS_LOCK_ACQUIRE(KIND, LOCK)
# define LIBXS_LOCK_RELREAD(KIND, LOCK) LIBXS_LOCK_RELEASE(KIND, LOCK)
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
/** Destroy the resources associated with this barrier. */
LIBXS_API void libxs_barrier_destroy(const libxs_barrier* barrier);
/** DEPRECATED: use libxs_barrier_destroy instead. */
#define libxs_barrier_release libxs_barrier_destroy

/** Mutex, which is eventually not based on LIBXS_LOCK_TYPE(LIBXS_LOCK_MUTEX). */
typedef struct LIBXS_RETARGETABLE libxs_mutex libxs_mutex;
LIBXS_API libxs_mutex* libxs_mutex_create(void);
LIBXS_API void libxs_mutex_destroy(const libxs_mutex* mutex);
LIBXS_API int libxs_mutex_trylock(libxs_mutex* mutex);
LIBXS_API void libxs_mutex_acquire(libxs_mutex* mutex);
LIBXS_API void libxs_mutex_release(libxs_mutex* mutex);

/** RW-lock, which is eventually not based on LIBXS_LOCK_TYPE(LIBXS_LOCK_RWLOCK). */
typedef struct LIBXS_RETARGETABLE libxs_rwlock libxs_rwlock;
LIBXS_API libxs_rwlock* libxs_rwlock_create(void);
LIBXS_API void libxs_rwlock_destroy(const libxs_rwlock* rwlock);
LIBXS_API int libxs_rwlock_trylock(libxs_rwlock* mutex);
LIBXS_API void libxs_rwlock_acquire(libxs_rwlock* rwlock);
LIBXS_API void libxs_rwlock_release(libxs_rwlock* rwlock);
LIBXS_API int libxs_rwlock_tryread(libxs_rwlock* mutex);
LIBXS_API void libxs_rwlock_acqread(libxs_rwlock* rwlock);
LIBXS_API void libxs_rwlock_relread(libxs_rwlock* rwlock);

/** Utility function to receive the process ID of the calling process. */
LIBXS_API unsigned int libxs_get_pid(void);
/**
 * Utility function to receive a Thread-ID (TID) for the calling thread.
 * The TID is not related to a specific threading runtime. TID=0 may not
 * represent the main thread. TIDs are zero-based and consecutive numbers.
 */
LIBXS_API unsigned int libxs_get_tid(void);

#endif /*LIBXS_SYNC_H*/
