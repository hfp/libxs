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

#include "libxs_main.h"

#if !defined(LIBXS_SYNC_FUTEX) && defined(__linux__)
# define LIBXS_SYNC_FUTEX
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_WIN32)
# include <windows.h>
# include <process.h>
#else
# if defined(LIBXS_SYNC_FUTEX) && defined(__linux__)
#   include <linux/futex.h>
# endif
# include <unistd.h>
# include <time.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE internal_sync_core_tag { /* per-core */
  uint8_t id;
  volatile uint8_t core_sense;
  volatile uint8_t* thread_senses;
  volatile uint8_t* my_flags[2];
  uint8_t** partner_flags[2];
  uint8_t parity;
  uint8_t sense;
} internal_sync_core_tag;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE internal_sync_thread_tag { /* per-thread */
  int core_tid;
  internal_sync_core_tag *core;
} internal_sync_thread_tag;

struct LIBXS_RETARGETABLE libxs_barrier {
  internal_sync_core_tag** cores;
  internal_sync_thread_tag** threads;
  int ncores, nthreads_per_core;
  int nthreads, ncores_log2;
  /* internal counter type which is guaranteed to be atomic when using certain methods */
  volatile int threads_waiting;
  /* thread-safety during initialization */
  volatile uint8_t init_done;
};


LIBXS_API libxs_barrier* libxs_barrier_create(int ncores, int nthreads_per_core)
{
  libxs_barrier *const barrier = (libxs_barrier*)malloc(sizeof(libxs_barrier));
#if defined(LIBXS_NO_SYNC)
  LIBXS_UNUSED(ncores); LIBXS_UNUSED(nthreads_per_core);
#else
  if (NULL != barrier && 1 < ncores && 1 <= nthreads_per_core) {
    barrier->ncores = ncores;
    barrier->ncores_log2 = LIBXS_LOG2((ncores << 1) - 1);
    barrier->nthreads_per_core = nthreads_per_core;
    barrier->nthreads = ncores * nthreads_per_core;
    barrier->threads = (internal_sync_thread_tag**)libxs_aligned_malloc(
      barrier->nthreads * sizeof(internal_sync_thread_tag*), LIBXS_CACHELINE);
    barrier->cores = (internal_sync_core_tag**)libxs_aligned_malloc(
      barrier->ncores * sizeof(internal_sync_core_tag*), LIBXS_CACHELINE);
    barrier->threads_waiting = barrier->nthreads; /* atomic */
    barrier->init_done = 0; /* false */    
  }
  else
#endif
  if (NULL != barrier) {
    barrier->nthreads = 1;
  }
  return barrier;
}


LIBXS_API void libxs_barrier_init(libxs_barrier* barrier, int tid)
{
#if defined(LIBXS_NO_SYNC)
  LIBXS_UNUSED(barrier); LIBXS_UNUSED(tid);
#else
  if (NULL != barrier && 1 < barrier->nthreads) {
    const int cid = tid / barrier->nthreads_per_core; /* this thread's core ID */
    internal_sync_core_tag* core = 0;
    int i;
    internal_sync_thread_tag* thread;

    /* we only initialize the barrier once */
    if (barrier->init_done == 2) {
      return;
    }

    /* allocate per-thread structure */
    thread = (internal_sync_thread_tag*)libxs_aligned_malloc(
      sizeof(internal_sync_thread_tag), LIBXS_CACHELINE);
    barrier->threads[tid] = thread;
    thread->core_tid = tid - (barrier->nthreads_per_core * cid); /* mod */
    /* each core's thread 0 does all the allocations */
    if (0 == thread->core_tid) {
      core = (internal_sync_core_tag*)libxs_aligned_malloc(
        sizeof(internal_sync_core_tag), LIBXS_CACHELINE);
      core->id = (uint8_t)cid;
      core->core_sense = 1;

      core->thread_senses = (uint8_t*)libxs_aligned_malloc(
        barrier->nthreads_per_core * sizeof(uint8_t), LIBXS_CACHELINE);
      for (i = 0; i < barrier->nthreads_per_core; ++i) core->thread_senses[i] = 1;

      for (i = 0; i < 2; ++i) {
        core->my_flags[i] = (uint8_t*)libxs_aligned_malloc(
          barrier->ncores_log2 * sizeof(uint8_t) * LIBXS_CACHELINE,
          LIBXS_CACHELINE);
        core->partner_flags[i] = (uint8_t**)libxs_aligned_malloc(
          barrier->ncores_log2 * sizeof(uint8_t*),
          LIBXS_CACHELINE);
      }

      core->parity = 0;
      core->sense = 1;
      barrier->cores[cid] = core;
    }

    /* barrier to let all the allocations complete */
    if (0 == LIBXS_ATOMIC_SUB_FETCH(&barrier->threads_waiting, 1, LIBXS_ATOMIC_RELAXED)) {
      barrier->threads_waiting = barrier->nthreads; /* atomic */
      barrier->init_done = 1; /* true */
    }
    else {
      while (0/*false*/ == barrier->init_done);
    }

    /* set required per-thread information */
    thread->core = barrier->cores[cid];

    /* each core's thread 0 completes setup */
    if (0 == thread->core_tid) {
      int di;
      for (i = di = 0; i < barrier->ncores_log2; ++i, di += LIBXS_CACHELINE) {
        /* find dissemination partner and link to it */
        const int dissem_cid = (cid + (1 << i)) % barrier->ncores;
        assert(0 != core); /* initialized under the same condition; see above */
        core->my_flags[0][di] = core->my_flags[1][di] = 0;
        core->partner_flags[0][i] = (uint8_t*)&barrier->cores[dissem_cid]->my_flags[0][di];
        core->partner_flags[1][i] = (uint8_t*)&barrier->cores[dissem_cid]->my_flags[1][di];
      }
    }

    /* barrier to let initialization complete */
    if (0 == LIBXS_ATOMIC_SUB_FETCH(&barrier->threads_waiting, 1, LIBXS_ATOMIC_RELAXED)) {
      barrier->threads_waiting = barrier->nthreads; /* atomic */
      barrier->init_done = 2;
    }
    else {
      while (2 != barrier->init_done);
    }    
  }
#endif
}


LIBXS_API LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
void libxs_barrier_wait(libxs_barrier* barrier, int tid)
{
#if defined(LIBXS_NO_SYNC)
  LIBXS_UNUSED(barrier); LIBXS_UNUSED(tid);
#else
  if (NULL != barrier && 1 < barrier->nthreads) {
    internal_sync_thread_tag *const thread = barrier->threads[tid];
    internal_sync_core_tag *const core = thread->core;

    /* first let's execute a memory fence */
    LIBXS_ATOMIC_SYNC(LIBXS_ATOMIC_SEQ_CST);

    /* first signal this thread's arrival */
    core->thread_senses[thread->core_tid] = (uint8_t)(0 == core->thread_senses[thread->core_tid] ? 1 : 0);

    /* each core's thread 0 syncs across cores */
    if (0 == thread->core_tid) {
      int i;
      /* wait for the core's remaining threads */
      for (i = 1; i < barrier->nthreads_per_core; ++i) {
        uint8_t core_sense = core->core_sense, thread_sense = core->thread_senses[i];
        while (core_sense == thread_sense) { /* avoid evaluation in unspecified order */
          LIBXS_SYNC_PAUSE;
          core_sense = core->core_sense;
          thread_sense = core->thread_senses[i];
        }
      }

      if (1 < barrier->ncores) {
        int di;
# if defined(__MIC__)
        /* cannot use LIBXS_ALIGNED since attribute may not apply to local non-static arrays */
        uint8_t sendbuffer[LIBXS_CACHELINE+LIBXS_CACHELINE-1];
        uint8_t *const sendbuf = LIBXS_ALIGN(sendbuffer, LIBXS_CACHELINE);
        __m512d m512d;
        _mm_prefetch((const char*)core->partner_flags[core->parity][0], _MM_HINT_ET1);
        sendbuf[0] = core->sense;
        m512d = LIBXS_INTRINSICS_MM512_LOAD_PD(sendbuf);
# endif

        for (i = di = 0; i < barrier->ncores_log2 - 1; ++i, di += LIBXS_CACHELINE) {
# if defined(__MIC__)
          _mm_prefetch((const char*)core->partner_flags[core->parity][i+1], _MM_HINT_ET1);
          _mm512_storenrngo_pd(core->partner_flags[core->parity][i], m512d);
# else
          *core->partner_flags[core->parity][i] = core->sense;
# endif
          while (core->my_flags[core->parity][di] != core->sense) LIBXS_SYNC_PAUSE;
        }

# if defined(__MIC__)
        _mm512_storenrngo_pd(core->partner_flags[core->parity][i], m512d);
# else
        *core->partner_flags[core->parity][i] = core->sense;
# endif
        while (core->my_flags[core->parity][di] != core->sense) LIBXS_SYNC_PAUSE;
        if (1 == core->parity) {
          core->sense = (uint8_t)(0 == core->sense ? 1 : 0);
        }
        core->parity = (uint8_t)(1 - core->parity);
      }

      /* wake up the core's remaining threads */
      core->core_sense = core->thread_senses[0];
    }
    else { /* other threads wait for cross-core sync to complete */
      uint8_t core_sense = core->core_sense, thread_sense = core->thread_senses[thread->core_tid];
      while (core_sense != thread_sense) { /* avoid evaluation in unspecified order */
        LIBXS_SYNC_PAUSE;
        core_sense = core->core_sense;
        thread_sense = core->thread_senses[thread->core_tid];
      }
    }
  }
#endif
}


LIBXS_API void libxs_barrier_destroy(const libxs_barrier* barrier)
{
#if !defined(LIBXS_NO_SYNC)
  if (NULL != barrier && 1 < barrier->nthreads) {
    if (2 == barrier->init_done) {
      int i;
      for (i = 0; i < barrier->ncores; ++i) {
        int j;
        libxs_free((const void*)barrier->cores[i]->thread_senses);
        for (j = 0; j < 2; ++j) {
          libxs_free((const void*)barrier->cores[i]->my_flags[j]);
          libxs_free(barrier->cores[i]->partner_flags[j]);
        }
        libxs_free(barrier->cores[i]);
      }
      for (i = 0; i < barrier->nthreads; ++i) {
        libxs_free(barrier->threads[i]);
      }
    }
    libxs_free(barrier->threads);
    libxs_free(barrier->cores);
  }
#endif
  free((libxs_barrier*)barrier);
}


#if !defined(LIBXS_NO_SYNC)
enum {
  INTERNAL_SYNC_LOCK_FREE = 0,
  INTERNAL_SYNC_LOCK_LOCKED = 1,
  INTERNAL_SYNC_LOCK_CONTESTED = 2,
  INTERNAL_SYNC_RWLOCK_READINC = 0x10000/*(USHRT_MAX+1)*/,
  INTERNAL_SYNC_FUTEX = 202
};
#endif


#if defined(LIBXS_LOCK_SYSTEM_SPINLOCK) && defined(LIBXS_SYNC_SYSTEM)
typedef LIBXS_LOCK_TYPE(LIBXS_LOCK_SPINLOCK) libxs_spinlock_state;
#else
typedef unsigned int libxs_spinlock_state;
#endif

struct LIBXS_RETARGETABLE libxs_spinlock {
#if defined(LIBXS_LOCK_SYSTEM_SPINLOCK) && defined(LIBXS_SYNC_SYSTEM)
  libxs_spinlock_state impl;
#else
  volatile libxs_spinlock_state state;
#endif
};


LIBXS_API libxs_spinlock* libxs_spinlock_create(void)
{
  libxs_spinlock *const result = (libxs_spinlock*)malloc(sizeof(libxs_spinlock));
  if (0 != result) {
#if defined(LIBXS_LOCK_SYSTEM_SPINLOCK) && defined(LIBXS_SYNC_SYSTEM)
    LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK_SPINLOCK) attr;
    LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK_SPINLOCK, &attr);
    LIBXS_LOCK_INIT(LIBXS_LOCK_SPINLOCK, &result->impl, &attr);
    LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK_SPINLOCK, &attr);
#elif !defined(LIBXS_NO_SYNC)
    result->state = INTERNAL_SYNC_LOCK_FREE;
#endif
  }
  return result;
}


LIBXS_API void libxs_spinlock_destroy(const libxs_spinlock* spinlock)
{
#if defined(LIBXS_LOCK_SYSTEM_SPINLOCK) && defined(LIBXS_SYNC_SYSTEM)
  LIBXS_LOCK_DESTROY(LIBXS_LOCK_SPINLOCK, (LIBXS_LOCK_TYPE(LIBXS_LOCK_SPINLOCK)*)&spinlock->impl);
#endif
  free((libxs_spinlock*)spinlock);
}


LIBXS_API int libxs_spinlock_trylock(libxs_spinlock* spinlock)
{
#if !defined(LIBXS_NO_SYNC)
# if defined(LIBXS_LOCK_SYSTEM_SPINLOCK) && defined(LIBXS_SYNC_SYSTEM)
  assert(0 != spinlock);
  return LIBXS_LOCK_TRYLOCK(LIBXS_LOCK_SPINLOCK, &spinlock->impl);
# elif 0
  /*const*/ libxs_spinlock_state lock_free = INTERNAL_SYNC_LOCK_FREE;
  assert(0 != spinlock);
  return 0/*false*/ == LIBXS_ATOMIC_CMPSWP(&spinlock->state, lock_free, INTERNAL_SYNC_LOCK_LOCKED, LIBXS_ATOMIC_RELAXED)
    ? (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_SPINLOCK) + 1) /* not acquired */
    : (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_SPINLOCK));
# else
  return LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_SPINLOCK) + !LIBXS_ATOMIC_TRYLOCK(&spinlock->state, LIBXS_ATOMIC_RELAXED);
# endif
#else
  LIBXS_UNUSED(spinlock);
  return LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_SPINLOCK);
#endif
}


LIBXS_API void libxs_spinlock_acquire(libxs_spinlock* spinlock)
{
#if !defined(LIBXS_NO_SYNC)
# if defined(LIBXS_LOCK_SYSTEM_SPINLOCK) && defined(LIBXS_SYNC_SYSTEM)
  assert(0 != spinlock);
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_SPINLOCK, &spinlock->impl);
# else
  LIBXS_SYNC_CYCLE_DECL(counter);
  assert(0 != spinlock);
  for (;;) {
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&spinlock->state, 1, LIBXS_ATOMIC_RELAXED)) break;
    while (INTERNAL_SYNC_LOCK_FREE != spinlock->state) LIBXS_SYNC_CYCLE(counter, LIBXS_SYNC_NPAUSE);
  }
  LIBXS_ATOMIC_SYNC(LIBXS_ATOMIC_SEQ_CST);
# endif
#else
  LIBXS_UNUSED(spinlock);
#endif
}


LIBXS_API void libxs_spinlock_release(libxs_spinlock* spinlock)
{
#if !defined(LIBXS_NO_SYNC)
  assert(0 != spinlock);
# if defined(LIBXS_LOCK_SYSTEM_SPINLOCK) && defined(LIBXS_SYNC_SYSTEM)
  LIBXS_LOCK_RELEASE(LIBXS_LOCK_SPINLOCK, &spinlock->impl);
# else
  LIBXS_ATOMIC_SYNC(LIBXS_ATOMIC_SEQ_CST);
  spinlock->state = INTERNAL_SYNC_LOCK_FREE;
# endif
#else
  LIBXS_UNUSED(spinlock);
#endif
}


#if defined(LIBXS_LOCK_SYSTEM_MUTEX) && defined(LIBXS_SYNC_SYSTEM)
typedef LIBXS_LOCK_TYPE(LIBXS_LOCK_MUTEX) libxs_mutex_state;
#elif defined(LIBXS_SYNC_FUTEX) && defined(__linux__)
typedef int libxs_mutex_state;
#else
typedef char libxs_mutex_state;
#endif

struct LIBXS_RETARGETABLE libxs_mutex {
#if defined(LIBXS_LOCK_SYSTEM_MUTEX) && defined(LIBXS_SYNC_SYSTEM)
  LIBXS_LOCK_TYPE(LIBXS_LOCK_MUTEX) impl;
#else
  volatile libxs_mutex_state state;
#endif
};


LIBXS_API libxs_mutex* libxs_mutex_create(void)
{
  libxs_mutex *const result = (libxs_mutex*)malloc(sizeof(libxs_mutex));
  if (0 != result) {
#if defined(LIBXS_LOCK_SYSTEM_MUTEX) && defined(LIBXS_SYNC_SYSTEM)
    LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK_MUTEX) attr;
    LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK_MUTEX, &attr);
    LIBXS_LOCK_INIT(LIBXS_LOCK_MUTEX, &result->impl, &attr);
    LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK_MUTEX, &attr);
#elif !defined(LIBXS_NO_SYNC)
    result->state = INTERNAL_SYNC_LOCK_FREE;
#endif
  }
  return result;
}


LIBXS_API void libxs_mutex_destroy(const libxs_mutex* mutex)
{
#if defined(LIBXS_LOCK_SYSTEM_MUTEX) && defined(LIBXS_SYNC_SYSTEM)
  LIBXS_LOCK_DESTROY(LIBXS_LOCK_MUTEX, (LIBXS_LOCK_TYPE(LIBXS_LOCK_MUTEX)*)&mutex->impl);
#endif
  free((libxs_mutex*)mutex);
}


LIBXS_API int libxs_mutex_trylock(libxs_mutex* mutex)
{
#if !defined(LIBXS_NO_SYNC)
  assert(0 != mutex);
# if defined(LIBXS_LOCK_SYSTEM_MUTEX) && defined(LIBXS_SYNC_SYSTEM)
  return LIBXS_LOCK_TRYLOCK(LIBXS_LOCK_MUTEX, &mutex->impl);
# else
  return LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_MUTEX) + !LIBXS_ATOMIC_TRYLOCK(&mutex->state, LIBXS_ATOMIC_RELAXED);
# endif
#else
  LIBXS_UNUSED(mutex);
  return LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_MUTEX);
#endif
}


LIBXS_API void libxs_mutex_acquire(libxs_mutex* mutex)
{
#if !defined(LIBXS_NO_SYNC)
# if defined(LIBXS_LOCK_SYSTEM_MUTEX) && defined(LIBXS_SYNC_SYSTEM)
  assert(0 != mutex);
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_MUTEX, &mutex->impl);
# else
#   if defined(_WIN32)
  LIBXS_SYNC_CYCLE_DECL(counter);
  assert(0 != mutex);
  while (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_MUTEX) != libxs_mutex_trylock(mutex)) {
    while (0 != (mutex->state & 1)) LIBXS_SYNC_CYCLE(counter, LIBXS_SYNC_NPAUSE);
  }
#   else
  libxs_mutex_state lock_free = INTERNAL_SYNC_LOCK_FREE, lock_state = INTERNAL_SYNC_LOCK_LOCKED;
  LIBXS_SYNC_CYCLE_DECL(counter);
  assert(0 != mutex);
  while (0/*false*/ == LIBXS_ATOMIC_CMPSWP(&mutex->state, lock_free, lock_state, LIBXS_ATOMIC_RELAXED)) {
    libxs_mutex_state state;
    for (state = mutex->state; INTERNAL_SYNC_LOCK_FREE != state; state = mutex->state) {
#     if defined(LIBXS_SYNC_FUTEX) && defined(__linux__)
      LIBXS_SYNC_CYCLE_ELSE(counter, LIBXS_SYNC_NPAUSE, {
        /*const*/ libxs_mutex_state state_locked = INTERNAL_SYNC_LOCK_LOCKED;
        if (INTERNAL_SYNC_LOCK_LOCKED != state || LIBXS_ATOMIC_CMPSWP(&mutex->state,
          state_locked, INTERNAL_SYNC_LOCK_CONTESTED, LIBXS_ATOMIC_RELAXED))
        {
          syscall(INTERNAL_SYNC_FUTEX, &mutex->state, FUTEX_WAIT, INTERNAL_SYNC_LOCK_CONTESTED, NULL, NULL, 0);
          lock_state = INTERNAL_SYNC_LOCK_CONTESTED;
        }}
      );
      break;
#     else
      LIBXS_SYNC_CYCLE(counter, LIBXS_SYNC_NPAUSE);
#     endif
    }
    lock_free = INTERNAL_SYNC_LOCK_FREE;
  }
#   endif
# endif
#else
  LIBXS_UNUSED(mutex);
#endif
}


LIBXS_API void libxs_mutex_release(libxs_mutex* mutex)
{
#if !defined(LIBXS_NO_SYNC)
  assert(0 != mutex);
# if defined(LIBXS_LOCK_SYSTEM_MUTEX) && defined(LIBXS_SYNC_SYSTEM)
  LIBXS_LOCK_RELEASE(LIBXS_LOCK_MUTEX, &mutex->impl);
# else
  LIBXS_ATOMIC_SYNC(LIBXS_ATOMIC_SEQ_CST);
#   if defined(LIBXS_SYNC_FUTEX) && defined(__linux__)
  if (INTERNAL_SYNC_LOCK_CONTESTED == LIBXS_ATOMIC_FETCH_SUB(&mutex->state, 1, LIBXS_ATOMIC_RELAXED)) {
    mutex->state = INTERNAL_SYNC_LOCK_FREE;
    syscall(INTERNAL_SYNC_FUTEX, &mutex->state, FUTEX_WAKE, 1, NULL, NULL, 0);
  }
#   else
  mutex->state = INTERNAL_SYNC_LOCK_FREE;
#   endif
# endif
#else
  LIBXS_UNUSED(mutex);
#endif
}


#if !defined(LIBXS_NO_SYNC) && !(defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM))
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE internal_sync_counter {
  struct {
    uint16_t writer;
    uint16_t reader;
  } kind;
  uint32_t bits;
} internal_sync_counter;
#endif


LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_rwlock {
#if !defined(LIBXS_NO_SYNC)
# if defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM)
  LIBXS_LOCK_TYPE(LIBXS_LOCK_RWLOCK) impl;
# else
  volatile internal_sync_counter completions;
  volatile internal_sync_counter requests;
# endif
#else
  int dummy;
#endif
};


LIBXS_API libxs_rwlock* libxs_rwlock_create(void)
{
  libxs_rwlock *const result = (libxs_rwlock*)malloc(sizeof(libxs_rwlock));
  if (0 != result) {
#if !defined(LIBXS_NO_SYNC)
# if defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM)
    LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK_RWLOCK) attr;
    LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK_RWLOCK, &attr);
    LIBXS_LOCK_INIT(LIBXS_LOCK_RWLOCK, &result->impl, &attr);
    LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK_RWLOCK, &attr);
# else
    memset((void*)&result->completions, 0, sizeof(internal_sync_counter));
    memset((void*)&result->requests, 0, sizeof(internal_sync_counter));
# endif
#else
    memset(result, 0, sizeof(libxs_rwlock));
#endif
  }
  return result;
}


LIBXS_API void libxs_rwlock_destroy(const libxs_rwlock* rwlock)
{
#if defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM)
  LIBXS_LOCK_DESTROY(LIBXS_LOCK_RWLOCK, (LIBXS_LOCK_TYPE(LIBXS_LOCK_RWLOCK)*)&rwlock->impl);
#endif
  free((libxs_rwlock*)rwlock);
}


#if !defined(LIBXS_NO_SYNC) && !(defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM))
LIBXS_API_INLINE int internal_rwlock_trylock(libxs_rwlock* rwlock, internal_sync_counter* prev)
{
  internal_sync_counter next;
  assert(0 != rwlock && 0 != prev);
  do {
    prev->bits = rwlock->requests.bits;
    next.bits = prev->bits;
    ++next.kind.writer;
  }
  while (0/*false*/ == LIBXS_ATOMIC_CMPSWP(&rwlock->requests.bits, prev->bits, next.bits, LIBXS_ATOMIC_RELAXED));
  return rwlock->completions.bits != prev->bits
    ? (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK) + 1) /* not acquired */
    : (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK));
}
#endif


LIBXS_API int libxs_rwlock_trylock(libxs_rwlock* rwlock)
{
#if !defined(LIBXS_NO_SYNC)
# if defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM)
  assert(0 != rwlock);
  return LIBXS_LOCK_TRYLOCK(LIBXS_LOCK_RWLOCK, &rwlock->impl);
# else
  internal_sync_counter prev;
  return internal_rwlock_trylock(rwlock, &prev);
# endif
#else
  LIBXS_UNUSED(rwlock);
  return LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK);
#endif
}


LIBXS_API void libxs_rwlock_acquire(libxs_rwlock* rwlock)
{
#if !defined(LIBXS_NO_SYNC)
# if defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM)
  assert(0 != rwlock);
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK_RWLOCK, &rwlock->impl);
# else
  internal_sync_counter prev;
  if (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK) != internal_rwlock_trylock(rwlock, &prev)) {
    LIBXS_SYNC_CYCLE_DECL(counter);
    while (rwlock->completions.bits != prev.bits) LIBXS_SYNC_CYCLE(counter, LIBXS_SYNC_NPAUSE);
  }
# endif
#else
  LIBXS_UNUSED(rwlock);
#endif
}


LIBXS_API void libxs_rwlock_release(libxs_rwlock* rwlock)
{
#if !defined(LIBXS_NO_SYNC)
  assert(0 != rwlock);
# if defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM)
  LIBXS_LOCK_RELEASE(LIBXS_LOCK_RWLOCK, &rwlock->impl);
# elif defined(_WIN32)
  _InterlockedExchangeAdd16((volatile short*)&rwlock->completions.kind.writer, 1);
# else
  LIBXS_ATOMIC_ADD_FETCH(&rwlock->completions.kind.writer, 1, LIBXS_ATOMIC_SEQ_CST);
# endif
#else
  LIBXS_UNUSED(rwlock);
#endif
}


#if !defined(LIBXS_NO_SYNC) && !(defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM))
LIBXS_API_INLINE int internal_rwlock_tryread(libxs_rwlock* rwlock, internal_sync_counter* prev)
{
#if !defined(LIBXS_NO_SYNC)
  assert(0 != rwlock && 0 != prev);
# if defined(_WIN32)
  prev->bits = InterlockedExchangeAdd((volatile LONG*)&rwlock->requests.bits, INTERNAL_SYNC_RWLOCK_READINC);
# else
  prev->bits = LIBXS_ATOMIC_FETCH_ADD(&rwlock->requests.bits, INTERNAL_SYNC_RWLOCK_READINC, LIBXS_ATOMIC_SEQ_CST);
# endif
  return rwlock->completions.kind.writer != prev->kind.writer
    ? (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK) + 1) /* not acquired */
    : (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK));
#else
  LIBXS_UNUSED(rwlock); LIBXS_UNUSED(prev);
  return LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK);
#endif
}
#endif


LIBXS_API int libxs_rwlock_tryread(libxs_rwlock* rwlock)
{
#if !defined(LIBXS_NO_SYNC)
# if defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM)
  assert(0 != rwlock);
  return LIBXS_LOCK_TRYREAD(LIBXS_LOCK_RWLOCK, &rwlock->impl);
# else
  internal_sync_counter prev;
  return internal_rwlock_tryread(rwlock, &prev);
# endif
#else
  LIBXS_UNUSED(rwlock);
  return LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK);
#endif
}


LIBXS_API void libxs_rwlock_acqread(libxs_rwlock* rwlock)
{
#if !defined(LIBXS_NO_SYNC)
# if defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM)
  assert(0 != rwlock);
  LIBXS_LOCK_ACQREAD(LIBXS_LOCK_RWLOCK, &rwlock->impl);
# else
  internal_sync_counter prev;
  if (LIBXS_LOCK_ACQUIRED(LIBXS_LOCK_RWLOCK) != internal_rwlock_tryread(rwlock, &prev)) {
    LIBXS_SYNC_CYCLE_DECL(counter);
    while (rwlock->completions.kind.writer != prev.kind.writer) LIBXS_SYNC_CYCLE(counter, LIBXS_SYNC_NPAUSE);
  }
# endif
#else
  LIBXS_UNUSED(rwlock);
#endif
}


LIBXS_API void libxs_rwlock_relread(libxs_rwlock* rwlock)
{
#if !defined(LIBXS_NO_SYNC)
  assert(0 != rwlock);
# if defined(LIBXS_LOCK_SYSTEM_RWLOCK) && defined(LIBXS_SYNC_SYSTEM)
  LIBXS_LOCK_RELREAD(LIBXS_LOCK_RWLOCK, &rwlock->impl);
# elif defined(_WIN32)
  _InterlockedExchangeAdd16((volatile short*)&rwlock->completions.kind.reader, 1);
# else
  LIBXS_ATOMIC_ADD_FETCH(&rwlock->completions.kind.reader, 1, LIBXS_ATOMIC_SEQ_CST);
# endif
#else
  LIBXS_UNUSED(rwlock);
#endif
}


LIBXS_API unsigned int libxs_get_pid(void)
{
#if defined(_WIN32)
  return (unsigned int)_getpid();
#else
  return (unsigned int)getpid();
#endif
}


LIBXS_API unsigned int libxs_get_tid(void)
{
  static LIBXS_TLS unsigned int tid = (unsigned int)(-1);
  if ((unsigned int)(-1) == tid) {
    tid = LIBXS_ATOMIC_ADD_FETCH(&libxs_threads_count, 1, LIBXS_ATOMIC_RELAXED) - 1;
  }
  return tid;
}

