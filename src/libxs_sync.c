/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_main.h"

#if !defined(LIBXS_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
# define LIBXS_SYNC_FUTEX
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdint.h>
#if defined(_WIN32)
# include <process.h>
#else
# if defined(LIBXS_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
#   include <linux/futex.h>
# endif
# include <unistd.h>
# include <time.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_SYNC_RWLOCK_BITS)
# if defined(__MINGW32__)
#   define LIBXS_SYNC_RWLOCK_BITS 32
# else
#   define LIBXS_SYNC_RWLOCK_BITS 16
# endif
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
  int nthreads, ncores_nbits; /* nbits(ncores) != log2(ncores) */
  /* internal counter type which is guaranteed to be atomic when using certain methods */
  volatile int threads_waiting;
  /* thread-safety during initialization */
  volatile uint8_t init_done;
};


LIBXS_API libxs_barrier* libxs_barrier_create(int ncores, int nthreads_per_core)
{
  libxs_barrier *const barrier = (libxs_barrier*)malloc(sizeof(libxs_barrier));
#if (0 == LIBXS_SYNC)
  LIBXS_UNUSED(ncores); LIBXS_UNUSED(nthreads_per_core);
#else
  if (NULL != barrier && 1 < ncores && 1 <= nthreads_per_core) {
    barrier->ncores = ncores;
    barrier->ncores_nbits = (int)LIBXS_NBITS(ncores);
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
#if (0 == LIBXS_SYNC)
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
          barrier->ncores_nbits * sizeof(uint8_t) * LIBXS_CACHELINE,
          LIBXS_CACHELINE);
        core->partner_flags[i] = (uint8_t**)libxs_aligned_malloc(
          barrier->ncores_nbits * sizeof(uint8_t*),
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
      while (0/*false*/ == barrier->init_done) {} /* empty block instead of semicolon */
    }

    /* set required per-thread information */
    thread->core = barrier->cores[cid];

    /* each core's thread 0 completes setup */
    if (0 == thread->core_tid) {
      int di;
      for (i = di = 0; i < barrier->ncores_nbits; ++i, di += LIBXS_CACHELINE) {
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
      while (2 != barrier->init_done) {} /* empty block instead of semicolon */
    }
  }
#endif
}


LIBXS_API LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
void libxs_barrier_wait(libxs_barrier* barrier, int tid)
{
#if (0 == LIBXS_SYNC)
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

        for (i = di = 0; i < barrier->ncores_nbits - 1; ++i, di += LIBXS_CACHELINE) {
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
#if (0 != LIBXS_SYNC)
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


LIBXS_API unsigned int libxs_get_pid(void)
{
#if defined(_WIN32)
  return (unsigned int)_getpid();
#else
  return (unsigned int)getpid();
#endif
}


LIBXS_API_INTERN unsigned int internal_get_tid(void);
LIBXS_API_INTERN unsigned int internal_get_tid(void)
{
  const unsigned int nthreads = LIBXS_ATOMIC_ADD_FETCH(&libxs_thread_count, 1, LIBXS_ATOMIC_RELAXED);
#if !defined(NDEBUG)
  static int error_once = 0;
  if (LIBXS_NTHREADS_MAX < nthreads
    && 0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: maximum number of threads is exhausted!\n");
  }
#endif
  LIBXS_ASSERT(LIBXS_ISPOT(LIBXS_NTHREADS_MAX));
  return LIBXS_MOD2(nthreads - 1, LIBXS_NTHREADS_MAX);
}


LIBXS_API unsigned int libxs_get_tid(void)
{
#if (0 != LIBXS_SYNC)
# if defined(_OPENMP) && defined(LIBXS_SYNC_OMP)
  return (unsigned int)omp_get_thread_num();
# else
  static LIBXS_TLS unsigned int tid = 0xFFFFFFFF;
  if (0xFFFFFFFF == tid) tid = internal_get_tid();
  return tid;
# endif
#else
  return 0;
#endif
}
