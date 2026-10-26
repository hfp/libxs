/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_sync.h>
#include "libxs_main.h"

#include <stdint.h>
#if defined(_WIN32)
# include <process.h>
#else
# include <sys/file.h>
# include <sys/stat.h>
# include <fcntl.h>
# include <unistd.h>
# include <time.h>
#endif
#if defined(__linux__)
# include <linux/futex.h>
# include <sys/syscall.h>
#endif

#if !defined(INTERNAL_LIBXS_SINGLETON_MAX)
# define INTERNAL_LIBXS_SINGLETON_MAX 8
#endif
#if !defined(INTERNAL_LIBXS_SINGLETON_ATTEMPTS)
# define INTERNAL_LIBXS_SINGLETON_ATTEMPTS 4
#endif
#if !defined(_WIN32) && defined(O_CLOEXEC)
# define INTERNAL_LIBXS_SINGLETON_OFLAGS (O_RDWR | O_CREAT | O_CLOEXEC)
# define INTERNAL_LIBXS_SLOT_OFLAGS (O_RDWR | O_CLOEXEC)
#elif !defined(_WIN32)
# define INTERNAL_LIBXS_SINGLETON_OFLAGS (O_RDWR | O_CREAT)
# define INTERNAL_LIBXS_SLOT_OFLAGS O_RDWR
#endif
/* node-local slots (one per process), and missing slot files ending a count */
#if !defined(INTERNAL_LIBXS_SLOT_MAX)
# define INTERNAL_LIBXS_SLOT_MAX 4096
#endif
#if !defined(INTERNAL_LIBXS_SLOT_GAP)
# define INTERNAL_LIBXS_SLOT_GAP 16
#endif
/* pauses before a ring side sleeps: a short wait costs no system call */
#if !defined(INTERNAL_LIBXS_RING_NSPIN)
# define INTERNAL_LIBXS_RING_NSPIN 2048
#endif
/* longest sleep in one go: bounds what a wake-up lost to libxs_ring_close costs */
#if !defined(INTERNAL_LIBXS_RING_NAP)
# define INTERNAL_LIBXS_RING_NAP 10
#endif
#define INTERNAL_LIBXS_RING_MIN 64
#define INTERNAL_LIBXS_RING_SKIP 0xFFFFFFFFU
#define INTERNAL_LIBXS_RING_RECORD(NBYTES) (8 + LIBXS_UP2((unsigned int)(NBYTES), 8))
#define INTERNAL_LIBXS_RING_DATA(RING) ((char*)((RING) + 1))
#if defined(__linux__) && defined(SYS_futex) && defined(FUTEX_WAIT) && defined(FUTEX_WAKE)
# define INTERNAL_LIBXS_RING_FUTEX
#endif


LIBXS_EXTERN_C typedef struct internal_libxs_singleton_t {
#if defined(_WIN32)
  HANDLE handle;
#else
  char fname[256];
  int handle;
#endif
  int state; /* 0: free, 1: claimed */
} internal_libxs_singleton_t;


LIBXS_APIVAR_DEFINE(internal_libxs_singleton_t internal_libxs_singletons[INTERNAL_LIBXS_SINGLETON_MAX]);
/* descriptor (plus one) and index of the node-local slot of this process (zero: none) */
LIBXS_APIVAR_DEFINE(int internal_libxs_slot_handle);
LIBXS_APIVAR_DEFINE(unsigned int internal_libxs_slot_index);


#if !defined(_WIN32)
/* Name of a slot file: per user as the singleton's, but not a name a singleton key can form. */
LIBXS_API_INLINE int internal_libxs_slot_fname(char fname[], size_t size, unsigned int index)
{
  const int nchars = LIBXS_SNPRINTF(fname, size, "/tmp/.libxs.slot.%u.%u", (unsigned int)getuid(), index);
  return (0 < nchars && (int)size > nchars) ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif


LIBXS_API_INTERN void internal_libxs_slot_acquire(void)
{
#if !defined(_WIN32)
  unsigned int index = 0;
  for (; index < INTERNAL_LIBXS_SLOT_MAX && 0 == internal_libxs_slot_handle; ++index) {
    char fname[256];
    if (EXIT_SUCCESS == internal_libxs_slot_fname(fname, sizeof(fname), index)) {
      const int fd = open(fname, INTERNAL_LIBXS_SINGLETON_OFLAGS, S_IRUSR | S_IWUSR);
      if (0 <= fd) {
        struct stat file_stat, path_stat;
        /* a lock on a file that is no longer at the path is stale (see libxs_singleton_acquire) */
        if (0 == flock(fd, LOCK_EX | LOCK_NB) && 0 == fstat(fd, &file_stat) && 0 == stat(fname, &path_stat)
          && file_stat.st_ino == path_stat.st_ino && file_stat.st_dev == path_stat.st_dev)
        {
          internal_libxs_slot_handle = fd + 1;
          internal_libxs_slot_index = index;
        }
        else close(fd);
      }
    }
  }
#endif
}


LIBXS_API_INTERN void internal_libxs_slot_release(void)
{
#if !defined(_WIN32)
  if (0 != internal_libxs_slot_handle) {
    char fname[256];
    if (EXIT_SUCCESS == internal_libxs_slot_fname(fname, sizeof(fname), internal_libxs_slot_index)) {
      unlink(fname); /* before the lock is dropped */
    }
    close(internal_libxs_slot_handle - 1);
    internal_libxs_slot_handle = 0;
  }
#endif
}


/**
 * Processes of this user on the node that hold a slot (libxs_init), i.e., slot files that
 * cannot be locked. A process probing its own slot counts it, since a lock of another open
 * file of the same file conflicts. Ends after INTERNAL_LIBXS_SLOT_GAP missing slot files.
 */
LIBXS_API_INLINE unsigned int internal_libxs_slot_count(void)
{
  unsigned int result = 0;
#if !defined(_WIN32)
  unsigned int index = 0, gap = 0;
  for (; index < INTERNAL_LIBXS_SLOT_MAX && INTERNAL_LIBXS_SLOT_GAP > gap; ++index) {
    char fname[256];
    const int fd = (EXIT_SUCCESS == internal_libxs_slot_fname(fname, sizeof(fname), index)
      ? open(fname, INTERNAL_LIBXS_SLOT_OFLAGS) : -1);
    if (0 <= fd) {
      if (0 != flock(fd, LOCK_EX | LOCK_NB)) ++result;
      close(fd); /* also drops the lock if it was taken (stale slot file) */
      gap = 0;
    }
    else ++gap;
  }
#endif
  return result;
}


LIBXS_API unsigned int libxs_nranks(void)
{
  /* Intel MPI and MPICH, Open MPI, MVAPICH2 */
  const char* env_nranks = getenv("MPI_LOCALNRANKS");
  int nranks;
  if (NULL == env_nranks) env_nranks = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (NULL == env_nranks) env_nranks = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  /* no launcher states it: the processes holding a node-local slot */
  nranks = (NULL != env_nranks ? atoi(env_nranks) : (int)internal_libxs_slot_count());
  return (unsigned int)LIBXS_MAX(nranks, 1);
}


LIBXS_API unsigned int libxs_nrank(void)
{
  const char* env_rank = getenv("MPI_LOCALRANKID");
  unsigned int result;
  if (NULL == env_rank) env_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (NULL == env_rank) env_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (NULL == env_rank) env_rank = getenv("SLURM_LOCALID");
  if (NULL == env_rank && 0 != internal_libxs_slot_handle) {
    result = internal_libxs_slot_index; /* unique on the node, not necessarily below libxs_nranks */
  }
  else {
    int nrank;
    /* the global PMI_RANK is the last resort: it is local only if ranks are placed in blocks */
    if (NULL == env_rank) env_rank = getenv("PMI_RANK");
    nrank = (NULL == env_rank ? 0 : atoi(env_rank));
    result = (unsigned int)LIBXS_MAX(nrank, 0) % libxs_nranks();
  }
  return result;
}


LIBXS_API unsigned int libxs_rid(void)
{
  return 1 < libxs_nranks() ? libxs_nrank() : libxs_pid();
}


LIBXS_API unsigned int libxs_pid(void)
{
#if defined(_WIN32)
  return (unsigned int)_getpid();
#else
  return (unsigned int)getpid();
#endif
}


LIBXS_API unsigned int libxs_tid(void)
{
#if (0 != LIBXS_SYNC)
  static LIBXS_TLS unsigned int tid = 0xFFFFFFFF;
  if (0xFFFFFFFF == tid) tid = LIBXS_ATOMIC_ADD_FETCH(&libxs_thread_count, 1, LIBXS_ATOMIC_RELAXED) - 1;
  return tid;
#else
  return 0;
#endif
}


LIBXS_API void libxs_stdio_acquire(void)
{
#if !defined(_WIN32)
  if (0 < libxs_stdio_handle) {
    flock(libxs_stdio_handle - 1, LOCK_EX);
  }
  else
#endif
  {
    LIBXS_FLOCK(stdout);
    LIBXS_FLOCK(stderr);
  }
}


LIBXS_API void libxs_stdio_release(void)
{
#if !defined(_WIN32)
  if (0 < libxs_stdio_handle) {
    flock(libxs_stdio_handle - 1, LOCK_UN);
  }
  else
#endif
  {
    LIBXS_FUNLOCK(stderr);
    LIBXS_FUNLOCK(stdout);
  }
}


LIBXS_API int libxs_singleton_acquire(const char key[])
{
  int result = -1, slot = 0;
  if (NULL != key && NULL == strchr(key, '/') && NULL == strchr(key, '\\')) {
    for (; slot < INTERNAL_LIBXS_SINGLETON_MAX; ++slot) {
      int expected = 0;
      if (LIBXS_ATOMIC_CMPSWP(&internal_libxs_singletons[slot].state, expected, 1,
        LIBXS_ATOMIC_SEQ_CST))
      {
        break;
      }
    }
  }
  else slot = INTERNAL_LIBXS_SINGLETON_MAX;
  if (INTERNAL_LIBXS_SINGLETON_MAX > slot) {
    internal_libxs_singleton_t *const entry = internal_libxs_singletons + slot;
#if defined(_WIN32)
    char name[256];
    const int nchars = LIBXS_SNPRINTF(name, sizeof(name), "%s%s",
      '\0' != *key ? "LIBXS." : "GlobalLIBXS", key);
    entry->handle = ((0 < nchars && (int)sizeof(name) > nchars)
      ? CreateMutexA(NULL, TRUE, name) : NULL);
    if (NULL != entry->handle) {
      /* CreateMutex succeeds for an existing mutex as well, without taking ownership */
      if (ERROR_ALREADY_EXISTS != GetLastError()) result = slot;
      else {
        CloseHandle(entry->handle);
        entry->handle = NULL;
      }
    }
#else
    const unsigned int userid = (unsigned int)getuid();
    const int nchars = LIBXS_SNPRINTF(entry->fname, sizeof(entry->fname), "/tmp/.libxs.%u%s%s",
      /*rely on user id to avoid permission issues in case of left-over files*/userid,
      '\0' != *key ? "." : "", key);
    int attempt = 0;
    entry->handle = -1;
    if (0 < nchars && (int)sizeof(entry->fname) > nchars) {
      for (; attempt < INTERNAL_LIBXS_SINGLETON_ATTEMPTS && 0 > result; ++attempt) {
        const int fd = open(entry->fname, INTERNAL_LIBXS_SINGLETON_OFLAGS, S_IRUSR | S_IWUSR);
        if (0 > fd) break;
        /* flock belongs to the open file: a second acquisition in the same process fails as well */
        if (0 == flock(fd, LOCK_EX | LOCK_NB)) {
          struct stat file_stat, path_stat;
          /**
           * A releasing owner unlinks the file while holding the lock. A lock taken on a
           * file that is no longer at the path is stale, and two processes would own the key.
           */
          if (0 == fstat(fd, &file_stat) && 0 == stat(entry->fname, &path_stat)
            && file_stat.st_ino == path_stat.st_ino && file_stat.st_dev == path_stat.st_dev)
          {
            entry->handle = fd;
            result = slot;
          }
          else close(fd);
        }
        else {
          close(fd);
          break; /* owned by another process */
        }
      }
    }
#endif
    if (0 > result) LIBXS_ATOMIC_STORE(&entry->state, 0, LIBXS_ATOMIC_SEQ_CST);
  }
  return result;
}


LIBXS_API void libxs_singleton_release(int handle)
{
  if (0 <= handle && INTERNAL_LIBXS_SINGLETON_MAX > handle
    && 0 != LIBXS_ATOMIC_LOAD(&internal_libxs_singletons[handle].state, LIBXS_ATOMIC_SEQ_CST))
  {
    internal_libxs_singleton_t *const entry = internal_libxs_singletons + handle;
#if defined(_WIN32)
    ReleaseMutex(entry->handle);
    CloseHandle(entry->handle);
    entry->handle = NULL;
#else
    unlink(entry->fname); /* before the lock is dropped, see libxs_singleton_acquire */
    close(entry->handle);
    entry->handle = -1;
#endif
    LIBXS_ATOMIC_STORE(&entry->state, 0, LIBXS_ATOMIC_SEQ_CST);
  }
}


/* Milliseconds of a monotonic clock: a deadline must not depend on calibrating a cycle counter. */
LIBXS_API_INTERN int64_t internal_libxs_ring_msec(void);
LIBXS_API_INTERN int64_t internal_libxs_ring_msec(void)
{
  int64_t result;
#if defined(_WIN32)
  result = (int64_t)GetTickCount64();
#else
  struct timespec now;
  if (0 == clock_gettime(CLOCK_MONOTONIC, &now)) {
    result = (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
  }
  else result = (int64_t)(1E3 * libxs_timer_duration(0, libxs_timer_tick()));
#endif
  return result;
}


/* Sleeps while the word holds what was seen (wake is zero), or wakes whoever sleeps on the word. */
LIBXS_API_INTERN void internal_libxs_ring_futex(const libxs_ring_t* ring, volatile unsigned int* word,
  int wake, unsigned int seen, int nap_ms);
LIBXS_API_INTERN void internal_libxs_ring_futex(const libxs_ring_t* ring, volatile unsigned int* word,
  int wake, unsigned int seen, int nap_ms)
{
#if defined(INTERNAL_LIBXS_RING_FUTEX)
# if defined(FUTEX_PRIVATE_FLAG)
  const int flag = (LIBXS_RING_SHARED != ring->info.v.kind ? FUTEX_PRIVATE_FLAG : 0);
# else
  const int flag = 0;
# endif
  if (0 != wake) syscall(SYS_futex, word, FUTEX_WAKE | flag, INT_MAX, NULL, NULL, 0);
  else {
    struct timespec nap;
    nap.tv_sec = nap_ms / 1000;
    nap.tv_nsec = (long)(nap_ms % 1000) * 1000000L;
    syscall(SYS_futex, word, FUTEX_WAIT | flag, seen, &nap, NULL, 0);
  }
#else
  LIBXS_UNUSED(ring); LIBXS_UNUSED(word); LIBXS_UNUSED(seen); LIBXS_UNUSED(nap_ms);
  if (0 == wake) LIBXS_SYNC_YIELD;
#endif
}


/**
 * Waits until the other side moved far enough: the producer for need bytes of room,
 * the consumer for a record. The sleeper raises its flag before it reads the other
 * side's position for the last time, and the other side publishes its position
 * before it reads the flag. Both in sequential consistency, one of the two sees the
 * other, hence a wake-up is not lost to the order of the two.
 */
LIBXS_API_INTERN int internal_libxs_ring_wait(libxs_ring_t* ring, int producer, unsigned int need, int timeout_ms);
LIBXS_API_INTERN int internal_libxs_ring_wait(libxs_ring_t* ring, int producer, unsigned int need, int timeout_ms)
{
  libxs_ring_side_t* const self = (0 != producer ? &ring->head : &ring->tail);
  volatile unsigned int* const other = (0 != producer ? &ring->tail.v.pos : &ring->head.v.pos);
  const unsigned int capacity = ring->info.v.capacity, own = self->v.pos;
  unsigned int seen = LIBXS_ATOMIC_LOAD(other, LIBXS_ATOMIC_SEQ_CST);
  int closed = LIBXS_ATOMIC_LOAD(&ring->info.v.closed, LIBXS_ATOMIC_SEQ_CST);
  int result = (need <= (0 != producer ? (capacity - (own - seen)) : (seen - own)) && (0 == producer || 0 == closed));
  if (0 == result && 0 == closed && 0 != timeout_ms) {
    const int64_t start = internal_libxs_ring_msec();
    int npause = 0, expired = 0;
    while (0 == result && 0 == closed && 0 == expired) {
      if (INTERNAL_LIBXS_RING_NSPIN > npause) {
        LIBXS_SYNC_PAUSE;
        ++npause;
      }
      else {
        const int64_t elapsed = internal_libxs_ring_msec() - start;
        if (0 > timeout_ms || elapsed < timeout_ms) {
          const int nap = (0 > timeout_ms ? INTERNAL_LIBXS_RING_NAP
            : (int)LIBXS_MIN(INTERNAL_LIBXS_RING_NAP, timeout_ms - elapsed));
          LIBXS_ATOMIC_STORE(&self->v.wait, 1, LIBXS_ATOMIC_SEQ_CST);
          if (seen == LIBXS_ATOMIC_LOAD(other, LIBXS_ATOMIC_SEQ_CST)) {
            internal_libxs_ring_futex(ring, other, 0 /*wait*/, seen, LIBXS_MAX(nap, 1));
          }
          LIBXS_ATOMIC_STORE(&self->v.wait, 0, LIBXS_ATOMIC_SEQ_CST);
        }
        else expired = 1;
      }
      seen = LIBXS_ATOMIC_LOAD(other, LIBXS_ATOMIC_SEQ_CST);
      closed = LIBXS_ATOMIC_LOAD(&ring->info.v.closed, LIBXS_ATOMIC_SEQ_CST);
      result = (need <= (0 != producer ? (capacity - (own - seen)) : (seen - own)) && (0 == producer || 0 == closed));
    }
    if (0 == result && 0 == producer) result = (need <= seen - own); /* closed: what was committed is delivered */
  }
  return result;
}


LIBXS_API size_t libxs_ring_size(size_t capacity)
{
  return sizeof(libxs_ring_t) + (size_t)LIBXS_UP2POT(LIBXS_MAX(capacity, INTERNAL_LIBXS_RING_MIN));
}


LIBXS_API libxs_ring_t* libxs_ring_init(void* memory, size_t size, libxs_ring_kind_t kind)
{
  libxs_ring_t* result = NULL;
  if (NULL != memory && 0 == ((uintptr_t)memory) % LIBXS_CACHELINE
    && sizeof(libxs_ring_t) + INTERNAL_LIBXS_RING_MIN <= size)
  {
    const size_t limit = LIBXS_MIN(size - sizeof(libxs_ring_t), (size_t)1 << 31);
    unsigned int capacity = INTERNAL_LIBXS_RING_MIN;
    while ((size_t)capacity * 2 <= limit) capacity *= 2;
    result = (libxs_ring_t*)memory;
    memset(result, 0, sizeof(libxs_ring_t));
    result->info.v.capacity = capacity;
    result->info.v.kind = (int)kind;
    LIBXS_ATOMIC_SYNC(LIBXS_ATOMIC_SEQ_CST);
  }
  return result;
}


LIBXS_API void* libxs_ring_reserve(libxs_ring_t* ring, size_t nbytes, int timeout_ms)
{
  void* result = NULL;
  if (NULL != ring && nbytes + 8 <= ring->info.v.capacity / 2) {
    const unsigned int capacity = ring->info.v.capacity, pos = ring->head.v.pos;
    const unsigned int offset = pos & (capacity - 1), toend = capacity - offset;
    const unsigned int record = INTERNAL_LIBXS_RING_RECORD(nbytes);
    const unsigned int need = (record <= toend ? record : (toend + record)); /* a record never wraps */
    if (0 != internal_libxs_ring_wait(ring, 1 /*producer*/, need, timeout_ms)) {
      char* const data = INTERNAL_LIBXS_RING_DATA(ring);
      unsigned int* header = (unsigned int*)(void*)(data + offset);
      if (record > toend) {
        *header = INTERNAL_LIBXS_RING_SKIP;
        header = (unsigned int*)(void*)data;
      }
      *header = (unsigned int)nbytes;
      ring->head.v.size = need;
      result = (char*)header + 8;
    }
  }
  return result;
}


LIBXS_API void libxs_ring_commit(libxs_ring_t* ring)
{
  if (NULL != ring && 0 != ring->head.v.size) {
    LIBXS_ATOMIC_STORE(&ring->head.v.pos, ring->head.v.pos + ring->head.v.size, LIBXS_ATOMIC_SEQ_CST);
    ring->head.v.size = 0;
    if (0 != LIBXS_ATOMIC_LOAD(&ring->tail.v.wait, LIBXS_ATOMIC_SEQ_CST)) {
      internal_libxs_ring_futex(ring, &ring->head.v.pos, 1 /*wake*/, 0, 0);
    }
  }
}


LIBXS_API const void* libxs_ring_peek(libxs_ring_t* ring, size_t* nbytes, int timeout_ms)
{
  const void* result = NULL;
  if (NULL != ring && 0 != internal_libxs_ring_wait(ring, 0 /*consumer*/, 1, timeout_ms)) {
    const unsigned int capacity = ring->info.v.capacity;
    const char* const data = INTERNAL_LIBXS_RING_DATA(ring);
    unsigned int offset = ring->tail.v.pos & (capacity - 1), skip = 0;
    unsigned int size = *(const unsigned int*)(const void*)(data + offset);
    if (INTERNAL_LIBXS_RING_SKIP == size) {
      skip = capacity - offset;
      offset = 0;
      size = *(const unsigned int*)(const void*)data;
    }
    ring->tail.v.size = skip + INTERNAL_LIBXS_RING_RECORD(size);
    if (NULL != nbytes) *nbytes = size;
    result = data + offset + 8;
  }
  return result;
}


LIBXS_API void libxs_ring_release(libxs_ring_t* ring)
{
  if (NULL != ring && 0 != ring->tail.v.size) {
    LIBXS_ATOMIC_STORE(&ring->tail.v.pos, ring->tail.v.pos + ring->tail.v.size, LIBXS_ATOMIC_SEQ_CST);
    ring->tail.v.size = 0;
    if (0 != LIBXS_ATOMIC_LOAD(&ring->head.v.wait, LIBXS_ATOMIC_SEQ_CST)) {
      internal_libxs_ring_futex(ring, &ring->tail.v.pos, 1 /*wake*/, 0, 0);
    }
  }
}


LIBXS_API void libxs_ring_close(libxs_ring_t* ring)
{
  if (NULL != ring) {
    LIBXS_ATOMIC_STORE(&ring->info.v.closed, 1, LIBXS_ATOMIC_SEQ_CST);
    internal_libxs_ring_futex(ring, &ring->head.v.pos, 1 /*wake*/, 0, 0);
    internal_libxs_ring_futex(ring, &ring->tail.v.pos, 1 /*wake*/, 0, 0);
  }
}
