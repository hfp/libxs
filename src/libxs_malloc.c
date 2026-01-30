/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_hash.h"
#include <libxs_mem.h>
#include <libxs_sync.h>

#if !defined(LIBXS_MALLOC_SEED)
# define LIBXS_MALLOC_SEED 1051981
#endif
#if !defined(LIBXS_MALLOC_NLOCKS)
# define LIBXS_MALLOC_NLOCKS 16
#endif

LIBXS_APIVAR_DEFINE(volatile int internal_malloc_plocks[LIBXS_MALLOC_NLOCKS]);


LIBXS_API void libxs_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage)
{
  char* p = (char*)storage;
  volatile int* lock;
  unsigned int hash;
  size_t n, i = 0;
  LIBXS_ASSERT(0 < size && NULL != num && NULL != pool && NULL != storage);
  libxs_hash_init(); /* CRC-facility must be initialized upfront */
  hash = LIBXS_CRCPTR(LIBXS_MALLOC_SEED, pool);
  lock = internal_malloc_plocks + LIBXS_MOD2(hash, LIBXS_MALLOC_NLOCKS);
  LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
  for (n = *num; i < n; ++i, p += size) pool[i] = p;
  LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_SEQ_CST);
}


LIBXS_API void* libxs_pmalloc(void* pool[], size_t* i)
{
  const unsigned int hash = LIBXS_CRCPTR(LIBXS_MALLOC_SEED, pool);
  volatile int *const lock = internal_malloc_plocks + LIBXS_MOD2(hash, LIBXS_MALLOC_NLOCKS);
  void* pointer;
  LIBXS_ASSERT(NULL != pool && NULL != i);
  LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
  assert(0 < *i && ((size_t)-1) != *i); /* !LIBXS_ASSERT */
  pointer = pool[--(*i)];
  LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_SEQ_CST);
  LIBXS_ASSERT(NULL != pointer);
  return pointer;
}


LIBXS_API void libxs_pfree(const void* pointer, void* pool[], size_t* i)
{
  LIBXS_ASSERT(NULL != pool && NULL != i);
  if (NULL != pointer) {
    const unsigned int hash = LIBXS_CRCPTR(LIBXS_MALLOC_SEED, pool);
    volatile int *const lock = internal_malloc_plocks + LIBXS_MOD2(hash, LIBXS_MALLOC_NLOCKS);
    LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
    LIBXS_VALUE_ASSIGN(pool[*i], pointer); ++(*i);
    LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_SEQ_CST);
  }
}
