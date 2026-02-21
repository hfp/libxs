/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_reg.h>

#include <stdlib.h>
#include <string.h>

#if !defined(LIBXS_LOCK)
# define LIBXS_LOCK LIBXS_LOCK_DEFAULT
#endif

/* Entry states for open-addressing hash table. */
#define INTERNAL_REG_EMPTY 0
#define INTERNAL_REG_USED  1
#define INTERNAL_REG_TOMB  2 /* tombstone: deleted entry, skip during probe */

/* Load factor threshold (numerator/denominator) triggering growth. */
#define INTERNAL_REG_LOAD_NUM 3
#define INTERNAL_REG_LOAD_DEN 4

#define INTERNAL_REG_HASH_SEED 2166136261u
#define INTERNAL_REG_HASH_PRIME 16777619u

/* Enable thread-local cache when TLS and sync are available and cache is non-zero. */
#if !defined(LIBXS_NO_TLS) && (0 != LIBXS_SYNC) && (0 < (LIBXS_REGCACHE_NENTRIES))
# define INTERNAL_REG_CACHE
#endif


LIBXS_EXTERN_C typedef struct internal_regentry_t {
  void* value;          /* heap-allocated value buffer */
  size_t key_size;      /* actual key size in Bytes */
  size_t value_size;    /* allocated value size in Bytes */
  unsigned char state;  /* INTERNAL_REG_EMPTY|USED|TOMB */
  char key[LIBXS_REGKEY_MAXSIZE];
} internal_regentry_t;

struct libxs_registry_t {
  internal_regentry_t* entries;
  unsigned int capacity; /* always POT */
  unsigned int size;     /* number of USED entries */
#if (0 != LIBXS_SYNC)
  LIBXS_LOCK_TYPE(LIBXS_LOCK) lock;
#endif
};


#if defined(INTERNAL_REG_CACHE)
LIBXS_EXTERN_C typedef struct internal_regcache_entry_t {
  const libxs_registry_t* registry;
  unsigned int hash;
  size_t key_size;
  void* value;
  char key[LIBXS_REGKEY_MAXSIZE];
} internal_regcache_entry_t;

static LIBXS_TLS internal_regcache_entry_t
  internal_regcache[(unsigned int)LIBXS_UP2POT(LIBXS_REGCACHE_NENTRIES)];
#define INTERNAL_REG_CACHE_MASK \
  ((unsigned int)LIBXS_UP2POT(LIBXS_REGCACHE_NENTRIES) - 1)
#endif


/** FNV-1a hash: self-contained, no initialization needed. */
LIBXS_API_INLINE unsigned int internal_regkey_hash(
  const void* key, size_t key_size)
{
  const unsigned char* data = (const unsigned char*)key;
  unsigned int hash = INTERNAL_REG_HASH_SEED;
  size_t i;
  for (i = 0; i < key_size; ++i) {
    hash ^= (unsigned int)data[i];
    hash *= INTERNAL_REG_HASH_PRIME;
  }
  return hash;
}


/**
 * Find entry by key. Returns index of the matching USED entry,
 * or the index of the first available slot (EMPTY or TOMB) suitable
 * for insertion. Sets *found to 1 if an existing entry was found.
 */
LIBXS_API_INLINE unsigned int internal_registry_probe(
  const internal_regentry_t* entries, unsigned int capacity,
  const void* key, size_t key_size, int* found)
{
  const unsigned int mask = capacity - 1; /* capacity is POT */
  unsigned int i = internal_regkey_hash(key, key_size) & mask;
  unsigned int tomb = capacity; /* sentinel: no tombstone seen yet */
  unsigned int steps = 0;
  *found = 0;
  while (steps < capacity) {
    const internal_regentry_t* e = entries + i;
    if (INTERNAL_REG_EMPTY == e->state) {
      return (tomb < capacity) ? tomb : i; /* insert position */
    }
    if (INTERNAL_REG_USED == e->state
      && e->key_size == key_size
      && 0 == memcmp(e->key, key, key_size))
    {
      *found = 1;
      return i; /* exact match */
    }
    if (INTERNAL_REG_TOMB == e->state && tomb >= capacity) {
      tomb = i; /* remember first tombstone for potential insert */
    }
    i = (i + 1) & mask;
    ++steps;
  }
  /* table is full (should not happen if load factor is maintained) */
  return (tomb < capacity) ? tomb : 0;
}


/** Grow and rehash the table. Caller must hold the lock. Returns 0 on success. */
LIBXS_API_INLINE int internal_registry_grow(libxs_registry_t* registry)
{
  const unsigned int old_cap = registry->capacity;
  const unsigned int new_cap = old_cap << 1; /* double */
  internal_regentry_t* new_entries = (internal_regentry_t*)calloc(
    new_cap, sizeof(internal_regentry_t));
  unsigned int i;
  if (NULL == new_entries) return EXIT_FAILURE;
  for (i = 0; i < old_cap; ++i) {
    internal_regentry_t* e = registry->entries + i;
    if (INTERNAL_REG_USED == e->state) {
      int found = 0;
      const unsigned int j = internal_registry_probe(
        new_entries, new_cap, e->key, e->key_size, &found);
      LIBXS_ASSERT(0 == found); /* no duplicates during rehash */
      new_entries[j] = *e; /* shallow copy (value pointer transfers) */
    }
  }
  free(registry->entries);
  registry->entries = new_entries;
  registry->capacity = new_cap;
  return EXIT_SUCCESS;
}


LIBXS_API void libxs_registry_create(libxs_registry_t** registry)
{
  libxs_registry_t* r;
  const unsigned int nbuckets = (unsigned int)LIBXS_UP2POT(LIBXS_REGISTRY_NBUCKETS);
  LIBXS_ASSERT(NULL != registry);
  if (NULL == registry) return;
  r = (libxs_registry_t*)calloc(1, sizeof(libxs_registry_t));
  if (NULL != r) {
    r->entries = (internal_regentry_t*)calloc(
      nbuckets, sizeof(internal_regentry_t));
    if (NULL != r->entries) {
      r->capacity = nbuckets;
      r->size = 0;
#if (0 != LIBXS_SYNC)
      { LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK) attr;
        LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &attr);
        LIBXS_LOCK_INIT(LIBXS_LOCK, &r->lock, &attr);
        LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &attr);
      }
#endif
      *registry = r;
    }
    else {
      free(r);
      *registry = NULL;
    }
  }
  else {
    *registry = NULL;
  }
}


LIBXS_API void libxs_registry_destroy(libxs_registry_t* registry)
{
  if (NULL != registry) {
    if (NULL != registry->entries) {
      unsigned int i;
      for (i = 0; i < registry->capacity; ++i) {
        if (INTERNAL_REG_USED == registry->entries[i].state) {
          free(registry->entries[i].value);
        }
      }
      free(registry->entries);
    }
#if (0 != LIBXS_SYNC)
    LIBXS_LOCK_DESTROY(LIBXS_LOCK, &registry->lock);
#endif
    free(registry);
  }
}


LIBXS_API void* libxs_registry_set(libxs_registry_t* registry,
  const void* key, size_t key_size,
  size_t value_size, const void* value_init)
{
  void* result = NULL;
  if (NULL == registry || NULL == key || 0 == key_size || 0 == value_size
    || key_size > LIBXS_REGKEY_MAXSIZE)
  {
    return NULL;
  }
#if (0 != LIBXS_SYNC)
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &registry->lock);
#endif
  { int found = 0;
    unsigned int idx = internal_registry_probe(
      registry->entries, registry->capacity, key, key_size, &found);
    if (0 != found) { /* key exists: check value size */
      internal_regentry_t* e = registry->entries + idx;
      if (value_size <= e->value_size) { /* fits in existing allocation */
        if (NULL != value_init) memcpy(e->value, value_init, value_size);
        result = e->value;
      }
      /* else: value too large - return NULL (caller must _free first) */
    }
    else { /* new entry */
      /* grow if load factor exceeded */
      if (registry->size * INTERNAL_REG_LOAD_DEN
        >= registry->capacity * INTERNAL_REG_LOAD_NUM)
      {
        if (EXIT_SUCCESS != internal_registry_grow(registry)) goto done;
        /* re-probe after growth */
        idx = internal_registry_probe(
          registry->entries, registry->capacity, key, key_size, &found);
        LIBXS_ASSERT(0 == found);
      }
      { void* value_buf = malloc(value_size);
        if (NULL != value_buf) {
          internal_regentry_t* e = registry->entries + idx;
          if (NULL != value_init) {
            memcpy(value_buf, value_init, value_size);
          }
          else {
            memset(value_buf, 0, value_size);
          }
          memcpy(e->key, key, key_size);
          e->key_size = key_size;
          e->value = value_buf;
          e->value_size = value_size;
          e->state = INTERNAL_REG_USED;
          ++registry->size;
          result = value_buf;
        }
      }
    }
  }
done:
#if (0 != LIBXS_SYNC)
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &registry->lock);
#endif
#if defined(INTERNAL_REG_CACHE)
  if (NULL != result) { /* update TLS cache for this thread */
    const unsigned int hash = internal_regkey_hash(key, key_size);
    internal_regcache_entry_t* ce = internal_regcache + (hash & INTERNAL_REG_CACHE_MASK);
    ce->registry = registry;
    ce->hash = hash;
    ce->key_size = key_size;
    ce->value = result;
    memcpy(ce->key, key, key_size);
  }
#endif
  return result;
}


LIBXS_API void* libxs_registry_get(libxs_registry_t* registry,
  const void* key, size_t key_size)
{
  void* result = NULL;
#if defined(INTERNAL_REG_CACHE)
  unsigned int hash;
  internal_regcache_entry_t* ce;
#endif
  if (NULL == registry || NULL == key || 0 == key_size
    || key_size > LIBXS_REGKEY_MAXSIZE)
  {
    return NULL;
  }
#if defined(INTERNAL_REG_CACHE)
  hash = internal_regkey_hash(key, key_size);
  ce = internal_regcache + (hash & INTERNAL_REG_CACHE_MASK);
  if (ce->registry == registry && ce->hash == hash
    && ce->key_size == key_size
    && 0 == memcmp(ce->key, key, key_size))
  {
    return ce->value; /* TLS cache hit: no lock needed */
  }
#endif
#if (0 != LIBXS_SYNC)
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &registry->lock);
#endif
  { int found = 0;
    const unsigned int idx = internal_registry_probe(
      registry->entries, registry->capacity, key, key_size, &found);
    if (0 != found) {
      result = registry->entries[idx].value;
    }
  }
#if (0 != LIBXS_SYNC)
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &registry->lock);
#endif
#if defined(INTERNAL_REG_CACHE)
  if (NULL != result) { /* populate TLS cache on miss */
    ce->registry = registry;
    ce->hash = hash;
    ce->key_size = key_size;
    ce->value = result;
    memcpy(ce->key, key, key_size);
  }
#endif
  return result;
}


LIBXS_API void libxs_registry_free(libxs_registry_t* registry,
  const void* key, size_t key_size)
{
  if (NULL == registry || NULL == key || 0 == key_size
    || key_size > LIBXS_REGKEY_MAXSIZE)
  {
    return;
  }
#if (0 != LIBXS_SYNC)
  LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &registry->lock);
#endif
  { int found = 0;
    const unsigned int idx = internal_registry_probe(
      registry->entries, registry->capacity, key, key_size, &found);
    if (0 != found) {
      internal_regentry_t* e = registry->entries + idx;
      free(e->value);
      e->value = NULL;
      e->key_size = 0;
      e->value_size = 0;
      e->state = INTERNAL_REG_TOMB;
      LIBXS_ASSERT(0 < registry->size);
      --registry->size;
    }
  }
#if (0 != LIBXS_SYNC)
  LIBXS_LOCK_RELEASE(LIBXS_LOCK, &registry->lock);
#endif
#if defined(INTERNAL_REG_CACHE)
  { /* invalidate TLS cache entry for this thread */
    const unsigned int hash = internal_regkey_hash(key, key_size);
    internal_regcache_entry_t* ce = internal_regcache + (hash & INTERNAL_REG_CACHE_MASK);
    if (ce->registry == registry && ce->hash == hash
      && ce->key_size == key_size
      && 0 == memcmp(ce->key, key, key_size))
    {
      ce->registry = NULL; /* invalidate */
    }
  }
#endif
}


LIBXS_API void* libxs_registry_begin(libxs_registry_t* registry,
  const void** key)
{
  if (NULL != registry && NULL != registry->entries) {
    unsigned int i;
    for (i = 0; i < registry->capacity; ++i) {
      const internal_regentry_t* e = registry->entries + i;
      if (INTERNAL_REG_USED == e->state) {
        if (NULL != key) *key = e->key;
        return e->value;
      }
    }
  }
  if (NULL != key) *key = NULL;
  return NULL;
}


LIBXS_API void* libxs_registry_next(libxs_registry_t* registry,
  const void* regentry, const void** key)
{
  if (NULL != registry && NULL != registry->entries && NULL != regentry) {
    unsigned int i;
    int past = 0; /* set once we pass the current entry */
    for (i = 0; i < registry->capacity; ++i) {
      const internal_regentry_t* e = registry->entries + i;
      if (INTERNAL_REG_USED != e->state) continue;
      if (0 != past) { /* return next occupied entry */
        if (NULL != key) *key = e->key;
        return e->value;
      }
      if (e->value == regentry) past = 1;
    }
  }
  if (NULL != key) *key = NULL;
  return NULL;
}


LIBXS_API int libxs_registry_info(libxs_registry_t* registry,
  libxs_registry_info_t* info)
{
  if (NULL != registry && NULL != info && NULL != registry->entries) {
    info->capacity = registry->capacity;
    info->size = registry->size;
    info->nbytes = (size_t)registry->capacity * sizeof(internal_regentry_t)
      + sizeof(libxs_registry_t);
    { unsigned int i;
      for (i = 0; i < registry->capacity; ++i) {
        if (INTERNAL_REG_USED == registry->entries[i].state) {
          info->nbytes += registry->entries[i].value_size;
        }
      }
    }
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}


/** Backward-compatible stub (JIT heritage). */
LIBXS_API void libxs_release_kernel(const void* kernel)
{
  LIBXS_UNUSED(kernel);
}
