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

#define INTERNAL_REG_HASH_SEED  2166136261u
#define INTERNAL_REG_HASH_PRIME 16777619u

/* Enable thread-local cache when TLS and sync are available and cache is non-zero. */
#if !defined(LIBXS_NO_TLS) && (0 != LIBXS_SYNC) && (0 < (LIBXS_REGCACHE_NENTRIES))
# define INTERNAL_REG_CACHE
#endif
/* When cache is disabled, use rwlock so readers can proceed in parallel. */
#if !defined(INTERNAL_REG_CACHE) && (0 != LIBXS_SYNC)
# undef LIBXS_LOCK
# define LIBXS_LOCK LIBXS_LOCK_RWLOCK
#endif

#if defined(INTERNAL_REG_CACHE)
# define INTERNAL_REG_CACHE_MASK \
    ((unsigned int)LIBXS_UP2POT(LIBXS_REGCACHE_NENTRIES) - 1)
#endif

/* Small-value optimization: values <= sizeof(void*) are stored directly
   in the 'value' field (reinterpreted as a byte buffer), avoiding heap
   allocation. */
#define INTERNAL_REG_INLINE(E) ((E)->value_size <= sizeof((E)->value))


LIBXS_EXTERN_C typedef struct internal_libxs_regentry_t {
  void* value;          /* heap pointer or inline storage */
  size_t key_size;      /* actual key size in Bytes */
  size_t value_size;    /* allocated value size in Bytes */
  unsigned char state;  /* INTERNAL_REG_EMPTY|USED|TOMB */
  char key[LIBXS_REGKEY_MAXSIZE];
} internal_libxs_regentry_t;

struct libxs_registry_t {
  internal_libxs_regentry_t* entries;
  unsigned int capacity; /* always POT */
  unsigned int size;     /* number of USED entries */
#if (0 != LIBXS_SYNC)
  LIBXS_LOCK_TYPE(LIBXS_LOCK) lock;
#endif
};

#if defined(INTERNAL_REG_CACHE)
LIBXS_EXTERN_C typedef struct internal_libxs_regcache_entry_t {
  const libxs_registry_t* registry;
  unsigned int hash;
  size_t key_size;
  void* value;
  char key[LIBXS_REGKEY_MAXSIZE];
} internal_libxs_regcache_entry_t;
#endif

#if defined(INTERNAL_REG_CACHE)
static LIBXS_TLS internal_libxs_regcache_entry_t
  internal_libxs_regcache[(unsigned int)LIBXS_UP2POT(LIBXS_REGCACHE_NENTRIES)];
#endif


LIBXS_API_INLINE void* internal_value_ptr(internal_libxs_regentry_t* e) {
  return INTERNAL_REG_INLINE(e) ? (void*)&e->value : e->value;
}


/** FNV-1a hash: self-contained, no initialization needed. */
LIBXS_API_INLINE unsigned int internal_libxs_regkey_hash(
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
LIBXS_API_INLINE unsigned int internal_libxs_registry_probe(
  const internal_libxs_regentry_t* entries, unsigned int capacity,
  const void* key, size_t key_size, int* found)
{
  const unsigned int mask = capacity - 1; /* capacity is POT */
  unsigned int i = internal_libxs_regkey_hash(key, key_size) & mask;
  unsigned int tomb = capacity; /* sentinel: no tombstone seen yet */
  unsigned int result = 0, steps = 0;
  *found = 0;
  while (steps < capacity) {
    const internal_libxs_regentry_t* e = entries + i;
    if (INTERNAL_REG_EMPTY == e->state) {
      result = (tomb < capacity) ? tomb : i;
      break;
    }
    if (INTERNAL_REG_USED == e->state
      && e->key_size == key_size
      && 0 == memcmp(e->key, key, key_size))
    {
      *found = 1;
      result = i;
      break;
    }
    if (INTERNAL_REG_TOMB == e->state && tomb >= capacity) {
      tomb = i;
    }
    i = (i + 1) & mask;
    ++steps;
  }
  if (steps >= capacity) {
    result = (tomb < capacity) ? tomb : 0;
  }
  return result;
}


/** Grow and rehash the table. Caller must hold the lock. */
LIBXS_API_INLINE int internal_libxs_registry_grow(libxs_registry_t* registry)
{
  const unsigned int old_cap = registry->capacity;
  const unsigned int new_cap = old_cap << 1; /* double */
  internal_libxs_regentry_t* new_entries = (internal_libxs_regentry_t*)calloc(
    new_cap, sizeof(internal_libxs_regentry_t));
  int result = EXIT_FAILURE;
  if (NULL != new_entries) {
    unsigned int i;
    for (i = 0; i < old_cap; ++i) {
      internal_libxs_regentry_t* e = registry->entries + i;
      if (INTERNAL_REG_USED == e->state) {
        int found = 0;
        const unsigned int j = internal_libxs_registry_probe(
          new_entries, new_cap, e->key, e->key_size, &found);
        LIBXS_ASSERT(0 == found);
        new_entries[j] = *e; /* shallow copy (value pointer transfers) */
      }
    }
    free(registry->entries);
    registry->entries = new_entries;
    registry->capacity = new_cap;
    result = EXIT_SUCCESS;
  }
  return result;
}


/**
 * Core set logic (no locking). Caller must hold whatever lock is appropriate.
 * Returns the value pointer on success, NULL on failure.
 */
LIBXS_API_INLINE void* internal_libxs_registry_set_impl(
  libxs_registry_t* registry,
  const void* key, size_t key_size,
  const void* value_init, size_t value_size)
{
  void* result = NULL;
  int found = 0;
  unsigned int idx = internal_libxs_registry_probe(
    registry->entries, registry->capacity, key, key_size, &found);
  if (0 != found) { /* key exists: update in place */
    internal_libxs_regentry_t* e = registry->entries + idx;
    if (value_size > e->value_size) { /* need larger buffer */
      if (value_size <= sizeof(e->value)) { /* fits inline */
        if (!INTERNAL_REG_INLINE(e)) free(e->value);
      }
      else { /* heap allocation */
        void* new_buf = INTERNAL_REG_INLINE(e)
            ? malloc(value_size) : realloc(e->value, value_size);
        if (NULL != new_buf) {
          e->value = new_buf;
        }
      }
      e->value_size = value_size;
    }
    result = internal_value_ptr(e);
    if (NULL != result && NULL != value_init) {
      memcpy(result, value_init, value_size);
    }
  }
  else { /* new entry */
    /* grow if load factor exceeded */
    if (registry->size * INTERNAL_REG_LOAD_DEN
      >= registry->capacity * INTERNAL_REG_LOAD_NUM)
    {
      if (EXIT_SUCCESS != internal_libxs_registry_grow(registry)) {
        return result; /* NULL */
      }
      idx = internal_libxs_registry_probe(
        registry->entries, registry->capacity, key, key_size, &found);
      LIBXS_ASSERT(0 == found);
    }
    { internal_libxs_regentry_t* e = registry->entries + idx;
      void* value_buf;
      e->value_size = value_size;
      if (value_size <= sizeof(e->value)) { /* inline */
        e->value = NULL;
        value_buf = &e->value;
      }
      else { /* heap */
        value_buf = malloc(value_size);
        if (NULL == value_buf) {
          return result; /* NULL */
        }
        e->value = value_buf;
      }
      if (NULL != value_init) {
        memcpy(value_buf, value_init, value_size);
      }
      else {
        memset(value_buf, 0, value_size);
      }
      memcpy(e->key, key, key_size);
      e->key_size = key_size;
      e->state = INTERNAL_REG_USED;
      ++registry->size;
      result = value_buf;
    }
  }
  return result;
}


/** Core remove logic (no locking). Caller must hold whatever lock is appropriate. */
LIBXS_API_INLINE void internal_libxs_registry_remove_impl(
  libxs_registry_t* registry,
  const void* key, size_t key_size)
{
  int found = 0;
  const unsigned int idx = internal_libxs_registry_probe(
    registry->entries, registry->capacity, key, key_size, &found);
  if (0 != found) {
    internal_libxs_regentry_t* e = registry->entries + idx;
    if (!INTERNAL_REG_INLINE(e)) free(e->value);
    e->value = NULL;
    e->key_size = 0;
    e->value_size = 0;
    e->state = INTERNAL_REG_TOMB;
    LIBXS_ASSERT(0 < registry->size);
    --registry->size;
  }
}


/** Probe, optionally copy value, optionally remove. Caller must hold the lock. */
LIBXS_API_INLINE int internal_libxs_registry_fetch_impl(
  libxs_registry_t* registry,
  const void* key, size_t key_size,
  void* value_out, size_t value_size, int remove)
{
  int found = 0;
  const unsigned int idx = internal_libxs_registry_probe(
    registry->entries, registry->capacity, key, key_size, &found);
  if (0 != found) {
    internal_libxs_regentry_t* e = registry->entries + idx;
    if (NULL != value_out && 0 < value_size) {
      const size_t n = value_size < e->value_size ? value_size : e->value_size;
      memcpy(value_out, internal_value_ptr(e), n);
    }
    if (0 != remove) {
      if (!INTERNAL_REG_INLINE(e)) free(e->value);
      e->value = NULL;
      e->key_size = 0;
      e->value_size = 0;
      e->state = INTERNAL_REG_TOMB;
      LIBXS_ASSERT(0 < registry->size);
      --registry->size;
    }
  }
  return found;
}


LIBXS_API libxs_registry_t* libxs_registry_create(void)
{
  const unsigned int nbuckets =
      (unsigned int)LIBXS_UP2POT(LIBXS_REGISTRY_NBUCKETS);
  libxs_registry_t* result =
      (libxs_registry_t*)calloc(1, sizeof(libxs_registry_t));
  if (NULL != result) {
    result->entries = (internal_libxs_regentry_t*)calloc(
      nbuckets, sizeof(internal_libxs_regentry_t));
    if (NULL != result->entries) {
      result->capacity = nbuckets;
      result->size = 0;
#if (0 != LIBXS_SYNC)
      { LIBXS_LOCK_ATTR_TYPE(LIBXS_LOCK) attr;
        LIBXS_LOCK_ATTR_INIT(LIBXS_LOCK, &attr);
        LIBXS_LOCK_INIT(LIBXS_LOCK, &result->lock, &attr);
        LIBXS_LOCK_ATTR_DESTROY(LIBXS_LOCK, &attr);
      }
#endif
    }
    else {
      free(result);
      result = NULL;
    }
  }
  return result;
}


LIBXS_API libxs_lock_t* libxs_registry_lock(libxs_registry_t* registry)
{
#if (0 != LIBXS_SYNC)
  return (NULL != registry) ? &registry->lock : NULL;
#else
  LIBXS_UNUSED(registry);
  return NULL;
#endif
}


LIBXS_API void libxs_registry_destroy(libxs_registry_t* registry)
{
  if (NULL != registry) {
    if (NULL != registry->entries) {
      unsigned int i;
      for (i = 0; i < registry->capacity; ++i) {
        if (INTERNAL_REG_USED == registry->entries[i].state
          && !INTERNAL_REG_INLINE(registry->entries + i))
        {
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
  const void* value_init, size_t value_size, libxs_lock_t* lock)
{
  void* result = NULL;
  if (NULL != registry && NULL != key && 0 < key_size && 0 < value_size
    && key_size <= LIBXS_REGKEY_MAXSIZE)
  {
    if (NULL != lock) {
      LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    }
    result = internal_libxs_registry_set_impl(
      registry, key, key_size, value_init, value_size);
    if (NULL != lock) {
      LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_LOCKORDER);
    }
#if defined(INTERNAL_REG_CACHE)
    if (NULL != result) { /* update TLS cache */
      const unsigned int hash = internal_libxs_regkey_hash(key, key_size);
      internal_libxs_regcache_entry_t* ce =
          internal_libxs_regcache + (hash & INTERNAL_REG_CACHE_MASK);
      ce->registry = registry;
      ce->hash = hash;
      ce->key_size = key_size;
      ce->value = result;
      memcpy(ce->key, key, key_size);
    }
#endif
  }
  return result;
}


LIBXS_API void* libxs_registry_get(const libxs_registry_t* registry,
  const void* key, size_t key_size, libxs_lock_t* lock)
{
  void* result = NULL;
#if defined(INTERNAL_REG_CACHE)
  unsigned int hash = 0;
  internal_libxs_regcache_entry_t* ce = NULL;
#endif
  if (NULL != registry && NULL != key && 0 < key_size
    && key_size <= LIBXS_REGKEY_MAXSIZE)
  {
#if defined(INTERNAL_REG_CACHE)
    hash = internal_libxs_regkey_hash(key, key_size);
    ce = internal_libxs_regcache + (hash & INTERNAL_REG_CACHE_MASK);
    if (NULL == lock && ce->registry == registry && ce->hash == hash
      && ce->key_size == key_size
      && 0 == memcmp(ce->key, key, key_size))
    {
      result = ce->value; /* TLS cache hit */
    }
    else
#endif
    {
      if (NULL != lock) {
        LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
      }
      { int found = 0;
        const unsigned int idx = internal_libxs_registry_probe(
          registry->entries, registry->capacity, key, key_size, &found);
        if (0 != found) {
          result = internal_value_ptr(registry->entries + idx);
        }
      }
      if (NULL != lock) {
        LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_LOCKORDER);
      }
#if defined(INTERNAL_REG_CACHE)
      if (NULL != result) { /* populate TLS cache on miss */
        ce->registry = registry;
        ce->hash = hash;
        ce->key_size = key_size;
        ce->value = result;
        memcpy(ce->key, key, key_size);
      }
#endif
    }
  }
  return result;
}


LIBXS_API void libxs_registry_remove(libxs_registry_t* registry,
  const void* key, size_t key_size, libxs_lock_t* lock)
{
  if (NULL != registry && NULL != key && 0 < key_size
    && key_size <= LIBXS_REGKEY_MAXSIZE)
  {
    if (NULL != lock) {
      LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    }
    internal_libxs_registry_remove_impl(registry, key, key_size);
    if (NULL != lock) {
      LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_LOCKORDER);
    }
#if defined(INTERNAL_REG_CACHE)
    { const unsigned int hash = internal_libxs_regkey_hash(key, key_size);
      internal_libxs_regcache_entry_t* ce =
          internal_libxs_regcache + (hash & INTERNAL_REG_CACHE_MASK);
      if (ce->registry == registry && ce->hash == hash
        && ce->key_size == key_size
        && 0 == memcmp(ce->key, key, key_size))
      {
        ce->registry = NULL; /* invalidate */
      }
    }
#endif
  }
}


LIBXS_API int libxs_registry_extract(libxs_registry_t* registry,
  const void* key, size_t key_size,
  void* value_out, size_t value_size, libxs_lock_t* lock)
{
  int result = 0;
  if (NULL != registry && NULL != key && 0 < key_size
    && key_size <= LIBXS_REGKEY_MAXSIZE)
  {
    if (NULL != lock) {
      LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    }
    result = internal_libxs_registry_fetch_impl(
      registry, key, key_size, value_out, value_size, 1/*remove*/);
    if (NULL != lock) {
      LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_LOCKORDER);
    }
#if defined(INTERNAL_REG_CACHE)
    if (0 != result) {
      const unsigned int hash = internal_libxs_regkey_hash(key, key_size);
      internal_libxs_regcache_entry_t* ce =
          internal_libxs_regcache + (hash & INTERNAL_REG_CACHE_MASK);
      if (ce->registry == registry && ce->hash == hash
        && ce->key_size == key_size
        && 0 == memcmp(ce->key, key, key_size))
      {
        ce->registry = NULL;
      }
    }
#endif
  }
  return result;
}


LIBXS_API void* libxs_registry_begin(libxs_registry_t* registry,
  const void** key, size_t* cursor)
{
  void* result = NULL;
  if (NULL != registry && NULL != registry->entries) {
    unsigned int i;
    for (i = 0; i < registry->capacity; ++i) {
      if (INTERNAL_REG_USED == registry->entries[i].state) {
        if (NULL != key) *key = registry->entries[i].key;
        if (NULL != cursor) *cursor = (size_t)i;
        result = internal_value_ptr(registry->entries + i);
        break;
      }
    }
  }
  if (NULL == result) {
    if (NULL != key) *key = NULL;
    if (NULL != cursor) *cursor = 0;
  }
  return result;
}


LIBXS_API void* libxs_registry_next(libxs_registry_t* registry,
  const void** key, size_t* cursor)
{
  void* result = NULL;
  if (NULL != registry && NULL != registry->entries && NULL != cursor) {
    unsigned int i;
    for (i = (unsigned int)(*cursor) + 1; i < registry->capacity; ++i) {
      if (INTERNAL_REG_USED == registry->entries[i].state) {
        if (NULL != key) *key = registry->entries[i].key;
        *cursor = (size_t)i;
        result = internal_value_ptr(registry->entries + i);
        break;
      }
    }
  }
  if (NULL == result && NULL != key) {
    *key = NULL;
  }
  return result;
}


LIBXS_API int libxs_registry_has(libxs_registry_t* registry,
  const void* key, size_t key_size, libxs_lock_t* lock)
{
  int result = 0;
  if (NULL != registry && NULL != key && 0 < key_size
    && key_size <= LIBXS_REGKEY_MAXSIZE)
  {
    if (NULL != lock) {
      LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    }
    { int found = 0;
      internal_libxs_registry_probe(
        registry->entries, registry->capacity, key, key_size, &found);
      result = found;
    }
    if (NULL != lock) {
      LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_LOCKORDER);
    }
  }
  return result;
}


LIBXS_API size_t libxs_registry_value_size(libxs_registry_t* registry,
  const void* key, size_t key_size, libxs_lock_t* lock)
{
  size_t result = 0;
  if (NULL != registry && NULL != key && 0 < key_size
    && key_size <= LIBXS_REGKEY_MAXSIZE)
  {
    if (NULL != lock) {
      LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    }
    { int found = 0;
      const unsigned int idx = internal_libxs_registry_probe(
        registry->entries, registry->capacity, key, key_size, &found);
      if (0 != found) {
        result = registry->entries[idx].value_size;
      }
    }
    if (NULL != lock) {
      LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_LOCKORDER);
    }
  }
  return result;
}


LIBXS_API int libxs_registry_get_copy(const libxs_registry_t* registry,
  const void* key, size_t key_size,
  void* value_out, size_t value_size, libxs_lock_t* lock)
{
  int result = 0;
  if (NULL != registry && NULL != key && 0 < key_size
    && key_size <= LIBXS_REGKEY_MAXSIZE
    && NULL != value_out && 0 < value_size)
  {
    if (NULL != lock) {
      LIBXS_ATOMIC_ACQUIRE(lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER);
    }
    { libxs_registry_t* mutable_reg;
      memcpy(&mutable_reg, &registry, sizeof(registry));
      result = internal_libxs_registry_fetch_impl(
        mutable_reg, key, key_size, value_out, value_size, 0/*keep*/);
    }
    if (NULL != lock) {
      LIBXS_ATOMIC_RELEASE(lock, LIBXS_ATOMIC_LOCKORDER);
    }
  }
  return result;
}


LIBXS_API int libxs_registry_info(libxs_registry_t* registry,
  libxs_registry_info_t* info)
{
  int result = EXIT_FAILURE;
  if (NULL != registry && NULL != info && NULL != registry->entries) {
    unsigned int i;
    info->capacity = registry->capacity;
    info->size = registry->size;
    info->nbytes = (size_t)registry->capacity * sizeof(internal_libxs_regentry_t)
      + sizeof(libxs_registry_t);
    for (i = 0; i < registry->capacity; ++i) {
      if (INTERNAL_REG_USED == registry->entries[i].state) {
        info->nbytes += registry->entries[i].value_size;
      }
    }
    result = EXIT_SUCCESS;
  }
  return result;
}
