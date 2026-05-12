/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_REG_H
#define LIBXS_REG_H

#include "libxs_sync.h"

/** Maximum key size in Bytes (binary-reproducible keys). */
#if !defined(LIBXS_REGKEY_MAXSIZE)
# define LIBXS_REGKEY_MAXSIZE 64
#endif
/** Backward compatibility. */
#if !defined(LIBXS_DESCRIPTOR_MAXSIZE)
# define LIBXS_DESCRIPTOR_MAXSIZE LIBXS_REGKEY_MAXSIZE
#endif
/** Initial number of hash-table buckets (must be POT). */
#if !defined(LIBXS_REGISTRY_NBUCKETS)
# define LIBXS_REGISTRY_NBUCKETS 64
#endif
/** Thread-local cache entries per thread (POT, 0 to disable). */
#if !defined(LIBXS_REGCACHE_NENTRIES)
# define LIBXS_REGCACHE_NENTRIES 16
#endif


/** Opaque registry type. */
LIBXS_EXTERN_C typedef struct libxs_registry_t libxs_registry_t;

/** Structure to receive the status of the registry. */
LIBXS_EXTERN_C typedef struct libxs_registry_info_t {
  size_t capacity, size, nbytes;
} libxs_registry_info_t;


/** Create registry object. Returns NULL in case of an error. */
LIBXS_API libxs_registry_t* libxs_registry_create(void);

/** Destroy registry object (release all entries). */
LIBXS_API void libxs_registry_destroy(libxs_registry_t* registry);

/** Return pointer to the registry's internal lock (for use as lock argument). */
LIBXS_API libxs_lock_t* libxs_registry_lock(libxs_registry_t* registry);

/**
 * Enumerate registry. Caller must initialize *cursor to 0 before
 * the first call. Returns the value pointer of the first occupied
 * entry, or NULL when the registry is empty.
 */
LIBXS_API void* libxs_registry_begin(const libxs_registry_t* registry,
  const void** key, size_t* cursor);

/**
 * Advance to the next entry. Returns the value pointer of the
 * next occupied entry, or NULL when iteration is complete.
 */
LIBXS_API void* libxs_registry_next(const libxs_registry_t* registry,
  const void** key, size_t* cursor);

/**
 * Register user-defined key-value pair; value can be queried (libxs_registry_get).
 * Since the key-type is unknown to LIBXS, the key must be binary reproducible,
 * i.e., a structured type (can be padded) must be initialized like a binary blob
 * (memset) followed by an element-wise initialization. The size of the key is
 * limited to LIBXS_REGKEY_MAXSIZE. The given value is copied by the registry and
 * can be initialized prior to registration or when queried (returned pointer).
 * Registered data is released by libxs_registry_remove or libxs_registry_destroy.
 * Re-registering an existing key automatically reallocates if the new value
 * is larger than the currently stored one.
 * If lock is NULL, no locking is performed (caller guarantees exclusion);
 * otherwise the provided lock is acquired/released around the operation.
 * Use libxs_registry_lock(registry) to obtain the registry's internal lock.
 */
LIBXS_API void* libxs_registry_set(libxs_registry_t* registry, const void* key, size_t key_size,
  const void* value_init, size_t value_size, libxs_lock_t* LIBXS_ARGDEF(lock, NULL));

/** Query registered value by key; returns NULL if not found. */
LIBXS_API void* libxs_registry_get(const libxs_registry_t* registry, const void* key, size_t key_size,
  libxs_lock_t* LIBXS_ARGDEF(lock, NULL));

/**
 * Thread-safe query: copies up to value_size bytes of the stored value into
 * value_out under the lock. Returns non-zero if the key was found, zero
 * otherwise. Unlike libxs_registry_get, the caller never sees a raw pointer
 * into the registry's internal storage.
 */
LIBXS_API int libxs_registry_get_copy(const libxs_registry_t* registry, const void* key, size_t key_size,
  void* value_out, size_t value_size, libxs_lock_t* LIBXS_ARGDEF(lock, NULL));

/** Check whether a key exists. Returns non-zero if found, zero otherwise. */
LIBXS_API int libxs_registry_has(const libxs_registry_t* registry, const void* key, size_t key_size,
  libxs_lock_t* LIBXS_ARGDEF(lock, NULL));

/** Query the stored value size (Bytes) for a given key. Returns 0 if not found. */
LIBXS_API size_t libxs_registry_value_size(const libxs_registry_t* registry,
  const void* key, size_t key_size, libxs_lock_t* LIBXS_ARGDEF(lock, NULL));

/** Remove key-value pair from registry and release associated memory. */
LIBXS_API void libxs_registry_remove(libxs_registry_t* registry, const void* key, size_t key_size,
  libxs_lock_t* LIBXS_ARGDEF(lock, NULL));

/**
 * Atomically retrieve and remove a key-value pair. Copies up to value_size
 * bytes of the stored value into value_out (if non-NULL), then removes the
 * entry. Returns non-zero if the key was found, zero otherwise.
 */
LIBXS_API int libxs_registry_extract(libxs_registry_t* registry, const void* key, size_t key_size,
  void* value_out, size_t value_size, libxs_lock_t* LIBXS_ARGDEF(lock, NULL));

/** Get information about the registry. */
LIBXS_API int libxs_registry_info(const libxs_registry_t* registry, libxs_registry_info_t* info);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_REG_H*/
