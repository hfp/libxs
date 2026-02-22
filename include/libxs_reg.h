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
typedef struct libxs_registry_t libxs_registry_t;

/** Lock type for explicit-lock function variants. */
typedef LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_registry_lock_t;

/** Create registry object. Returns EXIT_SUCCESS or EXIT_FAILURE. */
LIBXS_API int libxs_registry_create(libxs_registry_t** registry);

/** Destroy registry object (release all entries). */
LIBXS_API void libxs_registry_destroy(libxs_registry_t* registry);

/**
 * Enumerate registry. Caller must initialize *cursor to 0 before
 * the first call. Returns the value pointer of the first occupied
 * entry, or NULL when the registry is empty.
 */
LIBXS_API void* libxs_registry_begin(libxs_registry_t* registry,
  const void** key, size_t* cursor);

/**
 * Advance to the next entry. Returns the value pointer of the
 * next occupied entry, or NULL when iteration is complete.
 */
LIBXS_API void* libxs_registry_next(libxs_registry_t* registry,
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
 */
LIBXS_API void* libxs_registry_set(libxs_registry_t* registry, const void* key, size_t key_size,
  const void* value_init, size_t value_size);
/** Similar to libxs_registry_set but using a caller-provided lock. */
LIBXS_API void* libxs_registry_set_lock(libxs_registry_t* registry, const void* key, size_t key_size,
  const void* value_init, size_t value_size, libxs_registry_lock_t* lock);

/** Query registered value by key; returns NULL if not found. */
LIBXS_API void* libxs_registry_get(libxs_registry_t* registry, const void* key, size_t key_size);
/** Similar to libxs_registry_get but using a caller-provided lock. */
LIBXS_API void* libxs_registry_get_lock(libxs_registry_t* registry, const void* key, size_t key_size,
  libxs_registry_lock_t* lock);

/** Check whether a key exists. Returns non-zero if found, zero otherwise. */
LIBXS_API int libxs_registry_has(libxs_registry_t* registry, const void* key, size_t key_size);
/** Similar to libxs_registry_has but using a caller-provided lock. */
LIBXS_API int libxs_registry_has_lock(libxs_registry_t* registry, const void* key, size_t key_size,
  libxs_registry_lock_t* lock);

/** Query the stored value size (Bytes) for a given key. Returns 0 if not found. */
LIBXS_API size_t libxs_registry_value_size(libxs_registry_t* registry,
  const void* key, size_t key_size);
/** Similar to libxs_registry_value_size but using a caller-provided lock. */
LIBXS_API size_t libxs_registry_value_size_lock(libxs_registry_t* registry,
  const void* key, size_t key_size, libxs_registry_lock_t* lock);

/** Remove key-value pair from registry and release associated memory. */
LIBXS_API void libxs_registry_remove(libxs_registry_t* registry, const void* key, size_t key_size);
/** Similar to libxs_registry_remove but using a caller-provided lock. */
LIBXS_API void libxs_registry_remove_lock(libxs_registry_t* registry, const void* key, size_t key_size,
  libxs_registry_lock_t* lock);

/** Structure to receive the status of the registry. */
LIBXS_EXTERN_C typedef struct libxs_registry_info_t {
  size_t capacity, size, nbytes;
} libxs_registry_info_t;

/** Get information about the registry. */
LIBXS_API int libxs_registry_info(libxs_registry_t* registry, libxs_registry_info_t* info);

#endif /*LIBXS_REG_H*/
