/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
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
 * Only the first key_size Bytes of *key are meaningful; a registry
 * holding keys of differing size must be enumerated with
 * libxs_registry_begin_length to recover each entry's key size.
 * *key is suitably aligned to be cast to the caller's key type.
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
 * Like libxs_registry_begin, but also yields the size of the entry's key
 * (key_size may be NULL). Needed when a single registry holds keys of
 * differing size, since the Bytes of *key beyond key_size are undefined.
 */
LIBXS_API void* libxs_registry_begin_length(const libxs_registry_t* registry,
  const void** key, size_t* key_size, size_t* cursor);

/** Like libxs_registry_next, but also yields the entry's key size. */
LIBXS_API void* libxs_registry_next_length(const libxs_registry_t* registry,
  const void** key, size_t* key_size, size_t* cursor);

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
 * Hash a key the way this registry does. The seed is per-registry, hence the
 * result is only valid for the given registry. Pass it to the _hashed flavors
 * to hash a key once and reuse the value across several operations (and to
 * index caller-side tables keyed by the same shape).
 *
 * Zero for arguments the registry itself would reject, i.e. a key_size outside
 * 1 to LIBXS_REGKEY_MAXSIZE: only a storable key is hashed, so the function
 * never reads a byte the caller did not promise.
 */
LIBXS_API unsigned int libxs_registry_hash(const libxs_registry_t* registry,
  const void* key, size_t key_size);

/**
 * libxs_registry_get with a precomputed hash (libxs_registry_hash).
 * The thread-local lookup cache is consulted before the lock is taken, hence a
 * repeated query of the same key costs no lock. That fast path is restricted to
 * values too large for the registry's inline storage, whose buffer does not move
 * when the table is rehashed. It assumes the entry is not removed, nor its value
 * resized, while another thread queries it (true if the registry only grows).
 * libxs_registry_get is unaffected and keeps taking the lock whenever one
 * is given.
 */
LIBXS_API void* libxs_registry_get_hashed(const libxs_registry_t* registry,
  const void* key, size_t key_size, unsigned int hash,
  libxs_lock_t* LIBXS_ARGDEF(lock, NULL));

/** libxs_registry_set with a precomputed hash (libxs_registry_hash). */
LIBXS_API void* libxs_registry_set_hashed(libxs_registry_t* registry,
  const void* key, size_t key_size, unsigned int hash,
  const void* value_init, size_t value_size, libxs_lock_t* LIBXS_ARGDEF(lock, NULL));

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

/** Number of entries (libxs_registry_info without inspecting entries). */
LIBXS_API size_t libxs_registry_size(const libxs_registry_t* registry);

/** Get information about the registry. */
LIBXS_API int libxs_registry_info(const libxs_registry_t* registry, libxs_registry_info_t* info);

/**
 * Save registry to a binary buffer.
 * buffer: destination (may be NULL to query required size).
 * size: on input, available buffer size in bytes;
 *       on output, bytes written (or required if buffer is NULL).
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
LIBXS_API int libxs_registry_save(const libxs_registry_t* registry,
  void* buffer, size_t* size);

/**
 * Load registry from a binary buffer (previously saved with libxs_registry_save).
 * When fixup is NULL, values are loaded lazily: keys are materialized immediately
 * but value data is read from the buffer on first access (libxs_registry_get);
 * the buffer must remain valid for the lifetime of the returned registry.
 * When fixup is non-NULL, each entry is heap-allocated immediately and fixup
 * is called once per entry, allowing the caller to rewrite values that contain
 * pointers or handles (e.g., re-registering function pointers by key identity).
 * Returns a new registry, or NULL on failure.
 */
LIBXS_API libxs_registry_t* libxs_registry_load(const void* buffer, size_t size,
  void (*fixup)(void* value, const void* key, size_t key_size,
    size_t value_size, void* udata),
  void* LIBXS_ARGDEF(udata, NULL));

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_GEMM_H) && !defined(LIBXS_PERM_H) \
 && !defined(LIBXS_PREDICT_H) && !defined(LIBXS_TOKEN_H) \
 && !defined(LIBXS_NGRAM_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_REG_H*/
