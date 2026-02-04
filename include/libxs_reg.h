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


/** TODO */
LIBXS_API void* libxs_registry_create();

/** TODO */
LIBXS_API void* libxs_registry_destroy();

/** Enumerate registry; result can be NULL (no entry found). */
LIBXS_API void* libxs_registry_begin(const void** key);

/** Receive next (or NULL) based on given entry (see libxs_registry_begin). */
LIBXS_API void* libxs_registry_next(const void* regentry, const void** key);

/**
 * Register user-defined key-value; value can be queried (libxs_registry_get).
 * Since the key-type is unknown to LIBXS, the key must be binary reproducible,
 * i.e., a structured type (can be padded) must be initialized like a binary blob
 * (memset) followed by an element-wise initialization. The size of the
 * key is limited (see documentation). The given value is copied by LIBXS and
 * can be initialized prior to registration or whenever queried. Registered data
 * is released when the program terminates but can released if needed
 * (libxs_registry_free), .e.g., in case of a larger value reusing the same key.
 */
LIBXS_API void* libxs_registry_set(const void* key, size_t key_size,
  size_t value_size, const void* value_init);

/** Query user-defined value from LIBXS's code registry. */
LIBXS_API void* libxs_registry_get(const void* key, size_t key_size);

/** Remove key-value pair from code registry and release memory. */
LIBXS_API void libxs_registry_free(const void* key, size_t key_size);

/** Structure to receive the status of the code registry. */
LIBXS_EXTERN_C typedef struct libxs_registry_info_t {
  size_t capacity, size, nbytes, nstatic, ncache;
} libxs_registry_info_t;

/** Get information about the code registry. */
LIBXS_API int libxs_registry_info(libxs_registry_info_t* info);

#endif /*LIBXS_REG_H*/
