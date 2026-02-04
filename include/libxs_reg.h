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


/** Structure to receive information about the code registry status (libxs_get_registry_info). */
LIBXS_EXTERN_C typedef struct libxs_registry_info {
  size_t capacity, size, nbytes, nstatic, ncache;
} libxs_registry_info;

/** Get information about the code registry. */
LIBXS_API int libxs_get_registry_info(libxs_registry_info* info);
/** Enumerate registry; result can be NULL (no entry found). */
LIBXS_API void* libxs_get_registry_begin(const void** key);
/** Receive next (or NULL) based on given entry (see libxs_get_registry_begin). */
LIBXS_API void* libxs_get_registry_next(const void* regentry, const void** key);

/**
 * Register user-defined key-value; value can be queried (libxs_xdispatch).
 * Since the key-type is unknown to LIBXS, the key must be binary reproducible,
 * i.e., a structured type (can be padded) must be initialized like a binary blob
 * (memset) followed by an element-wise initialization. The size of the
 * key is limited (see documentation). The given value is copied by LIBXS and
 * can be initialized prior to registration or whenever queried. Registered data
 * is released when the program terminates but can released if needed
 * (libxs_xrelease), .e.g., in case of a larger value reusing the same key.
 */
LIBXS_API void* libxs_xregister(const void* key, size_t key_size,
  size_t value_size, const void* value_init);
/** Query user-defined value from LIBXS's code registry. */
LIBXS_API void* libxs_xdispatch(const void* key, size_t key_size);
/** Remove key-value pair from code registry and release memory. */
LIBXS_API void libxs_xrelease(const void* key, size_t key_size);

#endif /*LIBXS_REG_H*/
