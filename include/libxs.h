/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_H
#define LIBXS_H

#include "libxs_macros.h"

/**
 * Strings to denote the version of LIBXS (libxs_config.h).
 * LIBXS_VERSION: Name of the version (stringized version numbers).
 * LIBXS_BRANCH:  Name of the branch this version is derived from.
 */
#define LIBXS_VERSION LIBXS_CONFIG_VERSION
#define LIBXS_BRANCH  LIBXS_CONFIG_BRANCH

/**
 * Semantic version according to https://semver.org/ (see also libxs_config.h).
 * LIBXS_VERSION_MAJOR:  Major version derived from the most recent RCS-tag.
 * LIBXS_VERSION_MINOR:  Minor version derived from the most recent RCS-tag.
 * LIBXS_VERSION_UPDATE: Update number derived from the most recent RCS-tag.
 * LIBXS_VERSION_PATCH:  Patch number based on distance to most recent RCS-tag.
 */
#define LIBXS_VERSION_MAJOR  LIBXS_CONFIG_VERSION_MAJOR
#define LIBXS_VERSION_MINOR  LIBXS_CONFIG_VERSION_MINOR
#define LIBXS_VERSION_UPDATE LIBXS_CONFIG_VERSION_UPDATE
#define LIBXS_VERSION_PATCH  LIBXS_CONFIG_VERSION_PATCH


/** Initialize the library; pay for setup cost at a specific point. */
LIBXS_API void libxs_init(void);
/** De-initialize the library and free internal memory (optional). */
LIBXS_API void libxs_finalize(void);

/** Get the level of verbosity. */
LIBXS_API int libxs_get_verbosity(void);
/**
 * Set the level of verbosity (0: off, positive value: verbosity level,
 * negative value: maximum verbosity, which also dumps JIT-code)
 */
LIBXS_API void libxs_set_verbosity(int level);

/** Enumerates primitive element/data types. */
typedef enum libxs_datatype {
  LIBXS_DATATYPE_F64,
  LIBXS_DATATYPE_F32,
  LIBXS_DATATYPE_I64,
  LIBXS_DATATYPE_U64,
  LIBXS_DATATYPE_I32,
  LIBXS_DATATYPE_U32,
  LIBXS_DATATYPE_I16,
  LIBXS_DATATYPE_U16,
  LIBXS_DATATYPE_I8,
  LIBXS_DATATYPE_U8,
  LIBXS_DATATYPE_UNSUPPORTED
} libxs_datatype;

/** Returns the type-size of the type (libxs_datatype). */
LIBXS_API int libxs_typesize(libxs_datatype datatype);
/** Returns the name of the type (libxs_datatype). */
LIBXS_API const char* libxs_get_typename(libxs_datatype datatype);
/** Determines the given value in double-precision. */
LIBXS_API int libxs_dvalue(libxs_datatype datatype, const void* value, double* dvalue);

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

#endif /*LIBXS_H*/
