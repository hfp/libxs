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
 * Semantic version according to https://semver.org/.
 * LIBXS_VERSION_MAJOR:  Major version derived from the most recent RCS-tag.
 * LIBXS_VERSION_MINOR:  Minor version derived from the most recent RCS-tag.
 * LIBXS_VERSION_UPDATE: Update number derived from the most recent RCS-tag.
 */
#define LIBXS_VERSION_MAJOR  1
#define LIBXS_VERSION_MINOR  0
#define LIBXS_VERSION_UPDATE 0

/** String to denote the version of LIBXS. */
#define LIBXS_VERSION \
  LIBXS_STRINGIFY(LIBXS_VERSION_MAJOR) "." \
  LIBXS_STRINGIFY(LIBXS_VERSION_MINOR) "." \
  LIBXS_STRINGIFY(LIBXS_VERSION_UPDATE)

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
  LIBXS_DATATYPE_UNKNOWN
} libxs_datatype;

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

/** Returns the type-size of the type (libxs_datatype). */
LIBXS_API int libxs_typesize(libxs_datatype datatype);

/** Returns the name of the type (libxs_datatype). */
LIBXS_API const char* libxs_typename(libxs_datatype datatype);

/** Determines the given value in double-precision. */
LIBXS_API int libxs_dvalue(libxs_datatype datatype, const void* value, double* dvalue);

#endif /*LIBXS_H*/
