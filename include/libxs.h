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

/** Semantic version according to https://semver.org/. */
#include "libxs_version.h"
#include "libxs_macros.h"

/* Construct an enumerator (libxs_datatype) from a built-in type (float, double, etc.). */
#define LIBXS_DATATYPE(TYPE) LIBXS_CONCATENATE(LIBXS_DATATYPE_, LIBXS_TYPESYMBOL(TYPE))
/** Helper macro for type postfixes. */
#define LIBXS_TYPESYMBOL(TYPE) LIBXS_CONCATENATE(LIBXS_TYPESYMBOL_, TYPE)
#define LIBXS_TYPESYMBOL_double F64
#define LIBXS_TYPESYMBOL_float F32
#define LIBXS_TYPESYMBOL_int I32
#define LIBXS_TYPESYMBOL_short I16
#define LIBXS_TYPESYMBOL_char I8

/** Determine the type-size of the type (libxs_datatype). */
#define LIBXS_TYPESIZE(ENUM) ((ENUM) >> 4)

/** Enumerate primitive element/data types. */
typedef enum libxs_datatype {
  LIBXS_DATATYPE_UNKNOWN = 0,
  LIBXS_DATATYPE_F64 =  1 | (8 << 4),
  LIBXS_DATATYPE_F32 =  2 | (4 << 4),
  LIBXS_DATATYPE_I64 =  3 | (8 << 4),
  LIBXS_DATATYPE_U64 =  4 | (8 << 4),
  LIBXS_DATATYPE_I32 =  5 | (4 << 4),
  LIBXS_DATATYPE_U32 =  6 | (4 << 4),
  LIBXS_DATATYPE_I16 =  7 | (2 << 4),
  LIBXS_DATATYPE_U16 =  8 | (2 << 4),
  LIBXS_DATATYPE_I8  =  9 | (1 << 4),
  LIBXS_DATATYPE_U8  = 10 | (1 << 4)
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

/** Return the name of the type (libxs_datatype). */
LIBXS_API const char* libxs_typename(libxs_datatype datatype);

/** Determine the given value in double-precision. */
LIBXS_API int libxs_dvalue(libxs_datatype datatype, const void* value, double* dvalue);

#endif /*LIBXS_H*/
