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

/* Construct an enumerator (libxs_data_t) from a built-in type (float, double, etc.). */
#define LIBXS_DATATYPE(TYPE) LIBXS_CONCATENATE(LIBXS_DATATYPE_, LIBXS_TYPESYMBOL(TYPE))
/** Helper macro for type postfixes. */
#define LIBXS_TYPESYMBOL(TYPE) LIBXS_CONCATENATE(LIBXS_TYPESYMBOL_, TYPE)
#define LIBXS_TYPESYMBOL_double F64
#define LIBXS_TYPESYMBOL_float F32
#define LIBXS_TYPESYMBOL_int I32
#define LIBXS_TYPESYMBOL_short I16
#define LIBXS_TYPESYMBOL_char I8

/** Determine the type-size of the type (libxs_data_t). */
#define LIBXS_TYPESIZE(ENUM) ((ENUM) >> 4)

/** Removes type-size info from type (libxs_data_t). */
#define LIBXS_TYPEORDER(ENUM) (0xF & (ENUM))


/** Enumerate primitive element/data types. */
typedef enum libxs_data_t {
  LIBXS_DATATYPE_F64 = 0 | (8 << 4),
  LIBXS_DATATYPE_F32 = 1 | (4 << 4),
  LIBXS_DATATYPE_C64 = 2 | (16 << 4),
  LIBXS_DATATYPE_C32 = 3 | (8 << 4),
  LIBXS_DATATYPE_I64 = 4 | (8 << 4),
  LIBXS_DATATYPE_U64 = 5 | (8 << 4),
  LIBXS_DATATYPE_I32 = 6 | (4 << 4),
  LIBXS_DATATYPE_U32 = 7 | (4 << 4),
  LIBXS_DATATYPE_I16 = 8 | (2 << 4),
  LIBXS_DATATYPE_U16 = 9 | (2 << 4),
  LIBXS_DATATYPE_I8  = 10 | (1 << 4),
  LIBXS_DATATYPE_U8  = 11 | (1 << 4),
  LIBXS_DATATYPE_UNKNOWN = 12
} libxs_data_t;

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

/** Return the name of the type (libxs_data_t). */
LIBXS_API const char* libxs_typename(libxs_data_t datatype);

/** Determine the given value in double-precision. */
LIBXS_API int libxs_dvalue(libxs_data_t datatype, const void* value, double* dvalue);

/* header-only: include implementation when not inside another LIBXS header
 * (deferred to the end of the outermost header that triggered the chain). */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_MATH_H) && !defined(LIBXS_CPUID_H) && !defined(LIBXS_GEMM_H) \
 && !defined(LIBXS_MHD_H) && !defined(LIBXS_TIMER_H) && !defined(LIBXS_MEM_H) \
 && !defined(LIBXS_SYNC_H) && !defined(LIBXS_UTILS_H) && !defined(LIBXS_RNG_H) \
 && !defined(LIBXS_HIST_H) && !defined(LIBXS_MALLOC_H) && !defined(LIBXS_REG_H) \
 && !defined(LIBXS_PERM_H) && !defined(LIBXS_STR_H) && !defined(LIBXS_HASH_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_H*/
