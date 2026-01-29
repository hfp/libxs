/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_UTILS_MHD_H
#define LIBXS_UTILS_MHD_H

#include "../libxs_typedefs.h"


/** Denotes the element/pixel type of an image/channel. */
typedef enum libxs_mhd_elemtype {
  LIBXS_MHD_ELEMTYPE_F64  = LIBXS_DATATYPE_F64,   /* MET_DOUBLE */
  LIBXS_MHD_ELEMTYPE_F32  = LIBXS_DATATYPE_F32,   /* MET_FLOAT */
  LIBXS_MHD_ELEMTYPE_F16  = LIBXS_DATATYPE_F16,   /* MET_HALF */
  LIBXS_MHD_ELEMTYPE_BF16 = LIBXS_DATATYPE_BF16,  /* MET_BFLOAT */
  LIBXS_MHD_ELEMTYPE_BF8  = LIBXS_DATATYPE_BF8,   /* MET_BFLOAT8 */
  LIBXS_MHD_ELEMTYPE_I64  = LIBXS_DATATYPE_I64,   /* MET_LONG */
  LIBXS_MHD_ELEMTYPE_I32  = LIBXS_DATATYPE_I32,   /* MET_INT */
  LIBXS_MHD_ELEMTYPE_I16  = LIBXS_DATATYPE_I16,   /* MET_SHORT */
  LIBXS_MHD_ELEMTYPE_I8   = LIBXS_DATATYPE_I8,    /* MET_CHAR */
  LIBXS_MHD_ELEMTYPE_U64  = LIBXS_DATATYPE_U64,   /* MET_ULONG */
  LIBXS_MHD_ELEMTYPE_U32  = LIBXS_DATATYPE_U32,   /* MET_UINT */
  LIBXS_MHD_ELEMTYPE_U16  = LIBXS_DATATYPE_U16,   /* MET_USHORT */
  LIBXS_MHD_ELEMTYPE_U8   = LIBXS_DATATYPE_U8,    /* MET_UCHAR */
  LIBXS_MHD_ELEMTYPE_UNKNOWN = LIBXS_DATATYPE_UNSUPPORTED
} libxs_mhd_elemtype;

typedef enum libxs_mhd_element_conversion_hint {
  LIBXS_MHD_ELEMENT_CONVERSION_DEFAULT,
  LIBXS_MHD_ELEMENT_CONVERSION_MODULUS
} libxs_mhd_element_conversion_hint;

LIBXS_EXTERN_C typedef struct libxs_mhd_element_handler_info {
  libxs_mhd_elemtype type;
  libxs_mhd_element_conversion_hint hint;
} libxs_mhd_element_handler_info;

/**
 * Function type used for custom data-handler or element conversion.
 * The value-range (src_min, src_max) may be used to scale values
 * in case of a type-conversion.
 */
LIBXS_EXTERN_C typedef int (*libxs_mhd_element_handler)(void* dst,
  const libxs_mhd_element_handler_info* dst_info, libxs_mhd_elemtype src_type,
  const void* src, const void* src_min, const void* src_max);

/**
 * Predefined function to perform element data conversion.
 * Scales source-values in case of non-NULL src_min and src_max,
 * or otherwise clamps to the destination-type.
 */
LIBXS_API int libxs_mhd_element_conversion(void* dst,
  const libxs_mhd_element_handler_info* dst_info, libxs_mhd_elemtype src_type,
  const void* src, const void* src_min, const void* src_max);

/**
 * Predefined function to check a buffer against file content.
 * In case of different types, libxs_mhd_element_conversion
 * is performed to compare values using the source-type.
 */
LIBXS_API int libxs_mhd_element_comparison(void* dst,
  const libxs_mhd_element_handler_info* dst_info, libxs_mhd_elemtype src_type,
  const void* src, const void* src_min, const void* src_max);


/** Returns the name of the element type; result may be NULL/0 in case of an unknown type. */
LIBXS_API const char* libxs_mhd_typename(libxs_mhd_elemtype type, const char** ctypename);

/** Returns the type of the element for a given type-name, e.g., "MET_FLOAT". */
LIBXS_API libxs_mhd_elemtype libxs_mhd_typeinfo(const char elemname[]);

/** Returns the size of the element-type in question. */
LIBXS_API size_t libxs_mhd_typesize(libxs_mhd_elemtype type);


/**
 * Parse the header of an MHD-file. The header can be part of the data file (local),
 * or separately stored (header: MHD, data MHA or RAW).
 */
LIBXS_API int libxs_mhd_read_header(
  /* Filename referring to the header-file (may also contain the data). */
  const char header_filename[],
  /* Maximum length of path/file name. */
  size_t filename_max_length,
  /* Filename containing the data (may be the same as the header-file). */
  char filename[],
  /* Yields the maximum/possible number of dimensions on input,
   * and the actual number of dimensions on output. */
  size_t* ndims,
  /* Image extents ("ndims" number of entries). */
  size_t size[],
  /* Number of interleaved image channels. */
  size_t* ncomponents,
  /* Type of the image elements (pixel type). */
  libxs_mhd_elemtype* type,
  /* Size of the header in bytes; may be used to skip the header,
   * when reading content; can be a NULL-argument (optional). */
  size_t* header_size,
  /* Size (in Bytes) of an user-defined extended data record;
   * can be a NULL-argument (optional). */
  size_t* extension_size);


/**
 * Loads the data file, and optionally allows data conversion.
 * Conversion is performed such that values are clamped to fit
 * into the destination.
 */
LIBXS_API int libxs_mhd_read(
  /* Filename referring to the data. */
  const char filename[],
  /* Offset within pitched buffer (NULL: no offset). */
  const size_t offset[],
  /* Image dimensions (extents). */
  const size_t size[],
  /* Leading buffer dimensions (NULL: same as size). */
  const size_t pitch[],
  /* Dimensionality (number of entries in size). */
  size_t ndims,
  /* Number of interleaved image channels. */
  size_t ncomponents,
  /* Used to skip the header, and to only read the data. */
  size_t header_size,
  /* Data element type as stored (pixel type). */
  libxs_mhd_elemtype type_stored,
  /* Buffer where the data is read into. */
  void* data,
  /**
   * Destination info including type. If given, data
   * is type-converted (no custom handler necessary).
   */
  const libxs_mhd_element_handler_info* handler_info,
  /**
   * Optional callback executed per entry when reading the data.
   * May assign the value to the left-most argument, but also
   * allows to only compare with present data. Can be used to
   * avoid allocating an actual destination.
   */
  libxs_mhd_element_handler handler,
  /* Post-content data (extension, optional). */
  char extension[],
  /* Size of the extension; can be zero. */
  size_t extension_size);


/**
 * Save a file using an extended data format, which is compatible with the Meta Image Format (MHD).
 * The file is suitable for visual inspection using, e.g., ITK-SNAP or ParaView.
 */
LIBXS_API int libxs_mhd_write(const char filename[],
  /* Offset within pitched buffer (NULL: no offset). */
  const size_t offset[],
  /* Image dimensions (extents). */
  const size_t size[],
  /* Leading buffer dimensions (NULL: same as size). */
  const size_t pitch[],
  /* Dimensionality, i.e., number of entries in data_size/size. */
  size_t ndims,
  /* Number of pixel components. */
  size_t ncomponents,
  /* Type (input). */
  libxs_mhd_elemtype type_data,
  /* Raw data to be saved. */
  const void* data,
  /**
   * Destination info including type. If given, data
   * is type-converted (no custom handler necessary).
   */
  const libxs_mhd_element_handler_info* handler_info,
  /**
   * Optional callback executed per entry when reading the data.
   * May assign the value to the left-most argument, but also
   * allows to only compare with present data. Can be used to
   * avoid allocating an actual destination.
   */
  libxs_mhd_element_handler handler,
  /* Size of the header; can be a NULL-argument (optional). */
  size_t* header_size,
  /* Extension header data; can be NULL. */
  const char extension_header[],
  /* Extension data stream; can be NULL. */
  const void* extension,
  /* Extension data size; can be NULL. */
  size_t extension_size);

#endif /*LIBXS_UTILS_MHD_H*/
