/******************************************************************************
** Copyright (c) 2009-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
#ifndef LIBXS_MHD_H
#define LIBXS_MHD_H

#include <libxs.h>


/** Denotes the element/pixel type of an image/channel. */
typedef enum libxs_mhd_elemtype {
  LIBXS_MHD_ELEMTYPE_F64  = LIBXS_DATATYPE_F64,     /* MET_DOUBLE */
  LIBXS_MHD_ELEMTYPE_F32  = LIBXS_DATATYPE_F32,     /* MET_FLOAT */
  LIBXS_MHD_ELEMTYPE_I32  = LIBXS_DATATYPE_I32,     /* MET_INT */
  LIBXS_MHD_ELEMTYPE_I16  = LIBXS_DATATYPE_I16,     /* MET_SHORT */
  LIBXS_MHD_ELEMTYPE_U8   = LIBXS_DATATYPE_I8,      /* MET_UCHAR */
  LIBXS_MHD_ELEMTYPE_I8,                              /* MET_CHAR */
  LIBXS_MHD_ELEMTYPE_CHAR = LIBXS_MHD_ELEMTYPE_I8,  /* MET_CHAR */
  LIBXS_MHD_ELEMTYPE_U16, LIBXS_MHD_ELEMTYPE_U32,   /* MET_USHORT, MET_UINT */
  LIBXS_MHD_ELEMTYPE_U64, LIBXS_MHD_ELEMTYPE_I64,   /* MET_ULONG,  MET_LONG */
  LIBXS_MHD_ELEMTYPE_UNKNOWN
} libxs_mhd_elemtype;


/** Function type used for custom data-handler or element conversion. */
typedef LIBXS_RETARGETABLE int (*libxs_mhd_element_handler)(
  void* dst, libxs_mhd_elemtype dst_type,
  const void* src, libxs_mhd_elemtype src_type);

/** Predefined function to check a buffer against file content. */
LIBXS_API int libxs_mhd_element_comparison(void* dst, libxs_mhd_elemtype dst_type, const void* src, libxs_mhd_elemtype src_type);


/** Returns the name and size of the element type; result may be NULL/0 in case of an unknown type. */
LIBXS_API const char* libxs_mhd_typename(libxs_mhd_elemtype type, size_t typesize[], const char** ctypename);

/** Returns the type of the element for a given type-name. */
LIBXS_API libxs_mhd_elemtype libxs_mhd_typeinfo(const char elemname[]);


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
  /* Number of image components (channels). */
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
  /** Offset within pitched buffer (NULL: no offset). */
  const size_t offset[],
  /** Image dimensions (extents). */
  const size_t size[],
  /** Leading buffer dimensions (NULL: same as size). */
  const size_t pitch[],
  /* Dimensionality (number of entries in size). */
  size_t ndims,
  /* Number of image components (channels). */
  size_t ncomponents,
  /* Used to skip the header, and to only read the data. */
  size_t header_size,
  /* Data element type as stored (pixel type). */
  libxs_mhd_elemtype type_stored,
  /* Storage type (data conversion, optional). */
  const libxs_mhd_elemtype* type_data,
  /* Buffer where the data is read into. */
  void* data,
  /**
   * Optional callback executed per entry when reading the data.
   * May assign the value to the left-most argument, but also
   * allows to only compare with present data. Can be used to
   * avoid allocating an actual destination.
   */
  libxs_mhd_element_handler handle_element,
  /* Post-content data (extension, optional). */
  char extension[],
  /* Size of the extension; can be zero. */
  size_t extension_size);


/**
 * Save a file using an extended data format, which is compatible with the Meta Image Format (MHD).
 * The file is suitable for visual inspection using e.g., ITK-SNAP or ParaView.
 */
LIBXS_API int libxs_mhd_write(const char filename[],
  /** Offset within pitched buffer (NULL: no offset). */
  const size_t offset[],
  /** Image dimensions (extents). */
  const size_t size[],
  /** Leading buffer dimensions (NULL: same as size). */
  const size_t pitch[],
  /** Dimensionality i.e., number of entries in data_size/size. */
  size_t ndims,
  /** Number of pixel components. */
  size_t ncomponents,
  /** Storage type. */
  libxs_mhd_elemtype type,
  /** Raw data to be saved. */
  const void* data,
  /** Size of the header; can be a NULL-argument (optional). */
  size_t* header_size,
  /** Extension header data; can be NULL. */
  const char extension_header[],
  /** Extension data stream; can be NULL. */
  const void* extension,
  /** Extension data size; can be NULL. */
  size_t extension_size);

#endif /*LIBXS_MHD_H*/
