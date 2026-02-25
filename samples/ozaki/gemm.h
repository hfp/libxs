/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_math.h>
#include <libxs_mhd.h>

#if !defined(GEMM_REAL_TYPE)
# define GEMM_REAL_TYPE double
#endif
#if !defined(GEMM_INT_TYPE)
# define GEMM_INT_TYPE int
#endif

/* Precision detection: GEMM_IS_DOUBLE expands to 1 for double, 0 for float.
 * Uses token-pasting to produce a preprocessor-evaluable integer constant,
 * since LIBXS_DATATYPE enum values are invisible to #if directives. */
#define GEMM_ISDOUBLE_double 1
#define GEMM_ISDOUBLE_float 0
#define GEMM_IS_DOUBLE LIBXS_CONCATENATE(GEMM_ISDOUBLE_, GEMM_REAL_TYPE)
/* GEMM symbol (dgemm_ for double, sgemm_ for float) */
#if !defined(GEMM)
# define GEMM LIBXS_FSYMBOL(LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm))
#endif
/* Complex GEMM symbol (zgemm_ for double, cgemm_ for float) */
#if !defined(ZGEMM)
# define ZGEMM LIBXS_FSYMBOL(LIBXS_CPREFIX(GEMM_REAL_TYPE, gemm))
#endif

/* Common GEMM argument list macros to reduce boilerplate */
#define GEMM_ARGDECL \
  const char* transa, const char* transb, \
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k, \
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda, \
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb, \
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc
#define GEMM_ARGPASS transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc

/* Precision-specific name redirects for public/driver-visible symbols */
#define gemm_diff      LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_diff)
#define gemm_ozaki     LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_ozaki)
#define gemm_stat      LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_stat)
#define gemm_verbose   LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_verbose)
#define print_gemm     LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_print)
#define print_diff     LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_print_diff)
#define gemm_mhd_read  LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_mhd_read)
#define gemm_mhd_write LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm_mhd_write)

/* Size of the serialized settings block */
#define GEMM_MHD_SETTINGS_SIZE (4 * (int)sizeof(int) + 2 * (int)sizeof(double))

/** Real GEMM entry point (dgemm_ or sgemm_). */
LIBXS_API void GEMM(GEMM_ARGDECL);
/** Complex GEMM entry point (zgemm_ or cgemm_). */
LIBXS_API void ZGEMM(GEMM_ARGDECL);

/** Print GEMM arguments. */
LIBXS_API void print_gemm(FILE* ostream, GEMM_ARGDECL);
/** Print statistics. */
LIBXS_API void print_diff(FILE* ostream, const libxs_matdiff_info_t* diff);

/** Ozaki settings stored in MHD extension (optional). */
typedef struct gemm_mhd_settings_t {
  int ozaki, ozn, ozflags, ozcutoff;
  double eps, rsq;
} gemm_mhd_settings_t;

/** Read a GEMM matrix from an MHD file.
 *  Extension: [trans:char][ld:int][scalar:ncomp*real][settings(opt)].
 *  If data is NULL, reads header/extension only (info pass).
 *  If data is non-NULL, also reads the matrix data.
 *  If settings is non-NULL, fills it from extension (zeros if absent).
 *  All output pointers may be NULL.  Returns EXIT_SUCCESS on success. */
LIBXS_API_INLINE int gemm_mhd_read(const char* filename,
  GEMM_INT_TYPE* rows, GEMM_INT_TYPE* cols,
  char* trans, GEMM_INT_TYPE* ld,
  GEMM_REAL_TYPE scalar[2], size_t* ncomp,
  gemm_mhd_settings_t* settings,
  GEMM_REAL_TYPE* data)
{
  char ext[sizeof(char) + sizeof(GEMM_INT_TYPE) + 2 * sizeof(GEMM_REAL_TYPE)
         + GEMM_MHD_SETTINGS_SIZE];
  const size_t ext1 = sizeof(char) + sizeof(GEMM_INT_TYPE) + sizeof(GEMM_REAL_TYPE);
  const size_t ext2 = ext1 + sizeof(GEMM_REAL_TYPE);
  libxs_mhd_info_t info = { 2, 0, LIBXS_DATATYPE_UNKNOWN, 0 };
  size_t size[2], ext_size = sizeof(ext), base_size;
  GEMM_INT_TYPE file_ld;
  char data_filename[1024];
  int has_settings;
  int result;
  strncpy(data_filename, filename, sizeof(data_filename) - 1);
  data_filename[sizeof(data_filename) - 1] = '\0';
  result = libxs_mhd_read_header(filename, sizeof(data_filename) - 1,
    data_filename, &info, size, ext, &ext_size);
  if (EXIT_SUCCESS != result || 2 != info.ndims
    || LIBXS_DATATYPE(GEMM_REAL_TYPE) != info.type
    || (1 != info.ncomponents && 2 != info.ncomponents))
  {
    return EXIT_FAILURE;
  }
  base_size = (1 == info.ncomponents) ? ext1 : ext2;
  has_settings = (ext_size == base_size + GEMM_MHD_SETTINGS_SIZE);
  if (ext_size != base_size && 0 == has_settings) {
    return EXIT_FAILURE;
  }
  memcpy(&file_ld, ext + sizeof(char), sizeof(GEMM_INT_TYPE));
  if (NULL != rows) *rows = (GEMM_INT_TYPE)size[0];
  if (NULL != cols) *cols = (GEMM_INT_TYPE)size[1];
  if (NULL != trans) *trans = *(const char*)ext;
  if (NULL != ld) *ld = file_ld;
  if (NULL != scalar) {
    memcpy(scalar, ext + sizeof(char) + sizeof(GEMM_INT_TYPE),
      info.ncomponents * sizeof(GEMM_REAL_TYPE));
    if (1 == info.ncomponents) scalar[1] = 0;
  }
  if (NULL != ncomp) *ncomp = info.ncomponents;
  if (NULL != settings) {
    if (0 != has_settings) {
      const char* p = ext + base_size;
      memcpy(&settings->ozaki, p, sizeof(int)); p += sizeof(int);
      memcpy(&settings->ozn, p, sizeof(int)); p += sizeof(int);
      memcpy(&settings->ozflags, p, sizeof(int)); p += sizeof(int);
      memcpy(&settings->ozcutoff, p, sizeof(int)); p += sizeof(int);
      memcpy(&settings->eps, p, sizeof(double)); p += sizeof(double);
      memcpy(&settings->rsq, p, sizeof(double));
    }
    else memset(settings, 0, sizeof(*settings));
  }
  if (NULL != data) {
    size_t pitch[2];
    pitch[0] = file_ld; pitch[1] = size[1];
    result = libxs_mhd_read(data_filename, NULL/*offset*/, size, pitch,
      &info, data, NULL/*handler_info*/, NULL/*handler*/);
  }
  return result;
}

/** Write a GEMM matrix to an MHD file (with extension data).
 *  Extension: [trans:char][ld:int][scalar:ncomp*real][settings(opt)].
 *  If settings is non-NULL, the Ozaki settings are appended. */
LIBXS_API_INLINE int gemm_mhd_write(const char* filename,
  const GEMM_REAL_TYPE* data, GEMM_INT_TYPE rows, GEMM_INT_TYPE cols,
  GEMM_INT_TYPE ld, char trans, const GEMM_REAL_TYPE* scalar, size_t ncomp,
  const gemm_mhd_settings_t* settings)
{
  char ext[sizeof(char) + sizeof(GEMM_INT_TYPE) + 2 * sizeof(GEMM_REAL_TYPE)
         + GEMM_MHD_SETTINGS_SIZE];
  size_t ext_size = sizeof(char) + sizeof(GEMM_INT_TYPE)
                  + ncomp * sizeof(GEMM_REAL_TYPE);
  libxs_mhd_info_t mhd_info;
  size_t size[2], pitch[2];
  char* p;
  mhd_info.type = LIBXS_DATATYPE(GEMM_REAL_TYPE);
  mhd_info.ncomponents = ncomp;
  mhd_info.header_size = 0;
  mhd_info.ndims = 2;
  size[0] = rows; size[1] = cols;
  pitch[0] = ld; pitch[1] = cols;
  *(char*)ext = trans;
  memcpy(ext + sizeof(char), &ld, sizeof(GEMM_INT_TYPE));
  memcpy(ext + sizeof(char) + sizeof(GEMM_INT_TYPE),
    scalar, ncomp * sizeof(GEMM_REAL_TYPE));
  if (NULL != settings) {
    p = ext + ext_size;
    memcpy(p, &settings->ozaki, sizeof(int)); p += sizeof(int);
    memcpy(p, &settings->ozn, sizeof(int)); p += sizeof(int);
    memcpy(p, &settings->ozflags, sizeof(int)); p += sizeof(int);
    memcpy(p, &settings->ozcutoff, sizeof(int)); p += sizeof(int);
    memcpy(p, &settings->eps, sizeof(double)); p += sizeof(double);
    memcpy(p, &settings->rsq, sizeof(double));
    ext_size += GEMM_MHD_SETTINGS_SIZE;
  }
  return libxs_mhd_write(filename, NULL/*offset*/, size, pitch,
    &mhd_info, data, NULL/*handler_info*/, NULL/*handler*/,
    NULL/*extension_header*/, ext, ext_size);
}

LIBXS_APIVAR_PUBLIC(libxs_matdiff_info_t gemm_diff);
LIBXS_APIVAR_PUBLIC(int gemm_ozaki);
LIBXS_APIVAR_PUBLIC(int gemm_verbose);
LIBXS_APIVAR_PUBLIC(int gemm_stat);
