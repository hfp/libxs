/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_mhd.h>
#include <libxs_mem.h>


int main(int argc, char* argv[])
{
  const char *const filename = (1 < argc ? argv[1] : "mhd_image.mhd");
  /* take some block-sizes, which are used to test leading dimensions */
  const int bw = LIBXS_MAX(2 < argc ? atoi(argv[2]) : 64, 1);
  const int bh = LIBXS_MAX(3 < argc ? atoi(argv[3]) : 64, 1);
  size_t size[3], pitch[3], offset[3], extension_size;
  libxs_mhd_info_t info = { 3 };
  libxs_mhd_element_handler_info_t dst_info;
  char data_filename[1024];
  void* data = NULL;
  int result;

  /* Read header information; function includes various sanity checks. */
  result = libxs_mhd_read_header(filename, sizeof(data_filename),
    data_filename, &info, size,
    NULL/*extension*/, &extension_size);

  /* Allocate data according to the header information. */
  if (EXIT_SUCCESS == result) {
    size_t nelements = 0;
    pitch[0] = LIBXS_UP(size[0], bw);
    pitch[1] = LIBXS_UP(size[1], bh);
    pitch[2] = size[2];
    /* center the image inside of the (pitched) buffer */
    offset[0] = (pitch[0] - size[0]) / 2;
    offset[1] = (pitch[1] - size[1]) / 2;
    offset[2] = 0;
    libxs_offset(info.ndims, NULL/*offset*/, pitch, &nelements);
    data = malloc(info.ncomponents * nelements * libxs_mhd_typesize(info.type));
  }

  /* perform tests with libxs_mhd_element_conversion (int2signed) */
  if (EXIT_SUCCESS == result) {
    short src = 2507, src_min = 0, src_max = 5000;
    float dst_f32; /* destination range is implicit due to type */
    signed char dst_i8; /* destination range is implicit due to type */
    LIBXS_MEMZERO(&dst_info);
    dst_info.type = LIBXS_DATATYPE_F32;
    result = libxs_mhd_element_conversion(
      &dst_f32, &dst_info, LIBXS_DATATYPE_I16/*src_type*/,
      &src, NULL/*src_min*/, NULL/*src_max*/);
    if (EXIT_SUCCESS == result && src != dst_f32) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      result = libxs_mhd_element_conversion(
        &dst_f32, &dst_info, LIBXS_DATATYPE_I16/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && src != dst_f32) result = EXIT_FAILURE;
    }
    dst_info.type = LIBXS_DATATYPE_I8;
    if (EXIT_SUCCESS == result) {
      result = libxs_mhd_element_conversion(
        &dst_i8, &dst_info, LIBXS_DATATYPE_I16/*src_type*/,
        &src, NULL/*src_min*/, NULL/*src_max*/);
      if (EXIT_SUCCESS == result && LIBXS_MIN(127, src) != dst_i8) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      result = libxs_mhd_element_conversion(
        &dst_i8, &dst_info, LIBXS_DATATYPE_I16/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && 64 != dst_i8) result = EXIT_FAILURE;
    }
  }

  /* perform tests with libxs_mhd_element_conversion (float2int) */
  if (EXIT_SUCCESS == result) {
    double src = 1975, src_min = -25071975, src_max = 1981;
    short dst_i16; /* destination range is implicit due to type */
    unsigned char dst_u8; /* destination range is implicit due to type */
    dst_info.type = LIBXS_DATATYPE_I16;
    result = libxs_mhd_element_conversion(
      &dst_i16, &dst_info, LIBXS_DATATYPE_F64/*src_type*/,
      &src, NULL/*src_min*/, NULL/*src_max*/);
    if (EXIT_SUCCESS == result && src != dst_i16) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) {
      result = libxs_mhd_element_conversion(
        &dst_i16, &dst_info, LIBXS_DATATYPE_F64/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && 2 != dst_i16) result = EXIT_FAILURE;
    }
    dst_info.type = LIBXS_DATATYPE_U8;
    if (EXIT_SUCCESS == result) {
      result = libxs_mhd_element_conversion(
        &dst_u8, &dst_info, LIBXS_DATATYPE_F64/*src_type*/,
        &src, NULL/*src_min*/, NULL/*src_max*/);
      if (EXIT_SUCCESS == result && LIBXS_MIN(255, src) != dst_u8) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      result = libxs_mhd_element_conversion(
        &dst_u8, &dst_info, LIBXS_DATATYPE_F64/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && 255 != dst_u8) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      src = -src;
      result = libxs_mhd_element_conversion(
        &dst_u8, &dst_info, LIBXS_DATATYPE_F64/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && 0 != dst_u8) result = EXIT_FAILURE;
    }
    dst_info.type = LIBXS_DATATYPE_I16;
    if (EXIT_SUCCESS == result) {
      result = libxs_mhd_element_conversion(
        &dst_i16, &dst_info, LIBXS_DATATYPE_F64/*src_type*/,
        &src, &src_min, &src_max);
      if (EXIT_SUCCESS == result && -3 != dst_i16) result = EXIT_FAILURE;
    }
  }

  /* Read the data according to the header into the allocated buffer. */
  if (EXIT_SUCCESS == result) {
    result = libxs_mhd_read(data_filename,
      offset, size, pitch, &info, data,
      NULL/*handler_info*/, NULL/*handler*/);
  }

  /* Write the data into a new file; update header_size. */
  if (EXIT_SUCCESS == result) {
    result = libxs_mhd_write("mhd_test.mhd", NULL/*offset*/, pitch, pitch,
      &info, data, NULL/*no conversion*/, NULL/*handler*/,
      NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
  }

  /* Check the written data against the buffer. */
  if (EXIT_SUCCESS == result) {
    info.header_size = 0;
    result = libxs_mhd_read(data_filename,
      offset, size, pitch, &info, data,
      NULL/*handler_info*/, libxs_mhd_element_comparison);
  }

  /* Check the written data against the buffer with conversion. */
  if (EXIT_SUCCESS == result) {
    void* buffer = NULL;
    size_t nelements;
    dst_info.type = LIBXS_DATATYPE_F64;
    libxs_offset(info.ndims, NULL/*offset*/, pitch, &nelements);
    buffer = malloc(info.ncomponents * nelements * libxs_mhd_typesize(dst_info.type));
    info.header_size = 0;
    result = libxs_mhd_read(data_filename,
      offset, size, pitch, &info, buffer,
      &dst_info, NULL/*libxs_mhd_element_comparison*/);
    free(buffer);
  }

  /* Deallocate the buffer. */
  free(data);

  return result;
}
