/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
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
#include <libxs_mhd.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if 1
# define MALLOC(SIZE) libxs_aligned_malloc(SIZE, 0/*auto*/)
# define FREE(POINTER) libxs_free(POINTER)
#else
# define MALLOC(SIZE) malloc(SIZE)
# define FREE(POINTER) free(POINTER)
#endif

#if !defined(USE_OVERWRITE)
/*# define USE_OVERWRITE*/
#endif


int main(int argc, char* argv[])
{
  const char *const filename_in   = (1 < argc ? argv[1] : "iconv_in.mhd");
  const char *const filename_out  = (2 < argc ? argv[2] : "iconv_out.mhd");
  const int kh = (3 < argc ? atoi(argv[3]) : 39);
  const int kw = (4 < argc ? atoi(argv[4]) : kh);
  int result = 0 != strcmp(filename_in, filename_out) ? EXIT_SUCCESS : EXIT_FAILURE;
  size_t ndims = 3, size[3], pitch[2], ncomponents = 0, header_size = 0, extension_size;
  void *conv_input_buffer = 0, *conv_filter_buffer = 0, *conv_output_buffer = 0;
  libxs_dnn_buffer *conv_input = 0, *conv_output = 0;
  libxs_dnn_filter *conv_filter = 0;
  libxs_dnn_datatype type_out = LIBXS_DNN_DATATYPE_F32;
  libxs_mhd_elemtype type_in = LIBXS_MHD_ELEMTYPE_UNKNOWN;
  libxs_dnn_conv_desc descriptor = { 0 };
  libxs_dnn_layer* handle = 0;
  libxs_dnn_err_t status;
  size_t size1 = 0, typesize = 0;
  size_t conv_output_size1 = 0;
  static int error_once = 0;
  char filename_data[1024];
  void *filter = 0;
  void *image = 0;

  /* Read MHD-header information; function includes various sanity checks. */
  if (EXIT_SUCCESS == result) {
    result = libxs_mhd_read_header(filename_in, sizeof(filename_data),
      filename_data, &ndims, size, &ncomponents, &type_in,
      &header_size, &extension_size);
  }

  /* Only accept 2-d images or a single slice of a 3d-image. */
  if (2 == ndims || (3 == ndims && 1 == size[2])) {
    size1 = size[0] * size[1];
    pitch[0] = size[0];
    pitch[1] = size[1];
  }
  else {
    result = EXIT_FAILURE;
  }

  /* Allocate image data according to the MHD-header information. */
  if (EXIT_SUCCESS == result) {
    /* DNN type: assume that MHD I/O provides a super-set of types */
    if (0 != libxs_mhd_typename((libxs_mhd_elemtype)type_out, &typesize)) {
      const size_t filter_size = ncomponents * kh * kw;
      image = MALLOC(size1 * ncomponents * typesize);
      if (0 == image) result = EXIT_FAILURE;
      filter = MALLOC(filter_size * typesize);
      if (0 != filter) {
        size_t i;
        switch (type_out) {
          case LIBXS_DNN_DATATYPE_F32: {
            float *const f = (float*)filter;
            for (i = 0; i < filter_size; ++i) {
              f[i] = (float)(0.05 - ((double)rand() / RAND_MAX) * 0.1);
            }
          } break;
          default: result = EXIT_FAILURE;
        }
      }
      else {
        result = EXIT_FAILURE;
      }
    }
    else {
      result = EXIT_FAILURE;
    }
  }

  /* Read the image data according to the header into the allocated buffer. */
  if (EXIT_SUCCESS == result) {
    result = libxs_mhd_read(filename_data,
      size, pitch, ndims, ncomponents, header_size, type_in,
      /* eventually perform a type-conversion (type_in != type_out) */
      (const libxs_mhd_elemtype*)&type_out, image,
      0/*handle_element*/, 0/*extension*/, 0/*extension_size*/);
  }

  /* Setup convolution descriptor. */
  if (EXIT_SUCCESS == result) {
    descriptor.threads = 1;
    descriptor.N = 1; /* number of images */
    descriptor.R = kh; /* kernel height */
    descriptor.S = kw; /* kernel width */
    descriptor.C = (int)ncomponents; /* in */
    descriptor.K = descriptor.C; /* no reduction */
    descriptor.u = 1; /* H-stride */
    descriptor.v = 1; /* W-stride */
    descriptor.H = (int)size[1];
    descriptor.W = (int)size[0];
    descriptor.pad_h_out = ((descriptor.u - 1) * descriptor.H + descriptor.R - descriptor.u) / 2;
    descriptor.pad_w_out = ((descriptor.v - 1) * descriptor.W + descriptor.S - descriptor.v) / 2;
    descriptor.pad_h_in = 0;
    descriptor.pad_w_in = 0;
    descriptor.pad_h = 0; /* H-pad */
    descriptor.pad_w = 0; /* W-pad */
    descriptor.algo = LIBXS_DNN_CONV_ALGO_DIRECT/*LIBXS_DNN_CONV_ALGO_AUTO*/;
    descriptor.buffer_format = LIBXS_DNN_TENSOR_FORMAT_LIBXS;
    descriptor.filter_format = LIBXS_DNN_TENSOR_FORMAT_LIBXS;
    descriptor.fuse_ops = LIBXS_DNN_CONV_FUSE_NONE;
#if defined(USE_OVERWRITE)
    descriptor.options = LIBXS_DNN_CONV_OPTION_OVERWRITE;
#else
    descriptor.options = LIBXS_DNN_CONV_OPTION_NONE;
#endif
    descriptor.fuse_ops = LIBXS_DNN_CONV_FUSE_NONE;
    descriptor.datatype = LIBXS_DNN_DATATYPE_F32;
    handle = libxs_dnn_create_conv_layer(descriptor, &status);
    if (LIBXS_DNN_SUCCESS != status) {
      const char *const error_message = libxs_dnn_get_error(status);
      fprintf(stderr, "%s\n", error_message);
      if (LIBXS_DNN_WARN_FALLBACK != status) result = EXIT_FAILURE;
    }
  }

  /* Link buffers and convert NCHW-image and KCRS-filter to internal format. */
  if (EXIT_SUCCESS == result) {
    /* Input buffer */
    conv_input_buffer = MALLOC(descriptor.N * descriptor.C * (descriptor.H + 2 * descriptor.pad_h_in) * (descriptor.W + 2 * descriptor.pad_w_in) * typesize);
    if (0 == conv_input_buffer) result = EXIT_FAILURE;
    conv_input = libxs_dnn_link_buffer(handle, LIBXS_DNN_INPUT, conv_input_buffer, LIBXS_DNN_TENSOR_FORMAT_LIBXS_PTR, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_bind_buffer(handle, conv_input, LIBXS_DNN_REGULAR_INPUT);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_copyin_buffer(conv_input, image, LIBXS_DNN_TENSOR_FORMAT_NCHW);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    /* Filter buffer */
    conv_filter_buffer = MALLOC(descriptor.K * descriptor.C * descriptor.R * descriptor.S * typesize);
    if (0 == conv_filter_buffer) result = EXIT_FAILURE;
    conv_filter = libxs_dnn_link_filter(handle, LIBXS_DNN_FILTER, conv_filter_buffer, LIBXS_DNN_TENSOR_FORMAT_LIBXS_PTR, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_bind_filter(handle, conv_filter, LIBXS_DNN_REGULAR_FILTER);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_copyin_filter(conv_filter, filter, LIBXS_DNN_TENSOR_FORMAT_KCRS);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    /* Output buffer */
    conv_output_size1 = descriptor.N * descriptor.K * (descriptor.H + 2 * descriptor.pad_h_out) * (descriptor.W + 2 * descriptor.pad_w_out);
    conv_output_buffer = MALLOC(conv_output_size1 * typesize);
    if (0 == conv_output_buffer) result = EXIT_FAILURE;
    conv_output = libxs_dnn_link_buffer(handle, LIBXS_DNN_OUTPUT, conv_output_buffer, LIBXS_DNN_TENSOR_FORMAT_LIBXS_PTR, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_bind_buffer(handle, conv_output, LIBXS_DNN_REGULAR_OUTPUT);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
  }

  /* Attempt to run the convolution. */
  if (EXIT_SUCCESS == result) {
#if !defined(USE_OVERWRITE)
    memset(conv_output_buffer, 0, conv_output_size1 * typesize);
#endif
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
#if !defined(NDEBUG)
      const libxs_dnn_err_t r =
#endif
      libxs_dnn_execute_st(handle, LIBXS_DNN_COMPUTE_KIND_FWD, 0, tid);
#if !defined(NDEBUG)
      if (LIBXS_DNN_SUCCESS != r && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        const char *const error_message = libxs_dnn_get_error(r);
        fprintf(stderr, "%s\n", error_message);
        result = EXIT_FAILURE;
      }
#endif
    }
  }

  /* Copy-out image into original format. */
  if (EXIT_SUCCESS == result) {
    status = libxs_dnn_copyout_buffer(conv_output, image, LIBXS_DNN_TENSOR_FORMAT_NCHW);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
  }

  /* Write the image into a different file. */
  if (EXIT_SUCCESS == result) {
    result = libxs_mhd_write(filename_out, size, pitch, 2, ncomponents,
      /* DNN type: assume that MHD I/O provides a super-set of types */
      (libxs_mhd_elemtype)type_out, image,
      0/*extension_header*/,
      0/*extension*/,
      0/*extension_size*/);
  }

  /* Release resources. */
  if (LIBXS_DNN_SUCCESS != libxs_dnn_destroy_filter(conv_filter)) result = EXIT_FAILURE;
  if (LIBXS_DNN_SUCCESS != libxs_dnn_destroy_buffer(conv_output)) result = EXIT_FAILURE;
  if (LIBXS_DNN_SUCCESS != libxs_dnn_destroy_buffer(conv_input)) result = EXIT_FAILURE;
  if (LIBXS_DNN_SUCCESS != libxs_dnn_destroy_conv_layer(handle)) result = EXIT_FAILURE;
  FREE(conv_output_buffer);
  FREE(conv_filter_buffer);
  FREE(conv_input_buffer);
  FREE(filter);
  FREE(image);

  return result;
}

