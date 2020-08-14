/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_sync.h>
#include <libxs_mhd.h>
#include <libxs.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if defined(_WIN32)
# include <io.h>
# if !defined(F_OK)
#   define F_OK 0
# endif
# define FEXIST(FILENAME) _access(FILENAME, F_OK)
#else
# include <unistd.h>
# define FEXIST(FILENAME) access(FILENAME, F_OK)
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if 1
# define MALLOC(SIZE) libxs_aligned_malloc(SIZE, 0/*auto*/)
# define FREE(POINTER) libxs_free(POINTER)
# define SCRATCH_MALLOC(SIZE) libxs_aligned_scratch(SIZE, 0/*auto*/)
# define SCRATCH_FREE(POINTER) libxs_free(POINTER)
#else
# define MALLOC(SIZE) malloc(SIZE)
# define FREE(POINTER) free(POINTER)
# define SCRATCH_MALLOC(SIZE) MALLOC(SIZE)
# define SCRATCH_FREE(POINTER) FREE(POINTER)
#endif

#if !defined(USE_OUTPUT_PADDING) && 0
# define USE_OUTPUT_PADDING
#endif

#if !defined(USE_OVERWRITE)
# define USE_OVERWRITE
#endif


/**
 * This code mainly demonstrates MHD image I/O but happens to also perform an image convolution.
 * The latter is *not* a showcase of LIBXS's Deeplearning as the code only runs over a single image.
 */
int main(int argc, char* argv[])
{
  const char* filename_in = (1 < argc ? argv[1] : "mhd_in.mhd");
  const size_t nrepeat = (size_t)LIBXS_MAX(2 < argc ? strtoul(argv[2], 0, 10) : 1, 1);
  const int kw = LIBXS_MAX(3 < argc ? atoi(argv[3]) : 32, 1);
  const int kh = LIBXS_MAX(4 < argc ? atoi(argv[4]) : kw, 1);
  const char *const filename_out = (5 < argc ? argv[5] : "mhd_out.mhd");
  int result = (0 != strcmp(filename_in, filename_out) ? EXIT_SUCCESS : EXIT_FAILURE);
  size_t ndims = 2, size_in[] = { 0, 0 }, size_out[] = { 0, 0 }, pitch[2], offset[2], ncomponents = 1, header_size = 0, extension_size;
  void *conv_input_buffer = 0, *conv_filter_buffer = 0, *conv_output_buffer = 0;
  libxs_dnn_tensor *conv_input = 0, *conv_output = 0, *conv_filter = 0;
  libxs_mhd_elemtype type_in = LIBXS_MHD_ELEMTYPE_UNKNOWN;
  libxs_dnn_datatype type_dnn = LIBXS_DNN_DATATYPE_F32;
  libxs_dnn_conv_desc descriptor;
  libxs_dnn_layer* handle = 0;
  libxs_dnn_err_t status;
  size_t size1 = 0, typesize_dnn = 0;
  size_t conv_output_size1 = 0, i, j;
  unsigned long long start;
  char filename[1024];
  double duration = 0;
  void *scratch = 0;
  void *filter = 0;
  void *image = 0;
#if !defined(NDEBUG)
  static int error_once = 0;
#endif

  const char *const env_mult = getenv("MULT"), *const env_orig = getenv("ORIG");
  /* extents of result image become multiples of block-size */
  const int mult = ((NULL == env_mult || 0 == *env_mult) ? 64/*default*/ : LIBXS_MAX(atoi(env_mult), 0));
  /* save result with original pixel-type of input (type_in), otherwise save with compute-type (type_dnn) */
  const int orig = ((NULL == env_orig || 0 == *env_orig) ? 1/*enabled*/ : atoi(env_orig));

  /* Generate an input file if a pseudo filename (resolution) is given. */
  if (0 != FEXIST(filename_in) && 0 < atoi(filename_in)) {
    const char* split = strchr(filename_in, 'x');
    if (0 == split) split = strchr(filename_in, 'X');
    size_in[0] = atoi(filename_in);
    size_in[1] = (0 != split ? atoi(split + 1) : 0);
    if (0 == size_in[1]) size_in[1] = size_in[0];
    image = MALLOC(size_in[0] * size_in[1]);
    if (0 < sprintf(filename, "%s.mhd", filename_in) && 0 != image) {
      const int c0 = 0, c1 = 255, r = LIBXS_MAX(kw, kh);
      for (i = 0; i < size_in[1]; ++i) {
        for (j = 0; j < size_in[0]; ++j) {
          ((unsigned char*)image)[i*size_in[0]+j] = (unsigned char)(0 == (i + j) % r ? c1 : c0);
        }
      }
      result = libxs_mhd_write(filename, NULL/*offset*/, size_in, size_in,
        2/*ndims*/, 1/*ncomponents*/, LIBXS_MHD_ELEMTYPE_U8, NULL/*conversion*/, image,
        0/*header_size*/, NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
      if (EXIT_SUCCESS == result) filename_in = filename;
    }
    else {
      result = EXIT_FAILURE;
    }
    FREE(image);
  }

  /* Read MHD-header information; function includes various sanity checks. */
  if (EXIT_SUCCESS == result) {
    result = libxs_mhd_read_header(filename_in, sizeof(filename),
      filename, &ndims, size_in, &ncomponents, &type_in,
      &header_size, &extension_size);
  }

  /* Only accept 2d-images (maybe a slice of a 3d-image). */
  if (2 == ndims) {
    const int m = LIBXS_MAX(mult, 1);
    offset[0] = ((size_in[0] + LIBXS_MAX(kw, m) - 1) / m * m - size_in[0] + kw) / 2;
    offset[1] = ((size_in[1] + LIBXS_MAX(kh, m) - 1) / m * m - size_in[1] + kh) / 2;
    /* center image inside of (pitched) buffer */
    size_out[0] = size_in[0] + 2 * offset[0];
    size_out[1] = size_in[1] + 2 * offset[1];
#if defined(USE_OUTPUT_PADDING)
    size_out[0] -= (kw / 2) * 2;
    size_out[1] -= (kh / 2) * 2;
    pitch[0] = size_out[0];
    pitch[1] = size_out[1];
#else
    pitch[0] = size_out[0];
    pitch[1] = size_out[1];
    size_out[0] -= (kw / 2) * 2;
    size_out[1] -= (kh / 2) * 2;
#endif
    size1 = pitch[0] * pitch[1] * ncomponents;
  }
  else {
    result = EXIT_FAILURE;
  }

  /* Allocate image data according to the MHD-header information. */
  if (EXIT_SUCCESS == result) {
    const char* ctypename;
    /* DNN type: assume that MHD I/O provides a super-set of types */
    if (0 != libxs_mhd_typename((libxs_mhd_elemtype)type_dnn, &typesize_dnn, &ctypename)) {
      const size_t filter_size = ncomponents * kh * kw;
      /* print some information about the workload */
      fprintf(stdout, "filename=%s resolution=%ux%u kernel=%ix%i size_in=%.fMB nrepeat=%u (%s)\n",
        filename, (unsigned int)size_in[0], (unsigned int)size_in[1], kw, kh,
        1.0 * (size1 * typesize_dnn) / (1 << 20),
        (unsigned int)nrepeat, ctypename);
      image = MALLOC(size1 * typesize_dnn);
      filter = MALLOC(filter_size * typesize_dnn);
      if (0 != image && 0 != filter) {
        FILE *const file = fopen("mhd_in.txt", "r"); /* convolution-matrix (kh x kw) */
        double weight;
        switch (type_dnn) {
          case LIBXS_DNN_DATATYPE_F64: {
            for (i = 0; i < filter_size; ++i) {
              ((double*)filter)[i] = (double)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (0.05 - ((double)rand() / RAND_MAX) * 0.1) : weight);
            }
          } break;
          case LIBXS_DNN_DATATYPE_F32: {
            for (i = 0; i < filter_size; ++i) {
              ((float*)filter)[i] = (float)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (0.05 - ((double)rand() / RAND_MAX) * 0.1) : weight);
            }
          } break;
          case LIBXS_DNN_DATATYPE_I32: {
            for (i = 0; i < filter_size; ++i) {
              ((int*)filter)[i] = (int)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (255 * (0.05 - ((double)rand() / RAND_MAX) * 0.1)) : weight);
            }
          } break;
          case LIBXS_DNN_DATATYPE_I16: {
            for (i = 0; i < filter_size; ++i) {
              ((short*)filter)[i] = (short)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (255 * (0.05 - ((double)rand() / RAND_MAX) * 0.1)) : weight);
            }
          } break;
          case LIBXS_DNN_DATATYPE_I8: {
            for (i = 0; i < filter_size; ++i) {
              ((unsigned char*)filter)[i] = (unsigned char)((0 == file || 1 > fscanf(file, "%lf", &weight)) ?
                (255 * (0.05 - ((double)rand() / RAND_MAX) * 0.1)) : weight);
            }
          } break;
          default: result = EXIT_FAILURE;
        }
        if (0 != file && 0 != fclose(file)) result = EXIT_FAILURE;
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
    const void *const pv = &type_dnn;
    result = libxs_mhd_read(filename,
      offset, size_in, pitch, ndims, ncomponents, header_size, type_in,
      /* eventually perform a type-conversion (type_in != type_dnn) */
      (const libxs_mhd_elemtype*)pv, image,
      NULL/*handle_element*/, NULL/*extension*/, 0/*extension_size*/);
  }

  /* Setup convolution descriptor. */
  memset(&descriptor, 0, sizeof(descriptor));
  if (EXIT_SUCCESS == result) {
#if defined(_OPENMP)
    descriptor.threads = omp_get_max_threads();
#else
    descriptor.threads = 1;
#endif
    descriptor.N = 1; /* number of images */
    descriptor.R = kh; /* kernel height */
    descriptor.S = kw; /* kernel width */
    descriptor.C = (int)ncomponents; /* in */
    descriptor.K = descriptor.C; /* no reduction */
    descriptor.u = 1; /* H-stride */
    descriptor.v = 1; /* W-stride */
    descriptor.H = (int)pitch[1];
    descriptor.W = (int)pitch[0];
    descriptor.pad_h = 0;
    descriptor.pad_w = 0;
    descriptor.pad_h_in = 0;
    descriptor.pad_w_in = 0;
#if defined(USE_OUTPUT_PADDING)
    descriptor.pad_h_out = ((descriptor.u - 1) * descriptor.H + descriptor.R - descriptor.u) / 2;
    descriptor.pad_w_out = ((descriptor.v - 1) * descriptor.W + descriptor.S - descriptor.v) / 2;
#else
    descriptor.pad_h_out = 0;
    descriptor.pad_w_out = 0;
#endif
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
    descriptor.datatype_in = LIBXS_DNN_DATATYPE_F32;
    descriptor.datatype_out = LIBXS_DNN_DATATYPE_F32;
    handle = libxs_dnn_create_conv_layer(descriptor, &status);
    if (LIBXS_DNN_SUCCESS != status) {
      const char *const error_message = libxs_dnn_get_error(status);
      fprintf(stderr, "%s\n", error_message);
      if (LIBXS_DNN_WARN_FALLBACK != status) result = EXIT_FAILURE;
    }
  }

  /* Link buffers and convert NCHW-image and KCRS-filter to internal format. */
  if (EXIT_SUCCESS == result) {
    libxs_dnn_tensor_datalayout* layout;
    size_t scratch_size;

    /* Input buffer */
    conv_input_buffer = MALLOC(descriptor.N * descriptor.C * (descriptor.H + 2 * descriptor.pad_h_in) * (descriptor.W + 2 * descriptor.pad_w_in) * typesize_dnn);
    if (0 == conv_input_buffer) result = EXIT_FAILURE;
    layout = libxs_dnn_create_tensor_datalayout(handle, LIBXS_DNN_INPUT, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    conv_input = libxs_dnn_link_tensor(layout, conv_input_buffer, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_destroy_tensor_datalayout(layout);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_bind_tensor(handle, conv_input, LIBXS_DNN_REGULAR_INPUT);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_copyin_tensor(conv_input, image, LIBXS_DNN_TENSOR_FORMAT_NCHW);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;

    /* Filter buffer */
    conv_filter_buffer = MALLOC(descriptor.K * descriptor.C * descriptor.R * descriptor.S * typesize_dnn);
    if (0 == conv_filter_buffer) result = EXIT_FAILURE;
    layout = libxs_dnn_create_tensor_datalayout(handle, LIBXS_DNN_FILTER, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    conv_filter = libxs_dnn_link_tensor(layout, conv_filter_buffer, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_destroy_tensor_datalayout(layout);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_bind_tensor(handle, conv_filter, LIBXS_DNN_REGULAR_FILTER);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_copyin_tensor(conv_filter, filter, LIBXS_DNN_TENSOR_FORMAT_KCRS);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;

    /* Output buffer */
    conv_output_size1 = descriptor.N * descriptor.K * (descriptor.H + 2 * descriptor.pad_h_out) * (descriptor.W + 2 * descriptor.pad_w_out);
    conv_output_buffer = MALLOC(conv_output_size1 * typesize_dnn);
    if (0 == conv_output_buffer) result = EXIT_FAILURE;
    layout = libxs_dnn_create_tensor_datalayout(handle, LIBXS_DNN_OUTPUT, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    conv_output = libxs_dnn_link_tensor(layout, conv_output_buffer, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_destroy_tensor_datalayout(layout);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    status = libxs_dnn_bind_tensor(handle, conv_output, LIBXS_DNN_REGULAR_OUTPUT);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;

    /* allocate and bind scratch memory */
    scratch_size = libxs_dnn_get_scratch_size(handle, LIBXS_DNN_COMPUTE_KIND_ALL, &status);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
    scratch = SCRATCH_MALLOC(scratch_size);
    if (0 == scratch) result = EXIT_FAILURE;
    status = libxs_dnn_bind_scratch(handle, LIBXS_DNN_COMPUTE_KIND_ALL, scratch);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
  }

  /* Attempt to run the convolution. */
  start = libxs_timer_tick();
  for (i = 0; i < nrepeat && EXIT_SUCCESS == result; ++i) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
#if !defined(USE_OVERWRITE)
      memset(conv_output_buffer, 0, conv_output_size1 * typesize_dnn);
#endif
      {
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
  }
  duration = libxs_timer_duration(start, libxs_timer_tick());

  /* Copy-out image into original format. */
  if (EXIT_SUCCESS == result) {
    status = libxs_dnn_copyout_tensor(conv_output, image, LIBXS_DNN_TENSOR_FORMAT_NCHW);
    if (LIBXS_DNN_SUCCESS != status) result = EXIT_FAILURE;
  }

  /* Write the image into a different file. */
  if (EXIT_SUCCESS == result) {
    if (0 == mult) {
      offset[0] = (size_out[0] - size_in[0]) / 2;
      offset[1] = (size_out[1] - size_in[1]) / 2;
    }
    else {
      offset[0] = offset[1] = 0;
      size_in[0] = size_out[0];
      size_in[1] = size_out[1];
    }

    /* write result image without offset/padding. */
    result = libxs_mhd_write(filename_out, NULL/*offset*/, size_in, size_in, 2/*ndims*/, ncomponents,
      (libxs_mhd_elemtype)type_dnn/* assume super-set of DNN types */, 0 != orig ? &type_in : NULL, image,
      0/*header_size*/, NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
  }

  /* Release resources. */
  if (LIBXS_DNN_SUCCESS != libxs_dnn_release_tensor(handle, LIBXS_DNN_REGULAR_INPUT)) result = EXIT_FAILURE;
  if (LIBXS_DNN_SUCCESS != libxs_dnn_release_tensor(handle, LIBXS_DNN_REGULAR_FILTER)) result = EXIT_FAILURE;
  if (LIBXS_DNN_SUCCESS != libxs_dnn_release_tensor(handle, LIBXS_DNN_REGULAR_OUTPUT)) result = EXIT_FAILURE;
  if (LIBXS_DNN_SUCCESS != libxs_dnn_destroy_tensor(conv_filter)) result = EXIT_FAILURE;
  if (LIBXS_DNN_SUCCESS != libxs_dnn_destroy_tensor(conv_output)) result = EXIT_FAILURE;
  if (LIBXS_DNN_SUCCESS != libxs_dnn_destroy_tensor(conv_input)) result = EXIT_FAILURE;
  if (LIBXS_DNN_SUCCESS != libxs_dnn_destroy_conv_layer(handle)) result = EXIT_FAILURE;
  SCRATCH_FREE(scratch);
  FREE(conv_output_buffer);
  FREE(conv_filter_buffer);
  FREE(conv_input_buffer);
  FREE(filter);
  FREE(image);

  if (EXIT_SUCCESS == result) {
    if (0 < duration) {
      fprintf(stdout, "\tfrequency: %.0f Hz\n", nrepeat / duration);
    }
    assert(0 != nrepeat);
    fprintf(stdout, "\tduration: %.0f ms\n", 1E3 * duration / nrepeat);
    fprintf(stdout, "Finished.\n");
  }
  else {
    fprintf(stdout, "FAILED.\n");
  }

  assert(EXIT_SUCCESS == result);
  return result;
}

