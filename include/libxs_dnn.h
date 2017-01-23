/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
#ifndef LIBXS_DNN_H
#define LIBXS_DNN_H

#include "libxs_macros.h"
#include "libxs_typedefs.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/** Opaque handles which represents convolutions and LIBXS datatypes */
typedef struct LIBXS_RETARGETABLE libxs_dnn_layer libxs_dnn_layer;
typedef struct LIBXS_RETARGETABLE libxs_dnn_buffer libxs_dnn_buffer;
typedef struct LIBXS_RETARGETABLE libxs_dnn_bias libxs_dnn_bias;
typedef struct LIBXS_RETARGETABLE libxs_dnn_filter libxs_dnn_filter;
typedef unsigned int libxs_dnn_err_t;

/** Define error and warning codes */
#define LIBXS_DNN_SUCCESS                             0
#define LIBXS_DNN_WARN_FALLBACK                   90000
#define LIBXS_DNN_ERR_GENERAL                    100000
#define LIBXS_DNN_ERR_CREATE_HANDLE              100001
#define LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE       100002
#define LIBXS_DNN_ERR_INVALID_BLOCKING           100003
#define LIBXS_DNN_ERR_INVALID_HANDLE             100004
#define LIBXS_DNN_ERR_DATA_NOT_BOUND             100005
#define LIBXS_DNN_ERR_CREATE_BUFFER              100006
#define LIBXS_DNN_ERR_INVALID_BUFFER             100007
#define LIBXS_DNN_ERR_CREATE_FILTER              100008
#define LIBXS_DNN_ERR_INVALID_FILTER             100009
#define LIBXS_DNN_ERR_CREATE_BIAS                100010
#define LIBXS_DNN_ERR_INVALID_BIAS               100011
#define LIBXS_DNN_ERR_MISMATCH_BUFFER            100012
#define LIBXS_DNN_ERR_INVALID_HANDLE_BUFFER      100013
#define LIBXS_DNN_ERR_MISMATCH_FILTER            100014
#define LIBXS_DNN_ERR_INVALID_HANDLE_FILTER      100015
#define LIBXS_DNN_ERR_INVALID_KIND               100016
#define LIBXS_DNN_ERR_INVALID_FORMAT_NCHW        100017
#define LIBXS_DNN_ERR_UNSUPPORTED_DST_FORMAT     100018
#define LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT     100019
#define LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE    100020
#define LIBXS_DNN_ERR_INVALID_FORMAT_KCRS        100021
#define LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL     100022
#define LIBXS_DNN_ERR_CREATE_LAYOUT              100023
#define LIBXS_DNN_ERR_INVALID_LAYOUT             100024
#define LIBXS_DNN_ERR_UNSUPPORTED_ARCH           100025
#define LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED        100026
#define LIBXS_DNN_ERR_UNKNOWN_BUFFER_TYPE        100027
#define LIBXS_DNN_ERR_UNKNOWN_FILTER_TYPE        100028

/** Kinds of supported compute flavor operations. */
typedef enum libxs_dnn_compute_kind {
  /** Forward path */
  LIBXS_DNN_COMPUTE_KIND_FWD,
  /** Backward path */
  LIBXS_DNN_COMPUTE_KIND_BWD,
  /** Updated weights. */
  LIBXS_DNN_COMPUTE_KIND_UPD,
  /** All routines, need for some init routines. */
  LIBXS_DNN_COMPUTE_KIND_ALL
} libxs_dnn_compute_kind;

/** type/meaning of dimension in a LIBXS DNN tensor */
typedef enum libxs_dnn_tensor_dimtype {
  /** Mini-batch */
  LIBXS_DNN_TENSOR_DIMTYPE_N,
  /** Image Height */
  LIBXS_DNN_TENSOR_DIMTYPE_H,
  /** Image Width */
  LIBXS_DNN_TENSOR_DIMTYPE_W,
  /** channles or input channels */
  LIBXS_DNN_TENSOR_DIMTYPE_C,
  /** output channels */
  LIBXS_DNN_TENSOR_DIMTYPE_K,
  /** kernel height */
  LIBXS_DNN_TENSOR_DIMTYPE_R,
  /** kernel width */
  LIBXS_DNN_TENSOR_DIMTYPE_S
} libxs_dnn_tensor_dimtype;

/** types of different buffers */
typedef enum libxs_dnn_buffer_type {
  /** regular input buffer */
  LIBXS_DNN_REGULAR_INPUT,
  /** gradient input buffer */
  LIBXS_DNN_GRADIENT_INPUT,
  /** regular output buffer */
  LIBXS_DNN_REGULAR_OUTPUT,
  /** gradient output buffer */
  LIBXS_DNN_GRADIENT_OUTPUT
} libxs_dnn_buffer_type;

/** types of different filters */
typedef enum libxs_dnn_filter_type {
  /* regular filter */
  LIBXS_DNN_REGULAR_FILTER,
  /* gradient filter */
  LIBXS_DNN_GRADIENT_FILTER
} libxs_dnn_filter_type;

/** layout descriptor to allow external data allocation
    outside of LIBXS */
typedef struct LIBXS_RETARGETABLE libxs_dnn_tensor_datalayout {
  libxs_dnn_tensor_dimtype* dim_type;
  unsigned int* dim_size;
  unsigned int num_dims;
} libxs_dnn_tensor_datalayout;

typedef enum libxs_dnn_conv_fuse_op {
  /* we fuse nothing into convolution */
  LIBXS_DNN_CONV_FUSE_NONE = 0
#if 0
  ,
  /* we fuse fuse bias init into convolution */
  LIBXS_DNN_CONV_FUSE_BIAS = 1,
  /* we fase fase ReLU calculation into convolution Op */
  LIBXS_DNN_CONV_FUSE_RELU = 2
#endif
} libxs_dnn_conv_fuse_op;

/** Type of algorithm used for convolutions. */
typedef enum libxs_dnn_conv_algo {
  /** let the library decide */
  LIBXS_DNN_CONV_ALGO_AUTO,   /* ignored for now */
  /** direct convolution. */
  LIBXS_DNN_CONV_ALGO_DIRECT
} libxs_dnn_conv_algo;

/** Structure which describes the input and output of data (DNN). */
typedef struct LIBXS_RETARGETABLE libxs_dnn_conv_desc {
  int N;                                       /* number of images in mini-batch */
  int C;                                       /* number of input feature maps */
  int H;                                       /* height of input image */
  int W;                                       /* width of input image */
  int K;                                       /* number of output feature maps */
  int R;                                       /* height of filter kernel */
  int S;                                       /* width of filter kernel */
  int u;                                       /* vertical stride */
  int v;                                       /* horizontal stride */
  int pad_h;                                   /* height of logical rim padding to input for adjusting output height */
  int pad_w;                                   /* width of logical rim padding to input for adjusting output width */
  int pad_h_in;                                /* height of zero-padding in input buffer, must equal to pad_h for direct conv */
  int pad_w_in;                                /* width of zero-padding in input buffer, must equal to pad_w for direct conv */
  int pad_h_out;                               /* height of zero-padding in output buffer */
  int pad_w_out;                               /* width of zero-padding in output buffer */
  int threads;                                 /* number of threads to use when running convolution */
  libxs_dnn_datatype datatype;               /* datatypes use for all input and outputs */
  libxs_dnn_tensor_format buffer_format;     /* format which is for buffer buffers */
  libxs_dnn_tensor_format filter_format;     /* format which is for filter buffers */
  libxs_dnn_conv_algo algo;                  /* convolution algorithm used */
  libxs_dnn_conv_option options;             /* additional options */
  libxs_dnn_conv_fuse_op fuse_ops;           /* used ops into convolutions */
} libxs_dnn_conv_desc;

/** get string of error code */
LIBXS_API const char* libxs_dnn_get_error(libxs_dnn_err_t code);
LIBXS_API size_t libxs_dnn_typesize(libxs_dnn_datatype datatype);
LIBXS_API size_t libxs_dnn_get_simd_width(libxs_dnn_datatype datatype);

/** Create a handle (non-NULL if successful), and pre-build all JIT-code versions. */
LIBXS_API libxs_dnn_layer* libxs_dnn_create_conv_handle(
  libxs_dnn_conv_desc     conv_desc,
  libxs_dnn_err_t*        status );

/** Release the given convolution handle. */
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_conv_handle(const libxs_dnn_layer* handle);

/** Create buffers, filters and bias (non-NULL if successful) */
LIBXS_API libxs_dnn_buffer* libxs_dnn_link_buffer(const libxs_dnn_layer* handle, const libxs_dnn_buffer_type type, const void* data, libxs_dnn_tensor_format in_format, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_filter* libxs_dnn_link_filter(const libxs_dnn_layer* handle, const libxs_dnn_filter_type type, const void* data, libxs_dnn_tensor_format in_format, libxs_dnn_err_t* status);

/** get layout description of buffers and fiters from handle */
LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_get_buffer_datalayout(const libxs_dnn_layer* handle, const libxs_dnn_buffer_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_get_filter_datalayout(const libxs_dnn_layer* handle, const libxs_dnn_filter_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_datalayout(libxs_dnn_tensor_datalayout* layout);

/** scratch pad management */
LIBXS_API size_t libxs_dnn_get_scratch_size(const libxs_dnn_layer* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_bind_scratch(libxs_dnn_layer* handle, const libxs_dnn_compute_kind kind, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_release_scratch(libxs_dnn_layer* handle, const libxs_dnn_compute_kind kind);

/** Bind buffers, filters and bias to convolutions operation */
LIBXS_API libxs_dnn_err_t libxs_dnn_bind_buffer(libxs_dnn_layer* handle, const libxs_dnn_buffer* input, const libxs_dnn_buffer_type type);
LIBXS_API libxs_dnn_err_t libxs_dnn_bind_filter(libxs_dnn_layer* handle, const libxs_dnn_filter* filter, const libxs_dnn_filter_type type);

/** Release buffers, filters and bias from convolutions operation */
LIBXS_API libxs_dnn_err_t libxs_dnn_release_buffer(libxs_dnn_layer* handle, const libxs_dnn_buffer_type type);
LIBXS_API libxs_dnn_err_t libxs_dnn_release_filter(libxs_dnn_layer* handle, const libxs_dnn_filter_type type);

/** Release the given layer, filters, bias handle. */
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_buffer(const libxs_dnn_buffer* buffer);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_filter(const libxs_dnn_filter* filter);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_bias(const libxs_dnn_bias* bias);

/**
 * Copy-in from a plain format such as input := [N][C][H][W] or [N][H][W][C]
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXS_API libxs_dnn_err_t libxs_dnn_copyin_buffer(const libxs_dnn_buffer* buffer, const void* data, libxs_dnn_tensor_format in_format);
LIBXS_API libxs_dnn_err_t libxs_dnn_copyin_filter(const libxs_dnn_filter* filter, const void* data, libxs_dnn_tensor_format in_format);
/*LIBXS_API libxs_dnn_err_t libxs_conv_copyin_bias(const libxs_dnn_bias* bias, const void* data);*/
LIBXS_API libxs_dnn_err_t libxs_dnn_zero_buffer(const libxs_dnn_buffer* layer);

/**
 * Copy-out into a plain format such as output := [N][C][H][W] or [N][H][W][C]
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXS_API libxs_dnn_err_t libxs_dnn_copyout_buffer(const libxs_dnn_buffer* buffer, void* data, libxs_dnn_tensor_format out_format);
LIBXS_API libxs_dnn_err_t libxs_dnn_copyout_filter(const libxs_dnn_filter* filter, void* data, libxs_dnn_tensor_format out_format);
/*LIBXS_API libxs_dnn_err_t libxs_dnn_copyout_bias(const libxs_dnn_bias* bias, void* data);*/
/** Run the convolution identified by the handle; may use threads internally. */
LIBXS_API void libxs_dnn_execute(libxs_dnn_layer* handle, libxs_dnn_compute_kind kind);
LIBXS_API libxs_dnn_err_t libxs_dnn_transpose_filter(libxs_dnn_layer* handle, const libxs_dnn_filter_type type);
LIBXS_API libxs_dnn_err_t libxs_dnn_reduce_wu_filters(libxs_dnn_layer* handle, const libxs_dnn_filter_type type);
LIBXS_API libxs_dnn_err_t libxs_dnn_get_codegen_success(libxs_dnn_layer* handle, libxs_dnn_compute_kind kind);
LIBXS_API libxs_dnn_err_t libxs_dnn_get_parallel_tasks(libxs_dnn_layer* handle, libxs_dnn_compute_kind kind, unsigned int* num_tasks);

/** Run the convolution identified by the handle; takes a thread id. */
LIBXS_API libxs_dnn_err_t libxs_dnn_execute_st(libxs_dnn_layer* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid );

#if defined(LIBXS_BUILD) || defined(LIBXS_DNN_INTERNAL_API) /* Internal API */
/** Function type used for convolutions (single-precision); the actual signature depends on the kind of convolution. */
typedef LIBXS_RETARGETABLE void (*libxs_sconvfunction)(const float* input1, const float* input2, float* output,
                                                           const float* ipf1, const float* ipf2, const float* opf);

typedef LIBXS_RETARGETABLE void (*libxs_wconvfunction)(const short* input1, const short* input2, int* output,
                                                           const short* ipf1, const short* ipf2, const int* opf);

typedef LIBXS_RETARGETABLE void (*libxs_busconvfunction)(const unsigned char* input1, const char* input2, short* output,
                                                             const unsigned char* ipf1, const char* ipf2, const short* opf);

typedef LIBXS_RETARGETABLE void (*libxs_budconvfunction)(const unsigned char* input1, const char* input2, int* output,
                                                             const unsigned char* ipf1, const char* ipf2, const int* opf);

/** Function type which is either libxs_sconvfunction or libxs_wconvfunction (weak-typed). */
typedef union LIBXS_RETARGETABLE libxs_xconvfunction { libxs_sconvfunction sconv; libxs_wconvfunction wconv; libxs_busconvfunction busconv; libxs_busconvfunction budconv; } libxs_xconvfunction;

/** Code generation routine for a forward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API libxs_sconvfunction libxs_create_sconv_forward(const libxs_convolution_forward_descriptor* descriptor);

/** Code generation routine for a backward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API libxs_sconvfunction libxs_create_sconv_backward(const libxs_convolution_backward_descriptor* descriptor);

/** Code generation routine for a convolution kernel as specified by descriptor. */
LIBXS_API libxs_sconvfunction libxs_create_sconv_update_weights(const libxs_convolution_weight_update_descriptor* descriptor);

/** Code generation routine for a forward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API void* libxs_create_xconv_forward(const libxs_convolution_forward_descriptor* descriptor);

/** Code generation routine for a backward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API void* libxs_create_xconv_backward(const libxs_convolution_backward_descriptor* descriptor);

/** Code generation routine for a convolution kernel as specified by descriptor. */
LIBXS_API void* libxs_create_xconv_update_weights(const libxs_convolution_weight_update_descriptor* descriptor);

#endif
#endif /*LIBXS_DNN_H*/
