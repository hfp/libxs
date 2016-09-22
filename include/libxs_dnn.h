/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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

/** Opaque handles which represents convolutions and LIBXS datatypes */
typedef struct LIBXS_RETARGETABLE libxs_dnn_conv_handle libxs_dnn_conv_handle;
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

/** Kinds of supported convolution operations. */
typedef enum libxs_dnn_conv_kind {
  /** Forward convolution. */
  LIBXS_DNN_CONV_KIND_FWD,
  /** Backward convolution. */
  LIBXS_DNN_CONV_KIND_BWD,
  /** Updated weights. */
  LIBXS_DNN_CONV_KIND_UPD
} libxs_dnn_conv_kind;

typedef enum libxs_dnn_conv_fuse_ops {
  /* we fuse nothing into convolution */
  LIBXS_DNN_CONV_FUSE_NONE = 0
#if 0
  ,
  /* we fuse fuse bias init into convolution */
  LIBXS_DNN_CONV_FUSE_BIAS = 1,
  /* we fase fase ReLU calculation into convolution Op */
  LIBXS_DNN_CONV_FUSE_RELU = 2
#endif
} libxs_dnn_conv_fuse_ops;

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
  int pad_h_in;                                /* height of zero-padding in input buffer, ignored */
  int pad_w_in;                                /* width of zero-padding in input buffer, ignored */
  int pad_h_out;                               /* height of zero-padding in output buffer */
  int pad_w_out;                               /* width of zero-padding in output buffer */
  int splits;                                  /* number of splits */
  libxs_dnn_conv_algo algo;                  /* convolution algorithm used */
  libxs_dnn_conv_format buffer_format;       /* format which is for buffer buffers */
  libxs_dnn_conv_format filter_format;       /* format which is for filter buffers */
  libxs_dnn_conv_fuse_ops fuse_ops;          /* used ops into convolutoions */
  libxs_dnn_datatype datatype;               /* dataytpes use for all buffers */
} libxs_dnn_conv_desc;

/** get string of error code */
LIBXS_API const char* libxs_dnn_get_error(libxs_dnn_err_t code);

/** Create a handle (non-NULL if successful), and pre-build all JIT-code versions. */
LIBXS_API libxs_dnn_conv_handle* libxs_dnn_create_conv_handle(
  libxs_dnn_conv_desc     conv_desc );

LIBXS_API libxs_dnn_conv_handle* libxs_dnn_create_conv_handle_check(
  libxs_dnn_conv_desc     conv_desc,
  libxs_dnn_err_t*        status );

/** Release the given convolution handle. */
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_conv_handle(const libxs_dnn_conv_handle* handle);

/** Create buffers, filters and bias (non-NULL if successful) */
LIBXS_API libxs_dnn_buffer* libxs_dnn_create_input_buffer(const libxs_dnn_conv_handle* handle);
LIBXS_API libxs_dnn_buffer* libxs_dnn_create_output_buffer(const libxs_dnn_conv_handle* handle);
LIBXS_API libxs_dnn_filter* libxs_dnn_create_filter(const libxs_dnn_conv_handle* handle);
LIBXS_API libxs_dnn_bias*   libxs_dnn_create_bias(const libxs_dnn_conv_handle* handle);
LIBXS_API libxs_dnn_buffer* libxs_dnn_link_input_buffer(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format);
LIBXS_API libxs_dnn_buffer* libxs_dnn_link_output_buffer(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format);
LIBXS_API libxs_dnn_filter* libxs_dnn_link_filter(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format);

LIBXS_API libxs_dnn_buffer* libxs_dnn_create_input_buffer_check(const libxs_dnn_conv_handle* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_buffer* libxs_dnn_create_output_buffer_check(const libxs_dnn_conv_handle* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_filter* libxs_dnn_create_filter_check(const libxs_dnn_conv_handle* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_bias*   libxs_dnn_create_bias_check(const libxs_dnn_conv_handle* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_buffer* libxs_dnn_link_input_buffer_check(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_buffer* libxs_dnn_link_output_buffer_check(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_filter* libxs_dnn_link_filter_check(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format, libxs_dnn_err_t* status);

/** Bind layers, filters and bias to convolutions operation */
LIBXS_API libxs_dnn_err_t libxs_dnn_bind_input_buffer(libxs_dnn_conv_handle* handle, const libxs_dnn_buffer* input);
LIBXS_API libxs_dnn_err_t libxs_dnn_bind_output_buffer(libxs_dnn_conv_handle* handle, const libxs_dnn_buffer* output);
LIBXS_API libxs_dnn_err_t libxs_dnn_bind_filter(libxs_dnn_conv_handle* handle, const libxs_dnn_filter* filter);

/** Release layers, filters and bias from convolutions operation */
#if 0
LIBXS_API libxs_dnn_err_t libxs_dnn_release_input_buffer(libxs_dnn_conv_handle* handle);
LIBXS_API libxs_dnn_err_t libxs_dnn_release_output_buffer(libxs_dnn_conv_handle* handle);
LIBXS_API libxs_dnn_err_t libxs_dnn_release_filter(libxs_dnn_conv_handle* handle);
#endif

/** Release the given layer, filters, bias handle. */
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_buffer(const libxs_dnn_buffer* buffer);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_filter(const libxs_dnn_filter* filter);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_bias(const libxs_dnn_bias* bias);

/**
 * Copy-in from a plain format such as input := [img][splits][ofm][ifm].
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXS_API libxs_dnn_err_t libxs_dnn_copyin_buffer(const libxs_dnn_buffer* buffer, const void* data, libxs_dnn_conv_format in_format);
LIBXS_API libxs_dnn_err_t libxs_dnn_copyin_filter(const libxs_dnn_filter* filter, const void* data, libxs_dnn_conv_format in_format);
/*LIBXS_API libxs_dnn_err_t libxs_conv_copyin_bias(const libxs_dnn_bias* bias, const void* data);*/
LIBXS_API libxs_dnn_err_t libxs_dnn_zero_buffer(const libxs_dnn_buffer* layer);

/**
 * Copy-out into a plain format such as output := [img][splits][ofm][ifm].
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXS_API libxs_dnn_err_t libxs_dnn_copyout_buffer(const libxs_dnn_buffer* buffer, void* data, libxs_dnn_conv_format out_format);
LIBXS_API libxs_dnn_err_t libxs_dnn_copyout_filter(const libxs_dnn_filter* filter, void* data, libxs_dnn_conv_format out_format);
/*LIBXS_API libxs_dnn_err_t libxs_dnn_copyout_bias(const libxs_dnn_bias* bias, void* data);*/
/** Run the convolution identified by the handle; may use threads internally. */
LIBXS_API void libxs_dnn_convolve(libxs_dnn_conv_handle* handle, libxs_dnn_conv_kind kind);

/** Run the convolution identified by the handle; takes a thread id. */
LIBXS_API libxs_dnn_err_t libxs_dnn_convolve_st(libxs_dnn_conv_handle* handle, libxs_dnn_conv_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid, /*unsigned*/int num_threads);

#if defined(LIBXS_BUILD) || defined(LIBXS_DNN_INTERNAL_API) /* Internal API */
/** Function type used for convolutions (single-precision); the actual signature depends on the kind of convolution. */
typedef LIBXS_RETARGETABLE void (*libxs_sconvfunction)(const float* input1, const float* input2, float* output,
                                                           const float* ipf1, const float* ipf2, const float* opf);

typedef LIBXS_RETARGETABLE void (*libxs_wconvfunction)(const short* input1, const short* input2, int* output,
                                                           const short* ipf1, const short* ipf2, const int* opf);

/** Function type which is either libxs_sconvfunction or libxs_wconvfunction (weak-typed). */
typedef union LIBXS_RETARGETABLE libxs_xconvfunction { libxs_sconvfunction sconv; libxs_wconvfunction wconv; } libxs_xconvfunction;

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
