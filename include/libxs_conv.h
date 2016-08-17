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
#ifndef LIBXS_CONV_H
#define LIBXS_CONV_H

#include <libxs.h>

/** Opaque handles which represents convolutions and LIBXS datatypes */
typedef struct LIBXS_RETARGETABLE libxs_conv_handle libxs_conv_handle;
typedef struct LIBXS_RETARGETABLE libxs_conv_layer libxs_conv_layer;
typedef struct LIBXS_RETARGETABLE libxs_conv_bias libxs_conv_bias;
typedef struct LIBXS_RETARGETABLE libxs_conv_filter libxs_conv_filter;
typedef unsigned int libxs_conv_err_t;

/** Define error and warning codes */
#define LIBXS_CONV_SUCCESS                           0
#define LIBXS_CONV_WARN_FALLBACK                 90000
#define LIBXS_CONV_ERR_GENERAL                  100000
#define LIBXS_CONV_ERR_CREATE_HANDLE            100001
#define LIBXS_CONV_ERR_UNSUPPORTED_DATATYPE     100002
#define LIBXS_CONV_ERR_INVALID_BLOCKING         100003
#define LIBXS_CONV_ERR_INVALID_HANDLE           100004
#define LIBXS_CONV_ERR_DATA_NOT_BOUND           100005
#define LIBXS_CONV_ERR_CREATE_LAYER             100006
#define LIBXS_CONV_ERR_INVALID_LAYER            100007
#define LIBXS_CONV_ERR_CREATE_FILTER            100008
#define LIBXS_CONV_ERR_INVALID_FILTER           100009
#define LIBXS_CONV_ERR_CREATE_BIAS              100010
#define LIBXS_CONV_ERR_INVALID_BIAS             100011
#define LIBXS_CONV_ERR_MISMATCH_LAYER           100012
#define LIBXS_CONV_ERR_INVALID_HANDLE_LAYER     100013
#define LIBXS_CONV_ERR_MISMATCH_FILTER          100014
#define LIBXS_CONV_ERR_INVALID_HANDLE_FILTER    100015
#define LIBXS_CONV_ERR_INVALID_KIND             100016

/** Kinds of supported convolution operations. */
typedef enum libxs_conv_kind {
  /** Forward convolution. */
  LIBXS_CONV_KIND_FWD,
  /** Forward convolution, fused Bias */
  LIBXS_CONV_KIND_FWD_BIAS,
  /** Forward convolution, fused Bias and ReLU */
  LIBXS_CONV_KIND_FWD_BIAS_RELU,
  /** Backward convolution. */
  LIBXS_CONV_KIND_BWD,
  /** Backward convolution, fused ReLU */
  LIBXS_CONV_KIND_BWD_RELU,
  /** Updated weights. */
  LIBXS_CONV_KIND_UPD,
  /** Updated weights, fused Bias */
  LIBXS_CONV_KIND_UPD_BIAS
} libxs_conv_kind;

/** Typ of algorithm used for convolutions. */
typedef enum libxs_conv_algo {
  /** direct convolution. */
  LIBXS_CONV_ALGO_DIRECT
} libxs_conv_algo;

/** Denotes the element/pixel type of an image/channel. */
typedef enum libxs_conv_datatype {
  LIBXS_CONV_DATATYPE_FP32,
  LIBXS_CONV_DATATYPE_INT32,
  LIBXS_CONV_DATATYPE_INT16,
  LIBXS_CONV_DATATYPE_INT8
} libxs_conv_datatype;

/** struct which holds description of convolution */
typedef struct LIBXS_RETARGETABLE libxs_conv_desc {
  int N;           /* number of images in mini-batch */
  int C;           /* number of input feature maps */
  int H;           /* height of input image */
  int W;           /* width of input image */
  int K;           /* number of output feature maps */
  int R;           /* height of filter kernel */
  int S;           /* width of filter kernel */
  int u;           /* vertical stride */
  int v;           /* horizontal stride */
  int pad_h;       /* height of zero-padding */
  int pad_w;       /* width of zero-padding */
  int splits;      /* number of splits */
} libxs_conv_desc;

/** get string of error code */
LIBXS_API char* libxs_conv_get_error(libxs_conv_err_t code);

/** Create a handle (non-NULL if successful), and pre-build all JIT-code versions. */
LIBXS_API libxs_conv_handle* libxs_conv_create_handle(
  libxs_conv_desc     conv_desc,
  libxs_conv_datatype conv_datatype,
  libxs_conv_algo     conv_algo );

LIBXS_API libxs_conv_handle* libxs_conv_create_handle_check(
  libxs_conv_desc     conv_desc,
  libxs_conv_datatype conv_datatype,
  libxs_conv_algo     conv_algo,
  libxs_conv_err_t*   status );

/** Release the given convolution handle. */
LIBXS_API libxs_conv_err_t libxs_conv_destroy_handle(const libxs_conv_handle* handle);

/** Create layers, filters and bias (non-NULL if successful) */
LIBXS_API libxs_conv_layer* libxs_conv_create_input_layer(const libxs_conv_handle* handle);
LIBXS_API libxs_conv_layer* libxs_conv_create_output_layer(const libxs_conv_handle* handle);
LIBXS_API libxs_conv_filter* libxs_conv_create_filter(const libxs_conv_handle* handle);
LIBXS_API libxs_conv_bias* libxs_conv_create_bias(const libxs_conv_handle* handle);

LIBXS_API libxs_conv_layer* libxs_conv_create_input_layer_check(const libxs_conv_handle* handle, libxs_conv_err_t* status);
LIBXS_API libxs_conv_layer* libxs_conv_create_output_layer_check(const libxs_conv_handle* handle, libxs_conv_err_t* status);
LIBXS_API libxs_conv_filter* libxs_conv_create_filter_check(const libxs_conv_handle* handle, libxs_conv_err_t* status);
LIBXS_API libxs_conv_bias* libxs_conv_create_bias_check(const libxs_conv_handle* handle, libxs_conv_err_t* status);

/** Bind layers, filters and bias to convolutions operation */
LIBXS_API libxs_conv_err_t libxs_conv_bind_input_layer(libxs_conv_handle* handle, const libxs_conv_layer* layer);
LIBXS_API libxs_conv_err_t libxs_conv_bind_output_layer(libxs_conv_handle* handle, const libxs_conv_layer* layer);
LIBXS_API libxs_conv_err_t libxs_conv_bind_filter(libxs_conv_handle* handle, const libxs_conv_filter* filter);

/** Release layers, filters and bias from convolutions operation */
#if 0
LIBXS_API libxs_conv_err_t libxs_conv_release_input_layer(libxs_conv_handle* handle, const libxs_conv_layer* layer);
LIBXS_API libxs_conv_err_t libxs_conv_release_output_layer(libxs_conv_handle* handle, const libxs_conv_layer* layer);
LIBXS_API libxs_conv_err_t libxs_conv_release_filter(libxs_conv_handle* handle, const libxs_conv_filter* filter);
#endif

/** Release the given layer, filters, bias handle. */
LIBXS_API libxs_conv_err_t libxs_conv_destroy_layer(const libxs_conv_layer* layer);
LIBXS_API libxs_conv_err_t libxs_conv_destroy_filter(const libxs_conv_filter* filter);
LIBXS_API libxs_conv_err_t libxs_conv_destroy_bias(const libxs_conv_bias* bias);

/**
 * Copy-in from a plain format such as input := [img][splits][ofm][ifm].
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXS_API libxs_conv_err_t libxs_conv_copyin_layer(const libxs_conv_layer* layer, const void* data);
LIBXS_API libxs_conv_err_t libxs_conv_copyin_filter(const libxs_conv_filter* filter, const void* data);
/*LIBXS_API libxs_conv_err_t libxs_conv_copyin_bias(const libxs_conv_bias* bias, const void* data);*/
LIBXS_API libxs_conv_err_t libxs_conv_zero_layer(const libxs_conv_layer* layer);

/**
 * Copy-out into a plain format such as output := [img][splits][ofm][ifm].
 * The index specifies the actual channel number, and an eventual
 * padding is defined by the handle (pitch/stride).
 */
LIBXS_API libxs_conv_err_t libxs_conv_copyout_layer(const libxs_conv_layer* layer, void* data);
LIBXS_API libxs_conv_err_t libxs_conv_copyout_filter(const libxs_conv_filter* filter, void* data);
/*LIBXS_API libxs_conv_err_t libxs_conv_copyout_bias(const libxs_conv_bias* bias, void* data);*/
/** Run the convolution identified by the handle; may use threads internally. */
LIBXS_API void libxs_convolve(libxs_conv_handle* handle, libxs_conv_kind kind);

/** Run the convolution identified by the handle; takes a thread id. */
LIBXS_API libxs_conv_err_t libxs_convolve_st(libxs_conv_handle* handle, libxs_conv_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid, /*unsigned*/int num_threads);

#if defined(LIBXS_BUILD) || defined(LIBXS_CONV_INTERNAL_API) /* Internal API */
/** Function type used for convolutions (single-precision); the actual signature depends on the kind of convolution. */
typedef LIBXS_RETARGETABLE void (*libxs_sconvfunction)(const float* input1, const float* input2, float* output,
                                                           const float* ipf1, const float* ipf2, const float* opf);

/** Code generation routine for a forward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API libxs_sconvfunction libxs_create_sconv_forward(const libxs_convolution_forward_descriptor* descriptor);

/** Code generation routine for a backward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API libxs_sconvfunction libxs_create_sconv_backward(const libxs_convolution_backward_descriptor* descriptor);

/** Code generation routine for a convolution kernel as specified by descriptor. */
LIBXS_API libxs_sconvfunction libxs_create_sconv_update_weights(const libxs_convolution_weight_update_descriptor* descriptor);

#endif
#endif /*LIBXS_CONV_H*/
