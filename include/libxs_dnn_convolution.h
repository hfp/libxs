/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_CONVOLUTION_H
#define LIBXS_DNN_CONVOLUTION_H

#include "libxs_dnn.h"
#include "libxs_dnn_tensor.h"
#include "libxs_dnn_fusedbatchnorm.h"

/** Opaque handles which represents convolutions and LIBXS datatypes */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_layer libxs_dnn_layer;

typedef enum libxs_dnn_conv_fuse_op {
  /* we fuse nothing into convolution */
  LIBXS_DNN_CONV_FUSE_NONE = 0
} libxs_dnn_conv_fuse_op;

/** Type of algorithm used for convolutions. */
typedef enum libxs_dnn_conv_algo {
  /** let the library decide */
  LIBXS_DNN_CONV_ALGO_AUTO,
  /** direct convolution. */
  LIBXS_DNN_CONV_ALGO_DIRECT
} libxs_dnn_conv_algo;

/** Structure which describes the input and output of data (DNN). */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_conv_desc {
  int N;                                    /* number of images in mini-batch */
  int C;                                    /* number of input feature maps */
  int H;                                    /* height of input image */
  int W;                                    /* width of input image */
  int K;                                    /* number of output feature maps */
  int R;                                    /* height of filter kernel */
  int S;                                    /* width of filter kernel */
  int u;                                    /* vertical stride */
  int v;                                    /* horizontal stride */
  int pad_h;                                /* height of logical rim padding to input
                                               for adjusting output height */
  int pad_w;                                /* width of logical rim padding to input
                                               for adjusting output width */
  int pad_h_in;                             /* height of zero-padding in input buffer,
                                               must equal to pad_h for direct conv */
  int pad_w_in;                             /* width of zero-padding in input buffer,
                                               must equal to pad_w for direct conv */
  int pad_h_out;                            /* height of zero-padding in output buffer */
  int pad_w_out;                            /* width of zero-padding in output buffer */
  int threads;                              /* number of threads to use when running
                                               convolution */
  libxs_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxs_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxs_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
  libxs_dnn_tensor_format filter_format;  /* format which is for filter buffers */
  libxs_dnn_conv_algo algo;               /* convolution algorithm used */
  libxs_dnn_conv_option options;          /* additional options */
  libxs_dnn_conv_fuse_op fuse_ops;        /* used ops into convolutions */
} libxs_dnn_conv_desc;

/** Create a layer handle (non-NULL if successful), and pre-build all JIT-code versions. */
LIBXS_API libxs_dnn_layer* libxs_dnn_create_conv_layer(libxs_dnn_conv_desc conv_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_conv_layer(const libxs_dnn_layer* handle);

/** get layout description of buffers and filters from handle */
LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_create_tensor_datalayout(const libxs_dnn_layer* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

/** scratch pad management */
LIBXS_API size_t libxs_dnn_get_scratch_size(const libxs_dnn_layer* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_bind_scratch(libxs_dnn_layer* handle, const libxs_dnn_compute_kind kind, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_release_scratch(libxs_dnn_layer* handle, const libxs_dnn_compute_kind kind);

/** Bind/Release buffers, filters and bias to layer operation */
LIBXS_API libxs_dnn_err_t libxs_dnn_bind_tensor(libxs_dnn_layer* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_get_tensor(libxs_dnn_layer* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_release_tensor(libxs_dnn_layer* handle, const libxs_dnn_tensor_type type);

/** Run the layer identified by the handle; may use threads internally. */
LIBXS_API void libxs_dnn_execute(libxs_dnn_layer* handle, libxs_dnn_compute_kind kind);
LIBXS_API libxs_dnn_err_t libxs_dnn_execute_st(libxs_dnn_layer* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

/** some helper functions for framework integration */
LIBXS_API libxs_dnn_err_t libxs_dnn_trans_reg_filter(const libxs_dnn_layer* handle);
LIBXS_API libxs_dnn_err_t libxs_dnn_trans_reg_bf16_filter(const libxs_dnn_layer* handle);

#endif /*LIBXS_DNN_CONVOLUTION_H*/
