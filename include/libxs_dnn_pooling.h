/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_POOLING_H
#define LIBXS_DNN_POOLING_H

#include "libxs_dnn.h"
#include "libxs_dnn_tensor.h"

/** Opaque handles which represents LIBXS pooling */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_pooling libxs_dnn_pooling;

typedef enum libxs_dnn_pooling_type {
  LIBXS_DNN_POOLING_MAX = 1,
  LIBXS_DNN_POOLING_AVG = 2
} libxs_dnn_pooling_type;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_pooling_desc {
  int N;                                     /* number of images in mini-batch */
  int C;                                     /* number of input feature maps */
  int H;                                     /* height of input image */
  int W;                                     /* width of input image */
  int R;                                     /* kernel height */
  int S;                                     /* kernel width */
  int u;                                     /* vertical stride */
  int v;                                     /* horizontal stride */
  int pad_h;                                 /* height of logical padding of input buffer */
  int pad_w;                                 /* width of logical padding of input buffer */
  int pad_h_in;                              /* height of physical zero-padding in input buffer */
  int pad_w_in;                              /* width of physical zero-padding in input buffer */
  int pad_h_out;                             /* height of physical zero-padding in output buffer */
  int pad_w_out;                             /* width of physical zero-padding in output buffer */
  int threads;                               /* number of threads used */
  libxs_dnn_datatype datatype_in;          /* datatypes used for all input related buffer */
  libxs_dnn_datatype datatype_out;         /* datatypes used for all output related buffer */
  libxs_dnn_datatype datatype_mask;        /* datatypes used for the masks */
  libxs_dnn_tensor_format buffer_format;   /* format which is for activation buffers */
  libxs_dnn_pooling_type pooling_type;     /* type of pooling operation */
} libxs_dnn_pooling_desc;

LIBXS_API libxs_dnn_pooling* libxs_dnn_create_pooling(libxs_dnn_pooling_desc pooling_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_pooling(const libxs_dnn_pooling* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_pooling_create_tensor_datalayout(const libxs_dnn_pooling* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API size_t libxs_dnn_pooling_get_scratch_size(const libxs_dnn_pooling* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_pooling_bind_scratch(libxs_dnn_pooling* handle, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_pooling_release_scratch(libxs_dnn_pooling* handle);

LIBXS_API libxs_dnn_err_t libxs_dnn_pooling_bind_tensor(libxs_dnn_pooling* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_pooling_get_tensor(libxs_dnn_pooling* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_pooling_release_tensor(libxs_dnn_pooling* handle, const libxs_dnn_tensor_type type);

LIBXS_API libxs_dnn_err_t libxs_dnn_pooling_execute_st(libxs_dnn_pooling* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_POOLING_H*/
