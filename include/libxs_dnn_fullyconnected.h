/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_FULLYCONNECTED_H
#define LIBXS_DNN_FULLYCONNECTED_H

#include "libxs_dnn.h"
#include "libxs_dnn_tensor.h"

/** Opaque handles which represents LIBXS fullyconnected */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_fullyconnected libxs_dnn_fullyconnected;

typedef enum libxs_dnn_fullyconnected_fuse_op {
  /* the fuse order is: 1. BIAS, 2. Actitvation */
  LIBXS_DNN_FULLYCONNECTED_FUSE_NONE = 0,
  LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS = 1,
  LIBXS_DNN_FULLYCONNECTED_FUSE_RELU = 2,
  LIBXS_DNN_FULLYCONNECTED_FUSE_SIGMOID = 4,
  LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_RELU = 3,
  LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID = 5
} libxs_dnn_fullyconnected_fuse_op;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_fullyconnected_desc {
  int N;                                        /* number of images in mini-batch */
  int C;                                        /* number of input feature maps */
  int K;                                        /* number of output feature maps */
  int bn;
  int bk;
  int bc;
  int threads;                                  /* number of threads used */
  int compressed_A;
  int sparsity_factor_A;
  libxs_dnn_datatype datatype_in;             /* datatype used for all input related buffers */
  libxs_dnn_datatype datatype_out;            /* datatype used for all output related buffers */
  libxs_dnn_tensor_format buffer_format;      /* format which is for activation buffers */
  libxs_dnn_tensor_format filter_format;      /* format which is for filter buffers */
  libxs_dnn_fullyconnected_fuse_op fuse_ops;  /* fused operations */
} libxs_dnn_fullyconnected_desc;

LIBXS_API libxs_dnn_fullyconnected* libxs_dnn_create_fullyconnected(libxs_dnn_fullyconnected_desc fullyconnected_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_fullyconnected(const libxs_dnn_fullyconnected* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_fullyconnected_create_tensor_datalayout(const libxs_dnn_fullyconnected* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API void*  libxs_dnn_fullyconnected_get_scratch_ptr (const libxs_dnn_fullyconnected* handle, libxs_dnn_err_t* status);
LIBXS_API size_t libxs_dnn_fullyconnected_get_scratch_size(const libxs_dnn_fullyconnected* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_bind_scratch(libxs_dnn_fullyconnected* handle, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_release_scratch(libxs_dnn_fullyconnected* handle);

LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_bind_tensor(libxs_dnn_fullyconnected* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_fullyconnected_get_tensor(libxs_dnn_fullyconnected* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_release_tensor(libxs_dnn_fullyconnected* handle, const libxs_dnn_tensor_type type);

LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_execute_st(libxs_dnn_fullyconnected* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_FULLYCONNECTED_H*/
