/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_SGD_H
#define LIBXS_DNN_SGD_H

#include "libxs_dnn.h"
#include "libxs_dnn_tensor.h"

/** Opaque handles which represents LIBXS optimizer */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_optimizer libxs_dnn_optimizer;

typedef enum libxs_dnn_optimizer_type {
  LIBXS_DNN_OPTIMIZER_SGD = 1
} libxs_dnn_optimizer_type;


LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_optimizer_desc {
  int C;                                        /* number of feature maps */
  int K;                                        /* number of feature maps */
  int bc;
  int bk;
  float learning_rate;                          /* learning rate */
  int threads;                                  /* number of threads used */
  libxs_dnn_optimizer_type opt_type;
  libxs_dnn_datatype datatype_master;         /* datatype used for all input related buffers */
  libxs_dnn_datatype datatype;                /* datatype used for all input related buffers */
  libxs_dnn_tensor_format filter_format;      /* format which is for filter buffers */
} libxs_dnn_optimizer_desc;

LIBXS_API libxs_dnn_optimizer* libxs_dnn_create_optimizer(libxs_dnn_optimizer_desc optimizer_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_optimizer(const libxs_dnn_optimizer* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_optimizer_create_tensor_datalayout(const libxs_dnn_optimizer* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API void*  libxs_dnn_optimizer_get_scratch_ptr (const libxs_dnn_optimizer* handle, libxs_dnn_err_t* status);
LIBXS_API size_t libxs_dnn_optimizer_get_scratch_size(const libxs_dnn_optimizer* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_optimizer_bind_scratch(libxs_dnn_optimizer* handle, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_optimizer_release_scratch(libxs_dnn_optimizer* handle);

LIBXS_API libxs_dnn_err_t libxs_dnn_optimizer_bind_tensor(libxs_dnn_optimizer* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_optimizer_get_tensor(libxs_dnn_optimizer* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_optimizer_release_tensor(libxs_dnn_optimizer* handle, const libxs_dnn_tensor_type type);

LIBXS_API libxs_dnn_err_t libxs_dnn_optimizer_execute_st(libxs_dnn_optimizer* handle, /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_SGD_H*/
