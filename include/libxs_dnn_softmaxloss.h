/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_SOFTMAXLOSS_H
#define LIBXS_DNN_SOFTMAXLOSS_H

#include "libxs_dnn.h"
#include "libxs_dnn_tensor.h"

/** Opaque handles which represents LIBXS softmaxloss */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_softmaxloss libxs_dnn_softmaxloss;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_softmaxloss_desc {
  int N;                                        /* number of images in mini-batch */
  int C;                                        /* number of input feature maps */
  int bn;                                       /* requested N blocking for NCNC format */
  int bc;                                       /* requested C blocking for NCNC format */
  float loss_weight;                            /* loss weight */
  int threads;                                  /* number of threads used */
  libxs_dnn_datatype datatype;                /* datatype used for all buffers */
  libxs_dnn_tensor_format buffer_format;      /* format which is for activation buffers */
} libxs_dnn_softmaxloss_desc;

LIBXS_API libxs_dnn_softmaxloss* libxs_dnn_create_softmaxloss(libxs_dnn_softmaxloss_desc softmaxloss_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_softmaxloss(const libxs_dnn_softmaxloss* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_softmaxloss_create_tensor_datalayout(const libxs_dnn_softmaxloss* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API void*  libxs_dnn_softmaxloss_get_scratch_ptr (const libxs_dnn_softmaxloss* handle, libxs_dnn_err_t* status);
LIBXS_API size_t libxs_dnn_softmaxloss_get_scratch_size(const libxs_dnn_softmaxloss* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_softmaxloss_bind_scratch(libxs_dnn_softmaxloss* handle, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_softmaxloss_release_scratch(libxs_dnn_softmaxloss* handle);

LIBXS_API libxs_dnn_err_t libxs_dnn_softmaxloss_bind_tensor(libxs_dnn_softmaxloss* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_softmaxloss_get_tensor(libxs_dnn_softmaxloss* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_softmaxloss_release_tensor(libxs_dnn_softmaxloss* handle, const libxs_dnn_tensor_type type);

LIBXS_API libxs_dnn_err_t libxs_dnn_softmaxloss_execute_st(libxs_dnn_softmaxloss* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

LIBXS_API float libxs_dnn_softmaxloss_get_loss(const libxs_dnn_softmaxloss* handle, libxs_dnn_err_t* status);

#endif /*LIBXS_DNN_SOFTMAXLOSS_H*/
