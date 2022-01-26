/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_FUSEDBATCHNORM_H
#define LIBXS_DNN_FUSEDBATCHNORM_H

#include "libxs_dnn.h"
#include "libxs_dnn_tensor.h"

/** Opaque handles which represents LIBXS fusedbatchnorm */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_fusedbatchnorm libxs_dnn_fusedbatchnorm;

LIBXS_API libxs_dnn_fusedbatchnorm* libxs_dnn_create_fusedbatchnorm(libxs_dnn_fusedbatchnorm_desc fusedbatchnorm_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_fusedbatchnorm(const libxs_dnn_fusedbatchnorm* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_fusedbatchnorm_create_tensor_datalayout(const libxs_dnn_fusedbatchnorm* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API size_t libxs_dnn_fusedbatchnorm_get_scratch_size(const libxs_dnn_fusedbatchnorm* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_bind_scratch(libxs_dnn_fusedbatchnorm* handle, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_release_scratch(libxs_dnn_fusedbatchnorm* handle);

LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_bind_tensor(libxs_dnn_fusedbatchnorm* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_fusedbatchnorm_get_tensor(libxs_dnn_fusedbatchnorm* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_release_tensor(libxs_dnn_fusedbatchnorm* handle, const libxs_dnn_tensor_type type);

LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_execute_st(libxs_dnn_fusedbatchnorm* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);
LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_reduce_stats_st(libxs_dnn_fusedbatchnorm** handles, int num_handles, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_FUSEDBATCHNORM_H*/
