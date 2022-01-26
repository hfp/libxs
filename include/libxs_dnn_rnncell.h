/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_RNNCELL_H
#define LIBXS_DNN_RNNCELL_H

#include "libxs_dnn.h"
#include "libxs_dnn_tensor.h"

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_rnncell libxs_dnn_rnncell;

/** Type of algorithm used for convolutions. */
typedef enum libxs_dnn_rnncell_type {
  /** simple RNN cell with ReLU as activation function */
  LIBXS_DNN_RNNCELL_RNN_RELU,
  /** simple RNN cell with sigmoid as activation function */
  LIBXS_DNN_RNNCELL_RNN_SIGMOID,
  /** simple RNN cell with tanh as activation function */
  LIBXS_DNN_RNNCELL_RNN_TANH,
  /** LSTM cell */
  LIBXS_DNN_RNNCELL_LSTM,
  /** GRU cell */
  LIBXS_DNN_RNNCELL_GRU
} libxs_dnn_rnncell_type;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_rnncell_desc {
  int threads;
  libxs_blasint K;         /* number of outputs */
  libxs_blasint N;         /* size of the minibatch */
  libxs_blasint C;         /* number of inputs */
  libxs_blasint max_T;     /* number of time steps */
  libxs_blasint bk;
  libxs_blasint bn;
  libxs_blasint bc;
  int use_fwd_fused_impl;
  int fwd_block;
  int bwdupd_block;
  libxs_dnn_rnncell_type cell_type;       /* cell type RNN ReLU, RNN Sigmoid, RNN Tanh, LSTM, GRU */
  libxs_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxs_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxs_dnn_tensor_format buffer_format;  /* format which is for activation buffers */
  libxs_dnn_tensor_format filter_format;  /* format which is for filter buffers */
} libxs_dnn_rnncell_desc;

LIBXS_API libxs_dnn_rnncell* libxs_dnn_create_rnncell(libxs_dnn_rnncell_desc rnncell_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_rnncell(const libxs_dnn_rnncell* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_rnncell_create_tensor_datalayout(const libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API size_t libxs_dnn_rnncell_get_scratch_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API void*  libxs_dnn_rnncell_get_scratch_ptr (const libxs_dnn_rnncell* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind);

LIBXS_API size_t libxs_dnn_rnncell_get_internalstate_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API void*  libxs_dnn_rnncell_get_internalstate_ptr (const libxs_dnn_rnncell* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* internalstate);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind);

LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_allocate_forget_bias(libxs_dnn_rnncell* handle, const float forget_bias);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_rnncell_get_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type);

LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_set_sequence_length( libxs_dnn_rnncell* handle, const libxs_blasint T );
LIBXS_API libxs_blasint libxs_dnn_rnncell_get_sequence_length( libxs_dnn_rnncell* handle, libxs_dnn_err_t* status );

LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_execute_st(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_RNNCELL_H*/
