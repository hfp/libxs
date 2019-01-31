/******************************************************************************
** Copyright (c) 2017-2019, Intel Corporation                                **
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
#ifndef LIBXS_DNN_RNNCELL_H
#define LIBXS_DNN_RNNCELL_H

#include "libxs_dnn.h"


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
  LIBXS_DNN_RNNCELL_LSTM
} libxs_dnn_rnncell_type;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_rnncell_desc {
  libxs_blasint threads;
  libxs_blasint K;         /* number of outputs */
  libxs_blasint N;         /* size of the minibatch */
  libxs_blasint C;         /* number of inputs */
  libxs_blasint max_T;     /* number of time steps */
  libxs_blasint bk;
  libxs_blasint bn;
  libxs_blasint bc;
  libxs_dnn_rnncell_type cell_type;       /* cell type RNN ReLU, RNN Sigmoid, RNN Tanh, LSTM */
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
