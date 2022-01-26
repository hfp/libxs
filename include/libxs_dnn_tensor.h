/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_TENSOR_H
#define LIBXS_DNN_TENSOR_H

#include "libxs_typedefs.h"
#include "libxs_dnn.h"

/** Opaque handles which represents convolutions and LIBXS datatypes */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_tensor libxs_dnn_tensor;

typedef enum libxs_dnn_tensor_dimtype {
  /** Mini-batch */
  LIBXS_DNN_TENSOR_DIMTYPE_N,
  /** Image Height */
  LIBXS_DNN_TENSOR_DIMTYPE_H,
  /** Image Width */
  LIBXS_DNN_TENSOR_DIMTYPE_W,
  /** channels or input channels */
  LIBXS_DNN_TENSOR_DIMTYPE_C,
  /** output channels */
  LIBXS_DNN_TENSOR_DIMTYPE_K,
  /** kernel height */
  LIBXS_DNN_TENSOR_DIMTYPE_R,
  /** kernel width */
  LIBXS_DNN_TENSOR_DIMTYPE_S,
  /** sequence lenth counter */
  LIBXS_DNN_TENSOR_DIMTYPE_T,
  /** channle group counter */
  LIBXS_DNN_TENSOR_DIMTYPE_G,
  /** general counter */
  LIBXS_DNN_TENSOR_DIMTYPE_X
} libxs_dnn_tensor_dimtype;

/** types of different buffers */
typedef enum libxs_dnn_tensor_type {
  /** regular input buffer */
  LIBXS_DNN_REGULAR_INPUT,
  /** regular input buffer */
  LIBXS_DNN_REGULAR_INPUT_ADD,
  /** regular input buffer, transpose */
  LIBXS_DNN_REGULAR_INPUT_TRANS,
  /** gradient input buffer */
  LIBXS_DNN_GRADIENT_INPUT,
  /** gradient input buffer */
  LIBXS_DNN_GRADIENT_INPUT_ADD,
  /** regular output buffer */
  LIBXS_DNN_REGULAR_OUTPUT,
  /** gradient output buffer */
  LIBXS_DNN_GRADIENT_OUTPUT,
  /** general input type */
  LIBXS_DNN_INPUT,
  /** general output type */
  LIBXS_DNN_OUTPUT,
  /** general activation type */
  LIBXS_DNN_ACTIVATION,
  /* regular filter */
  LIBXS_DNN_REGULAR_FILTER,
  /* regular filter */
  LIBXS_DNN_REGULAR_FILTER_TRANS,
  /* gradient filter */
  LIBXS_DNN_GRADIENT_FILTER,
  /* master filter */
  LIBXS_DNN_MASTER_FILTER,
  /** general filter type */
  LIBXS_DNN_FILTER,
  /* regular bias */
  LIBXS_DNN_REGULAR_CHANNEL_BIAS,
  /* gradient bias */
  LIBXS_DNN_GRADIENT_CHANNEL_BIAS,
  /* bias */
  LIBXS_DNN_CHANNEL_BIAS,
  /* regular beta */
  LIBXS_DNN_REGULAR_CHANNEL_BETA,
  /* gradient beta */
  LIBXS_DNN_GRADIENT_CHANNEL_BETA,
  /* beta */
  LIBXS_DNN_CHANNEL_BETA,
  /* regular gamma */
  LIBXS_DNN_REGULAR_CHANNEL_GAMMA,
  /* gradient gamma */
  LIBXS_DNN_GRADIENT_CHANNEL_GAMMA,
  /* Gamma */
  LIBXS_DNN_CHANNEL_GAMMA,
  /* regular beta */
  LIBXS_DNN_CHANNEL_EXPECTVAL,
  /* regular beta */
  LIBXS_DNN_CHANNEL_RCPSTDDEV,
  /* variance */
  LIBXS_DNN_CHANNEL_VARIANCE,
  /** general bias type */
  LIBXS_DNN_CHANNEL_SCALAR,
  /** Labels */
  LIBXS_DNN_LABEL,
  /** batch stats */
  LIBXS_DNN_BATCH_STATS,
  LIBXS_DNN_MAX_STATS_FWD,
  LIBXS_DNN_MAX_STATS_BWD,
  LIBXS_DNN_MAX_STATS_UPD,
  /** pooling mask */
  LIBXS_DNN_POOLING_MASK,
  /** ReLU mask */
  LIBXS_DNN_RELU_MASK,
  /** general type, if needed might cause API issues in copy in/out API */
  LIBXS_DNN_TENSOR,

  /** regular input buffer */
  LIBXS_DNN_RNN_REGULAR_INPUT,
  /** regular previous cell state buffer */
  LIBXS_DNN_RNN_REGULAR_CS_PREV,
  /** regular previous hidden state buffer */
  LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE_PREV,
  /** regular weight (LSTM: wi, wc, wf, wo) */
  LIBXS_DNN_RNN_REGULAR_WEIGHT,
  /** regular recurrent weight (LSTM: ri, rc, rf, ro) */
  LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT,
  /** regular weight (LSTM: wi, wc, wf, wo) */
  LIBXS_DNN_RNN_REGULAR_WEIGHT_TRANS,
  /** regular recurrent weight (LSTM: ri, rc, rf, ro) */
  LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS,
  /** regular bias (LSTM: bi, bc, bf, bo) */
  LIBXS_DNN_RNN_REGULAR_BIAS,
  /** regular output cell state buffer */
  LIBXS_DNN_RNN_REGULAR_CS,
  /** regular hidden state buffer */
  LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE,
  /** gradient input buffer */
  LIBXS_DNN_RNN_GRADIENT_INPUT,
  /** gradient previous cell state buffer */
  LIBXS_DNN_RNN_GRADIENT_CS_PREV,
  /** gradient previous hidden state buffer */
  LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV,
  /** gradient weight */
  LIBXS_DNN_RNN_GRADIENT_WEIGHT,
  /** gradient recurrent weight */
  LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT,
  /** gradient bias */
  LIBXS_DNN_RNN_GRADIENT_BIAS,
  /** gradient output cell state buffer */
  LIBXS_DNN_RNN_GRADIENT_CS,
  /** gradient hidden state buffer */
  LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE,
  /** internal i buffer */
  LIBXS_DNN_RNN_INTERNAL_I,
  /** internal f buffer */
  LIBXS_DNN_RNN_INTERNAL_F,
  /** internal o buffer */
  LIBXS_DNN_RNN_INTERNAL_O,
  /** internal ci buffer */
  LIBXS_DNN_RNN_INTERNAL_CI,
  /** internal co buffer */
  LIBXS_DNN_RNN_INTERNAL_CO
} libxs_dnn_tensor_type;

/** layout descriptor to allow external data handling
    outside of LIBXS */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_tensor_datalayout {
  libxs_dnn_tensor_dimtype* dim_type;
  unsigned int* dim_size;
  unsigned int num_dims;
  libxs_dnn_tensor_format format;                /* format of activation buffer */
  libxs_dnn_datatype datatype;                   /* data type */
  libxs_dnn_tensor_type tensor_type;             /* tensor type */
} libxs_dnn_tensor_datalayout;

/** tensorlayout handling */
LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_duplicate_tensor_datalayout(const libxs_dnn_tensor_datalayout* layout, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_tensor_datalayout(libxs_dnn_tensor_datalayout* layout);
LIBXS_API unsigned int libxs_dnn_compare_tensor_datalayout(const libxs_dnn_tensor_datalayout* layout_a, const libxs_dnn_tensor_datalayout* layout_b, libxs_dnn_err_t* status);
LIBXS_API unsigned int libxs_dnn_get_tensor_size(const libxs_dnn_tensor_datalayout* layout, libxs_dnn_err_t* status);
LIBXS_API unsigned int libxs_dnn_get_tensor_elements(const libxs_dnn_tensor_datalayout* layout, libxs_dnn_err_t* status);

/** Create and manage buffers, filters and bias (non-NULL if successful) */
LIBXS_API libxs_dnn_tensor* libxs_dnn_link_tensor(const libxs_dnn_tensor_datalayout* layout, const void* data, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_tensor* libxs_dnn_link_qtensor(const libxs_dnn_tensor_datalayout* layout, const void* data, const unsigned char exp, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_set_tensor_data_ptr(libxs_dnn_tensor* tensor, const void* data);
LIBXS_API void* libxs_dnn_get_tensor_data_ptr(const libxs_dnn_tensor* tensor, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_get_tensor_datalayout(const libxs_dnn_tensor* tensor, libxs_dnn_err_t* status);
LIBXS_API unsigned char libxs_dnn_get_qtensor_scf(const libxs_dnn_tensor* tensor, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_set_qtensor_scf(libxs_dnn_tensor* tensor, const unsigned char scf);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_tensor(const libxs_dnn_tensor* tensor);
LIBXS_API libxs_dnn_err_t libxs_dnn_zero_tensor(const libxs_dnn_tensor* tensor);

/**
 * Copy-in/out from a plain format such [N][C][H][W] or [N][H][W][C]
 */
LIBXS_API libxs_dnn_err_t libxs_dnn_copyin_tensor(const libxs_dnn_tensor* tensor, const void* data, const libxs_dnn_tensor_format in_format);
LIBXS_API libxs_dnn_err_t libxs_dnn_copyout_tensor(const libxs_dnn_tensor* tensor, void* data, const libxs_dnn_tensor_format out_format);

#endif /*LIBXS_DNN_TENSOR_H*/
