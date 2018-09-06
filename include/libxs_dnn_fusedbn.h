/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
#ifndef LIBXS_DNN_FUSEDBN_H
#define LIBXS_DNN_FUSEDBN_H

#include "libxs_macros.h"
#include "libxs_typedefs.h"
#include "libxs_dnn.h"

/** Opaque handles which represents LIBXS fusedbn */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_fusedbn libxs_dnn_fusedbn;

typedef enum libxs_dnn_fusedbn_fuse_order {
  /* the fuse order is: 1. BN, 2. eltwise 3. RELU */
  LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU = 0
} libxs_dnn_fusedbn_fuse_order;

typedef enum libxs_dnn_fusedbn_fuse_op {
  /* the fuse order is: 1. BN, 2. eltwise 3. RELU */
  LIBXS_DNN_FUSEDBN_OPS_BN = 1,
  LIBXS_DNN_FUSEDBN_OPS_BNSCALE = 2,
  LIBXS_DNN_FUSEDBN_OPS_ELTWISE = 4,
  LIBXS_DNN_FUSEDBN_OPS_RELU = 8,
  LIBXS_DNN_FUSEDBN_OPS_BN_ELTWISE = LIBXS_DNN_FUSEDBN_OPS_BN | LIBXS_DNN_FUSEDBN_OPS_ELTWISE,
  LIBXS_DNN_FUSEDBN_OPS_BN_RELU = LIBXS_DNN_FUSEDBN_OPS_BN | LIBXS_DNN_FUSEDBN_OPS_RELU,
  LIBXS_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU = LIBXS_DNN_FUSEDBN_OPS_BN | LIBXS_DNN_FUSEDBN_OPS_ELTWISE | LIBXS_DNN_FUSEDBN_OPS_RELU,
  LIBXS_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE = LIBXS_DNN_FUSEDBN_OPS_BNSCALE | LIBXS_DNN_FUSEDBN_OPS_ELTWISE,
  LIBXS_DNN_FUSEDBN_OPS_BNSCALE_RELU = LIBXS_DNN_FUSEDBN_OPS_BNSCALE | LIBXS_DNN_FUSEDBN_OPS_RELU,
  LIBXS_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU = LIBXS_DNN_FUSEDBN_OPS_BNSCALE | LIBXS_DNN_FUSEDBN_OPS_ELTWISE | LIBXS_DNN_FUSEDBN_OPS_RELU
} libxs_dnn_fusedbn_fuse_op;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_fusedbn_desc {
  int N;                                     /* number of images in mini-batch */
  int C;                                     /* number of input feature maps */
  int H;                                     /* height of input image */
  int W;                                     /* width of input image */
  int u;                                     /* vertical stride */
  int v;                                     /* horizontal stride */
  int pad_h_in;                              /* height of physcial zero-padding in input buffer */
  int pad_w_in;                              /* width of physical zero-padding in input buffer */
  int pad_h_out;                             /* height of physical zero-padding in output buffer */
  int pad_w_out;                             /* width of physical zero-padding in output buffer */
  int threads;                               /* number of threads used */
  libxs_dnn_datatype datatype_in;          /* datatype used for all input related buffers */
  libxs_dnn_datatype datatype_out;         /* datatype used for all output related buffers */
  libxs_dnn_datatype datatype_stats;       /* datatype used for all stats related buffers */
  libxs_dnn_tensor_format buffer_format;   /* format which is for activation buffers */
  libxs_dnn_fusedbn_fuse_order fuse_order; /* additional options */
  libxs_dnn_fusedbn_fuse_op fuse_ops;      /* used ops into convolutions */
} libxs_dnn_fusedbn_desc;

LIBXS_API libxs_dnn_fusedbn* libxs_dnn_create_fusedbn(libxs_dnn_fusedbn_desc fusedbn_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_fusedbn(const libxs_dnn_fusedbn* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_fusedbn_create_tensor_datalayout(const libxs_dnn_fusedbn* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API size_t libxs_dnn_fusedbn_get_scratch_size(const libxs_dnn_fusedbn* handle, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbn_bind_scratch(libxs_dnn_fusedbn* handle, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbn_release_scratch(libxs_dnn_fusedbn* handle);

LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbn_bind_tensor(libxs_dnn_fusedbn* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_fusedbn_get_tensor(libxs_dnn_fusedbn* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbn_release_tensor(libxs_dnn_fusedbn* handle, const libxs_dnn_tensor_type type);

LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbn_execute_st(libxs_dnn_fusedbn* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_FUSEDBN_H*/
