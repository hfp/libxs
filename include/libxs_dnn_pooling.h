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
#ifndef LIBXS_DNN_POOLING_H
#define LIBXS_DNN_POOLING_H

#include "libxs_dnn.h"


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
