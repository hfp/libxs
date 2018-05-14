/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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

#include <libxs.h>
#include "libxs_main.h"

LIBXS_API libxs_dnn_rnncell* libxs_dnn_create_rnncell(libxs_dnn_rnncell_desc rnncell_desc, libxs_dnn_err_t* status) {
  libxs_dnn_rnncell* handle = 0;

  LIBXS_UNUSED( rnncell_desc );
  *status = LIBXS_DNN_SUCCESS;

  return handle;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_rnncell(const libxs_dnn_rnncell* handle) {
  LIBXS_UNUSED( handle );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_rnncell_create_tensor_datalayout(const libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor_datalayout* layout = 0;

  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( type );
  *status = LIBXS_DNN_SUCCESS;

  return layout;
}


LIBXS_API size_t libxs_dnn_rnncell_get_scratch_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status) {
  size_t size = 0;

  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );
  *status = LIBXS_DNN_SUCCESS;

  return size;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* scratch) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );
  LIBXS_UNUSED( scratch );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API size_t libxs_dnn_rnncell_get_internalstate_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status) {
  size_t size = 0;

  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );
  *status = LIBXS_DNN_SUCCESS;

  return size;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* internalstate) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );
  LIBXS_UNUSED( internalstate );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( tensor );
  LIBXS_UNUSED( type );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_tensor* libxs_dnn_rnncell_get_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor* tensor = 0;

  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( type );
  *status = LIBXS_DNN_SUCCESS;

  return tensor;
}

LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( type );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_execute_st(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );
  LIBXS_UNUSED( start_thread );
  LIBXS_UNUSED( tid );

  return LIBXS_DNN_SUCCESS;
}

