/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_CONVOLUTION_BACKWARD_H
#define LIBXS_DNN_CONVOLUTION_BACKWARD_H

#include <libxs_dnn_convolution.h>

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_custom_custom(libxs_dnn_layer* handle, int start_thread, int tid);

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_nhwc_rsck(libxs_dnn_layer* handle, int start_thread, int tid);

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_nhwc_custom(libxs_dnn_layer* handle, int start_thread, int tid);

#endif /* LIBXS_DNN_CONVOLUTION_BACKWARD_H */
