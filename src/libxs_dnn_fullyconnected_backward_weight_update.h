/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_FULLYCONNECTED_BACKWARD_WEIGHT_UPDATE_H
#define LIBXS_DNN_FULLYCONNECTED_BACKWARD_WEIGHT_UPDATE_H

#include <libxs_dnn_fullyconnected.h>

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_bwdupd_custom(libxs_dnn_fullyconnected* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_bwdupd_ncnc_kcck(libxs_dnn_fullyconnected* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_bwdupd_nhwc(libxs_dnn_fullyconnected* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);

#endif /* LIBXS_DNN_FULLYCONNECTED_BACKWARD_WEIGHT_UPDATE_H */
