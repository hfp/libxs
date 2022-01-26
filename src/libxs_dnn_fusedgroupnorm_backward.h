/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_FUSEDGROUPNORM_BACKWARD_H
#define LIBXS_DNN_FUSEDGROUPNORM_BACKWARD_H

#include <libxs_dnn_fusedgroupnorm.h>

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedgroupnorm_st_bwd_custom(libxs_dnn_fusedgroupnorm* handle, int start_thread, int tid);

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedgroupnorm_st_bwd_nhwc(libxs_dnn_fusedgroupnorm* handle, int start_thread, int tid);

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedgroupnorm_reduce_stats_st_bwd_custom(libxs_dnn_fusedgroupnorm** handles, int num_handles, int start_thread, int tid);

#endif /* LIBXS_DNN_FUSEDGROUPNORM_BACKWARD_H */
