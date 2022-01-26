/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_OPTIMIZER_SGD_H
#define LIBXS_DNN_OPTIMIZER_SGD_H

#include <libxs_dnn_optimizer.h>

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_optimizer_sgd_st(libxs_dnn_optimizer* handle, int start_thread, int tid);

#endif /* LIBXS_DNN_OPTIMIZER_SGD_H */
