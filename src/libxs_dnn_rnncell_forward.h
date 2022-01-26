/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_RNNCELL_FORWARD_H
#define LIBXS_DNN_RNNCELL_FORWARD_H

#include <libxs_dnn.h>
#include <libxs_dnn_rnncell.h>

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_ck(libxs_dnn_rnncell* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_ncnc_kcck(libxs_dnn_rnncell* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_kcck(libxs_dnn_rnncell* handle, int start_thread, int tid);

#endif /* LIBXS_DNN_RNNCELL_FORWARD_H */
