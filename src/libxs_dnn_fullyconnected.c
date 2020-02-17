/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_fullyconnected_backward_weight_update.h"
#include "libxs_dnn_fullyconnected_forward.h"
#include "libxs_main.h"

#define STRIDE_BRGEMM


LIBXS_API libxs_dnn_fullyconnected* libxs_dnn_create_fullyconnected(libxs_dnn_fullyconnected_desc fullyconnected_desc, libxs_dnn_err_t* status) {
  libxs_dnn_fullyconnected* handle = 0;

  if ( ((fullyconnected_desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (fullyconnected_desc.datatype_out == LIBXS_DNN_DATATYPE_BF16)) ||
       ((fullyconnected_desc.datatype_in == LIBXS_DNN_DATATYPE_F32)  && (fullyconnected_desc.datatype_out == LIBXS_DNN_DATATYPE_F32))  ||
       ((fullyconnected_desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (fullyconnected_desc.datatype_out == LIBXS_DNN_DATATYPE_F32))     ) {
    handle = (libxs_dnn_fullyconnected*)malloc(sizeof(libxs_dnn_fullyconnected));

    if (0 != handle) {
      *status = LIBXS_DNN_SUCCESS;
      /* zero entire content; not only safer but also sets data and code pointers to NULL */
      memset(handle, 0, sizeof(*handle));
      /* let's make the description persistent */
      handle->desc = fullyconnected_desc;
      /* @TODO perhaps we need a better switch here */
      if ( (handle->desc.buffer_format == LIBXS_DNN_TENSOR_FORMAT_NCPACKED) && (handle->desc.filter_format == LIBXS_DNN_TENSOR_FORMAT_CKPACKED) ) {
        handle->bk = handle->desc.bk;
        handle->bn = handle->desc.bn;
        handle->bc = handle->desc.bc;

        if ( handle->desc.N % handle->bn != 0 ) {
          handle->bn = handle->desc.N;
          *status = LIBXS_DNN_WARN_FC_SUBOPTIMAL_N_BLOCKING;
        }
        if ( handle->desc.C % handle->bc != 0 ) {
          handle->bc = handle->desc.C;
          *status = LIBXS_DNN_WARN_FC_SUBOPTIMAL_C_BLOCKING;
        }
        if ( handle->desc.K % handle->bk != 0 ) {
          handle->bk = handle->desc.K;
          *status = LIBXS_DNN_WARN_FC_SUBOPTIMAL_K_BLOCKING;
        }
        if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16) )  {
#if 0
          handle->fwd_bf = atoi(getenv("FWD_BF"));
          handle->bwd_bf = atoi(getenv("BWD_BF"));
          handle->upd_bf = atoi(getenv("UPD_BF"));
          handle->fwd_2d_blocking = atoi(getenv("FWD_2D_BLOCKING"));
          handle->bwd_2d_blocking = atoi(getenv("BWD_2D_BLOCKING"));
          handle->upd_2d_blocking = atoi(getenv("UPD_2D_BLOCKING"));
          handle->fwd_row_teams = atoi(getenv("FWD_ROW_TEAMS"));
          handle->fwd_column_teams = atoi(getenv("FWD_COLUMN_TEAMS"));
          handle->bwd_row_teams = atoi(getenv("BWD_ROW_TEAMS"));
          handle->bwd_column_teams = atoi(getenv("BWD_COLUMN_TEAMS"));
          handle->upd_row_teams = atoi(getenv("UPD_ROW_TEAMS"));
          handle->upd_column_teams = atoi(getenv("UPD_COLUMN_TEAMS"));
          handle->ifm_subtasks = atoi(getenv("IFM_SUBTASKS"));
          handle->ofm_subtasks = atoi(getenv("OFM_SUBTASKS"));
#endif
          /* Initialize with default values */
          handle->fwd_bf = 1;
          handle->bwd_bf = 1;
          handle->upd_bf = 1;
          handle->fwd_2d_blocking = 0;
          handle->bwd_2d_blocking = 0;
          handle->upd_2d_blocking = 0;
          handle->fwd_row_teams = 1;
          handle->fwd_column_teams = 1;
          handle->bwd_row_teams = 1;
          handle->bwd_column_teams = 1;
          handle->upd_row_teams = 1;
          handle->upd_column_teams = 1;
          handle->ifm_subtasks = 1;
          handle->ofm_subtasks = 1;

          if (handle->desc.C == 100 && handle->desc.K == 1024 && handle->desc.threads == 28) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 1;
            handle->fwd_row_teams = 14;
            handle->fwd_column_teams = 2;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 1 == 0) ? 1 : 1;
            handle->bwd_2d_blocking = 0;
            handle->bwd_row_teams = 1;
            handle->bwd_column_teams = 1;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 14 == 0) ? 14 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 1;
            handle->upd_column_teams = 1;
            handle->ifm_subtasks = ((handle->bc % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }

          if (handle->desc.C == 1024 && handle->desc.K == 1024 && handle->desc.threads == 28) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 1;
            handle->fwd_row_teams = 7;
            handle->fwd_column_teams = 4;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 8 == 0) ? 8 : 1;
            handle->bwd_2d_blocking = 0;
            handle->bwd_row_teams = 7;
            handle->bwd_column_teams = 4;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 14 == 0) ? 14 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 7;
            handle->upd_column_teams = 4;
            handle->ifm_subtasks = ((handle->bc % 2 == 0) && (handle->upd_2d_blocking == 0)) ? 2 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }

          if (handle->desc.C == 512 && handle->desc.K == 512 && handle->desc.threads == 28) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 0;
            handle->fwd_row_teams = 1;
            handle->fwd_column_teams = 1;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 4 == 0) ? 4 : 1;
            handle->bwd_2d_blocking = 0;
            handle->bwd_row_teams = 1;
            handle->bwd_column_teams = 1;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 14 == 0) ? 14 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 1;
            handle->upd_column_teams = 1;
            handle->ifm_subtasks = ((handle->bc % 2 == 0) && (handle->upd_2d_blocking == 0)) ? 2 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }

          if (handle->desc.C == 1024 && handle->desc.K == 1 && handle->desc.threads == 28) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 0;
            handle->fwd_row_teams = 1;
            handle->fwd_column_teams = 1;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 1 == 0) ? 1 : 1;
            handle->bwd_2d_blocking = 1;
            handle->bwd_row_teams = 14;
            handle->bwd_column_teams = 2;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 2 == 0) ? 2 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 1;
            handle->upd_column_teams = 1;
            handle->ifm_subtasks = ((handle->bc % 2 == 0) && (handle->upd_2d_blocking == 0)) ? 2 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }

          if (handle->desc.C == 1024 && handle->desc.K == 1024 && handle->desc.threads == 20) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 0;
            handle->fwd_row_teams = 5;
            handle->fwd_column_teams = 4;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 1 == 0) ? 1 : 1;
            handle->bwd_2d_blocking = 1;
            handle->bwd_row_teams = 5;
            handle->bwd_column_teams = 4;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 15 == 0) ? 15 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 5;
            handle->upd_column_teams = 4;
            handle->ifm_subtasks = ((handle->bc % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }

          if (handle->desc.C == 100 && handle->desc.K == 1024 && handle->desc.threads == 20) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 1;
            handle->fwd_row_teams = 5;
            handle->fwd_column_teams = 4;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 1 == 0) ? 1 : 1;
            handle->bwd_2d_blocking = 0;
            handle->bwd_row_teams = 1;
            handle->bwd_column_teams = 1;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 9 == 0) ? 9 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 1;
            handle->upd_column_teams = 1;
            handle->ifm_subtasks = ((handle->bc % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
            handle->ofm_subtasks = ((handle->bk % 2 == 0) && (handle->upd_2d_blocking == 0)) ? 2 : 1;
          }

          if (handle->desc.C == 1024 && handle->desc.K == 1024 && handle->desc.threads == 24) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 0;
            handle->fwd_row_teams = 6;
            handle->fwd_column_teams = 4;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 1 == 0) ? 1 : 1;
            handle->bwd_2d_blocking = 0;
            handle->bwd_row_teams = 6;
            handle->bwd_column_teams = 4;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 15 == 0) ? 15 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 6;
            handle->upd_column_teams = 4;
            handle->ifm_subtasks = ((handle->bc % 2 == 0) && (handle->upd_2d_blocking == 0)) ? 2 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }
          if (handle->desc.C == 100 && handle->desc.K == 1024 && handle->desc.threads == 24) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 0;
            handle->fwd_row_teams = 5;
            handle->fwd_column_teams = 4;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 1 == 0) ? 1 : 1;
            handle->bwd_2d_blocking = 1;
            handle->bwd_row_teams = 12;
            handle->bwd_column_teams = 2;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 15 == 0) ? 15 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 5;
            handle->upd_column_teams = 4;
            handle->ifm_subtasks = ((handle->bc % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }
          if (handle->desc.C == 512 && handle->desc.K == 512 && handle->desc.threads == 24) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 0;
            handle->fwd_row_teams = 5;
            handle->fwd_column_teams = 4;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 4 == 0) ? 4 : 1;
            handle->bwd_2d_blocking = 0;
            handle->bwd_row_teams = 5;
            handle->bwd_column_teams = 4;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 15 == 0) ? 15 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 5;
            handle->upd_column_teams = 4;
            handle->ifm_subtasks = ((handle->bc % 2 == 0) && (handle->upd_2d_blocking == 0)) ? 2 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }
          if (handle->desc.C == 512 && handle->desc.K == 512 && handle->desc.threads == 20) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 1;
            handle->fwd_row_teams = 5;
            handle->fwd_column_teams = 4;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 1 == 0) ? 1 : 1;
            handle->bwd_2d_blocking = 0;
            handle->bwd_row_teams = 1;
            handle->bwd_column_teams = 1;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 15 == 0) ? 15 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 1;
            handle->upd_column_teams = 1;
            handle->ifm_subtasks = ((handle->bc % 4 == 0) && (handle->upd_2d_blocking == 0)) ? 4 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }
          if (handle->desc.C == 1024 && handle->desc.K == 1 && handle->desc.threads == 24) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 0;
            handle->fwd_row_teams = 5;
            handle->fwd_column_teams = 4;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 1 == 0) ? 1 : 1;
            handle->bwd_2d_blocking = 0;
            handle->bwd_row_teams = 5;
            handle->bwd_column_teams = 4;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 1 == 0) ? 1 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 5;
            handle->upd_column_teams = 4;
            handle->ifm_subtasks = ((handle->bc % 4 == 0) && (handle->upd_2d_blocking == 0)) ? 4 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }
          if (handle->desc.C == 1024 && handle->desc.K == 1 && handle->desc.threads == 20) {
            handle->fwd_bf = ((handle->desc.C/handle->bc) % 1 == 0) ? 1 : 1;
            handle->fwd_2d_blocking = 0;
            handle->fwd_row_teams = 6;
            handle->fwd_column_teams = 4;
            handle->bwd_bf = ((handle->desc.K/handle->bk) % 1 == 0) ? 1 : 1;
            handle->bwd_2d_blocking = 1;
            handle->bwd_row_teams = 5;
            handle->bwd_column_teams = 4;
            handle->upd_bf = ((handle->desc.N/handle->bn) % 1 == 0) ? 1 : 1;
            handle->upd_2d_blocking = 0;
            handle->upd_row_teams = 6;
            handle->upd_column_teams = 4;
            handle->ifm_subtasks = ((handle->bc % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
            handle->ofm_subtasks = ((handle->bk % 1 == 0) && (handle->upd_2d_blocking == 0)) ? 1 : 1;
          }

        }
      } else {
        /* check that we cannot fuse */
        if ( handle->desc.fuse_ops != LIBXS_DNN_FULLYCONNECTED_FUSE_NONE  ) {
          free( handle );
          *status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
          return 0;
        }

        /* we need to compute the memory layout given the */
        if ( (handle->desc.C % 16 == 0) && (handle->desc.K % 16 == 0) ) {
          if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
            *status = libxs_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.K,
                &(handle->ifmblock), &(handle->ofmblock), &(handle->fm_lp_block),
                LIBXS_DNN_DATATYPE_F32, LIBXS_DNN_DATATYPE_F32 );
          } else if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
            *status = libxs_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.K,
                &(handle->ifmblock), &(handle->ofmblock), &(handle->fm_lp_block),
                handle->desc.datatype_in, handle->desc.datatype_out );
          } else {
            /* should not happen, not implemented */
          }
        } else if ( (handle->desc.C % 64 == 0) && (handle->desc.K == 1000) ) {
          /* @TODO this a hack for the last FC layer */
          handle->ifmblock = 64;
          handle->fm_lp_block = 1;
          handle->ofmblock = 10;
        } else if ( (handle->desc.C % 16 == 0) && (handle->desc.K == 1000) ) {
          /* @TODO this a hack for the last FC layer */
          handle->ifmblock = 16;
          handle->fm_lp_block = 1;
          handle->ofmblock = 10;
        } else {
          *status = LIBXS_DNN_ERR_CREATE_HANDLE;
          free( handle );
          return 0;
        }
        /* compute the outer blocks */
        handle->blocksifm = handle->desc.C / handle->ifmblock;
        handle->blocksofm = handle->desc.K / handle->ofmblock;
      }
      /* create barrier */
      handle->barrier = libxs_barrier_create(handle->desc.threads, 1);
      /* calculate scratch size for batchstats */
      if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
        handle->scratch_size = sizeof(float) * ( ( (size_t)handle->desc.C * (size_t)handle->desc.N ) + ( (size_t)handle->desc.C * (size_t)handle->desc.K ) );
      } else if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16)  ) {
        /* Let's allocate maximum required scratch  */
        size_t size_fwd = sizeof(float) * handle->desc.K * handle->desc.N;
        /* In case of K = 1 we pad A and B to "bk=2" */
        size_t size_bwd = (handle->desc.K != 1) ? ( sizeof(float) * handle->desc.C * handle->desc.N + sizeof(libxs_bfloat16) * handle->desc.C * handle->desc.K ) : ( sizeof(float) * handle->desc.C * handle->desc.N + sizeof(libxs_bfloat16) * handle->desc.C * 2 + sizeof(libxs_bfloat16) * 2 * handle->desc.N );
        size_t size_upd = sizeof(float) * handle->desc.C * handle->desc.K + sizeof(libxs_bfloat16) * handle->desc.threads * handle->bk * handle->bc + sizeof(libxs_bfloat16) * (handle->desc.N * (handle->desc.C + handle->desc.K));
        handle->scratch_size = LIBXS_MAX(LIBXS_MAX(size_fwd, size_bwd), size_upd);
      } else {
        handle->scratch_size = sizeof(float) * LIBXS_MAX( ((size_t)handle->desc.C + (size_t)handle->desc.K) * (size_t)handle->desc.N,
            (size_t)handle->desc.C * (size_t)handle->desc.K);
      }
      /* create code pointers in some special cases */
      if ( ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NCPACKED) > 0) && ((handle->desc.filter_format & LIBXS_DNN_TENSOR_FORMAT_CKPACKED) > 0)  ) {
        if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
          float alpha = 1.0f;
          /* beta is set to 1 for ncnc kcck format because ifm is split into 2 blocks */
          float beta  = 1.0f;
          float zerobeta  = 0.0f;
          libxs_blasint lda = (libxs_blasint)handle->bk;
          libxs_blasint ldb = (libxs_blasint)handle->bc;
          libxs_blasint ldc = (libxs_blasint)handle->bk;

#ifdef ADDRESS_BRGEMM
          handle->gemm_fwd.xgemm.smra = libxs_smmdispatch_reducebatch_addr(handle->bk, handle->bn, handle->bc, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
          handle->gemm_bwd.xgemm.smra = libxs_smmdispatch_reducebatch_addr(handle->bc, handle->bn, handle->bk, &ldb, &lda, &ldb, &alpha, &beta, NULL, NULL);
#endif
#ifdef OFFSET_BRGEMM
          handle->gemm_fwd.xgemm.smro = libxs_smmdispatch_reducebatch_offs(handle->bk, handle->bn, handle->bc, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
          handle->gemm_bwd.xgemm.smro = libxs_smmdispatch_reducebatch_offs(handle->bc, handle->bn, handle->bk, &ldb, &lda, &ldb, &alpha, &beta, NULL, NULL);
#endif
#ifdef STRIDE_BRGEMM
          handle->gemm_fwd.xgemm.smrs = libxs_smmdispatch_reducebatch_strd(handle->bk, handle->bn, handle->bc, handle->bk*handle->bc*sizeof(float), handle->bc*handle->bn*sizeof(float), &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
          handle->gemm_bwd.xgemm.smrs = libxs_smmdispatch_reducebatch_strd(handle->bc, handle->bn, handle->bk, handle->bk*handle->bc*sizeof(float), handle->bk*handle->bn*sizeof(float), &ldb, &lda, &ldb, &alpha, &beta, NULL, NULL);
          handle->gemm_bwd2.xgemm.smrs = libxs_smmdispatch_reducebatch_strd(handle->bc, handle->bn, handle->bk, handle->bk*handle->bc*sizeof(float), handle->bk*handle->bn*sizeof(float), &ldb, &lda, &ldb, &alpha, &zerobeta, NULL, NULL);
#endif
        } else if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16) ) {
          float alpha = 1.0f;
          float beta  = 1.0f;
          float zerobeta  = 0.0f;
          /* For UPD kernels we consider subtasking... */
          libxs_blasint M = handle->bk/handle->ofm_subtasks;
          libxs_blasint N = handle->bc/handle->ifm_subtasks;

          libxs_blasint lda = (libxs_blasint)handle->bk;
          libxs_blasint ldb = (libxs_blasint)handle->bc;
          libxs_blasint ldc = (libxs_blasint)handle->bk;

#ifdef ADDRESS_BRGEMM
          handle->gemm_fwd.xgemm.bmra = libxs_bmmdispatch_reducebatch_addr(handle->bk, handle->bn, handle->bc, &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
#endif
#ifdef OFFSET_BRGEMM
          handle->gemm_fwd.xgemm.bmro = libxs_bmmdispatch_reducebatch_offs(handle->bk, handle->bn, handle->bc, &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
#endif
#ifdef STRIDE_BRGEMM
          handle->gemm_fwd.xgemm.bsmrs = libxs_bsmmdispatch_reducebatch_strd(handle->bk, handle->bn, handle->bc, handle->bk*handle->bc*sizeof(libxs_bfloat16), handle->bc*handle->bn*sizeof(libxs_bfloat16), &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
          handle->gemm_fwd2.xgemm.bmrs = libxs_bmmdispatch_reducebatch_strd(handle->bk, handle->bn, handle->bc, handle->bk*handle->bc*sizeof(libxs_bfloat16), handle->bc*handle->bn*sizeof(libxs_bfloat16), &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
          /* Special bwd kernels for K == 1 */
          if (handle->desc.K == 1) {
            libxs_blasint _bk = 2;
            handle->gemm_bwd.xgemm.bsmrs = libxs_bsmmdispatch_reducebatch_strd(handle->bc, handle->bn, _bk, _bk*handle->bc*sizeof(libxs_bfloat16), _bk*handle->bn*sizeof(libxs_bfloat16), &ldb, &_bk, &ldb, &alpha, &beta, NULL, NULL);
            handle->gemm_bwd2.xgemm.bmrs = libxs_bmmdispatch_reducebatch_strd(handle->bc, handle->bn, _bk, _bk*handle->bc*sizeof(libxs_bfloat16), _bk*handle->bn*sizeof(libxs_bfloat16), &ldb, &_bk, &ldb, &alpha, &zerobeta, NULL, NULL);
          } else {
            handle->gemm_bwd.xgemm.bsmrs = libxs_bsmmdispatch_reducebatch_strd(handle->bc, handle->bn, handle->bk, handle->bk*handle->bc*sizeof(libxs_bfloat16), handle->bk*handle->bn*sizeof(libxs_bfloat16), &ldb, &lda, &ldb, &alpha, &beta, NULL, NULL);
            handle->gemm_bwd2.xgemm.bmrs = libxs_bmmdispatch_reducebatch_strd(handle->bc, handle->bn, handle->bk, handle->bk*handle->bc*sizeof(libxs_bfloat16), handle->bk*handle->bn*sizeof(libxs_bfloat16), &ldb, &lda, &ldb, &alpha, &zerobeta, NULL, NULL);
          }
          lda = (libxs_blasint)handle->bk;
          ldb = (libxs_blasint)handle->bn;
          ldc = (libxs_blasint)handle->bk;
          handle->gemm_upd.xgemm.bsmrs = libxs_bsmmdispatch_reducebatch_strd(M, N, handle->bn, handle->bk*handle->bn*sizeof(libxs_bfloat16), handle->bc*handle->bn*sizeof(libxs_bfloat16), &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
          handle->gemm_upd2.xgemm.bmrs = libxs_bmmdispatch_reducebatch_strd(M, N, handle->bn, handle->bk*handle->bn*sizeof(libxs_bfloat16), handle->bc*handle->bn*sizeof(libxs_bfloat16), &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
#endif
        } else {

        }
      }
    } else {
      *status = LIBXS_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return handle;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_fullyconnected(const libxs_dnn_fullyconnected* handle) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxs_barrier_release((const libxs_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_fullyconnected*)handle);
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_fullyconnected_create_tensor_datalayout(const libxs_dnn_fullyconnected* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor_datalayout* layout;

  *status = LIBXS_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxs_dnn_tensor_datalayout*) malloc(sizeof(libxs_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxs_dnn_tensor_datalayout));

      if ( (type == LIBXS_DNN_REGULAR_INPUT)     || (type == LIBXS_DNN_GRADIENT_INPUT)  || (type == LIBXS_DNN_INPUT)  ||
           (type == LIBXS_DNN_REGULAR_OUTPUT)    || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT)    ) {
        layout->format = handle->desc.buffer_format;
        if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(5*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXS_DNN_REGULAR_INPUT)     || (type == LIBXS_DNN_GRADIENT_INPUT)     || (type == LIBXS_DNN_INPUT)  ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else { /* coverity[dead_error_begin] */
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
            if ( (type == LIBXS_DNN_REGULAR_INPUT) || (type == LIBXS_DNN_GRADIENT_INPUT) || (type == LIBXS_DNN_INPUT) ) {
              layout->datatype = handle->desc.datatype_in;
              layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(5*sizeof(libxs_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
              if (0 != layout->dim_type && 0 != layout->dim_size) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
              }
            } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
              layout->datatype = handle->desc.datatype_out;
              layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(5*sizeof(libxs_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
              if (0 != layout->dim_type && 0 != layout->dim_size) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NHWC) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32)) ||
              ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32)) ||
              ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16))    ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXS_DNN_REGULAR_INPUT)     || (type == LIBXS_DNN_GRADIENT_INPUT)     || (type == LIBXS_DNN_INPUT)  )   {
                layout->dim_size[0] = handle->desc.C;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->desc.N;
              } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) )   {
                layout->dim_size[0] = handle->desc.K;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NCPACKED) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32)  && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32)) ||
              ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16))    ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXS_DNN_REGULAR_INPUT) || (type == LIBXS_DNN_GRADIENT_INPUT) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bc;
                layout->dim_size[1] = (unsigned int)handle->bn;
                layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
              } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bn;
                layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXS_DNN_REGULAR_FILTER)  || (type == LIBXS_DNN_GRADIENT_FILTER)  || (type == LIBXS_DNN_FILTER)  ) {
        layout->format = handle->desc.filter_format;
        layout->tensor_type = LIBXS_DNN_FILTER;

        if ((handle->desc.filter_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(6*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 6;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->ifmblock;
              layout->dim_size[2] = 1;
              layout->dim_size[3] = 1;
              layout->dim_size[4] = handle->blocksifm;
              layout->dim_size[5] = handle->blocksofm;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else if ( ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16) ) ||
              ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) )     ) {
            layout->datatype = LIBXS_DNN_DATATYPE_BF16;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(7*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 7;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[6] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->ifmblock/handle->fm_lp_block;
              layout->dim_size[3] = 1;
              layout->dim_size[4] = 1;
              layout->dim_size[5] = handle->blocksifm;
              layout->dim_size[6] = handle->blocksofm;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.filter_format & LIBXS_DNN_TENSOR_FORMAT_RSCK) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32))   ||
              ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32))  ||
              ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16))    ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_R;
              layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
              layout->dim_size[1] = handle->ifmblock * handle->blocksifm;
              layout->dim_size[2] = 1;
              layout->dim_size[3] = 1;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.filter_format & LIBXS_DNN_TENSOR_FORMAT_CKPACKED) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXS_DNN_REGULAR_FILTER) || (type == LIBXS_DNN_GRADIENT_FILTER) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bc;
                layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          } else if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_BF16;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(5*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 5;

              if ( (type == LIBXS_DNN_REGULAR_FILTER) || (type == LIBXS_DNN_GRADIENT_FILTER) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)2;
                layout->dim_size[1] = (unsigned int)handle->bk;
                layout->dim_size[2] = (unsigned int)handle->bc/2;
                layout->dim_size[3] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[4] = (unsigned int)(handle->desc.K / handle->bk);
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXS_DNN_REGULAR_CHANNEL_BIAS) || (type == LIBXS_DNN_GRADIENT_CHANNEL_BIAS) || (type == LIBXS_DNN_CHANNEL_BIAS) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXS_DNN_CHANNEL_SCALAR;

        if ( ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) || ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NCPACKED) > 0) ) {
          if ( (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) || (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(2*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->blocksofm;
            } else {
              free(layout->dim_type);
              free(layout->dim_size);
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
        }
      } else if ( (type == LIBXS_DNN_RELU_MASK) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXS_DNN_RELU_MASK;

        if ( ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) || ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NCPACKED) > 0) ) {
          layout->datatype = LIBXS_DNN_DATATYPE_I8;
          layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(1*sizeof(libxs_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(1*sizeof(unsigned int));

          if (0 != layout->dim_type && 0 != layout->dim_size) {
            layout->num_dims = 1;
            layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_X;
            layout->dim_size[0] = handle->desc.N * handle->desc.K;
          } else {
            free(layout->dim_type);
            free(layout->dim_size);
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
        }
      } else {
        free(layout);
        layout = 0; /* make sure a NULL is returned */
        *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
      }
    } else {
      *status = LIBXS_DNN_ERR_CREATE_LAYOUT;
    }
  }
  else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return layout;
}

LIBXS_API size_t libxs_dnn_fullyconnected_get_scratch_size(const libxs_dnn_fullyconnected* handle, libxs_dnn_err_t* status) {
  size_t l_scratch_size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    l_scratch_size = handle->scratch_size + 64; /* 64 byte extra in case the user code does not care about alignment */
  } else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXS_API void* libxs_dnn_fullyconnected_get_scratch_ptr(const libxs_dnn_fullyconnected* handle, libxs_dnn_err_t* status)
{
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    return handle->scratch;
  } else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return 0;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_bind_scratch(libxs_dnn_fullyconnected* handle, const void* scratch) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)scratch;
  size_t offset = 0;

  if (scratch == 0) {
    status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    /* align the internal scratch buffer if needed */
    if (address % 64 == 0) {
      handle->scratch = (void*)address;
    } else {
      offset = (64 - address % 64);
      handle->scratch = (void*)(address+offset);
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_release_scratch(libxs_dnn_fullyconnected* handle) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    handle->scratch = 0;
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_bind_tensor(libxs_dnn_fullyconnected* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_REGULAR_INPUT)        && (type != LIBXS_DNN_GRADIENT_INPUT)        &&
       (type != LIBXS_DNN_REGULAR_OUTPUT)       && (type != LIBXS_DNN_GRADIENT_OUTPUT)       &&
       (type != LIBXS_DNN_REGULAR_FILTER)       && (type != LIBXS_DNN_GRADIENT_FILTER)       &&
       (type != LIBXS_DNN_REGULAR_CHANNEL_BIAS) && (type != LIBXS_DNN_GRADIENT_CHANNEL_BIAS) &&
       (type != LIBXS_DNN_RELU_MASK)  ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxs_dnn_tensor_datalayout* handle_layout = libxs_dnn_fullyconnected_create_tensor_datalayout(handle, type, &status);

    if ( libxs_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXS_DNN_REGULAR_INPUT ) {
        handle->reg_input = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_INPUT ) {
        handle->grad_input = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_OUTPUT ) {
        handle->reg_output = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_OUTPUT ) {
        handle->grad_output = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_FILTER ) {
        handle->reg_filter = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_FILTER ) {
        handle->grad_filter = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_CHANNEL_BIAS ) {
        handle->reg_bias = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_CHANNEL_BIAS ) {
        handle->grad_bias = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RELU_MASK ) {
        handle->relumask = (libxs_dnn_tensor*)tensor;
      } else {
        /* cannot happen */
      }
    } else {
      status = LIBXS_DNN_ERR_MISMATCH_TENSOR;
    }

    libxs_dnn_destroy_tensor_datalayout( handle_layout );
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXS_API libxs_dnn_tensor* libxs_dnn_fullyconnected_get_tensor(libxs_dnn_fullyconnected* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor* return_tensor = 0;

  *status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_REGULAR_INPUT)        && (type != LIBXS_DNN_GRADIENT_INPUT)        &&
       (type != LIBXS_DNN_REGULAR_OUTPUT)       && (type != LIBXS_DNN_GRADIENT_OUTPUT)       &&
       (type != LIBXS_DNN_REGULAR_FILTER)       && (type != LIBXS_DNN_GRADIENT_FILTER)       &&
       (type != LIBXS_DNN_REGULAR_CHANNEL_BIAS) && (type != LIBXS_DNN_GRADIENT_CHANNEL_BIAS) &&
       (type != LIBXS_DNN_RELU_MASK)  ) {
    *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return return_tensor;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_REGULAR_INPUT ) {
      return_tensor = handle->reg_input;
    } else if ( type == LIBXS_DNN_GRADIENT_INPUT ) {
      return_tensor = handle->grad_input;
    } else if ( type == LIBXS_DNN_REGULAR_OUTPUT ) {
      return_tensor = handle->reg_output;
    } else if ( type == LIBXS_DNN_GRADIENT_OUTPUT ) {
      return_tensor = handle->grad_output;
    } else if ( type == LIBXS_DNN_REGULAR_FILTER ) {
      return_tensor = handle->reg_filter;
    } else if ( type == LIBXS_DNN_GRADIENT_FILTER ) {
      return_tensor = handle->grad_filter;
    } else if ( type == LIBXS_DNN_REGULAR_CHANNEL_BIAS ) {
      return_tensor = handle->reg_bias;
    } else if ( type == LIBXS_DNN_GRADIENT_CHANNEL_BIAS ) {
      return_tensor = handle->grad_bias;
    } else if ( type == LIBXS_DNN_RELU_MASK ) {
      return_tensor = handle->relumask;
    } else {
      /* cannot happen */
    }
  } else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return return_tensor;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_release_tensor(libxs_dnn_fullyconnected* handle, const libxs_dnn_tensor_type type) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_REGULAR_INPUT)        && (type != LIBXS_DNN_GRADIENT_INPUT)        &&
       (type != LIBXS_DNN_REGULAR_OUTPUT)       && (type != LIBXS_DNN_GRADIENT_OUTPUT)       &&
       (type != LIBXS_DNN_REGULAR_FILTER)       && (type != LIBXS_DNN_GRADIENT_FILTER)       &&
       (type != LIBXS_DNN_REGULAR_CHANNEL_BIAS) && (type != LIBXS_DNN_GRADIENT_CHANNEL_BIAS) &&
       (type != LIBXS_DNN_RELU_MASK)  ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_REGULAR_INPUT ) {
      handle->reg_input = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_INPUT ) {
      handle->grad_input = 0;
    } else if ( type == LIBXS_DNN_REGULAR_OUTPUT ) {
      handle->reg_output = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_OUTPUT ) {
      handle->grad_output = 0;
    } else if ( type == LIBXS_DNN_REGULAR_FILTER ) {
      handle->reg_filter = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_FILTER ) {
      handle->grad_filter = 0;
    } else if ( type == LIBXS_DNN_REGULAR_CHANNEL_BIAS ) {
      handle->reg_bias = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_CHANNEL_BIAS ) {
      handle->grad_bias = 0;
    } else if ( type == LIBXS_DNN_RELU_MASK ) {
      handle->relumask = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_fullyconnected_execute_st(libxs_dnn_fullyconnected* handle, libxs_dnn_compute_kind kind,
    /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  LIBXS_UNUSED( start_thread );
  LIBXS_UNUSED( tid );

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           if ( (handle->desc.buffer_format == LIBXS_DNN_TENSOR_FORMAT_LIBXS) && (handle->desc.filter_format == LIBXS_DNN_TENSOR_FORMAT_LIBXS) ) {
                                             status = libxs_dnn_fullyconnected_st_fwd_custom( handle, start_thread, tid );
                                           } else if ( (handle->desc.buffer_format == LIBXS_DNN_TENSOR_FORMAT_NCPACKED) && (handle->desc.filter_format == LIBXS_DNN_TENSOR_FORMAT_CKPACKED) ) {
                                             status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck( handle, start_thread, tid );
                                           } else {
                                             status = LIBXS_DNN_ERR_INVALID_FORMAT_FC;
                                           }
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_BWDUPD:
                                         {
                                           if ( (handle->desc.buffer_format == LIBXS_DNN_TENSOR_FORMAT_LIBXS) && (handle->desc.filter_format == LIBXS_DNN_TENSOR_FORMAT_LIBXS) ) {
                                             status = libxs_dnn_fullyconnected_st_bwdupd_custom( handle, kind, start_thread, tid );
                                           } else if ( (handle->desc.buffer_format == LIBXS_DNN_TENSOR_FORMAT_NCPACKED) && (handle->desc.filter_format == LIBXS_DNN_TENSOR_FORMAT_CKPACKED) ) {
                                             status = libxs_dnn_fullyconnected_st_bwdupd_ncnc_kcck( handle, kind, start_thread, tid );
                                           } else {
                                             status = LIBXS_DNN_ERR_INVALID_FORMAT_FC;
                                           }
                                         } break;
      default: {
                 status = LIBXS_DNN_ERR_INVALID_KIND;
               }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}

