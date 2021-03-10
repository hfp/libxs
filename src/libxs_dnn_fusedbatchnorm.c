/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_fusedbatchnorm_backward.h"
#include "libxs_dnn_fusedbatchnorm_forward.h"
#include "libxs_main.h"


LIBXS_API libxs_dnn_fusedbatchnorm* libxs_dnn_create_fusedbatchnorm(libxs_dnn_fusedbatchnorm_desc fusedbatchnorm_desc, libxs_dnn_err_t* status) {
  libxs_dnn_fusedbatchnorm* handle = 0;
  int lpb;

  /* init libxs */
  LIBXS_INIT

  if ( fusedbatchnorm_desc.partN > fusedbatchnorm_desc.fullN ) {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
    return handle;
  } else if ( (fusedbatchnorm_desc.partN != fusedbatchnorm_desc.fullN) && ((fusedbatchnorm_desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_BNSTATS_NORED) == 0 ) && ((fusedbatchnorm_desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_BNSCALE) == 0 ) ) {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
    return handle;
  } else {
  }

  if ( ((fusedbatchnorm_desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (fusedbatchnorm_desc.datatype_out == LIBXS_DNN_DATATYPE_BF16)) ||
       ((fusedbatchnorm_desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (fusedbatchnorm_desc.datatype_out == LIBXS_DNN_DATATYPE_F32))    ) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    handle = (libxs_dnn_fusedbatchnorm*)calloc(sizeof(libxs_dnn_fusedbatchnorm));

    if (0 != handle) {
      *status = LIBXS_DNN_SUCCESS;
      /* let's make the description persistent */
      handle->desc = fusedbatchnorm_desc;
      /* we need to compute the memory layout given the */
      *status = libxs_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.C,
                                                    &(handle->ifmblock), &(handle->ofmblock), &lpb,
                                                    handle->desc.datatype_in, handle->desc.datatype_out );
      /* compute the outer blocks */
      handle->blocksifm = handle->desc.C / handle->ifmblock;
      handle->blocksofm = handle->desc.C / handle->ofmblock;
      /* create barrier */
      handle->barrier = libxs_barrier_create(handle->desc.threads, 1);
      /* calculate scratch size for batchstats */
      handle->scratch_size = (sizeof(float) * 2 * handle->desc.C * handle->desc.partN);
    } else {
      *status = LIBXS_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return handle;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_fusedbatchnorm(const libxs_dnn_fusedbatchnorm* handle) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxs_barrier_release((const libxs_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_fusedbatchnorm*)handle);
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_fusedbatchnorm_create_tensor_datalayout(const libxs_dnn_fusedbatchnorm* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor_datalayout* layout;

  *status = LIBXS_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    layout = (libxs_dnn_tensor_datalayout*) calloc(sizeof(libxs_dnn_tensor_datalayout));

    if (layout != 0) {
      layout->format = handle->desc.buffer_format;

      if ( (type == LIBXS_DNN_REGULAR_INPUT)     || (type == LIBXS_DNN_GRADIENT_INPUT)  || (type == LIBXS_DNN_INPUT)  ||
           (type == LIBXS_DNN_REGULAR_OUTPUT)    || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ||
           (type == LIBXS_DNN_REGULAR_INPUT_ADD) || (type == LIBXS_DNN_GRADIENT_INPUT_ADD)                                  ) {
        if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
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
              if ( (type == LIBXS_DNN_REGULAR_INPUT)     || (type == LIBXS_DNN_GRADIENT_INPUT)     || (type == LIBXS_DNN_INPUT) ||
                   (type == LIBXS_DNN_REGULAR_INPUT_ADD) || (type == LIBXS_DNN_GRADIENT_INPUT_ADD)                                   ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->desc.W + (2*handle->desc.pad_w_in);
                layout->dim_size[2] = handle->desc.H + (2*handle->desc.pad_h_in);
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.partN;
              } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
                layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.partN;
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
          } else if ( (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_BF16;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(5*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXS_DNN_REGULAR_INPUT)     || (type == LIBXS_DNN_GRADIENT_INPUT)     || (type == LIBXS_DNN_INPUT) ||
                   (type == LIBXS_DNN_REGULAR_INPUT_ADD) || (type == LIBXS_DNN_GRADIENT_INPUT_ADD)                                   ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->desc.W + (2*handle->desc.pad_w_in);
                layout->dim_size[2] = handle->desc.H + (2*handle->desc.pad_h_in);
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.partN;
              } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
                layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.partN;
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
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NHWC) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32)) ||
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
              if ( (type == LIBXS_DNN_REGULAR_INPUT)     || (type == LIBXS_DNN_GRADIENT_INPUT)     || (type == LIBXS_DNN_INPUT) ||
                   (type == LIBXS_DNN_REGULAR_INPUT_ADD) || (type == LIBXS_DNN_GRADIENT_INPUT_ADD)                                      )   {
                layout->dim_size[0] = handle->desc.C;
                layout->dim_size[1] = handle->desc.W + (2*handle->desc.pad_w_in);
                layout->dim_size[2] = handle->desc.H + (2*handle->desc.pad_h_in);
                layout->dim_size[3] = handle->desc.partN;
              } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) )   {
                layout->dim_size[0] = handle->desc.C;
                layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
                layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
                layout->dim_size[3] = handle->desc.partN;
              } else { /* coverity[dead_error_begin] */
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
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXS_DNN_REGULAR_CHANNEL_BETA)  || (type == LIBXS_DNN_GRADIENT_CHANNEL_BETA)  || (type == LIBXS_DNN_CHANNEL_BETA)     ||
                  (type == LIBXS_DNN_REGULAR_CHANNEL_GAMMA) || (type == LIBXS_DNN_GRADIENT_CHANNEL_GAMMA) || (type == LIBXS_DNN_CHANNEL_GAMMA)    ||
                  (type == LIBXS_DNN_CHANNEL_EXPECTVAL)     || (type == LIBXS_DNN_CHANNEL_RCPSTDDEV)      || (type == LIBXS_DNN_CHANNEL_VARIANCE)    ) {
        layout->tensor_type = LIBXS_DNN_CHANNEL_SCALAR;

        if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( handle->desc.datatype_stats == LIBXS_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->desc.datatype_stats;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(2*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ifmblock;
              layout->dim_size[1] = handle->blocksifm;
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
        } else if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NHWC) > 0) {
          if ( handle->desc.datatype_stats == LIBXS_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->desc.datatype_stats;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(1*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(1*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 1;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->desc.C;
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
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXS_DNN_RELU_MASK) ) {
        layout->tensor_type = LIBXS_DNN_RELU_MASK;

        if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          layout->datatype = LIBXS_DNN_DATATYPE_I8;
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
            layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
            layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
            layout->dim_size[3] = handle->blocksofm;
            layout->dim_size[4] = handle->desc.partN;
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
          }
        } else if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NHWC) > 0) {
          layout->datatype = LIBXS_DNN_DATATYPE_I8;
          layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

          if (0 != layout->dim_type && 0 != layout->dim_size) {
            layout->num_dims = 6;
            layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_W;
            layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_H;
            layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_N;
            layout->dim_size[0] = handle->ofmblock*handle->blocksofm;
            layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
            layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
            layout->dim_size[3] = handle->desc.partN;
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
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

LIBXS_API size_t libxs_dnn_fusedbatchnorm_get_scratch_size(const libxs_dnn_fusedbatchnorm* handle, libxs_dnn_err_t* status) {
  size_t l_scratch_size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    l_scratch_size = handle->scratch_size + 64; /* 64 byte extra in case the user code does not care about alignment */
  } else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_bind_scratch(libxs_dnn_fusedbatchnorm* handle, const void* scratch) {
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


LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_release_scratch(libxs_dnn_fusedbatchnorm* handle) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    handle->scratch = 0;
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_bind_tensor(libxs_dnn_fusedbatchnorm* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_REGULAR_INPUT)         && (type != LIBXS_DNN_GRADIENT_INPUT)         &&
       (type != LIBXS_DNN_REGULAR_OUTPUT)        && (type != LIBXS_DNN_GRADIENT_OUTPUT)        &&
       (type != LIBXS_DNN_REGULAR_INPUT_ADD)     && (type != LIBXS_DNN_GRADIENT_INPUT_ADD)     &&
       (type != LIBXS_DNN_REGULAR_CHANNEL_BETA)  && (type != LIBXS_DNN_GRADIENT_CHANNEL_BETA)  &&
       (type != LIBXS_DNN_REGULAR_CHANNEL_GAMMA) && (type != LIBXS_DNN_GRADIENT_CHANNEL_GAMMA) &&
       (type != LIBXS_DNN_CHANNEL_EXPECTVAL)     && (type != LIBXS_DNN_CHANNEL_RCPSTDDEV)      &&
       (type != LIBXS_DNN_CHANNEL_VARIANCE)      && (type != LIBXS_DNN_RELU_MASK)                  ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxs_dnn_tensor_datalayout* handle_layout = libxs_dnn_fusedbatchnorm_create_tensor_datalayout(handle, type, &status);

    if ( libxs_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXS_DNN_REGULAR_INPUT ) {
        handle->reg_input = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_INPUT ) {
        handle->grad_input = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_OUTPUT ) {
        handle->reg_output = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_OUTPUT ) {
        handle->grad_output = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_INPUT_ADD ) {
        handle->reg_add = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_INPUT_ADD ) {
        handle->grad_add = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_CHANNEL_BETA ) {
        handle->reg_beta = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_CHANNEL_BETA ) {
        handle->grad_beta = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_CHANNEL_GAMMA ) {
        handle->reg_gamma = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_CHANNEL_GAMMA ) {
        handle->grad_gamma = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_CHANNEL_EXPECTVAL ) {
        handle->expvalue = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_CHANNEL_RCPSTDDEV ) {
        handle->rcpstddev = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_CHANNEL_VARIANCE ) {
        handle->variance = (libxs_dnn_tensor*)tensor;
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


LIBXS_API libxs_dnn_tensor* libxs_dnn_fusedbatchnorm_get_tensor(libxs_dnn_fusedbatchnorm* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor* return_tensor = 0;

  *status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_REGULAR_INPUT)         && (type != LIBXS_DNN_GRADIENT_INPUT)         &&
       (type != LIBXS_DNN_REGULAR_OUTPUT)        && (type != LIBXS_DNN_GRADIENT_OUTPUT)        &&
       (type != LIBXS_DNN_REGULAR_INPUT_ADD)     && (type != LIBXS_DNN_GRADIENT_INPUT_ADD)     &&
       (type != LIBXS_DNN_REGULAR_CHANNEL_BETA)  && (type != LIBXS_DNN_GRADIENT_CHANNEL_BETA)  &&
       (type != LIBXS_DNN_REGULAR_CHANNEL_GAMMA) && (type != LIBXS_DNN_GRADIENT_CHANNEL_GAMMA) &&
       (type != LIBXS_DNN_CHANNEL_EXPECTVAL)     && (type != LIBXS_DNN_CHANNEL_RCPSTDDEV)      &&
       (type != LIBXS_DNN_CHANNEL_VARIANCE)      && (type != LIBXS_DNN_RELU_MASK)                 ) {
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
    } else if ( type == LIBXS_DNN_REGULAR_INPUT_ADD ) {
      return_tensor = handle->reg_add;
    } else if ( type == LIBXS_DNN_GRADIENT_INPUT_ADD ) {
      return_tensor = handle->grad_add;
    } else if ( type == LIBXS_DNN_REGULAR_CHANNEL_BETA ) {
      return_tensor = handle->reg_beta;
    } else if ( type == LIBXS_DNN_GRADIENT_CHANNEL_BETA ) {
      return_tensor = handle->grad_beta;
    } else if ( type == LIBXS_DNN_REGULAR_CHANNEL_GAMMA ) {
      return_tensor = handle->reg_gamma;
    } else if ( type == LIBXS_DNN_GRADIENT_CHANNEL_GAMMA ) {
      return_tensor = handle->grad_gamma;
    } else if ( type == LIBXS_DNN_CHANNEL_EXPECTVAL ) {
      return_tensor = handle->expvalue;
    } else if ( type == LIBXS_DNN_CHANNEL_RCPSTDDEV ) {
      return_tensor = handle->rcpstddev;
    } else if ( type == LIBXS_DNN_CHANNEL_VARIANCE ) {
      return_tensor = handle->variance;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_release_tensor(libxs_dnn_fusedbatchnorm* handle, const libxs_dnn_tensor_type type) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_REGULAR_INPUT)         && (type != LIBXS_DNN_GRADIENT_INPUT)         &&
       (type != LIBXS_DNN_REGULAR_OUTPUT)        && (type != LIBXS_DNN_GRADIENT_OUTPUT)        &&
       (type != LIBXS_DNN_REGULAR_INPUT_ADD)     && (type != LIBXS_DNN_GRADIENT_INPUT_ADD)     &&
       (type != LIBXS_DNN_REGULAR_CHANNEL_BETA)  && (type != LIBXS_DNN_GRADIENT_CHANNEL_BETA)  &&
       (type != LIBXS_DNN_REGULAR_CHANNEL_GAMMA) && (type != LIBXS_DNN_GRADIENT_CHANNEL_GAMMA) &&
       (type != LIBXS_DNN_CHANNEL_EXPECTVAL)     && (type != LIBXS_DNN_CHANNEL_RCPSTDDEV)      &&
       (type != LIBXS_DNN_CHANNEL_VARIANCE)      && (type != LIBXS_DNN_RELU_MASK)                 ) {
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
    } else if ( type == LIBXS_DNN_REGULAR_INPUT_ADD ) {
      handle->reg_add = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_INPUT_ADD ) {
      handle->grad_add = 0;
    } else if ( type == LIBXS_DNN_REGULAR_CHANNEL_BETA ) {
      handle->reg_beta = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_CHANNEL_BETA ) {
      handle->grad_beta = 0;
    } else if ( type == LIBXS_DNN_REGULAR_CHANNEL_GAMMA ) {
      handle->reg_gamma = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_CHANNEL_GAMMA ) {
      handle->grad_gamma = 0;
    } else if ( type == LIBXS_DNN_CHANNEL_EXPECTVAL ) {
      handle->expvalue = 0;
    } else if ( type == LIBXS_DNN_CHANNEL_RCPSTDDEV ) {
      handle->rcpstddev = 0;
    } else if ( type == LIBXS_DNN_CHANNEL_VARIANCE ) {
      handle->variance = 0;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_execute_st(libxs_dnn_fusedbatchnorm* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        switch (handle->desc.buffer_format) {
          case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
            status = libxs_dnn_fusedbatchnorm_st_fwd_custom( handle, start_thread, tid );
          } break;
          default: {
            status = LIBXS_DNN_ERR_INVALID_FORMAT_FUSEDBN;
          }
        }
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD: {
        switch (handle->desc.buffer_format) {
          case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
            status = libxs_dnn_fusedbatchnorm_st_bwd_custom( handle, start_thread, tid );
          } break;
          default: {
            status = LIBXS_DNN_ERR_INVALID_FORMAT_FUSEDBN;
          }
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


LIBXS_API libxs_dnn_err_t libxs_dnn_fusedbatchnorm_reduce_stats_st(libxs_dnn_fusedbatchnorm** handles, int num_handles, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handles && num_handles > 0) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        switch (handles[0]->desc.buffer_format) {
          case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
            status = libxs_dnn_fusedbatchnorm_reduce_stats_st_fwd_custom( handles, num_handles, start_thread, tid );
          } break;
          default: {
            status = LIBXS_DNN_ERR_INVALID_FORMAT_FUSEDBN;
          }
        }
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD: {
        switch (handles[0]->desc.buffer_format) {
          case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
            status = libxs_dnn_fusedbatchnorm_reduce_stats_st_bwd_custom( handles, num_handles, start_thread, tid );
          } break;
          default: {
            status = LIBXS_DNN_ERR_INVALID_FORMAT_FUSEDBN;
          }
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
