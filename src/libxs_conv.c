/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#include "libxs_alloc.h"
#include "libxs_main.h"
#include "libxs_sync.h"
#include "libxs_conv_fwd.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API_DEFINITION const char* libxs_conv_get_error(libxs_conv_err_t code)
{
  switch (code) {
    case LIBXS_CONV_SUCCESS:
      return "LIBXS CONV Success!";
    case LIBXS_CONV_WARN_FALLBACK:
      return "LIBXS CONV Warning: Falling back to naive code as target is currently not supported by LIBXS!";
    case LIBXS_CONV_ERR_GENERAL:
      return "LIBXS CONV Error: General error occured!";
    case LIBXS_CONV_ERR_CREATE_HANDLE:
      return "LIBXS CONV Error: Handle creation failed!";
    case LIBXS_CONV_ERR_UNSUPPORTED_DATATYPE:
      return "LIBXS CONV Error: Requested datatype is not available!";
    case LIBXS_CONV_ERR_INVALID_BLOCKING:
      return "LIBXS CONV Error: Requested Input/Output layer size cannot be blocked!";
    case LIBXS_CONV_ERR_INVALID_HANDLE:
      return "LIBXS CONV Error: An invalid handle was proivded!";
    case LIBXS_CONV_ERR_DATA_NOT_BOUND:
      return "LIBXS CONV Error: Not all required sources and destinations have been bound to convolution!";
    case LIBXS_CONV_ERR_CREATE_LAYER:
      return "LIBXS CONV Error: Layer creation failed!";
    case LIBXS_CONV_ERR_INVALID_LAYER:
      return "LIBXS CONV Error: Invalid layer was specified!";
    case LIBXS_CONV_ERR_CREATE_FILTER:
      return "LIBXS CONV Error: Filter creation failed!";
    case LIBXS_CONV_ERR_INVALID_FILTER:
      return "LIBXS CONV Error: Invalid filter was specified!";
    case LIBXS_CONV_ERR_CREATE_BIAS:
      return "LIBXS CONV Error: Bias creation failed!";
    case LIBXS_CONV_ERR_INVALID_BIAS:
      return "LIBXS CONV Error: Invalid Bias was specified";
    case LIBXS_CONV_ERR_MISMATCH_LAYER:
      return "LIBXS CONV Error: Layer doesn't match handle it should be bind to!";
    case LIBXS_CONV_ERR_INVALID_HANDLE_LAYER:
      return "LIBXS CONV Error: Invalid hanlde or layer!";
    case LIBXS_CONV_ERR_MISMATCH_FILTER:
      return "LIBXS CONV Error: Filter doens't match handle it should be bind to!";
    case LIBXS_CONV_ERR_INVALID_HANDLE_FILTER:
      return "LIBXS CONV Error: Invalid handle or filter!";
    case LIBXS_CONV_ERR_INVALID_KIND:
      return "LIBXS CONV Error: Invalid convolution kind!";
    default:
      return "LIBXS CONV Error: Unknown error or warning occured!";
  }
}


LIBXS_API_DEFINITION libxs_conv_handle* libxs_conv_create_handle(
  libxs_conv_desc     conv_desc,
  libxs_conv_datatype conv_datatype,
  libxs_conv_algo     conv_algo)
{
  libxs_conv_err_t status;
  return libxs_conv_create_handle_check( conv_desc, conv_datatype, conv_algo, &status);
}


LIBXS_API_DEFINITION libxs_conv_handle* libxs_conv_create_handle_check(
  libxs_conv_desc     conv_desc,
  libxs_conv_datatype conv_datatype,
  libxs_conv_algo     conv_algo,
  libxs_conv_err_t*   status)
{
  libxs_conv_handle* handle = (libxs_conv_handle*)malloc(sizeof(libxs_conv_handle));
  int noarch = 1;
  int i = 0;
  *status = LIBXS_CONV_SUCCESS;

  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = conv_desc;
    /* at min. we have 1 split */
    handle->desc.splits = (conv_desc.splits <= 1) ? 1 : conv_desc.splits;
    handle->datatype = conv_datatype;
    handle->algo = conv_algo;
    /* derive additional values */
    handle->ifhp = conv_desc.H;
    handle->ifwp = conv_desc.W;
    handle->ofh = (conv_desc.H - conv_desc.R) / conv_desc.u + 1;
    handle->ofw = (conv_desc.W - conv_desc.S) / conv_desc.v + 1;
    handle->ofhp = handle->ofh + 2*conv_desc.pad_h;
    handle->ofwp = handle->ofw + 2*conv_desc.pad_w;

    /* now architecture specific */
    if (libxs_get_target_archid() == LIBXS_X86_AVX512_MIC  ||
        libxs_get_target_archid() == LIBXS_X86_AVX512_CORE)
    {
      noarch = 0;
#define LIBXS_FWD_OFH_BLOCKING
#if defined(LIBXS_FWD_OFH_BLOCKING)
      if ((handle->ofw < 15) && (handle->ofh % 2 == 0) && (conv_desc.S == 1)) {
        handle->fwd_ofw_rb = handle->ofw;
        handle->fwd_ofh_rb = 2;
      }
      else {
#endif
        for (i = 28; i > 1; --i) {
          if (handle->ofw % i == 0) break;
        }
        handle->fwd_ofw_rb = i;
        handle->fwd_ofh_rb = 1;
#if defined(LIBXS_FWD_OFH_BLOCKING)
      }
#endif

      if (handle->datatype == LIBXS_CONV_DATATYPE_FP32) {
        handle->ifmblock = (conv_desc.C >=16) ? 16 : conv_desc.C;
        handle->ofmblock = (conv_desc.K >=16) ? 16 : conv_desc.K;
      }
      else if (handle->datatype == LIBXS_CONV_DATATYPE_INT16) {
        handle->ifmblock = (conv_desc.C >=32) ? 32 : conv_desc.C;
        handle->ofmblock = (conv_desc.K >=16) ? 16 : conv_desc.K;
      }
      else if (handle->datatype == LIBXS_CONV_DATATYPE_INT8) {
        handle->ifmblock = (conv_desc.C >=64) ? 64 : conv_desc.C;
        handle->ofmblock = (conv_desc.K >=16) ? 16 : conv_desc.K;
      }
      else {
        *status = LIBXS_CONV_ERR_UNSUPPORTED_DATATYPE;
        free(handle);
        handle = 0;
        return handle;
      }
    }
    else {
      *status = LIBXS_CONV_WARN_FALLBACK;
      handle->ifmblock = (conv_desc.C >=8) ? 8 : conv_desc.C;
      handle->ofmblock = (conv_desc.K >=8) ? 8 : conv_desc.K;
    }
    /* Let's calculate how many blocks we need */
    handle->blocksifm = conv_desc.C / handle->ifmblock;
    handle->blocksofm = conv_desc.K / handle->ofmblock;
    /* Let's check that we can actually block */
    if (conv_desc.C % handle->ifmblock != 0 ||
        conv_desc.K % handle->ofmblock != 0)
    {
      *status = LIBXS_CONV_ERR_INVALID_BLOCKING;
      free(handle);
      handle = 0;
      return handle;
    }

    /* TODO: we need to add much more checks here .... */
    if (noarch == 0) {
      /* Forward path */
      { libxs_convolution_forward_descriptor descriptor;
        if (conv_desc.R == 1 && conv_desc.S == 1) {
          descriptor.unroll_kh = 1;
          descriptor.unroll_kw = 1;
        }
        else {
          descriptor.unroll_kh = 0;
          descriptor.unroll_kw = 1;
        }
        descriptor.ifh_padded = conv_desc.H;
        descriptor.ifw_padded = conv_desc.W;
        descriptor.kh = conv_desc.R;
        descriptor.kw = conv_desc.S;
        descriptor.stride_h = conv_desc.u;
        descriptor.stride_w = conv_desc.v;
        descriptor.ofm_block = handle->ofmblock;
        descriptor.ifm_block = handle->ifmblock;
        descriptor.ofh_padded = handle->ofhp;
        descriptor.ofw_padded = handle->ofwp;
        descriptor.ofh_rb = handle->fwd_ofh_rb;
        descriptor.ofw_rb = handle->fwd_ofw_rb;
        descriptor.prefetch = LIBXS_CONVOLUTION_PREFETCH_NONE;
        /* TODO check JIT errors */
        handle->code_fwd[0].sconv = libxs_create_sconv_forward(&descriptor);
        descriptor.prefetch = LIBXS_CONVOLUTION_PREFETCH_NO_WEIGHT;
        handle->code_fwd[1].sconv = libxs_create_sconv_forward(&descriptor);
        descriptor.prefetch = LIBXS_CONVOLUTION_PREFETCH_ALL;
        handle->code_fwd[2].sconv = libxs_create_sconv_forward(&descriptor);
        descriptor.prefetch = LIBXS_CONVOLUTION_PREFETCH_NO_OUTPUT;
        handle->code_fwd[3].sconv = libxs_create_sconv_forward(&descriptor);
      }
#if 0
      /* TODO Backward path */
      {
        libxs_convolution_backward_descriptor descriptor;
        descriptor.ifw_padded = handle->ifw;
        descriptor.ifh_padded = handle->ifh;
        descriptor.kw = handle->kw;
        descriptor.kh = handle->kh;
        descriptor.stride_w = handle->stridew;
        descriptor.stride_h = handle->strideh;
        handle->code_bwd.sconv = libxs_create_sconv_backward(&descriptor);
      }
      /* TODO weight update path */
      { libxs_convolution_weight_update_descriptor descriptor;
        descriptor.ifw_padded = handle->ifw;
        descriptor.ifh_padded = handle->ifh;
        descriptor.kw = handle->kw;
        /*descriptor.kh = handle->kh;*/
        descriptor.stride_w = handle->stridew;
        descriptor.stride_h = handle->strideh;
        handle->code_upd.sconv = libxs_create_sconv_update_weights(&descriptor);
      }
#endif
    }
    else {
      handle->code_fwd[0].sconv = 0;
      handle->code_fwd[1].sconv = 0;
      handle->code_fwd[2].sconv = 0;
      handle->code_fwd[3].sconv = 0;
      /* TODO Backward path */
      /* TODO weight update path */
    }
  }
  else {
    *status = LIBXS_CONV_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_destroy_handle(const libxs_conv_handle* handle)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (0 != handle) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    /* TODO */
    /* deallocate code known to be not registered; no index attached */
    /* do not use libxs_release_kernel here! */
    libxs_deallocate(handle->code_fwd[0].pmm);
    libxs_deallocate(handle->code_fwd[1].pmm);
    libxs_deallocate(handle->code_fwd[2].pmm);
    libxs_deallocate(handle->code_fwd[3].pmm);
    /* TODO Backward path */
    /* TODO weight update path */
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_conv_handle*)handle);
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_INLINE LIBXS_RETARGETABLE size_t internal_conv_typesize(libxs_conv_datatype datatype)
{
  switch (datatype) {
    case LIBXS_CONV_DATATYPE_FP32:  return 4;
    case LIBXS_CONV_DATATYPE_INT32: return 4;
    case LIBXS_CONV_DATATYPE_INT16: return 2;
    case LIBXS_CONV_DATATYPE_INT8:  return 1;
    /* no error expected as enumeration really arrives at an enum; compiler-checked */
    default: return 1;
  }
}


LIBXS_API_DEFINITION libxs_conv_layer* libxs_conv_create_input_layer(const libxs_conv_handle* handle)
{
  libxs_conv_err_t status;
  return libxs_conv_create_input_layer_check(handle, &status);
}


LIBXS_API_DEFINITION libxs_conv_layer* libxs_conv_create_input_layer_check(const libxs_conv_handle* handle, libxs_conv_err_t* status)
{
  libxs_conv_layer* layer = (libxs_conv_layer*)malloc(sizeof(libxs_conv_layer));
  int result = EXIT_SUCCESS;
  *status = LIBXS_CONV_SUCCESS;

  if (handle != 0 && layer != 0) {
    /* set properties of the layer according to convolution handle */
    layer->N = handle->desc.N;
    layer->splits = handle->desc.splits;
    layer->fmb = handle->blocksifm;
    layer->bfm = handle->ifmblock;
    layer->H = handle->ifhp;
    layer->W = handle->ifwp;
    layer->datatype = handle->datatype;
    /* allocate raw data */
    result = libxs_allocate(&layer->data,
        layer->N * layer->splits * layer->fmb * layer->bfm * layer->H * layer->W * internal_conv_typesize(layer->datatype),
        LIBXS_ALIGNMENT, LIBXS_ALLOC_FLAG_RW, 0/*extra*/, 0/*extra_size*/);
  }
  else {
    *status = LIBXS_CONV_ERR_CREATE_LAYER;
    layer = 0;
  }

  if (result != EXIT_SUCCESS) {
    *status = LIBXS_CONV_ERR_CREATE_LAYER;
    free((libxs_conv_layer*)layer);
    layer = 0;
  }

  return layer;
}


LIBXS_API_DEFINITION libxs_conv_layer* libxs_conv_create_output_layer(const libxs_conv_handle* handle)
{
  libxs_conv_err_t status;
  return libxs_conv_create_output_layer_check(handle, &status);
}


LIBXS_API_DEFINITION libxs_conv_layer* libxs_conv_create_output_layer_check(const libxs_conv_handle* handle, libxs_conv_err_t* status)
{
  libxs_conv_layer* layer = (libxs_conv_layer*)malloc(sizeof(libxs_conv_layer));
  int result = EXIT_SUCCESS;
  *status = LIBXS_CONV_SUCCESS;

  if (handle != 0 && layer != 0) {
    /* set properties of the layer according to convolution handle */
    layer->N = handle->desc.N;
    layer->splits = handle->desc.splits;
    layer->fmb = handle->blocksofm;
    layer->bfm = handle->ofmblock;
    layer->H = handle->ofhp;
    layer->W = handle->ofwp;
    if (handle->datatype == LIBXS_CONV_DATATYPE_FP32) {
      layer->datatype = handle->datatype;
    }
    else {
      layer->datatype = LIBXS_CONV_DATATYPE_INT32;
    }
    /* allocate raw data, we always have a 4 byte wide type!! */
    result = libxs_allocate(&layer->data,
        layer->N * layer->splits * layer->fmb * layer->bfm * layer->H * layer->W * internal_conv_typesize(layer->datatype),
        LIBXS_ALIGNMENT, LIBXS_ALLOC_FLAG_RW, 0/*extra*/, 0/*extra_size*/);
  }
  else {
    *status = LIBXS_CONV_ERR_CREATE_LAYER;
    layer = 0;
  }

  if (result != EXIT_SUCCESS) {
    *status = LIBXS_CONV_ERR_CREATE_LAYER;
    free((libxs_conv_layer*)layer);
    layer = 0;
  }

  return layer;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_destroy_layer(const libxs_conv_layer* layer)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (0 != layer) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    libxs_deallocate(layer->data);
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_conv_layer*)layer);
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_LAYER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_conv_filter* libxs_conv_create_filter(const libxs_conv_handle* handle)
{
  libxs_conv_err_t status;
  return libxs_conv_create_filter_check(handle, &status);
}


LIBXS_API_DEFINITION libxs_conv_filter* libxs_conv_create_filter_check(const libxs_conv_handle* handle, libxs_conv_err_t* status)
{
  libxs_conv_filter* filter = (libxs_conv_filter*)malloc(sizeof(libxs_conv_filter));
  int result = EXIT_SUCCESS;
  *status = LIBXS_CONV_SUCCESS;

  if (handle != 0 && filter != 0) {
    /* set properties of the layer according to convolution handle */
    filter->splits = handle->desc.splits;
    filter->ifmb = handle->blocksifm;
    filter->bifm = handle->ifmblock;
    filter->ofmb = handle->blocksofm;
    filter->bofm = handle->ofmblock;
    filter->R = handle->desc.R;
    filter->S = handle->desc.S;
    filter->datatype = handle->datatype;
    /* allocate raw data */
    result = libxs_allocate(&filter->data,
        filter->splits * filter->ifmb * filter->bifm * filter->ofmb * filter->bofm * filter->R * filter->S * internal_conv_typesize(filter->datatype),
        LIBXS_ALIGNMENT, LIBXS_ALLOC_FLAG_RW, 0/*extra*/, 0/*extra_size*/);
  }
  else {
    *status = LIBXS_CONV_ERR_CREATE_FILTER;
    filter = 0;
  }

  if (result != EXIT_SUCCESS) {
    *status = LIBXS_CONV_ERR_CREATE_FILTER;
    free((libxs_conv_filter*)filter);
    filter = 0;
  }

  return filter;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_destroy_filter(const libxs_conv_filter* filter)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (0 != filter) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    libxs_deallocate(filter->data);
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_conv_filter*)filter);
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_FILTER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_conv_bias* libxs_conv_create_bias(const libxs_conv_handle* handle)
{
  libxs_conv_err_t status;
  return libxs_conv_create_bias_check(handle, &status);
}


LIBXS_API_DEFINITION libxs_conv_bias* libxs_conv_create_bias_check(const libxs_conv_handle* handle, libxs_conv_err_t* status)
{
  libxs_conv_bias* bias = (libxs_conv_bias*)malloc(sizeof(libxs_conv_bias));
  int result = EXIT_SUCCESS;
  *status = LIBXS_CONV_SUCCESS;

  if (handle != 0 && bias != 0) {
    /* set properties of the layer according to convolution handle */
    bias->splits = handle->desc.splits;
    bias->fmb = handle->blocksifm;
    bias->bfm = handle->ifmblock;
    bias->datatype = handle->datatype;
    /* allocate raw data, we always have a 4 byte wide type!! */
    result = libxs_allocate(&bias->data,
        bias->splits * bias->fmb * bias->bfm * internal_conv_typesize(bias->datatype),
        LIBXS_ALIGNMENT, LIBXS_ALLOC_FLAG_RW, 0/*extra*/, 0/*extra_size*/);
  }
  else {
    *status = LIBXS_CONV_ERR_CREATE_BIAS;
    bias = 0;
  }

  if (result != EXIT_SUCCESS) {
    *status = LIBXS_CONV_ERR_CREATE_BIAS;
    free((libxs_conv_bias*)bias);
    bias = 0;
  }

  return bias;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_destroy_bias(const libxs_conv_bias* bias)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (0 != bias) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    libxs_deallocate(bias->data);
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_conv_bias*)bias);
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_BIAS;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_copyin_layer(const libxs_conv_layer* layer, const void* data)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (0 != layer) {
    /* we do for-loops such that we could potentially leverage NUMA in future */
    switch (layer->datatype) {
      case LIBXS_CONV_DATATYPE_FP32: {
        typedef float element_type;
        int i1, i2, i3, i4, i5, i6;
        int N = layer->N;
        int splits = layer->splits;
        int fmb = layer->fmb;
        int bfm = layer->bfm;
        int H = layer->H;
        int W = layer->W;
#if defined(LIBXS_VLA)
        typedef element_type (*LIBXS_RESTRICT handle_data_type)[splits][fmb][H][W][bfm];
        typedef element_type (*LIBXS_RESTRICT user_data_type)[splits][fmb*bfm][H][W];
        const handle_data_type handle_data = (handle_data_type)layer->data;
        const user_data_type user_data = (user_data_type)data;
#else
        element_type *const handle_data = (element_type*)layer->data;
        const element_type *const user_data = (const element_type*)data;
        unsigned int hindexn[6], uindexn[5];
        unsigned int hshape[6], ushape[5];
        /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
        hshape[0] = bfm; hshape[1] = W; hshape[2] = H; hshape[3] = fmb; hshape[4] = splits; hshape[5] = N;
        ushape[0] = W; ushape[1] = H; ushape[2] = fmb * bfm; ushape[3] = splits; ushape[4] = N;
#endif
        for (i1 = 0; i1 < N; ++i1) {
          for (i2 = 0; i2 < splits; ++i2) {
            for (i3 = 0; i3 < fmb; ++i3) {
              for (i4 = 0; i4 < H; ++i4) {
                for (i5 = 0; i5 < W; ++i5) {
                  for (i6 = 0; i6 < bfm; ++i6) {
#if defined(LIBXS_VLA)
                    handle_data[i1][i2][i3][i4][i5][i6] = user_data[i1][i2][i3*bfm+i6][i4][i5];
#else
                    size_t h, u;
                    /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
                    hindexn[0] = i6; hindexn[1] = i5; hindexn[2] = i4; hindexn[3] = i3; hindexn[4] = i2; hindexn[5] = i1;
                    uindexn[0] = i5; uindexn[1] = i4; uindexn[2] = i3 * bfm + i6; uindexn[3] = i2; uindexn[4] = i1;
                    LIBXS_CALC_INDEX1(size_t, h, 6, hindexn, hshape);
                    LIBXS_CALC_INDEX1(size_t, u, 5, uindexn, ushape);
                    handle_data[h] = user_data[u];
#endif
                  }
                }
              }
            }
          }
        }
      } break;
      default: {
        status = LIBXS_CONV_ERR_UNSUPPORTED_DATATYPE;
      }
    }
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_LAYER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_zero_layer(const libxs_conv_layer* layer)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;
  const size_t size = (size_t)layer->N * (size_t)layer->splits * (size_t)layer->fmb
                    * (size_t)layer->bfm * (size_t)layer->H * (size_t)layer->W;
  size_t i;

  if (0 != layer) {
    /* we do for-loops such that we could potentially leverage NUMA in future */
    switch (layer->datatype) {
      case LIBXS_CONV_DATATYPE_FP32: {
        float* fp32_data = (float*)layer->data;
        for (i = 0; i < size; ++i) fp32_data[i] = 0.0f;
      } break;
      case LIBXS_CONV_DATATYPE_INT32: {
        int* int32_data = (int*)layer->data;
        for (i = 0; i < size; ++i) int32_data[i] = 0;
      } break;
      case LIBXS_CONV_DATATYPE_INT16: {
        short* int16_data = (short*)layer->data;
        for (i = 0; i < size; ++i) int16_data[i] = 0;
      } break;
      case LIBXS_CONV_DATATYPE_INT8: {
        char* int8_data = (char*)layer->data;
        for (i = 0; i < size; ++i) int8_data[i] = 0;
      } break;
      default: {
        status = LIBXS_CONV_ERR_UNSUPPORTED_DATATYPE;
      }
    }
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_LAYER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_copyout_layer(const libxs_conv_layer* layer, void* data)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (0 != layer) {
    /* we do for-loops such that we could potentially leverage NUMA in future */
    switch (layer->datatype) {
      case LIBXS_CONV_DATATYPE_FP32: {
        typedef float element_type;
        int i1, i2, i3, i4, i5, i6;
        int N = layer->N;
        int splits = layer->splits;
        int fmb = layer->fmb;
        int bfm = layer->bfm;
        int H = layer->H;
        int W = layer->W;
#if defined(LIBXS_VLA)
        typedef element_type (*LIBXS_RESTRICT handle_data_type)[splits][fmb][H][W][bfm];
        typedef element_type (*LIBXS_RESTRICT user_data_type)[splits][fmb*bfm][H][W];
        const handle_data_type handle_data = (handle_data_type)layer->data;
        const user_data_type user_data = (user_data_type)data;
#else
        const element_type *const handle_data = (const element_type*)layer->data;
        element_type *const user_data = (element_type*)data;
        unsigned int hindexn[6], uindexn[5];
        unsigned int hshape[6], ushape[5];
        /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
        hshape[0] = bfm; hshape[1] = W; hshape[2] = H; hshape[3] = fmb; hshape[4] = splits; hshape[5] = N;
        ushape[0] = W; ushape[1] = H; ushape[2] = fmb * bfm; ushape[3] = splits; ushape[4] = N;
#endif
        for (i1 = 0; i1 < N; ++i1) {
          for (i2 = 0; i2 < splits; ++i2) {
            for (i3 = 0; i3 < fmb; ++i3) {
              for (i4 = 0; i4 < H; ++i4) {
                for (i5 = 0; i5 < W; ++i5) {
                  for (i6 = 0; i6 < bfm; ++i6) {
#if defined(LIBXS_VLA)
                    user_data[i1][i2][i3*bfm+i6][i4][i5] = handle_data[i1][i2][i3][i4][i5][i6];
#else
                    size_t h, u;
                    /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
                    hindexn[0] = i6; hindexn[1] = i5; hindexn[2] = i4; hindexn[3] = i3; hindexn[4] = i2; hindexn[5] = i1;
                    uindexn[0] = i5; uindexn[1] = i4; uindexn[2] = i3 * bfm + i6; uindexn[3] = i2; uindexn[4] = i1;
                    LIBXS_CALC_INDEX1(size_t, h, 6, hindexn, hshape);
                    LIBXS_CALC_INDEX1(size_t, u, 5, uindexn, ushape);
                    user_data[u] = handle_data[h];
#endif
                  }
                }
              }
            }
          }
        }
      } break;
      default: {
        status = LIBXS_CONV_ERR_UNSUPPORTED_DATATYPE;
      }
    }
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_LAYER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_copyin_filter(const libxs_conv_filter* filter, const void* data)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (0 != filter) {
    /* we do for-loops such that we could potentially leverage NUMA in future */
    switch (filter->datatype) {
      case LIBXS_CONV_DATATYPE_FP32: {
        typedef float element_type;
        int i1, i2, i3, i4, i5, i6, i7;
        int splits = filter->splits;
        int ifmb = filter->ifmb;
        int bifm = filter->bifm;
        int ofmb = filter->ofmb;
        int bofm = filter->bofm;
        int R = filter->R;
        int S = filter->S;
#if defined(LIBXS_VLA)
        typedef element_type (*LIBXS_RESTRICT handle_data_type)[ofmb][ifmb][R][S][bifm][bofm];
        typedef element_type (*LIBXS_RESTRICT user_data_type)[ofmb*bofm][ifmb*bifm][R][S];
        const handle_data_type handle_data = (handle_data_type)filter->data;
        const user_data_type user_data = (user_data_type)data;
#else
        element_type *const handle_data = (element_type*)filter->data;
        const element_type *const user_data = (const element_type*)data;
        unsigned int hindexn[7], uindexn[5];
        unsigned int hshape[7], ushape[5];
        /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
        hshape[0] = bofm; hshape[1] = bifm; hshape[2] = S; hshape[3] = R; hshape[4] = ifmb; hshape[5] = ofmb; hshape[6] = splits;
        ushape[0] = S; ushape[1] = R; ushape[2] = ifmb * bifm; ushape[3] = ofmb * bofm; ushape[4] = splits;
#endif
        for (i1 = 0; i1 < splits; ++i1) {
          for (i2 = 0; i2 < ofmb; ++i2) {
            for (i3 = 0; i3 < ifmb; ++i3) {
              for (i4 = 0; i4 < R; ++i4) {
                for (i5 = 0; i5 < S; ++i5) {
                  for (i6 = 0; i6 < bifm; ++i6) {
                    for (i7 = 0; i7 < bofm; ++i7) {
#if defined(LIBXS_VLA)
                      handle_data[i1][i2][i3][i4][i5][i6][i7] = user_data[i1][i2*bofm+i7][i3*bifm+i6][i4][i5];
#else
                      size_t h, u;
                      /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
                      hindexn[0] = i7; hindexn[1] = i6; hindexn[2] = i5; hindexn[3] = i4; hindexn[4] = i3; hindexn[5] = i2; hindexn[6] = i1;
                      uindexn[0] = i5; uindexn[1] = i4; uindexn[2] = i3 * bifm + i6; uindexn[3] = i2 * bofm + i7; uindexn[4] = i1;
                      LIBXS_CALC_INDEX1(size_t, h, 7, hindexn, hshape);
                      LIBXS_CALC_INDEX1(size_t, u, 5, uindexn, ushape);
                      handle_data[h] = user_data[u];
#endif
                    }
                  }
                }
              }
            }
          }
        }
      } break;
      default: {
        status = LIBXS_CONV_ERR_UNSUPPORTED_DATATYPE;
      }
    }
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_FILTER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_copyout_filter(const libxs_conv_filter* filter, void* data)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (0 != filter) {
    /* we do for-loops such that we could potentially leverage NUMA in future */
    switch (filter->datatype) {
      case LIBXS_CONV_DATATYPE_FP32: {
        typedef float element_type;
        int i1, i2, i3, i4, i5, i6, i7;
        int splits = filter->splits;
        int ifmb = filter->ifmb;
        int bifm = filter->bifm;
        int ofmb = filter->ofmb;
        int bofm = filter->bofm;
        int R = filter->R;
        int S = filter->S;
#if defined(LIBXS_VLA)
        typedef element_type (*LIBXS_RESTRICT handle_data_type)[ofmb][ifmb][R][S][bifm][bofm];
        typedef element_type (*LIBXS_RESTRICT user_data_type)[ofmb*bofm][ifmb*bifm][R][S];
        const handle_data_type handle_data = (handle_data_type)filter->data;
        const user_data_type user_data = (user_data_type)data;
#else
        const element_type *const handle_data = (const element_type*)filter->data;
        element_type *const user_data = (element_type*)data;
        unsigned int hindexn[7], uindexn[5];
        unsigned int hshape[7], ushape[5];
        /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
        hshape[0] = bofm; hshape[1] = bifm; hshape[2] = S; hshape[3] = R; hshape[4] = ifmb; hshape[5] = ofmb; hshape[6] = splits;
        ushape[0] = S; ushape[1] = R; ushape[2] = ifmb * bifm; ushape[3] = ofmb * bofm; ushape[4] = splits;
#endif
        for (i1 = 0; i1 < splits; ++i1) {
          for (i2 = 0; i2 < ofmb; ++i2) {
            for (i3 = 0; i3 < ifmb; ++i3) {
              for (i4 = 0; i4 < R; ++i4) {
                for (i5 = 0; i5 < S; ++i5) {
                  for (i6 = 0; i6 < bifm; ++i6) {
                    for (i7 = 0; i7 < bofm; ++i7) {
#if defined(LIBXS_VLA)
                      user_data[i1][i2*bofm+i7][i3*bifm+i6][i4][i5] = handle_data[i1][i2][i3][i4][i5][i6][i7];
#else
                      size_t h, u;
                      /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
                      hindexn[0] = i7; hindexn[1] = i6; hindexn[2] = i5; hindexn[3] = i4; hindexn[4] = i3; hindexn[5] = i2; hindexn[6] = i1;
                      uindexn[0] = i5; uindexn[1] = i4; uindexn[2] = i3 * bifm + i6; uindexn[3] = i2 * bofm + i7; uindexn[4] = i1;
                      LIBXS_CALC_INDEX1(size_t, h, 7, hindexn, hshape);
                      LIBXS_CALC_INDEX1(size_t, u, 5, uindexn, ushape);
                      user_data[u] = handle_data[h];
#endif
                    }
                  }
                }
              }
            }
          }
        }
      } break;
      default: {
        status = LIBXS_CONV_ERR_UNSUPPORTED_DATATYPE;
      }
    }
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_FILTER;
  }

  return status;
}


#if 0
LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_copyin_bias(const libxs_conv_bias* bias, const void* data)
{
  LIBXS_UNUSED(bias); LIBXS_UNUSED(data); /* TODO: libxs_conv_copyin_input */
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_copyout_bias(const libxs_conv_bias* bias, void* data)
{
  LIBXS_UNUSED(bias); LIBXS_UNUSED(data); /* TODO: libxs_conv_copyin_input */
}
#endif


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_bind_input_layer(libxs_conv_handle* handle, const libxs_conv_layer* layer)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (handle != 0 && layer != 0) {
    /* check if format matches */
    if ( handle->desc.N == layer->N
      && handle->desc.splits == layer->splits
      && handle->ifwp == layer->W
      && handle->ifhp == layer->H
      && handle->ifmblock == layer->bfm
      && handle->blocksifm == layer->fmb
      && handle->datatype == layer->datatype)
    {
      handle->input = (libxs_conv_layer*)layer;
    }
    else {
      status = LIBXS_CONV_ERR_MISMATCH_LAYER;
    }
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_HANDLE_LAYER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_bind_output_layer(libxs_conv_handle* handle, const libxs_conv_layer* layer)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (handle != 0 && layer != 0) {
    /* check if format matches */
    if ( handle->desc.N == layer->N
      && handle->desc.splits == layer->splits
      && handle->ofwp == layer->W
      && handle->ofhp == layer->H
      && handle->ofmblock == layer->bfm
      && handle->blocksofm == layer->fmb
      && ((handle->datatype == LIBXS_CONV_DATATYPE_FP32 && layer->datatype == LIBXS_CONV_DATATYPE_FP32)
        || (layer->datatype == LIBXS_CONV_DATATYPE_INT32)))
    {
      handle->output = (libxs_conv_layer*)layer;
    }
    else {
      status = LIBXS_CONV_ERR_MISMATCH_LAYER;
    }
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_HANDLE_LAYER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_conv_bind_filter(libxs_conv_handle* handle, const libxs_conv_filter* filter)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (handle != 0 && filter != 0) {
    /* check if format matches */
    if ( handle->desc.splits == filter->splits
      && handle->desc.R == filter->R
      && handle->desc.S == filter->S
      && handle->ifmblock == filter->bifm
      && handle->blocksifm == filter->ifmb
      && handle->ofmblock == filter->bofm
      && handle->blocksofm == filter->ofmb
      && handle->datatype == filter->datatype)
    {
      handle->filter = (libxs_conv_filter*)filter;
    }
    else {
      status = LIBXS_CONV_ERR_MISMATCH_FILTER;
    }
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_HANDLE_FILTER;
  }

  return status;
}


LIBXS_INLINE LIBXS_RETARGETABLE libxs_conv_err_t internal_convolve_st(libxs_conv_handle* handle,
  libxs_conv_kind kind, int start_thread, int tid, int num_threads)
{
  libxs_conv_err_t status = LIBXS_CONV_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_CONV_KIND_FWD: {
        libxs_convolve_st_fwd(handle, start_thread, tid, num_threads);
      } break;
      default: {
        status = LIBXS_CONV_ERR_INVALID_KIND;
      }
    }
  }
  else {
    status = LIBXS_CONV_ERR_INVALID_HANDLE;
  }

  return status;
#if 0
  libxs_sconvfunction convolution = 0;
  if (0 != handle) {
    /* TODO: implement support for bias */
    LIBXS_UNUSED(bias);
    switch (kind) {
      case LIBXS_CONV_KIND_FWD: if (
           0 != handle->data_input
        && 0 != handle->data_weight
        && 0 != handle->data_output)
      {
        convolution = handle->code_fwd.sconv;
      } break;
      case LIBXS_CONV_KIND_BWD: if (
           0 != handle->data_input
        && 0 != handle->data_weight
        && 0 != handle->data_output)
      {
        convolution = handle->code_bwd.sconv;
      } break;
      case LIBXS_CONV_KIND_UPD: if (
           0 != handle->data_input
        && 0 != handle->data_weight
        && 0 != handle->data_output)
      {
        convolution = handle->code_upd.sconv;
      } break;
    }
  }

  /* so far, no need to distinct convolutions (synchronization impl.'d only one time) */
  if (0 != convolution) { /* execute convolution */
    /* TODO: implement thread-synchronization */
    LIBXS_UNUSED(tid); LIBXS_UNUSED(num_threads);
    /* execute convolution */
    convolution(handle->data_input, handle->data_weight, handle->data_output,
      /* TODO: prefetch -> */ 0/*ipf1*/, 0/*ipf2*/, 0/*opf*/);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static LIBXS_TLS int error_handle = 0;
    if (0 == error_handle) {
      fprintf(stderr, "LIBXS: convolution failed to execute!\n");
      error_handle = 1;
    }
  }
#endif
#endif
}


LIBXS_API_DEFINITION void libxs_convolve(libxs_conv_handle* handle, libxs_conv_kind kind)
{
#if defined(_OPENMP)
# pragma omp parallel
  internal_convolve_st(handle, kind, 0, omp_get_thread_num(), omp_get_num_threads());
#else
  internal_convolve_st(handle, kind, 0/*start_thread*/, 0/*tid*/, 1/*num_threads*/);
#endif
}


LIBXS_API_DEFINITION libxs_conv_err_t libxs_convolve_st(libxs_conv_handle* handle,
  libxs_conv_kind kind, /*unsigned*/int start_thread, /*unsigned*/int tid, /*unsigned*/int num_threads)
{
  return internal_convolve_st(handle, kind, start_thread, tid, num_threads);
}


#if defined(LIBXS_BUILD) || defined(LIBXS_CONV_INTERNAL_API)

LIBXS_API_DEFINITION libxs_sconvfunction libxs_create_sconv_forward(
  const libxs_convolution_forward_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
  if (0 != descriptor) {
    libxs_build_request request;
    request.descriptor.cfwd = descriptor;
    request.kind = LIBXS_BUILD_KIND_CFWD;
    libxs_build(&request, LIBXS_REGSIZE/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static LIBXS_TLS int error_desc = 0;
    if (0 == error_desc) {
      fprintf(stderr, "LIBXS: invalid descriptor (forward convolution)!\n");
      error_desc = 1;
    }
  }
#endif
  return code.sconv;
}


LIBXS_API_DEFINITION libxs_sconvfunction libxs_create_sconv_backward(
  const libxs_convolution_backward_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
  if (0 != descriptor) {
    libxs_build_request request;
    request.descriptor.cbwd = descriptor;
    request.kind = LIBXS_BUILD_KIND_CBWD;
    libxs_build(&request, LIBXS_REGSIZE/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static LIBXS_TLS int error_desc = 0;
    if (0 == error_desc) {
      fprintf(stderr, "LIBXS: invalid descriptor (backward convolution)!\n");
      error_desc = 1;
    }
  }
#endif
  return code.sconv;
}


LIBXS_API_DEFINITION libxs_sconvfunction libxs_create_sconv_update_weights(
  const libxs_convolution_weight_update_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
  if (0 != descriptor) {
    libxs_build_request request;
    request.descriptor.cupd = descriptor;
    request.kind = LIBXS_BUILD_KIND_CUPD;
    libxs_build(&request, LIBXS_REGSIZE/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static LIBXS_TLS int error_desc = 0;
    if (0 == error_desc) {
      fprintf(stderr, "LIBXS: invalid convolution descriptor (weight update)!\n");
      error_desc = 1;
    }
  }
#endif
  return code.sconv;
}

#endif /*defined(LIBXS_BUILD) || defined(LIBXS_CONV_INTERNAL_API)*/
