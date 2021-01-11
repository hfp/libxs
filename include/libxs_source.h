/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_SOURCE_H
#define LIBXS_SOURCE_H

#if defined(LIBXS_MACROS_H)
# error Please do not include any LIBXS header other than libxs_source.h!
#endif
#if defined(LIBXS_BUILD)
# error LIBXS_BUILD cannot be defined for the header-only LIBXS!
#endif

/**
 * This header is intentionally called "libxs_source.h" since the followings block
 * includes *internal* files, and thereby exposes LIBXS's implementation.
 * The so-called "header-only" usage model gives up the clearly defined binary interface
 * (including support for hot-fixes after deployment), and requires to rebuild client
 * code for every (internal) change of LIBXS. Please make sure to only rely on the
 * public interface as the internal implementation may change without notice.
 */
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include "../src/generator_aarch64_instructions.c"
#include "../src/generator_common.c"
#include "../src/generator_common_aarch64.c"
#include "../src/generator_common_x86.c"
#include "../src/generator_gemm.c"
#include "../src/generator_gemm_aarch64.c"
#include "../src/generator_gemm_amx.c"
#include "../src/generator_gemm_amx_emu.c"
#include "../src/generator_gemm_amx_microkernel.c"
#include "../src/generator_gemm_amx_microkernel_emu.c"
#include "../src/generator_gemm_avx2_microkernel.c"
#include "../src/generator_gemm_avx512_microkernel.c"
#include "../src/generator_gemm_avx_microkernel.c"
#include "../src/generator_gemm_common.c"
#include "../src/generator_gemm_common_aarch64.c"
#include "../src/generator_gemm_noarch.c"
#include "../src/generator_gemm_sse_avx_avx2_avx512.c"
#include "../src/generator_gemm_sse_microkernel.c"
#include "../src/generator_matcopy.c"
#include "../src/generator_matcopy_avx_avx512.c"
#include "../src/generator_mateltwise.c"
#include "../src/generator_mateltwise_avx_avx512.c"
#include "../src/generator_mateltwise_cvtfp32bf16_act_avx_avx512.c"
#include "../src/generator_mateltwise_dropout_avx_avx512.c"
#include "../src/generator_mateltwise_reduce_avx_avx512.c"
#include "../src/generator_mateltwise_relu_avx_avx512.c"
#include "../src/generator_mateltwise_scale_avx_avx512.c"
#include "../src/generator_mateltwise_transform_avx_avx512.c"
#include "../src/generator_mateltwise_unary_avx_avx512.c"
#include "../src/generator_packed.c"
#include "../src/generator_packed_gemm_ac_rm.c"
#include "../src/generator_packed_gemm_ac_rm_aarch64.c"
#include "../src/generator_packed_gemm_ac_rm_avx_avx2_avx512.c"
#include "../src/generator_packed_gemm_avx_avx512.c"
#include "../src/generator_packed_gemm_bc_rm.c"
#include "../src/generator_packed_gemm_bc_rm_aarch64.c"
#include "../src/generator_packed_gemm_bc_rm_avx_avx2_avx512.c"
#include "../src/generator_packed_getrf_avx_avx512.c"
#include "../src/generator_packed_spgemm.c"
#include "../src/generator_packed_spgemm_csc_bsparse.c"
#include "../src/generator_packed_spgemm_csc_bsparse_aarch64.c"
#include "../src/generator_packed_spgemm_csc_bsparse_avx_avx2_avx512.c"
#include "../src/generator_packed_spgemm_csc_csparse.c"
#include "../src/generator_packed_spgemm_csc_csparse_avx_avx2_avx512.c"
#include "../src/generator_packed_spgemm_csr_asparse.c"
#include "../src/generator_packed_spgemm_csr_asparse_aarch64.c"
#include "../src/generator_packed_spgemm_csr_asparse_avx_avx2_avx512.c"
#include "../src/generator_packed_spgemm_csr_bsparse.c"
#include "../src/generator_packed_spgemm_csr_bsparse_aarch64.c"
#include "../src/generator_packed_spgemm_csr_bsparse_avx_avx2_avx512.c"
#include "../src/generator_packed_trmm_avx_avx512.c"
#include "../src/generator_packed_trsm_avx_avx512.c"
#include "../src/generator_spgemm.c"
#include "../src/generator_spgemm_csc_asparse.c"
#include "../src/generator_spgemm_csc_bsparse.c"
#include "../src/generator_spgemm_csc_reader.c"
#include "../src/generator_spgemm_csr_asparse.c"
#include "../src/generator_spgemm_csr_asparse_reg.c"
#include "../src/generator_spgemm_csr_reader.c"
#include "../src/generator_transpose.c"
#include "../src/generator_transpose_avx_avx512.c"
#include "../src/generator_x86_instructions.c"
#include "../src/libxs_blocked_gemm.c"
#include "../src/libxs_cpuid_x86.c"
#include "../src/libxs_dnn.c"
#include "../src/libxs_dnn_convolution.c"
#include "../src/libxs_dnn_convolution_backward.c"
#include "../src/libxs_dnn_convolution_forward.c"
#include "../src/libxs_dnn_convolution_weight_update.c"
#include "../src/libxs_dnn_elementwise.c"
#include "../src/libxs_dnn_fullyconnected.c"
#include "../src/libxs_dnn_fullyconnected_backward_weight_update.c"
#include "../src/libxs_dnn_fullyconnected_forward.c"
#include "../src/libxs_dnn_fusedbatchnorm.c"
#include "../src/libxs_dnn_fusedbatchnorm_backward.c"
#include "../src/libxs_dnn_fusedbatchnorm_forward.c"
#include "../src/libxs_dnn_fusedgroupnorm.c"
#include "../src/libxs_dnn_fusedgroupnorm_backward.c"
#include "../src/libxs_dnn_fusedgroupnorm_forward.c"
#include "../src/libxs_dnn_optimizer.c"
#include "../src/libxs_dnn_optimizer_sgd.c"
#include "../src/libxs_dnn_pooling.c"
#include "../src/libxs_dnn_pooling_backward.c"
#include "../src/libxs_dnn_pooling_forward.c"
#include "../src/libxs_dnn_rnncell.c"
#include "../src/libxs_dnn_rnncell_backward_weight_update.c"
#include "../src/libxs_dnn_rnncell_forward.c"
#include "../src/libxs_dnn_softmaxloss.c"
#include "../src/libxs_dnn_softmaxloss_backward.c"
#include "../src/libxs_dnn_softmaxloss_forward.c"
#include "../src/libxs_dnn_tensor.c"
#include "../src/libxs_ext.c"
#include "../src/libxs_ext_blocked_gemm.c"
#include "../src/libxs_ext_gemm.c"
#include "../src/libxs_ext_xcopy.c"
#include "../src/libxs_fsspmdm.c"
#include "../src/libxs_gemm.c"
#include "../src/libxs_generator.c"
#include "../src/libxs_hash.c"
#include "../src/libxs_main.c"
#include "../src/libxs_malloc.c"
#include "../src/libxs_math.c"
#include "../src/libxs_mem.c"
#include "../src/libxs_mhd.c"
#include "../src/libxs_perf.c"
#include "../src/libxs_python.c"
#include "../src/libxs_rng.c"
#include "../src/libxs_spmdm.c"
#include "../src/libxs_sync.c"
#include "../src/libxs_timer.c"
#include "../src/libxs_trace.c"
#include "../src/libxs_xcopy.c"
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#endif /*LIBXS_SOURCE_H*/
