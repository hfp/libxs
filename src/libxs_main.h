/******************************************************************************
** Copyright (c) 2014-2019, Intel Corporation                                **
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
#ifndef LIBXS_MAIN_H
#define LIBXS_MAIN_H

#include <libxs.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXS_CAPACITY_REGISTRY) /* must be POT */
# define LIBXS_CAPACITY_REGISTRY 524288 /* 524287: Mersenne Prime number (2^19-1) */
#endif

#if !defined(LIBXS_MAX_NTHREADS)
# define LIBXS_MAX_NTHREADS 1024
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS)
# define LIBXS_MALLOC_SCRATCH_MAX_NPOOLS LIBXS_MAX_NTHREADS
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_LIMIT)
# define LIBXS_MALLOC_SCRATCH_LIMIT (4ULL << 30) /* 4 GB */
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_MMAP) && 0
# define LIBXS_MALLOC_SCRATCH_MMAP
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_SCALE)
# if defined(LIBXS_MALLOC_SCRATCH_MMAP)
#   define LIBXS_MALLOC_SCRATCH_SCALE 1.3
# else
#   define LIBXS_MALLOC_SCRATCH_SCALE 1.0
# endif
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_INTERNAL_SITE)
# define LIBXS_MALLOC_SCRATCH_INTERNAL_SITE ((uintptr_t)-1)
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_INTERNAL)
# define LIBXS_MALLOC_SCRATCH_INTERNAL ((const char*)(LIBXS_MALLOC_SCRATCH_INTERNAL_SITE))
#endif

#if !defined(LIBXS_VERBOSITY_HIGH)
# define LIBXS_VERBOSITY_HIGH 3 /* secondary warning or info-verbosity */
#endif
#if !defined(LIBXS_VERBOSITY_WARN)
# define LIBXS_VERBOSITY_WARN ((LIBXS_VERBOSITY_HIGH) - LIBXS_MIN(1, LIBXS_VERBOSITY_HIGH))
#endif

#if !defined(LIBXS_LOCK)
# define LIBXS_LOCK LIBXS_LOCK_DEFAULT
#endif

#if !defined(LIBXS_EXT_MIN_NTASKS)
# define LIBXS_MIN_NTASKS(NT) 1
#endif
#if !defined(LIBXS_OVERHEAD)
# define LIBXS_OVERHEAD(NT) 0
#endif
#if !defined(LIBXS_NOOP_ARGS)
# define LIBXS_NOOP_ARGS(...)
#endif
#if !defined(LIBXS_NOOP)
# define LIBXS_NOOP
#endif

/** Check if M, N, K, or LDx fits into the descriptor. */
#if (0 != LIBXS_ILP64)
# define LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K) ( \
    ((unsigned int)(-1)) >= ((unsigned int)(M)) && \
    ((unsigned int)(-1)) >= ((unsigned int)(N)) && \
    ((unsigned int)(-1)) >= ((unsigned int)(K)))
#else /* always fits */
# define LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K) 1
#endif

#if defined(LIBXS_ASSERT) /* assert available */
# define LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K) LIBXS_ASSERT(LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K))
#else
# define LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K)
#endif

#if (defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)) /* TODO: full support for Windows calling convention */
# define LIBXS_GEMM_DESCRIPTOR_PREFETCH(DESCRIPTOR, PREFETCH) LIBXS_UNUSED(PREFETCH); \
            (DESCRIPTOR).prefetch = (unsigned short)(LIBXS_GEMM_PREFETCH_NONE)
#else
# define LIBXS_GEMM_DESCRIPTOR_PREFETCH(DESCRIPTOR, PREFETCH) (DESCRIPTOR).prefetch = (unsigned short)(PREFETCH)
#endif

/** Low-level/internal GEMM descriptor initialization. */
#define LIBXS_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K); LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(LDA, LDB, LDC); \
  (DESCRIPTOR).lda = (unsigned int)(LDA); (DESCRIPTOR).ldb = (unsigned int)(LDB); (DESCRIPTOR).ldc = (unsigned int)(LDC); \
  (DESCRIPTOR).m   = (unsigned int)(M);   (DESCRIPTOR).n   = (unsigned int)(N);   (DESCRIPTOR).k   = (unsigned int)(K); \
  (DESCRIPTOR).datatype = (unsigned char)(DATA_TYPE); (DESCRIPTOR).iflags = 0; (DESCRIPTOR).pad0 = 0; (DESCRIPTOR).pad1 = 0; \
  (DESCRIPTOR).flags = (unsigned short)((FLAGS) \
    /*| (LIBXS_NEQ(0, ALPHA) ? 0 : LIBXS_GEMM_FLAG_ALPHA_0)*/ \
    | (LIBXS_NEQ(0, BETA)  ? 0 : LIBXS_GEMM_FLAG_BETA_0)); \
    LIBXS_GEMM_DESCRIPTOR_PREFETCH(DESCRIPTOR, PREFETCH)
/** Similar to LIBXS_GEMM_DESCRIPTOR, but separately taking the input-/output-precision. */
#define LIBXS_GEMM_DESCRIPTOR2(DESCRIPTOR, IPREC, OPREC, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXS_GEMM_DESCRIPTOR(DESCRIPTOR, LIBXS_GETENUM(IPREC, OPREC), FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

/** Declare and construct a GEMM descriptor. */
#define LIBXS_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  libxs_gemm_descriptor DESCRIPTOR; LIBXS_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, \
    FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)
/** Similar to LIBXS_GEMM_DESCRIPTOR_TYPE, but separately taking the input-/output-precision. */
#define LIBXS_GEMM_DESCRIPTOR2_TYPE(DESCRIPTOR, IPREC, OPREC, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXS_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, LIBXS_GETENUM(IPREC, OPREC), FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)


/**
* Structure, which stores the argument description of GEMM routines.
* This structure must be ordered by the size of the members (packed).
* The size of the structure matches LIBXS_DESCRIPTOR_MAXSIZE.
*/
LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_gemm_descriptor {
  /** Leading dimensions are general offsets. */
  unsigned int lda, ldb, ldc;
  /** Extents of the matrix. */
  unsigned int m, n, k;
  /** Set of flags. */
  unsigned short flags;
  /** Prefetch strategy enumeration. */
  unsigned short prefetch;
  /** Denotes the data-type. */
  unsigned char datatype;
  /** LIBXS_DESCRIPTOR_MAXSIZE. */
  unsigned char pad0, pad1;
  /** INTERNAL (last member!) */
  unsigned char iflags;
};

/** Structure storing the matcopy argument description. */
LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_mcopy_descriptor { /* 20 Byte */
  /** LDx, M, and N. */
  unsigned int m, n, ldi, ldo;
  /** Size of data element. */
  unsigned char typesize;
  /** Level of unrolling. */
  unsigned char unroll_level;
  /** Boolean value (@TODO fix this). */
  unsigned char prefetch;
  /** Set of flags. */
  unsigned char flags;
};

/** Structure storing the transpose argument description. */
LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_trans_descriptor { /* 13 Byte */
  /** LD, M, and N. */
  unsigned int m, n, ldo;
  /** Size of data element. */
  unsigned char typesize;
};

/** Structure storing arguments of packed TRSM. */
LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_trsm_descriptor { /* 30 Byte */
  union { double d; float s; } alpha;
  unsigned int m, n, lda, ldb;
  unsigned char typesize;
  unsigned char layout;
  char diag, side, uplo;
  char transa;
};

/** Structure storing arguments of packed GEMM. */
LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_pgemm_descriptor { /* 30 Byte */
  unsigned int m, n, k, lda, ldb, ldc;
  unsigned char typesize;
  unsigned char layout;
  char transa, transb;
  char alpha_val;
};

/** Structure storing arguments of packed TRSM. */
LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_trmm_descriptor { /* 30 Byte */
  union { double d; float s; } alpha;
  unsigned int m, n, lda, ldb;
  unsigned char typesize;
  unsigned char layout;
  char diag, side, uplo;
  char transa;
};

/** Structure storing arguments of packed GETRF. */
LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_getrf_descriptor { /* 30 Byte */
  unsigned int m, n, lda;
  unsigned char typesize;
  unsigned char layout;
};

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_csr_soa_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxs_csr_soa_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_csc_soa_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* column_ptr;
  const unsigned int* row_idx;
  const void* values;
} libxs_csc_soa_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_rm_ac_soa_descriptor {
  const libxs_gemm_descriptor* gemm;
} libxs_rm_ac_soa_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_rm_bc_soa_descriptor {
  const libxs_gemm_descriptor* gemm;
} libxs_rm_bc_soa_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_csr_reg_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxs_csr_reg_descriptor;

/** Function type used for convolutions (single-precision); the actual signature depends on the kind of convolution. */
LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_sconvfunction)(
  const float* input1, const float* input2, float* output,
  const float* ipf1, const float* ipf2, const float* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_bf16convfunction)(
  const libxs_bfloat16* input1, const libxs_bfloat16* input2, libxs_bfloat16* output,
  const libxs_bfloat16* ipf1, const libxs_bfloat16* ipf2, const libxs_bfloat16* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_bf16f32convfunction)(
  const libxs_bfloat16* input1, const float* input2, libxs_bfloat16* output,
  const libxs_bfloat16* ipf1, const float* ipf2, const libxs_bfloat16* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_wconvfunction)(
  const short* input1, const short* input2, int* output,
  const short* ipf1, const short* ipf2, const int* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_wsconvfunction)(
  const short* input1, const short* input2, float* output,
  const short* ipf1, const short* ipf2, const float* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_uwsconvfunction)(
  short* input1, float* input2, short* output,
  short* ipf1, float* ipf2, short* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_bdbconvfunction)(
  unsigned char* input1, int* input2, unsigned char* output,
  unsigned char* ipf1, int* ipf2, unsigned char* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_busconvfunction)(
  const unsigned char* input1, const char* input2, short* output,
  const unsigned char* ipf1, const char* ipf2, const short* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_budconvfunction)(
  const unsigned char* input1, const char* input2, int* output,
  const unsigned char* ipf1, const char* ipf2, const int* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_wconvfunction_bwd)(
  int* input1, const short* input2, const short* output,
  const int* ipf1, const short* ipf2, const short* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_busconvfunction_bwd)(
  const unsigned short* input1, const char* input2, const char* output,
  const unsigned short* ipf1, const char* ipf2, const char* opf, ...);

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void (*libxs_budconvfunction_bwd)(
  const unsigned int* input1, const char* input2, const char* output,
  const unsigned int* ipf1, const char* ipf2, const char* opf, ...);

/** Function type which is either libxs_sconvfunction or libxs_wconvfunction (weak-typed). */
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_xconvfunction {
  libxs_sconvfunction sconv;
  libxs_bf16convfunction bf16conv;
  libxs_bf16f32convfunction bf1632conv;
  libxs_wsconvfunction wsconv;
  libxs_uwsconvfunction uwsconv;
  libxs_wconvfunction wconv;
  libxs_bdbconvfunction bdbconv;
  libxs_busconvfunction busconv;
  libxs_budconvfunction budconv;
  libxs_wconvfunction_bwd wconvb;
  libxs_busconvfunction_bwd busconvb;
  libxs_budconvfunction_bwd budconvb;
} libxs_xconvfunction;

LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_code_pointer {
  void (*ptr_fn)(LIBXS_VARIADIC);
  const void* ptr_const;
  void* pmm;
  uintptr_t uval;
  intptr_t ival;
  libxs_xmmfunction xgemm; /* GEMM: smm, dmm, wimm, wsmm, or void-function */
  libxs_xmcopyfunction xmatcopy;
  libxs_xtransfunction xtrans;
  libxs_xconvfunction xconv;
  libxs_xtrsmfunction xtrsm;
  libxs_xtrmmfunction xtrmm;
} libxs_code_pointer;

/** Structure which describes all tensors in LIBXS's DNN module */
LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_dnn_tensor {
  libxs_dnn_tensor_datalayout* layout;           /* data-layout descriptor */
  void* data;                                      /* pointer to data */
  unsigned char scf;                               /* fix point scaling factor for this tensor */
};

/* Structure to record segment in stream of code  */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE segment_t {
  int segment_type;
  int n_convs;
  int aux_index;
} segment_t;

LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_dnn_layer {
  libxs_dnn_datatype datatype_in;
  libxs_dnn_datatype datatype_out;
  libxs_dnn_conv_desc desc;
  libxs_dnn_conv_algo algo;
  libxs_dnn_tensor_format buffer_format;
  libxs_dnn_tensor_format filter_format;
  libxs_dnn_conv_fuse_op fuse_ops;
  libxs_dnn_conv_option options;
  libxs_convolution_winograd_descriptor cwino_fwd;
  libxs_convolution_winograd_descriptor cwino_bwd;
  libxs_convolution_winograd_descriptor cwino_upd;
  libxs_dnn_internal_format custom_format_type;    /* Specifies internal LIBXS format to be used */
  /* These are the batchnorm handles in case of fusion  */
  libxs_dnn_fusedbatchnorm* pre_bn;
  libxs_dnn_fusedbatchnorm* post_bn;

  /* additional size for internal data types */
  int ifhp;
  int ifwp;
  int ofh;
  int ofw;
  int ofhp;
  int ofwp;
  int ifmblock;
  int ifmblock_hp;
  int ofmblock;
  int ofmblock_lp;
  int blocksifm;
  int blocksofm;
  int blocksifm_lp;
  int blocksofm_lp;
  int fwd_ofw_rb;
  int fwd_ofw_rb_2;
  int fwd_ofh_rb;
  int fwd_ofh_rb_2;
  int bwd_ofw_rb;
  int bwd_ofh_rb;
  int upd_ofw_rb;
  int upd_ofh_rb;
  int fm_lp_block; /* additional blocking for low precision datatypes of feature maps */
  int upd_use_thread_fil;
  int upd_use_external_reduce;
  int filter_transposed;
  int nBImg;
  int nbImg;
  int blocksifm_blocking;
  int blocksofm_blocking;
  int blocksimg_blocking;
  int use_accumulation_scratch;
  int use_nts_fwd;
  int use_nts_bwd;
  int use_nts_upd;
  int avoid_acc_load;
  int pack_input;
  int pack_input_bwd;
  int spread_input_bwd;
  int use_fwd_for_bwd;
  int exploit_duality;
  int qfma_input_pad;
  int resize_input;
  int ifhp_resized;
  int ifwp_resized;
  int use_fastpath;
  int use_hybrid_wu_parallelism;
  int weight_copies;
  int compute_batch_stats_in_kernel_fwd;
  int compute_batch_stats_in_kernel_bwd;
  int compute_eltwise_in_kernel_bwd;
  int perform_relu_in_kernel;
  int compute_max_in_kernel_fwd;
  int compute_max_in_kernel_bwd;
  int fuse_batchstats_fwd;
  int fuse_batchstats_bwd;
  int fuse_eltwise_bwd;
  int fuse_relu_bwd;
  int use_lp_kernel;
  int output_lp_padding;
  int reduce_weights;
  int use_vperm_transposes;
  int avoid_output_trans;
  int avoid_input_trans;
  int enforce_sfma_kernel;
  int n_variants;
  int w_variants;
  int h_variants;
  int loop_order;
  int f32_bf16_cvt_rne;
  int fwd_img_par;
  int use_ofm_parallelization;
  int use_ifm_parallelization;
  int avoid_fmas_in_rim;
  int avoid_init_weights;
  int upd_use_batchreduce;
  int upd_pack_input;
  int upd_img_br_block;
  int upd_loop_order;
  int upd_linearized_tasklist;
  int upd_avoid_rim_fmas;

  /* internal data representation */
  libxs_dnn_tensor* reg_input;
  libxs_dnn_tensor* reg_output;
  libxs_dnn_tensor* reg_filter;
  libxs_dnn_tensor* grad_input;
  libxs_dnn_tensor* grad_output;
  libxs_dnn_tensor* grad_filter;
  libxs_dnn_tensor* reg_bias;
  libxs_dnn_tensor* grad_bias;
  /* internal data representations for copies of tensors */
  libxs_dnn_tensor* reg_input_tr;
  libxs_dnn_tensor* reg_filter_tr;
  /* batchnorm stats */
  libxs_dnn_tensor* batch_stats;
  /* maxstats used in low-precision kernels */
  libxs_dnn_tensor* maxstats_fwd;
  libxs_dnn_tensor* maxstats_bwd;
  libxs_dnn_tensor* maxstats_upd;

  /* barrier */
  libxs_barrier* barrier;

  /* scratch */
  void* scratch1;
  size_t scratch1_size;
  void* scratch2;
  size_t scratch2_size;
  void* scratch3;
  size_t scratch3_size;
  void* scratch4;             /* TLS: used to reduce weights */
  size_t scratch4_size;
  void* scratch5;             /* TLS: copy-buffer (if padding is needed), or [H][W][c-block]-tensor (generic FWD/BWD) */
  size_t max_scratch5_size;
  void* scratch6;             /* TLS: output_scratch (generic WU), or float-accumulation buffer */
  size_t scratch6_size;
  void* scratch7;             /* TLS: filter_scratch (generic WU) */
  size_t scratch7_size;
  size_t minibatch_scratch_size;
  size_t fwdbwd_scratch_size;
  int padding_flag;           /* Flag that dictates if we should apply padding in the input */
  void* scratchIw;            /* Winograd input buffer */
  size_t scratchIw_size;
  void* scratchOw;            /* Winograd output buffer */
  size_t scratchOw_size;
  void* scratchVk;            /* Winograd weight buffer */
  size_t scratchVk_size;

  /* JIT-generated convolution code */
  int use_fwd_generic;
  int use_bwd_generic;
  int use_upd_generic;
  /*
  libxs_convolution_forward_descriptor       fwd_desc;
  libxs_convolution_forward_descriptor       bwd_desc;
  libxs_convolution_weight_update_descriptor wu_desc;
  */
  libxs_code_pointer code_fwd[3];
  libxs_code_pointer code_bwd[3];
  libxs_code_pointer code_upd[2];

  libxs_code_pointer matcopy_fwd[4];
  libxs_code_pointer matcopy_bwd[4];
  libxs_code_pointer matcopy_upd[3];

  /* Data structures and metadata related to per-thread private JITing */
  int trans_ofw_ifm;

  int *n_entries_fwd;
  int **compute_fwd_indices_ptrs;
  int **bn_stats_indices_ptrs;
  int **bn_aux_stats_indices_ptrs;
  int **bn_aux_input_indices_ptrs;
  char **kernel_fwd_variant_ptrs;
  int block_fwd_oj;
  int block_fwd_oi;
  int block_fwd_ifm;
  int block_fwd_ofm;
  int *n_fwd_code_segments;
  segment_t **fwd_code_segments;
  int *ofh_fwd_start;
  int *ofh_fwd_end;

  int *n_entries_bwd;
  int **compute_bwd_indices_ptrs;
  char **kernel_bwd_variant_ptrs;
  int block_bwd_oj;
  int block_bwd_oi;
  int block_bwd_ifm;
  int block_bwd_ofm;
  int *n_bwd_code_segments;
  segment_t **bwd_code_segments;
  int *n_entries_trans_bwd;
  int **transpose_bwd_indices_ptrs;
  int *ofh_bwd_start;
  int *ofh_bwd_end;

  int *n_entries_upd;
  int block_upd_ifm;
  int block_upd_ofm;
  int **compute_upd_indices_ptrs;
  char **kernel_upd_variant_ptrs;
  int *n_upd_code_segments;
  segment_t **upd_code_segments;
  int *n_entries_init_upd;
  int **init_upd_indices_ptrs;
  int *n_entries_copy_upd;
  int **copy_upd_indices_ptrs;
};

LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_dnn_fusedbatchnorm {
  libxs_dnn_fusedbatchnorm_desc desc;
  libxs_dnn_tensor* reg_input;      /* input tensor */
  libxs_dnn_tensor* reg_output;     /* output tensor */
  libxs_dnn_tensor* grad_input;     /* grad input tensor */
  libxs_dnn_tensor* grad_output;    /* grad output tensor */
  libxs_dnn_tensor* reg_add;        /* elementwise tensor */
  libxs_dnn_tensor* grad_add;       /* grad elementwise tensor */
  libxs_dnn_tensor* reg_beta;       /* beta tensor */
  libxs_dnn_tensor* reg_gamma;      /* gamma tensor */
  libxs_dnn_tensor* grad_beta;      /* grad beta tensor */
  libxs_dnn_tensor* grad_gamma;     /* grad gamma tensor */
  libxs_dnn_tensor* expvalue;       /* expected value */
  libxs_dnn_tensor* rcpstddev;      /* reciprocal of standard derivation */
  libxs_dnn_tensor* variance;       /* variance */
  libxs_barrier* barrier;           /* barrier */
  int ifmblock;
  int ifmblock_hp;
  int ofmblock;
  int ofmblock_lp;
  int blocksifm;
  int blocksofm;
  int blocksifm_lp;  /* not used */
  int blocksofm_lp;  /* not used */
  int fm_lp_block;
  size_t scratch_size;
  void* scratch;
};

LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_dnn_fullyconnected {
  libxs_dnn_fullyconnected_desc desc;
  libxs_dnn_tensor* reg_input;      /* input tensor */
  libxs_dnn_tensor* reg_output;     /* output tensor */
  libxs_dnn_tensor* grad_input;     /* grad input tensor */
  libxs_dnn_tensor* grad_output;    /* grad output tensor */
  libxs_dnn_tensor* reg_filter;     /* filter tensor */
  libxs_dnn_tensor* grad_filter;    /* grad filter tensor */
  libxs_barrier* barrier;           /* barrier */
  int ifmblock;
  int ifmblock_hp;
  int ofmblock;
  int ofmblock_lp;
  int blocksifm;
  int blocksofm;
  int blocksifm_lp;  /* not used */
  int blocksofm_lp;  /* not used */
  int fm_lp_block;
  int bn;
  int bk;
  int bc;
  size_t scratch_size;
  void* scratch;
};

LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_dnn_pooling {
  libxs_dnn_pooling_desc desc;
  libxs_dnn_tensor* reg_input;      /* input tensor */
  libxs_dnn_tensor* reg_output;     /* output tensor */
  libxs_dnn_tensor* grad_input;     /* grad input tensor */
  libxs_dnn_tensor* grad_output;    /* grad output tensor */
  libxs_dnn_tensor* mask;           /* elementwise tensor */
  libxs_barrier* barrier;           /* barrier */
  int ifmblock;
  int ifmblock_hp;
  int ofmblock;
  int ofmblock_lp;
  int blocksifm;
  int blocksofm;
  int blocksifm_lp;  /* not used */
  int blocksofm_lp;  /* not used */
  int fm_lp_block;
  int ofh;
  int ofw;
  size_t scratch_size;
  void* scratch;
};

LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_dnn_rnncell {
  libxs_dnn_rnncell_desc desc;
  libxs_dnn_internal_format custom_format_type; /* required only for comparing layouts  */
  libxs_blasint T;                              /* sequnece length, must be smaller than max sequence length in desc */
  libxs_blasint bk;
  libxs_blasint bn;
  libxs_blasint bc;
  /* extrenal tensors */
  libxs_dnn_tensor* xt;
  libxs_dnn_tensor* csp;
  libxs_dnn_tensor* hp;
  libxs_dnn_tensor* w;
  libxs_dnn_tensor* wt;
  libxs_dnn_tensor* r;
  libxs_dnn_tensor* rt;
  libxs_dnn_tensor* b;
  libxs_dnn_tensor* cst;
  libxs_dnn_tensor* ht;
  libxs_dnn_tensor* dxt;
  libxs_dnn_tensor* dcsp;
  libxs_dnn_tensor* dhp;
  libxs_dnn_tensor* dw;
  libxs_dnn_tensor* dr;
  libxs_dnn_tensor* db;
  libxs_dnn_tensor* dcs;
  libxs_dnn_tensor* dht;
  libxs_dnn_tensor* it;
  libxs_dnn_tensor* ft;
  libxs_dnn_tensor* ot;
  libxs_dnn_tensor* cit;
  libxs_dnn_tensor* cot;
  float forget_bias;
  /* internal  state */
  void* internal_z;
  /* scratch pointers */
  void* scratch_base;
  void* scratch_wT;
  void* scratch_rT;
  void* scratch_w;
  void* scratch_r;
  void* scratch_xT;
  void* scratch_hT;
  void* scratch_deltat;
  void* scratch_di;
  void* scratch_df;
  void* scratch_do;
  void* scratch_dci;
  void* scratch_diB;
  void* scratch_dfB;
  void* scratch_dpB;
  void* scratch_dciB;
  void* scratch_dx;
  void* scratch_dhp;
  void* scratch_db;
  void* scratch_t1;
  void* scratch_t2;
  void* csp_scratch;
  void* cst_scratch;
  void* ht_scratch;
  void* it_scratch;
  void* ft_scratch;
  void* ot_scratch;
  void* cit_scratch;
  void* cot_scratch;
  /* options */
  int fwd_generic;
  int bwdupd_generic;
  /* barrier */
  libxs_barrier* barrier; /* barrier */
};

struct LIBXS_RETARGETABLE libxs_dfsspmdm {
  int M;
  int N;
  int K;
  int ldb;
  int ldc;
  int N_chunksize;
  double* a_dense;
  libxs_dmmfunction kernel;
};

struct LIBXS_RETARGETABLE libxs_sfsspmdm {
  int M;
  int N;
  int K;
  int ldb;
  int ldc;
  int N_chunksize;
  float* a_dense;
  libxs_smmfunction kernel;
};

typedef enum libxs_build_kind {
  LIBXS_BUILD_KIND_GEMM     = LIBXS_KERNEL_KIND_MATMUL,
  LIBXS_BUILD_KIND_MCOPY    = LIBXS_KERNEL_KIND_MCOPY,
  LIBXS_BUILD_KIND_TRANS    = LIBXS_KERNEL_KIND_TRANS,
  LIBXS_BUILD_KIND_TRSM     = LIBXS_KERNEL_KIND_TRSM,
  LIBXS_BUILD_KIND_TRMM     = LIBXS_KERNEL_KIND_TRMM,
  LIBXS_BUILD_KIND_RMACSOA  = LIBXS_KERNEL_KIND_INVALID,
  LIBXS_BUILD_KIND_RMBCSOA,
  LIBXS_BUILD_KIND_SRSOA,
  LIBXS_BUILD_KIND_SCSOA,
  LIBXS_BUILD_KIND_SREG,
  LIBXS_BUILD_KIND_CFWD,
  LIBXS_BUILD_KIND_CUPD,
  LIBXS_BUILD_KIND_CWFWD,
  LIBXS_BUILD_KIND_CWBWD,
  LIBXS_BUILD_KIND_CWUPD
} libxs_build_kind;

LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_build_descriptor {
  const libxs_gemm_descriptor* gemm;
  const libxs_csr_soa_descriptor* srsoa;
  const libxs_csc_soa_descriptor* scsoa;
  const libxs_rm_ac_soa_descriptor* rmacsoa;
  const libxs_rm_bc_soa_descriptor* rmbcsoa;
  const libxs_csr_reg_descriptor* sreg;
  const libxs_convolution_forward_descriptor* cfwd;
  const libxs_convolution_weight_update_descriptor* cupd;
  const libxs_convolution_winograd_descriptor* cwino;
  const libxs_mcopy_descriptor* matcopy;
  const libxs_trans_descriptor* trans;
  const libxs_trsm_descriptor* trsm;
  const libxs_trmm_descriptor* trmm;
} libxs_build_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_build_request {
  libxs_build_descriptor descriptor;
  libxs_build_kind kind;
} libxs_build_request;

typedef enum libxs_malloc_flags {
  LIBXS_MALLOC_FLAG_DEFAULT = 0,
  LIBXS_MALLOC_FLAG_SCRATCH = 1,
  LIBXS_MALLOC_FLAG_PRIVATE = 2,
  LIBXS_MALLOC_FLAG_MMAP    = 4,
  LIBXS_MALLOC_FLAG_R       = 8,
  LIBXS_MALLOC_FLAG_W       = 16,
  LIBXS_MALLOC_FLAG_X       = 32,
  LIBXS_MALLOC_FLAG_RW  = LIBXS_MALLOC_FLAG_R | LIBXS_MALLOC_FLAG_W,
  LIBXS_MALLOC_FLAG_WX  = LIBXS_MALLOC_FLAG_X | LIBXS_MALLOC_FLAG_W,
  LIBXS_MALLOC_FLAG_RWX = LIBXS_MALLOC_FLAG_X | LIBXS_MALLOC_FLAG_RW
} libxs_malloc_flags;

/** Returns the type-size of data-type (can be also libxs_gemm_precision). */
LIBXS_API unsigned char libxs_typesize(libxs_datatype datatype);

/** Returns the type-name of data-type (can be also libxs_gemm_precision). */
LIBXS_API const char* libxs_typename(libxs_datatype datatype);

/** Determines the generic value given in double-precision. */
LIBXS_API int libxs_cast(libxs_datatype datatype, double dvalue, void* value);

/** Retrieve internal information about a buffer (default memory domain). */
LIBXS_API int libxs_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra);

/** Calculates an alignment depending on supposedly allocated size; alignment can be zero ("auto"). */
LIBXS_API_INTERN size_t libxs_alignment(size_t size, size_t alignment);

/** Same as libxs_set_default_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xset_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Same as libxs_get_default_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xget_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/** Same as libxs_set_scratch_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xset_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Same as libxs_get_scratch_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xget_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/**
 * Attribute memory allocation and protect with only the necessary flags.
 * This procedure is expected to run only one time per buffer, and may
 * relocate the given memory.
 */
LIBXS_API_INTERN int libxs_malloc_attrib(void** memory, int flags,
  /** If a name is given, an executable buffer will be dumped into a file. */
  const char* name);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXS_API_INTERN int libxs_xmalloc(void** memory, size_t size, size_t alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
/** Release memory, which was allocated using libxs_[*]malloc. */
LIBXS_API_INTERN int libxs_xfree(const void* memory);

/** Determines the given value in double-precision based on the given type. */
LIBXS_API_INTERN int libxs_dvalue(libxs_datatype datatype, const void* value, double* dvalue);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXS_CAPACITY_REGISTRY for unmanaged code). */
LIBXS_API_INTERN int libxs_build(const libxs_build_request* request, unsigned int regindex, libxs_code_pointer* code);

LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_kernel_info {
  libxs_gemm_descriptor xgemm;
  libxs_mcopy_descriptor mcopy;
  libxs_trans_descriptor trans;
  libxs_trsm_descriptor trsm;
  libxs_trmm_descriptor trmm;
} libxs_kernel_info;

/** Attempts to receive information about JIT-generated code. */
LIBXS_API const libxs_kernel_info* libxs_get_kernel_info(libxs_code_pointer code, libxs_kernel_kind* kind, size_t* size);

/** Updates counters of the statistic, which is shown at program termination. */
LIBXS_API unsigned int libxs_update_mmstatistic(libxs_gemm_precision precision,
  libxs_blasint m, libxs_blasint n, libxs_blasint k, unsigned int ntry, unsigned int ncol);

/** Returns the current tick of a (monotonic) platform-specific counter; not necessarily CPU cycles. */
LIBXS_API_INTERN libxs_timer_tickint libxs_timer_tick_rtc(void);

LIBXS_API_INTERN void libxs_dnn_init(int target_arch);
LIBXS_API_INTERN void libxs_dnn_finalize(void);

/** Code generation routine for a forward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API_INTERN libxs_sconvfunction libxs_create_sconv_forward(const libxs_convolution_forward_descriptor* descriptor);

/** Code generation routine for a backward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API_INTERN libxs_sconvfunction libxs_create_sconv_backward(const libxs_convolution_backward_descriptor* descriptor);

/** Code generation routine for a convolution kernel as specified by descriptor. */
LIBXS_API_INTERN libxs_sconvfunction libxs_create_sconv_update_weights(const libxs_convolution_weight_update_descriptor* descriptor);

/** Code generation routine for a forward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API_INTERN void* libxs_create_xconv_forward(const libxs_convolution_forward_descriptor* descriptor);

/** Code generation routine for a backward-convolution kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API_INTERN void* libxs_create_xconv_backward(const libxs_convolution_backward_descriptor* descriptor);

/** Code generation routine for a convolution kernel as specified by descriptor. */
LIBXS_API_INTERN void* libxs_create_xconv_update_weights(const libxs_convolution_weight_update_descriptor* descriptor);

/** Code generation routine for a forward-convolution Winograd kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API_INTERN void* libxs_create_xconv_wino_forward(const libxs_convolution_winograd_descriptor* descriptor);

/** Code generation routine for a backward-convolution Winograd kernel. Call libxs_release_kernel in order to deallocate the JIT'ted code. */
LIBXS_API_INTERN void* libxs_create_xconv_wino_backward(const libxs_convolution_winograd_descriptor* descriptor);

/** Code generation routine for a weight-update-convolution Winograd kernel as specified by descriptor. */
LIBXS_API_INTERN void* libxs_create_xconv_wino_update_weights(const libxs_convolution_winograd_descriptor* descriptor);

/** Global lock; create an own lock for an independent domain. */
LIBXS_APIVAR_ALIGNED(LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_lock_global);
/** Target architecture (libxs_get_target_archid, libxs_set_target_archid). */
LIBXS_APIVAR_ALIGNED(int libxs_target_archid);
/** Determines whether a threaded implementation is synchronized or not. */
LIBXS_APIVAR_ALIGNED(int libxs_nosync);
/** Number of threads per core. */
LIBXS_APIVAR_ALIGNED(int libxs_nt);

/** Function used to allocate default memory. */
LIBXS_APIVAR(libxs_malloc_function libxs_default_malloc_fn);
/** Function used to allocate scratch memory. */
LIBXS_APIVAR(libxs_malloc_function libxs_scratch_malloc_fn);
/** Function used to release default memory. */
LIBXS_APIVAR(libxs_free_function libxs_default_free_fn);
/** Function used to release scratch memory. */
LIBXS_APIVAR(libxs_free_function libxs_scratch_free_fn);
/** If non-NULL, this context is used by the context-form of memory allocation. */
LIBXS_APIVAR(void* libxs_default_allocator_context);
/** If non-NULL, this context is used by the context-form of memory allocation. */
LIBXS_APIVAR(void* libxs_scratch_allocator_context);
/** Number of discovered threads (per libxs_get_tid) */
LIBXS_APIVAR(unsigned int libxs_threads_count);
/** Number of scratch memory pools used; clamped against internal maximum. */
LIBXS_APIVAR(unsigned int libxs_scratch_pools);
/** Maximum total size of the scratch memory domain. */
LIBXS_APIVAR(size_t libxs_scratch_limit);
/** Growth factor used to scale the scratch memory in case of reallocation. */
LIBXS_APIVAR(double libxs_scratch_scale);
/** Number of seconds per RDTSC-cycle (zero if RDTSC is not used for wall-clock) */
LIBXS_APIVAR(double libxs_timer_scale);

#endif /*LIBXS_MAIN_H*/
