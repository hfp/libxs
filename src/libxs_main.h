/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MAIN_H
#define LIBXS_MAIN_H

#include <libxs.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXS_CAPACITY_REGISTRY) /* must be POT */
# define LIBXS_CAPACITY_REGISTRY 131072
#endif
#if !defined(LIBXS_CAPACITY_CACHE) /* must be POT */
# define LIBXS_CAPACITY_CACHE 16
#endif

#if !defined(LIBXS_PAGE_MINSIZE)
# define LIBXS_PAGE_MINSIZE 4096 /* 4 KB */
#endif

#if !defined(LIBXS_NTHREADS_MAX)
# if (0 != LIBXS_SYNC)
#   define LIBXS_NTHREADS_MAX 1024
# else
#   define LIBXS_NTHREADS_MAX 1
# endif
#endif
/* code relies on LIBXS_NTHREADS_MAX or v/forks */
#if !defined(LIBXS_NTHREADS_USE) && 1
# define LIBXS_NTHREADS_USE
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS)
# define LIBXS_MALLOC_SCRATCH_MAX_NPOOLS LIBXS_NTHREADS_MAX
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_SCALE)
# define LIBXS_MALLOC_SCRATCH_SCALE 1.0
#endif
#if !defined(LIBXS_MALLOC_LIMIT)
# define LIBXS_MALLOC_LIMIT (2U << 20) /* 2 MB */
#endif
#if !defined(LIBXS_MALLOC_INTERNAL_CALLER_ID)
# define LIBXS_MALLOC_INTERNAL_CALLER_ID ((uintptr_t)LIBXS_UNLIMITED)
#endif
#if !defined(LIBXS_MALLOC_INTERNAL_CALLER)
# define LIBXS_MALLOC_INTERNAL_CALLER ((const void*)(LIBXS_MALLOC_INTERNAL_CALLER_ID))
#endif

#if !defined(LIBXS_INTERCEPT_DYNAMIC) && defined(LIBXS_BUILD) && \
  (defined(__GNUC__) || defined(_CRAYC)) && !defined(_WIN32) && !defined(__CYGWIN__) && \
  !(defined(__APPLE__) && defined(__MACH__) && LIBXS_VERSION3(6, 1, 0) >= \
    LIBXS_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
# define LIBXS_INTERCEPT_DYNAMIC
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
# define LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K) (0xFFFFFFFF >= (M) && 0xFFFFFFFF >= (N) && 0xFFFFFFFF >= (K))
#else /* always fits */
# define LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K) 1
#endif

#if defined(LIBXS_ASSERT) /* assert available */
# define LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K) LIBXS_ASSERT(LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K))
#else
# define LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K)
#endif

#if defined(LIBXS_UNPACKED)
# define LIBXS_DESCRIPTOR_CLEAR_AUX(DST, SIZE) LIBXS_MEMSET127(DST, 0, SIZE)
#else
# define LIBXS_DESCRIPTOR_CLEAR_AUX(DST, SIZE)
#endif
#define LIBXS_DESCRIPTOR_CLEAR(BLOB) \
  LIBXS_ASSERT((LIBXS_DESCRIPTOR_MAXSIZE) == sizeof(*(BLOB))); \
  LIBXS_DESCRIPTOR_CLEAR_AUX(BLOB, LIBXS_DESCRIPTOR_MAXSIZE)

/** Low-level/internal GEMM descriptor initialization. */
#define LIBXS_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(LDA, LDB, LDC); \
  LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K); \
  LIBXS_DESCRIPTOR_CLEAR_AUX(&(DESCRIPTOR), sizeof(DESCRIPTOR)); \
  (DESCRIPTOR).datatype = (unsigned char)(DATA_TYPE); (DESCRIPTOR).prefetch = (unsigned char)(PREFETCH); \
  (DESCRIPTOR).flags = (unsigned int)((FLAGS) \
    /*| (LIBXS_NEQ(0, ALPHA) ? 0 : LIBXS_GEMM_FLAG_ALPHA_0)*/ \
    | (LIBXS_NEQ(0, BETA) ? 0 : LIBXS_GEMM_FLAG_BETA_0)); \
  (DESCRIPTOR).m   = (unsigned int)(M);   (DESCRIPTOR).n   = (unsigned int)(N);   (DESCRIPTOR).k   = (unsigned int)(K); \
  (DESCRIPTOR).lda = (unsigned int)(LDA); (DESCRIPTOR).ldb = (unsigned int)(LDB); (DESCRIPTOR).ldc = (unsigned int)(LDC); \
  LIBXS_PAD((DESCRIPTOR).pad = 0) (DESCRIPTOR).c1 = 0; (DESCRIPTOR).c2 = 0; (DESCRIPTOR).c3 = 0

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

#define LIBXS_REGDESC_DEFAULT
#define LIBXS_REGDESC(START, MODIFIER) \
  START libxs_gemm_descriptor MODIFIER gemm; \
  START libxs_mcopy_descriptor MODIFIER mcopy; \
  START libxs_trans_descriptor MODIFIER trans; \
  START libxs_pgemm_descriptor MODIFIER pgemm; \
  START libxs_getrf_descriptor MODIFIER getrf; \
  START libxs_trmm_descriptor MODIFIER trmm; \
  START libxs_trsm_descriptor MODIFIER trsm


/**
* Packed structure, which stores the argument description of GEMM routines.
* The size of the structure is padded to LIBXS_DESCRIPTOR_MAXSIZE.
*/
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_gemm_descriptor {
  /** Extents of the matrix. */
  unsigned int m, n, k;
  /** Leading dimensions. */
  unsigned int lda, ldb, ldc;
  /** Set of flags. */
  unsigned int flags;
  /** Prefetch strategy. */
  unsigned char prefetch;
  /** Denotes the data-type. */
  unsigned char datatype;
  /** Ignored entry. */
  LIBXS_PAD(unsigned short pad)
  /** multipurpose 64bit field, currently used for: a) stride_a in brgemm */
  unsigned long long c1;
  /** multipurpose 64bit field, currently used for: a) stride_b in brgemm */
  unsigned long long c2;
  /** multipurpose 8bit field, currently used for: a) unroll hint in brgemm */
  unsigned char c3;
};

/** Packed structure storing the matcopy argument description. */
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_mcopy_descriptor {
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

/** Packed structure storing the transpose argument description. */
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_trans_descriptor {
  /** LD, M, and N. */
  unsigned int m, n, ldo;
  /** Size of data element. */
  unsigned char typesize;
};

/** Packed structure storing arguments of packed GEMM. */
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_pgemm_descriptor {
  unsigned int m, n, k, lda, ldb, ldc;
  unsigned char typesize;
  unsigned char layout;
  char transa, transb;
  char alpha_val;
};

/** Packed structure storing arguments of packed GETRF. */
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_getrf_descriptor {
  unsigned int m, n, lda;
  unsigned char typesize;
  unsigned char layout;
};

/** Packed structure storing arguments of packed TRSM. */
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_trmm_descriptor {
  union { double d; float s; } alpha;
  unsigned int m, n, lda, ldb;
  unsigned char typesize;
  unsigned char layout;
  char diag, side, uplo;
  char transa;
};

/** Packed structure storing arguments of packed TRSM. */
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_trsm_descriptor {
  union { double d; float s; } alpha;
  unsigned int m, n, lda, ldb;
  unsigned char typesize;
  unsigned char layout;
  char diag, side, uplo;
  char transa;
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

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_pgemm_ac_rm_descriptor {
  const libxs_gemm_descriptor* gemm;
  unsigned int packed_width;
} libxs_pgemm_ac_rm_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_pgemm_bc_rm_descriptor {
  const libxs_gemm_descriptor* gemm;
  unsigned int packed_width;
} libxs_pgemm_bc_rm_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_csr_reg_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxs_csr_reg_descriptor;

LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_code_pointer {
  void (*ptr_fn)(LIBXS_VARIADIC);
  const void* ptr_const;
  void* ptr;
  uintptr_t uval;
  intptr_t ival;
  libxs_xmmfunction xgemm; /* GEMM: smm, dmm, wimm, or void-function */
  libxs_xmcopyfunction xmatcopy;
  libxs_xtransfunction xtrans;
  libxs_pgemm_xfunction xpgemm;
  libxs_getrf_xfunction xgetrf;
  libxs_trmm_xfunction xtrmm;
  libxs_trsm_xfunction xtrsm;
} libxs_code_pointer;

/** Structure which describes all tensors in LIBXS's DNN module */
LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_dnn_tensor {
  libxs_dnn_tensor_datalayout* layout;           /* data-layout descriptor */
  void* data;                                      /* pointer to data */
  unsigned char scf;                               /* fix point scaling factor for this tensor */
};

/* Structure to record segment in stream of code */
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

  /* These are the batchnorm handles in case of fusion */
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
  int ofmblock;
  int blocksifm;
  int blocksofm;
  int fwd_ofw_rb;
  int fwd_ofw_rb_2;
  int fwd_ofh_rb;
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
  int avoid_acc_load;
  int avoid_acc_load_bwd;
  int pack_input;
  int pack_input_bwd;
  int spread_input_bwd;
  int weight_copies;
  int fuse_batchstats_fwd;
  int fuse_batchstats_bwd;
  int fuse_eltwise_bwd;
  int fuse_relu_bwd;
  int use_lp_kernel;
  int use_vperm_transposes;
  int loop_order;
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
  int fwd_flags;
  int shuffle_filter_accesses;
  int use_fallback_fwd_loops;
  int use_fallback_bwd_loops;
  int input_pixels;
  int output_pixels;
  int n_used_pixels;
  int pixel_blocking;
  int use_intermediate_f32_wt_tensor;
  int upd_linearized_pixels;
  int ifwp_extended;
  int ofwp_extended;
  int batchreduce_h_pixels;
  int on_the_fly_input_packing;
  int upd_pack_input_upfront;
  int use_hybrid_imgofm_parallelization;
  int compute_pixels;
  int upd_trans_w_only;

  libxs_xtransfunction tr_kernel;

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

  libxs_code_pointer gemm_fwd;     /* ability to hoist forward GEMMs */
  libxs_code_pointer gemm_fwd2;     /* ability to hoist forward GEMMs */
  unsigned long long *A_offsets;
  unsigned long long *B_offsets;

  /* JIT-generated convolution code */
  libxs_code_pointer code_fwd[3];
  libxs_code_pointer code_bwd[3];
  libxs_code_pointer code_upd[2];

  libxs_code_pointer matcopy_fwd[4];
  libxs_code_pointer matcopy_bwd[4];
  libxs_code_pointer matcopy_upd[3];

  /* Data structures and metadata related to per-thread private JITing */
  int block_fwd_oj;
  int block_fwd_ifm;
  int block_fwd_ofm;
  int block_bwd_oj;
  int block_bwd_ifm;
  int block_bwd_ofm;
  int block_upd_ifm;
  int block_upd_ofm;
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
  libxs_dnn_tensor* relumask;       /* relumask */
  libxs_barrier* barrier;           /* barrier */
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  size_t scratch_size;
  void* scratch;
};

LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_dnn_fusedgroupnorm {
  libxs_dnn_fusedgroupnorm_desc desc;
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
  libxs_dnn_tensor* relumask;       /* relumask */
  libxs_barrier* barrier;           /* barrier */
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
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
  libxs_dnn_tensor* reg_bias;       /* bias tensor */
  libxs_dnn_tensor* grad_bias;      /* grad bais tensor */
  libxs_dnn_tensor* relumask;       /* relumask */
  libxs_barrier* barrier;           /* barrier */
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  /* Parameters to tune/specialize FC algorithms */
  int fwd_2d_blocking;
  int bwd_2d_blocking;
  int upd_2d_blocking;
  int fwd_bf;
  int bwd_bf;
  int upd_bf;
  int fwd_row_teams;
  int fwd_column_teams;
  int bwd_row_teams;
  int bwd_column_teams;
  int upd_row_teams;
  int upd_column_teams;
  int ifm_subtasks;
  int ofm_subtasks;

  int fm_lp_block;
  int bn;
  int bk;
  int bc;
  size_t scratch_size;
  void* scratch;

  libxs_code_pointer gemm_fwd;     /* ability to hoist forward GEMMs */
  libxs_code_pointer gemm_fwd2;     /* ability to hoist forward GEMMs */
  libxs_code_pointer gemm_bwd;     /* ability to hoist backward GEMMs */
  libxs_code_pointer gemm_bwd2;    /* ability to hoist backward GEMMs */
  libxs_code_pointer gemm_upd;     /* ability to hoist update GEMMs */
  libxs_code_pointer gemm_upd2;    /* ability to hoist update GEMMs */
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
  int ofmblock;
  int blocksifm;
  int blocksofm;
  int ofh;
  int ofw;
  size_t scratch_size;
  void* scratch;
};

LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_dnn_rnncell {
  libxs_dnn_rnncell_desc desc;
  libxs_blasint T;                              /* sequence length, must be smaller than max sequence length in desc */
  libxs_blasint bk;
  libxs_blasint bn;
  libxs_blasint bc;
  libxs_blasint lpb;

  /* external tensors */
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

  /* Ability to hoist GEMMs */
  libxs_bsmmfunction_reducebatch_strd fwd_kernela;
  libxs_bsmmfunction_reducebatch_strd fwd_kernelb;
  libxs_bsmmfunction_reducebatch_strd bwdupd_kernela;
  libxs_bsmmfunction_reducebatch_strd bwdupd_kernelb;
  libxs_bsmmfunction_reducebatch_strd bwdupd_kernelc;
  libxs_bsmmfunction_reducebatch_strd bwdupd_kerneld;

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
  LIBXS_BUILD_KIND_GEMM       = LIBXS_KERNEL_KIND_MATMUL,
  LIBXS_BUILD_KIND_MCOPY      = LIBXS_KERNEL_KIND_MCOPY,
  LIBXS_BUILD_KIND_TRANS      = LIBXS_KERNEL_KIND_TRANS,
  LIBXS_BUILD_KIND_PGEMM      = LIBXS_KERNEL_KIND_PGEMM,
  LIBXS_BUILD_KIND_GETRF      = LIBXS_KERNEL_KIND_GETRF,
  LIBXS_BUILD_KIND_TRMM       = LIBXS_KERNEL_KIND_TRMM,
  LIBXS_BUILD_KIND_TRSM       = LIBXS_KERNEL_KIND_TRSM,
  LIBXS_BUILD_KIND_PGEMMRMAC  = LIBXS_KERNEL_UNREGISTERED,
  LIBXS_BUILD_KIND_PGEMMRMBC,
  LIBXS_BUILD_KIND_SRSOA,
  LIBXS_BUILD_KIND_SCSOA,
  LIBXS_BUILD_KIND_SREG
} libxs_build_kind;

/** Integral type (libxs_kernel_kind, libxs_build_kind). */
#if defined(LIBXS_UNPACKED)
typedef size_t libxs_descriptor_kind;
#else
typedef unsigned char libxs_descriptor_kind;
#endif

/** All descriptor types, which are valid for code-registration. */
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_descriptor {
  char data[LIBXS_DESCRIPTOR_MAXSIZE];
  libxs_descriptor_kind kind; /* kind: must be the first member */
  LIBXS_REGDESC(LIBXS_PACKED(struct) { libxs_descriptor_kind /*repeated kind*/ pad; , desc; });
} libxs_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_build_request {
  union {
    const void* ptr; /* raw content */
    LIBXS_REGDESC(LIBXS_REGDESC_DEFAULT, const*);
    const libxs_csr_soa_descriptor* srsoa;
    const libxs_csc_soa_descriptor* scsoa;
    const libxs_pgemm_ac_rm_descriptor* pgemmacrm;
    const libxs_pgemm_bc_rm_descriptor* pgemmbcrm;
    const libxs_csr_reg_descriptor* sreg;
  } descriptor;
  libxs_build_kind kind;
} libxs_build_request;

typedef enum libxs_malloc_flags {
  LIBXS_MALLOC_FLAG_DEFAULT = 0,
  LIBXS_MALLOC_FLAG_SCRATCH = 1,
  LIBXS_MALLOC_FLAG_PRIVATE = 2,
  LIBXS_MALLOC_FLAG_REALLOC = 4,
  LIBXS_MALLOC_FLAG_MMAP    = 8,
  LIBXS_MALLOC_FLAG_R       = 16,
  LIBXS_MALLOC_FLAG_W       = 32,
  LIBXS_MALLOC_FLAG_X       = 64,
  LIBXS_MALLOC_FLAG_RW  = LIBXS_MALLOC_FLAG_R | LIBXS_MALLOC_FLAG_W,
  LIBXS_MALLOC_FLAG_WX  = LIBXS_MALLOC_FLAG_X | LIBXS_MALLOC_FLAG_W,
  LIBXS_MALLOC_FLAG_RWX = LIBXS_MALLOC_FLAG_X | LIBXS_MALLOC_FLAG_RW,
  LIBXS_MALLOC_FLAG_VALID       = LIBXS_MALLOC_FLAG_SCRATCH |
      LIBXS_MALLOC_FLAG_PRIVATE | LIBXS_MALLOC_FLAG_REALLOC |
      LIBXS_MALLOC_FLAG_MMAP    | LIBXS_MALLOC_FLAG_RWX
} libxs_malloc_flags;

/** Format for instance an amount of Bytes like libxs_format_size(nbytes, "KMGT", "B", 10). */
LIBXS_API_INTERN const char* libxs_format_size(size_t nbytes, const char scale[], const char* unit, int base);

/** Returns the type-name of data-type (can be also libxs_gemm_precision). */
LIBXS_API_INTERN const char* libxs_typename(libxs_datatype datatype);

/** Returns the type-size of data-type (can be also libxs_gemm_precision). */
LIBXS_API unsigned char libxs_typesize(libxs_datatype datatype);

/** Retrieve internal information about a buffer (default memory domain). */
LIBXS_API int libxs_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra);

/** Initializes malloc hooks and other internals. */
LIBXS_API_INTERN void libxs_malloc_init(void);
LIBXS_API_INTERN void libxs_malloc_finalize(void);

/** Calculates an alignment depending on supposedly allocated size; alignment can be zero ("auto"). */
LIBXS_API_INTERN size_t libxs_alignment(size_t size, size_t alignment);

/** Same as libxs_set_default_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xset_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Same as libxs_get_default_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xget_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/** Same as libxs_set_scratch_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xset_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Same as libxs_get_scratch_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xget_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/** intern function to calculate blockings, that's private API hence it's in this function */
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_get_feature_map_blocks( int C, int K, int* C_block, int* K_block, int* fm_lp_block, libxs_dnn_datatype datatype_in, libxs_dnn_datatype datatype_out );

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
LIBXS_API_INTERN void libxs_xfree(const void* memory, int check);

/** Like libxs_release_scratch, but takes a lock (can be NULL). */
LIBXS_API_INTERN void libxs_xrelease_scratch(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock);

/** Determines the given value in double-precision based on the given type. */
LIBXS_API_INTERN int libxs_dvalue(libxs_datatype datatype, const void* value, double* dvalue);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXS_CAPACITY_REGISTRY for unmanaged code). */
LIBXS_API_INTERN int libxs_build(const libxs_build_request* request, unsigned int regindex, libxs_code_pointer* code);

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_kernel_xinfo {
  /** Non-zero of kernel is registered. */
  unsigned int registered;
  /** Number of FLoating Point OPerationS (FLOPS). */
  unsigned int nflops;
} libxs_kernel_xinfo;

/** Receive information about JIT-generated code. */
LIBXS_API_INTERN const libxs_kernel_xinfo* libxs_get_kernel_xinfo(libxs_code_pointer code, const libxs_descriptor** desc, size_t* code_size);

/** Calculates duration in seconds from given RTC ticks. */
LIBXS_API_INTERN double libxs_timer_duration_rtc(libxs_timer_tickint tick0, libxs_timer_tickint tick1);
/** Returns the current tick of a (monotonic) platform-specific real-time clock. */
LIBXS_API_INTERN libxs_timer_tickint libxs_timer_tick_rtc(void);
/** Returns the current tick of a (monotonic) platform-specific counter. */
LIBXS_API_INTERN libxs_timer_tickint libxs_timer_tick_tsc(void);

LIBXS_API_INTERN void libxs_memory_init(int target_arch);
LIBXS_API_INTERN void libxs_memory_finalize(void);

LIBXS_API_INTERN void libxs_dnn_init(int target_arch);
LIBXS_API_INTERN void libxs_dnn_finalize(void);

/** Global lock; create an own lock for an independent domain. */
LIBXS_APIVAR_ALIGNED(LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_lock_global);
/** Target architecture (libxs_get_target_archid, libxs_set_target_archid). */
LIBXS_APIVAR_ALIGNED(int libxs_target_archid);
/** Determines whether a threaded implementation is synchronized or not. */
LIBXS_APIVAR_ALIGNED(int libxs_nosync);

/** Function used to allocate default memory. */
LIBXS_APIVAR(libxs_malloc_function libxs_default_malloc_fn);
/** Function used to allocate scratch memory. */
LIBXS_APIVAR(libxs_malloc_function libxs_scratch_malloc_fn);
/** Function used to release default memory. */
LIBXS_APIVAR(libxs_free_function libxs_default_free_fn);
/** Function used to release scratch memory. */
LIBXS_APIVAR(libxs_free_function libxs_scratch_free_fn);
/** If non-NULL, this context is used by the context-form of memory allocation. */
LIBXS_APIVAR(const void* libxs_default_allocator_context);
/** If non-NULL, this context is used by the context-form of memory allocation. */
LIBXS_APIVAR(const void* libxs_scratch_allocator_context);
/** Number of scratch memory pools used; clamped against internal maximum. */
LIBXS_APIVAR(unsigned int libxs_scratch_pools);
/** Growth factor used to scale the scratch memory in case of reallocation. */
LIBXS_APIVAR(double libxs_scratch_scale);
/** Counts the number of attempts to create an SPMDM-handle. */
LIBXS_APIVAR(unsigned int libxs_statistic_num_spmdm);
/** Number of seconds per RDTSC-cycle (zero or negative if RDTSC is not constant/available). */
LIBXS_APIVAR(double libxs_timer_scale);
/** Security-enhanced environment. */
LIBXS_APIVAR(int libxs_se);

#endif /*LIBXS_MAIN_H*/
