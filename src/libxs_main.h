/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
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

#include <libxs_typedefs.h>
#include <libxs_generator.h>
#include <libxs_malloc.h>
#include <libxs_sync.h>
#include <libxs_dnn.h>

#include <stddef.h>
#include <stdint.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXS_CAPACITY_REGISTRY) /* must be POT */
# define LIBXS_CAPACITY_REGISTRY 524288 /* 524287: Mersenne Prime number (2^19-1) */
#endif

#if !defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS)
# define LIBXS_MALLOC_SCRATCH_MAX_NPOOLS 16
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_SCALE)
# define LIBXS_MALLOC_SCRATCH_SCALE 1.4
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_INTERNAL)
# define LIBXS_MALLOC_SCRATCH_INTERNAL ((const void*)-1)
#endif

#if !defined(LIBXS_CACHELINE_SIZE)
# define LIBXS_CACHELINE_SIZE 64
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

/* Helper macro to eventually (if defined) call libxs_init */
#if !defined(LIBXS_CTOR) && !defined(LIBXS_INIT)
# define LIBXS_INIT libxs_init();
#elif !defined(LIBXS_INIT)
# define LIBXS_INIT
#endif

typedef union LIBXS_RETARGETABLE libxs_code_pointer {
  const void* const_pmm;
  void* pmm;
  uintptr_t uimm;
  intptr_t imm;
  libxs_xmmfunction xmm;
  libxs_smmfunction smm;
  libxs_wmmfunction wmm;
  void (*vmm)(const void* a, const void* b, void* c, ...);
#if defined(LIBXS_BUILD) || defined(LIBXS_DNN_INTERNAL_API)
  libxs_xconvfunction xconv;
#endif
  libxs_xmatcopyfunction xmatcopy;
  libxs_xtransfunction xtrans;
} libxs_code_pointer;

typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_csr_soa_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxs_csr_soa_descriptor;

typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_csr_reg_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxs_csr_reg_descriptor;

/** Structure which describes an activation layer. */
struct LIBXS_RETARGETABLE libxs_dnn_buffer {
  int N;                            /* number of images in mini-batch */
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int H;                            /* height of image */
  int W;                            /* width of image */
  int lpb;                          /* low precision blocking factor */
  int bimg;                         /* size of blocked images */
  libxs_dnn_tensor_format format; /* format of activation buffer */
  libxs_dnn_internal_format custom_format_type;
  libxs_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
  char exp;                         /* fix point exponent for this tensor */
};

/** Structure which describes a bias. */
struct LIBXS_RETARGETABLE libxs_dnn_bias {
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int lpb;                          /* low precision blocking factor */
  libxs_dnn_tensor_format format; /* format of activation buffer */
  libxs_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
  char exp;                         /* fix point exponent for this tensor */
};

/** Structure which describes a filter */
struct LIBXS_RETARGETABLE libxs_dnn_filter {
  int ifmb;                         /* number of feature map blocks */
  int bifm;                         /* sized of blocked feature maps, in a block */
  int ofmb;                         /* number of feature map blocks */
  int bofm;                         /* sized of blocked feature maps, in a block */
  int R;                            /* height of filter kernel */
  int S;                            /* width of filter kernel */
  int lpb;                          /* low precision blocking factor */
  libxs_dnn_tensor_format format; /* format of filter buffer */
  libxs_dnn_internal_format custom_format_type;
  libxs_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
  char exp;                         /* fix point exponent for this tensor */
};

struct LIBXS_RETARGETABLE libxs_dnn_layer {
  libxs_dnn_datatype datatype;
  libxs_dnn_datatype datatype_itm;
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
  /* additional size for iternal data types */
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
  int fm_lp_block;              /* additional blocking for low precision datatypes of feature maps */
  int upd_use_thread_fil;
  int upd_use_external_reduce;
  int filter_transposed;
  int nBImg;
  int nbImg;

  /* internal data representation */
  libxs_dnn_buffer* reg_input;
  libxs_dnn_buffer* reg_output;
  libxs_dnn_filter* reg_filter;
  libxs_dnn_buffer* grad_input;
  libxs_dnn_buffer* grad_output;
  libxs_dnn_filter* grad_filter;
  libxs_dnn_bias* reg_bias;
  libxs_dnn_bias* grad_bias;

  /* barrier */
  libxs_barrier* barrier;

  /* scratch */
  void* scratch1;
  size_t scratch1_size;
  void* scratch3;
  size_t scratch3_size;
  void* scratch4;
  size_t scratch4_size;
  void* scratch5;             /* This scratch is used as a copy buffer when padding needs to be applied */
  size_t minibatch_scratch_size;
  size_t fwdbwd_scratch_size;
  size_t max_scratch5_size;
  int padding_flag;           /* Flag that dictates if we should apply padding in the input */
  void* scratch6;
  size_t scratch6_size;
  void* scratch7;             /* This scratch is used for low precision intermediate buffer for input in backward pass */
  size_t scratch7_size;
  void* scratchIw;
  size_t scratchIw_size;
  void* scratchOw;
  size_t scratchOw_size;
  void* scratchVk;
  size_t scratchVk_size;
  void* scratchInput;
  size_t scratchInput_size;
  void* scratchTemp;
  int flag_reuseInput;        /* This flag is set to 1 when we want to reuse the input in Winograd domain between forward pass and weight update */
  /* JIT-generated convolution code */
  /*
  libxs_convolution_forward_descriptor       fwd_desc;
  libxs_convolution_forward_descriptor       bwd_desc;
  libxs_convolution_weight_update_descriptor wu_desc;
  */
  int avx512avx2fallback;
  libxs_code_pointer code_fwd[4];
  libxs_code_pointer code_bwd[4];
  libxs_code_pointer code_upd[6];

  libxs_code_pointer matcopy_fwd[1];
  libxs_code_pointer matcopy_bwd[2];
  libxs_code_pointer matcopy_upd[2];
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
  LIBXS_BUILD_KIND_GEMM,
  LIBXS_BUILD_KIND_SSOA,
  LIBXS_BUILD_KIND_SREG,
  LIBXS_BUILD_KIND_CFWD,
  LIBXS_BUILD_KIND_CBWD,
  LIBXS_BUILD_KIND_CUPD,
  LIBXS_BUILD_KIND_CWFWD,
  LIBXS_BUILD_KIND_CWBWD,
  LIBXS_BUILD_KIND_CWUPD,
  LIBXS_BUILD_KIND_MCOPY,
  LIBXS_BUILD_KIND_TRANS
} libxs_build_kind;

typedef union LIBXS_RETARGETABLE libxs_build_descriptor {
  const libxs_gemm_descriptor* gemm;
  const libxs_csr_soa_descriptor* ssoa;
  const libxs_csr_reg_descriptor* sreg;
  const libxs_convolution_forward_descriptor* cfwd;
  const libxs_convolution_backward_descriptor* cbwd;
  const libxs_convolution_weight_update_descriptor* cupd;
  const libxs_convolution_winograd_descriptor* cwino;
  const libxs_matcopy_descriptor* matcopy;
  const libxs_transpose_descriptor* trans;
} libxs_build_descriptor;

typedef struct LIBXS_RETARGETABLE libxs_build_request {
  libxs_build_descriptor descriptor;
  libxs_build_kind kind;
} libxs_build_request;

typedef enum libxs_malloc_flags {
  LIBXS_MALLOC_FLAG_DEFAULT = 0,
  LIBXS_MALLOC_FLAG_SCRATCH = 1,
  LIBXS_MALLOC_FLAG_MMAP = 2,
  LIBXS_MALLOC_FLAG_R = 4,
  LIBXS_MALLOC_FLAG_W = 8,
  LIBXS_MALLOC_FLAG_X = 16,
  LIBXS_MALLOC_FLAG_RW  = LIBXS_MALLOC_FLAG_R | LIBXS_MALLOC_FLAG_W,
  LIBXS_MALLOC_FLAG_RWX = LIBXS_MALLOC_FLAG_RW | LIBXS_MALLOC_FLAG_X
} libxs_malloc_flags;

/** Greatest common divisor. */
LIBXS_API size_t libxs_gcd(size_t a, size_t b);
/** Least common multiple. */
LIBXS_API size_t libxs_lcm(size_t a, size_t b);
/** Calculates an alignment depending on supposedly allocated size; alignment can be zero ("auto"). */
LIBXS_API size_t libxs_alignment(size_t size, size_t alignment);

/** Same as libxs_set_default_allocator, but takes a lock (can be NULL). */
LIBXS_API int libxs_xset_default_allocator(LIBXS_LOCK_TYPE* lock,
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Same as libxs_get_default_allocator, but takes a lock (can be NULL). */
LIBXS_API int libxs_xget_default_allocator(LIBXS_LOCK_TYPE* lock,
  void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/** Same as libxs_set_scratch_allocator, but takes a lock (can be NULL). */
LIBXS_API int libxs_xset_scratch_allocator(LIBXS_LOCK_TYPE* lock,
  void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Same as libxs_get_scratch_allocator, but takes a lock (can be NULL). */
LIBXS_API int libxs_xget_scratch_allocator(LIBXS_LOCK_TYPE* lock,
  void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/** Retrieve internal information about a buffer (default memory domain). */
LIBXS_API int libxs_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXS_API int libxs_xmalloc(void** memory, size_t size, size_t alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
/** Release memory, which was allocated using libxs_[*]malloc. */
LIBXS_API int libxs_xfree(const void* memory);

/**
 * Attribute memory allocation and protect with only the necessary flags.
 * This procedure is expected to run only one time per buffer, and may
 * relocate the given memory.
 */
LIBXS_API int libxs_malloc_attrib(void** memory, int flags,
  /** If a name is given, an executable buffer will be dumped into a file. */
  const char* name);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXS_CAPACITY_REGISTRY for unmanaged code). */
LIBXS_API int libxs_build(const libxs_build_request* request, unsigned regindex, libxs_code_pointer* code);

/** Updates counters of the statistic, which is shown at program termination. */
LIBXS_API unsigned int libxs_update_mmstatistic(int flags, int m, int n, int k, unsigned int ntry, unsigned int ncol);

LIBXS_API void libxs_dnn_init(int target_arch);
LIBXS_API void libxs_dnn_finalize(void);

LIBXS_API_VARIABLE LIBXS_LOCK_TYPE libxs_lock_global;
/** Function used to allocate default memory. */
LIBXS_API_VARIABLE libxs_malloc_function libxs_default_malloc_fn;
/** Function used to allocate scratch memory. */
LIBXS_API_VARIABLE libxs_malloc_function libxs_scratch_malloc_fn;
/** Function used to release default memory. */
LIBXS_API_VARIABLE libxs_free_function libxs_default_free_fn;
/** Function used to release scratch memory. */
LIBXS_API_VARIABLE libxs_free_function libxs_scratch_free_fn;
/** If non-NULL, this context used for the context-form of the malloc/free function. */
LIBXS_API_VARIABLE void* libxs_default_allocator_context;
/** If non-NULL, this context used for the context-form of the malloc/free function. */
LIBXS_API_VARIABLE void* libxs_scratch_allocator_context;
/** Number of scratch memory pools used; clamped against internal maximum. */
LIBXS_API_VARIABLE unsigned int libxs_scratch_pools;
/** Growth factor used to scale the scratch memory in case of reallocation. */
LIBXS_API_VARIABLE double libxs_scratch_scale;
/** Number of seconds per RDTSC-cycle (zero if RDTSC is not used for wallclock) */
LIBXS_API_VARIABLE double libxs_timer_scale;
/** Stores the verbosity level (libxs_get_verbosity, libxs_set_verbosity). */
LIBXS_API_VARIABLE int libxs_verbosity;
/** Target architecture (libxs_get_target_archid, libxs_set_target_archid). */
LIBXS_API_VARIABLE int libxs_target_archid;
/** Determines whether a threaded implementation is synchronized or not. */
LIBXS_API_VARIABLE int libxs_sync;
/** Number of threads per core. */
LIBXS_API_VARIABLE int libxs_nt;

#endif /*LIBXS_MAIN_H*/
