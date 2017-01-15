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

#include <stddef.h>
#include <stdint.h>

#include <libxs_typedefs.h>
#include <libxs_generator.h>
#include <libxs_dnn.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXS_REGCAPACITY) /* must be POT */
# define LIBXS_REGCAPACITY 524288 /* 524287: Mersenne Prime number (2^19-1) */
#endif
#if !defined(LIBXS_CPU_DCACHESIZE)
# define LIBXS_CPU_DCACHESIZE 32768
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
  uintptr_t imm;
#if defined(LIBXS_BUILD) || defined(LIBXS_DNN_INTERNAL_API)
  libxs_xconvfunction xconv;
#endif
  libxs_xmmfunction xmm;
} libxs_code_pointer;

typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_csr_soa_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxs_csr_soa_descriptor;

/** Structure which describes an activation layer. */
struct LIBXS_RETARGETABLE libxs_dnn_buffer {
  int N;                            /* number of images in mini-batch */
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int H;                            /* height of image */
  int W;                            /* width of image */
  int lpb;                          /* low precision blocking factor */
  libxs_dnn_conv_format format;   /* format of activation buffer */
  libxs_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
};

/** Structure which describes a bias. */
struct LIBXS_RETARGETABLE libxs_dnn_bias {
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int lpb;                          /* low precision blocking factor */
  libxs_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
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
  libxs_dnn_conv_format format;   /* format of filter buffer */
  libxs_dnn_datatype datatype;    /* data type */
  void* data;                       /* pointer to data */
};

struct LIBXS_RETARGETABLE libxs_dnn_conv_handle {
  libxs_dnn_datatype datatype_in;
  libxs_dnn_datatype datatype_out;
  libxs_dnn_conv_desc desc;
  libxs_dnn_conv_algo algo;
  libxs_dnn_conv_format buffer_format;
  libxs_dnn_conv_format filter_format;
  libxs_dnn_conv_fuse_op fuse_ops;
  libxs_dnn_conv_option options;

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

  /* internal data representation */
  libxs_dnn_buffer* input;
  libxs_dnn_buffer* output;
  libxs_dnn_buffer* input_relu;
  libxs_dnn_filter* filter;
  libxs_dnn_bias* bias;
  void* scratch1;
  void* scratch2;
/*#ifdef LIBXS_WU_TRANSPOSE_OFW_IFM*/
  void* scratch3;
/*#endif*/
  void* scratch4;

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
};

typedef enum libxs_build_kind {
  LIBXS_BUILD_KIND_GEMM,
  LIBXS_BUILD_KIND_SSOA,
  LIBXS_BUILD_KIND_CFWD,
  LIBXS_BUILD_KIND_CBWD,
  LIBXS_BUILD_KIND_CUPD
} libxs_build_kind;

typedef union LIBXS_RETARGETABLE libxs_build_descriptor {
  const libxs_gemm_descriptor* gemm;
  const libxs_csr_soa_descriptor* ssoa;
  const libxs_convolution_forward_descriptor* cfwd;
  const libxs_convolution_backward_descriptor* cbwd;
  const libxs_convolution_weight_update_descriptor* cupd;
} libxs_build_descriptor;

typedef struct LIBXS_RETARGETABLE libxs_build_request {
  libxs_build_descriptor descriptor;
  libxs_build_kind kind;
} libxs_build_request;

typedef enum libxs_malloc_flags {
  LIBXS_MALLOC_FLAG_R = 1,
  LIBXS_MALLOC_FLAG_W = 2,
  LIBXS_MALLOC_FLAG_X = 4,
  LIBXS_MALLOC_FLAG_MMAP = 8,
  LIBXS_MALLOC_FLAG_RW  = LIBXS_MALLOC_FLAG_R | LIBXS_MALLOC_FLAG_W,
  LIBXS_MALLOC_FLAG_RWX = LIBXS_MALLOC_FLAG_RW | LIBXS_MALLOC_FLAG_X,
  /** LIBXS_MALLOC_FLAG_DEFAULT is an alias for setting no flag bits. */
  LIBXS_MALLOC_FLAG_DEFAULT = LIBXS_MALLOC_FLAG_RW
} libxs_malloc_flags;

/** Greatest common divisor. */
LIBXS_API size_t libxs_gcd(size_t a, size_t b);
/** Least common multiple. */
LIBXS_API size_t libxs_lcm(size_t a, size_t b);
/** Calculates an alignment depending on supposedly allocated size; alignment can be zero ("auto"). */
LIBXS_API size_t libxs_alignment(size_t size, size_t alignment);

/** Receive the size, the flags, or the extra attachment of the given buffer. */
LIBXS_API int libxs_malloc_info(const volatile void* memory, size_t* size, int* flags, void** extra);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXS_API int libxs_xmalloc(void** memory, size_t size, int alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
LIBXS_API int libxs_xfree(const volatile void* memory);

/**
 * Attribute memory allocation and protect with only the necessary flags.
 * This procedure is expected to run only one time per buffer, and may
 * relocate the given memory.
 */
LIBXS_API int libxs_malloc_attrib(void** memory, int flags,
  /** If a name is given, an executable buffer will be dumped into a file. */
  const char* name);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXS_REGCAPACITY for unmanaged code). */
LIBXS_API int libxs_build(const libxs_build_request* request, unsigned regindex, libxs_code_pointer* code);

/** Updates counters of the statistic, which is shown at program termination. */
LIBXS_API unsigned int libxs_update_mmstatistic(int flags, int m, int n, int k, unsigned int ntry, unsigned int ncol);

LIBXS_API int libxs_gemm_prefetch2uid(libxs_gemm_prefetch_type prefetch);
LIBXS_API libxs_gemm_prefetch_type libxs_gemm_uid2prefetch(int uid);

LIBXS_API size_t libxs_dnn_typesize(libxs_dnn_datatype datatype);

/** Stores the verbosity level (libxs_get_verbosity, libxs_set_verbosity). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_verbosity;
/** Target architecture (libxs_get_target_archid, libxs_set_target_archid). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_target_archid;
/** Try-lock property of the code registry (0: off, 1: enabled). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_dispatch_trylock;
/** Determines the prefetch strategy, which is used in case of LIBXS_PREFETCH_AUTO. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_gemm_auto_prefetch;
/** Determines if (OpenMP-)tasks are preferred over thread-style parallelization. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_tasks;
/** Kind of parallel support (0: none, 1: sequential, 2: parallelized). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_mt;
/** Number of threads per core. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_nt;

#endif /*LIBXS_MAIN_H*/
