/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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

#include <libxs_conv.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXS_REGSIZE) /* must be POT */
# define LIBXS_REGSIZE 524288 /* 524287: Mersenne Prime number (2^19-1) */
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


typedef union LIBXS_RETARGETABLE libxs_code_pointer {
#if defined(LIBXS_BUILD) || defined(LIBXS_CONV_INTERNAL_API)
  libxs_sconvfunction sconv;
#endif
  libxs_xmmfunction xmm;
  /*const*/void* pmm;
  uintptr_t imm;
} libxs_code_pointer;

typedef struct LIBXS_RETARGETABLE libxs_csr_soa_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxs_csr_soa_descriptor;

/** struct which holds description of a layer */
struct LIBXS_RETARGETABLE libxs_conv_layer {
  int N;                            /* number of images in mini-batch */
  int splits;                       /* number of splits */
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  int H;                            /* height of image */
  int W;                            /* width of image */
  libxs_conv_datatype datatype;   /* data type */
  void* data;                       /* pointer to data */
};

/** struct which holds description of a bias */
struct LIBXS_RETARGETABLE libxs_conv_bias {
  int splits;                       /* number of splits */
  int fmb;                          /* number of feature map blocks */
  int bfm;                          /* sized of blocked feature maps, in a block */
  libxs_conv_datatype datatype;   /* data type */
  void* data;                       /* pointer to data */
};

/** struct which holds description of a filter */
struct LIBXS_RETARGETABLE libxs_conv_filter {
  int splits;                       /* number of splits */
  int ifmb;                         /* number of feature map blocks */
  int bifm;                         /* sized of blocked feature maps, in a block */
  int ofmb;                         /* number of feature map blocks */
  int bofm;                         /* sized of blocked feature maps, in a block */
  int R;                            /* height of filter kernel */
  int S;                            /* width of filter kernel */
  libxs_conv_datatype datatype;   /* data type */
  void* data;                       /* pointer to data */
};

struct LIBXS_RETARGETABLE libxs_conv_handle {
  libxs_conv_datatype datatype;
  libxs_conv_desc desc;
  libxs_conv_algo algo;

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
  int fwd_ofh_rb;

  /* internal data representation */
  libxs_conv_layer* input;
  libxs_conv_layer* output;
  libxs_conv_layer* input_relu;
  libxs_conv_filter* filter;
  libxs_conv_bias* bias;
  void* scratch;

  /* JIT-generated convolution code */
  /*
  libxs_convolution_forward_descriptor       fwd_desc;
  libxs_convolution_forward_descriptor       bwd_desc;
  libxs_convolution_weight_update_descriptor wu_desc;
  */
  libxs_code_pointer code_fwd[4];
  libxs_code_pointer code_bwd[8];
  libxs_code_pointer code_upd[4];
};

typedef enum libxs_build_kind {
  LIBXS_BUILD_KIND_GEMM,
  LIBXS_BUILD_KIND_SSOA,
  LIBXS_BUILD_KIND_CFWD,
  LIBXS_BUILD_KIND_CBWD,
  LIBXS_BUILD_KIND_CUPD
} libxs_build_kind;

typedef struct LIBXS_RETARGETABLE libxs_build_request {
  union LIBXS_RETARGETABLE {
    const libxs_gemm_descriptor* gemm;
    const libxs_csr_soa_descriptor* ssoa;
    const libxs_convolution_forward_descriptor* cfwd;
    const libxs_convolution_backward_descriptor* cbwd;
    const libxs_convolution_weight_update_descriptor* cupd;
  } descriptor;
  libxs_build_kind kind;
} libxs_build_request;


typedef enum libxs_malloc_flags {
  LIBXS_MALLOC_FLAG_R = 1,
  LIBXS_MALLOC_FLAG_W = 2,
  LIBXS_MALLOC_FLAG_X = 4,
  LIBXS_MALLOC_FLAG_RWX = LIBXS_MALLOC_FLAG_R | LIBXS_MALLOC_FLAG_W | LIBXS_MALLOC_FLAG_X,
  LIBXS_MALLOC_FLAG_RW  = LIBXS_MALLOC_FLAG_R | LIBXS_MALLOC_FLAG_W,
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
LIBXS_API int libxs_malloc_info(const void* memory, size_t* size, int* flags, void** extra);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXS_API int libxs_xmalloc(void** memory, size_t size, int alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
LIBXS_API int libxs_xfree(const volatile void* memory);

/** Attribute memory allocation such as to revoke protection flags. */
LIBXS_API int libxs_malloc_attrib(const volatile void* memory, int flags,
  /** If a name is given, an executable buffer will be dumped into a file. */
  const char* name);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXS_REGSIZE for unmanaged code). */
LIBXS_API void libxs_build(const libxs_build_request* request, unsigned regindex, libxs_code_pointer* code);

/** Determines whether (OpenMP-)tasks are preferred over thread-style parallelization. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_tasks /*= 0*/;
/** Kind of parallel support (0: none, 1: sequential, 2: parallelized). */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_mp /*= 0*/;
/** Number of threads per core. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_nt /*= 2*/;

#endif /*LIBXS_MAIN_H*/
