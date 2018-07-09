/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
#ifndef LIBXS_TRANS_H
#define LIBXS_TRANS_H

#include <libxs.h>

#if !defined(LIBXS_TRANS_COLLAPSE)
# if !defined(_CRAYC)
#   define LIBXS_TRANS_COLLAPSE 1/*2*/
# else
#   define LIBXS_TRANS_COLLAPSE 1
# endif
#endif

/* kernel uses consecutive stores and consecutive loads (copy) */
#define LIBXS_MCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((INDEX_J) * (LDI) + (INDEX_I))); \
              TYPE *const DST = (TYPE*)(((const char*)(OUT)) + (TYPESIZE) * ((INDEX_J) * (LDO) + (INDEX_I)))
/* call JIT-kernel (matrix-copy) */
#define LIBXS_MCOPY_CALL_NOPF(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) (KERNEL)(SRC, LDI, DST, LDO)
/* call JIT-kernel (matrix-copy with prefetch) */
#define LIBXS_MCOPY_CALL(PRFT_KERNEL, TYPESIZE, SRC, LDI, DST, LDO) (PRFT_KERNEL)(SRC, LDI, DST, LDO, \
  ((const char*)(SRC)) + (TYPESIZE) * (*(LDI))) /* prefetch next line*/
/* kernel uses consecutive stores and strided loads (transpose) */
#define LIBXS_TCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((INDEX_J) * (LDI) + (INDEX_I))); \
              TYPE *const DST = (TYPE*)(((const char*)(OUT)) + (TYPESIZE) * ((INDEX_I) * (LDO) + (INDEX_J)))
/* call JIT-kernel (transpose) */
#define LIBXS_TCOPY_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) (KERNEL)(SRC, LDI, DST, LDO)

#define LIBXS_XCOPY_LOOP_UNALIGNED(A)
#define LIBXS_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, HINT_ALIGNED, OUT, IN, LDI, LDO, M0, M1, N0, N1) { \
  /*const*/int libxs_xcopy_loop_native_ = (sizeof(TYPE) == (TYPESIZE)); /* mute warning (constant conditional) */ \
  libxs_blasint libxs_xcopy_loop_i_, libxs_xcopy_loop_j_; \
  if (0 != libxs_xcopy_loop_native_) { \
    for (libxs_xcopy_loop_i_ = M0; libxs_xcopy_loop_i_ < (libxs_blasint)(M1); ++libxs_xcopy_loop_i_) { \
      LIBXS_PRAGMA_NONTEMPORAL HINT_ALIGNED(OUT) \
      for (libxs_xcopy_loop_j_ = N0; libxs_xcopy_loop_j_ < (libxs_blasint)(N1); ++libxs_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxs_xcopy_loop_i_, libxs_xcopy_loop_j_, \
          libxs_xcopy_loop_src_, libxs_xcopy_loop_dst_); *libxs_xcopy_loop_dst_ = *libxs_xcopy_loop_src_; \
      } \
    } \
  } \
  else { \
    unsigned int libxs_xcopy_loop_k_; \
    for (libxs_xcopy_loop_i_ = M0; libxs_xcopy_loop_i_ < (libxs_blasint)(M1); ++libxs_xcopy_loop_i_) { \
      LIBXS_PRAGMA_NONTEMPORAL HINT_ALIGNED(OUT) \
      for (libxs_xcopy_loop_j_ = N0; libxs_xcopy_loop_j_ < (libxs_blasint)(N1); ++libxs_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxs_xcopy_loop_i_, libxs_xcopy_loop_j_, \
          libxs_xcopy_loop_src_, libxs_xcopy_loop_dst_); \
        for (libxs_xcopy_loop_k_ = 0; libxs_xcopy_loop_k_ < (TYPESIZE); ++libxs_xcopy_loop_k_) { \
          libxs_xcopy_loop_dst_[libxs_xcopy_loop_k_] = libxs_xcopy_loop_src_[libxs_xcopy_loop_k_]; \
        } \
      } \
    } \
  } \
}

#define LIBXS_XCOPY_XALIGN(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1) { \
  if (0 == LIBXS_MOD2((N0) * (TYPESIZE), LIBXS_ALIGNMENT) && \
      0 == LIBXS_MOD2((LDO) * (TYPESIZE), LIBXS_ALIGNMENT) && \
      0 == LIBXS_MOD2((uintptr_t)(OUT), LIBXS_ALIGNMENT)) \
  { \
    LIBXS_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXS_PRAGMA_VALIGNED_VAR, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
  } \
  else { /* unaligned store */ \
    LIBXS_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXS_XCOPY_LOOP_UNALIGNED, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
  } \
}

#define LIBXS_XCOPY_NONJIT(XKERNEL, OUT, IN, TYPESIZE, LDI, LDO, M0, M1, N0, N1) { \
  switch(TYPESIZE) { \
    case 2: { \
      LIBXS_XCOPY_XALIGN(short, 2, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 4: { \
      LIBXS_XCOPY_XALIGN(float, 4, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 8: { \
      LIBXS_XCOPY_XALIGN(double, 8, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 16: { \
      typedef struct /*libxs_xcopy_nonjit_elem_t*/ { double value[2]; } libxs_xcopy_nonjit_elem_t; \
      LIBXS_XCOPY_XALIGN(libxs_xcopy_nonjit_elem_t, 16, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    default: { \
      LIBXS_XCOPY_XALIGN(char, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
  } \
}

#if 1
# define LIBXS_XCOPY_PRECOND(COND)
#else
# define LIBXS_XCOPY_PRECOND(COND) COND
#endif

#define LIBXS_XCOPY(PARALLEL, LOOP_START, KERNEL_START, SYNC, \
  XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1) { \
  PARALLEL \
  { \
    libxs_blasint libxs_xcopy_i_ = M0, libxs_xcopy_j_ = N0; \
    if (0 != (KERNEL)) { /* inner tiles with JIT */ \
      LOOP_START(LIBXS_TRANS_COLLAPSE) \
      for (libxs_xcopy_i_ = M0; libxs_xcopy_i_ < (libxs_blasint)((M1) - (TILE_M) + 1); libxs_xcopy_i_ += TILE_M) { \
        for (libxs_xcopy_j_ = N0; libxs_xcopy_j_ < (libxs_blasint)((N1) - (TILE_N) + 1); libxs_xcopy_j_ += TILE_N) { \
          KERNEL_START(firstprivate(libxs_xcopy_i_, libxs_xcopy_j_) untied) \
          { \
            XKERNEL(char, TYPESIZE, OUT, IN, LDI, LDO, libxs_xcopy_i_, libxs_xcopy_j_, libxs_xcopy_src_, libxs_xcopy_dst_); \
            KERNEL_CALL(KERNEL, TYPESIZE, libxs_xcopy_src_, &(LDI), libxs_xcopy_dst_, &(LDO)); \
          } \
        } \
      } \
    } \
    else { /* inner tiles without JIT */ \
      LOOP_START(LIBXS_TRANS_COLLAPSE) \
      for (libxs_xcopy_i_ = M0; libxs_xcopy_i_ < (libxs_blasint)((M1) - (TILE_M) + 1); libxs_xcopy_i_ += TILE_M) { \
        for (libxs_xcopy_j_ = N0; libxs_xcopy_j_ < (libxs_blasint)((N1) - (TILE_N) + 1); libxs_xcopy_j_ += TILE_N) { \
          KERNEL_START(firstprivate(libxs_xcopy_i_, libxs_xcopy_j_) untied) \
          { \
            LIBXS_XCOPY_NONJIT(XKERNEL, OUT, IN, TYPESIZE, LDI, LDO, \
              libxs_xcopy_i_, libxs_xcopy_i_ + (TILE_M), \
              libxs_xcopy_j_, libxs_xcopy_j_ + (TILE_N)); \
          } \
        } \
      } \
    } \
    LIBXS_XCOPY_PRECOND(if (libxs_xcopy_j_ < (N1))) { \
      LOOP_START(1/*COLLAPSE*/) \
      for (libxs_xcopy_i_ = M0; libxs_xcopy_i_ < (libxs_blasint)((M1) - (TILE_M) + 1); libxs_xcopy_i_ += TILE_M) { \
        KERNEL_START(firstprivate(libxs_xcopy_i_) untied) \
        LIBXS_XCOPY_NONJIT(XKERNEL, OUT, IN, TYPESIZE, LDI, LDO, \
          libxs_xcopy_i_, libxs_xcopy_i_ + (TILE_M), \
          libxs_xcopy_j_, N1); \
      } \
    } \
    LIBXS_XCOPY_PRECOND(if (libxs_xcopy_i_ < (M1))) { \
      LOOP_START(1/*COLLAPSE*/) \
      for (libxs_xcopy_j_ = N0; libxs_xcopy_j_ < (libxs_blasint)((N1) - (TILE_N)); libxs_xcopy_j_ += TILE_N) { \
        KERNEL_START(firstprivate(libxs_xcopy_j_) untied) \
        LIBXS_XCOPY_NONJIT(XKERNEL, OUT, IN, TYPESIZE, LDI, LDO, \
          libxs_xcopy_i_, M1, \
          libxs_xcopy_j_, libxs_xcopy_j_ + (TILE_N)); \
      } \
    } \
    LIBXS_XCOPY_PRECOND(if (libxs_xcopy_i_ < (M1) && libxs_xcopy_j_ < (N1))) { \
      LIBXS_XCOPY_NONJIT(XKERNEL, OUT, IN, TYPESIZE, LDI, LDO, \
        libxs_xcopy_i_, M1, \
        libxs_xcopy_j_, N1); \
    } \
    SYNC \
  } \
}


/** Initializes the transpose functionality; NOT thread-safe. */
LIBXS_API_INTERN void libxs_trans_init(int archid);

/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXS_API_INTERN void libxs_trans_finalize(void);


/** Determines whether JIT-kernels are used or not (0: none, 1: matcopy, 2: transpose, 3: matcopy+transpose). */
LIBXS_APIVAR(int libxs_trans_jit);
/** Configuration table containing the tile sizes separate for DP and SP. */
LIBXS_APIVAR(/*const*/ unsigned int(*libxs_trans_tile)[2/*M,N*/][8/*size-range*/]);

#endif /*LIBXS_TRANS_H*/
