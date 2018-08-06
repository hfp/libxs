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

#if !defined(LIBXS_TRANS_CHECK) && !defined(NDEBUG)
# define LIBXS_TRANS_CHECK
#endif
#if !defined(LIBXS_TRANS_TASKSCALE)
# define LIBXS_TRANS_TASKSCALE 2
#endif

/* kernel uses consecutive stores and consecutive loads (copy) */
#define LIBXS_MCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((size_t)(INDEX_J) * (LDI) + (INDEX_I))); \
        TYPE *const DST = (      TYPE*)(((      char*)(OUT)) + (TYPESIZE) * ((size_t)(INDEX_J) * (LDO) + (INDEX_I)))
/* call JIT-kernel (matrix-copy) */
#define LIBXS_MCOPY_CALL_NOPF(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
  const unsigned int libxs_mcopy_call_nopf_uldi_ = (unsigned int)(LDI); \
  const unsigned int libxs_mcopy_call_nopf_uldo_ = (unsigned int)(LDO); \
  (KERNEL)(SRC, &libxs_mcopy_call_nopf_uldi_, DST, &libxs_mcopy_call_nopf_uldo_); \
}
/* call JIT-kernel (matrix-copy with prefetch) */
#define LIBXS_MCOPY_CALL(PRFT_KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
  const unsigned int libxs_mcopy_call_uldi_ = (unsigned int)(LDI); \
  const unsigned int libxs_mcopy_call_uldo_ = (unsigned int)(LDO); \
  (PRFT_KERNEL)(SRC, &libxs_mcopy_call_uldi_, DST, &libxs_mcopy_call_uldo_, \
    /*prefetch next line*/((const char*)(SRC)) + (TYPESIZE) * (size_t)(LDI)); \
}
/* kernel uses consecutive stores and strided loads (transpose) */
#define LIBXS_TCOPY_KERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, INDEX_I, INDEX_J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*) (IN)) + (TYPESIZE) * ((size_t)(INDEX_J) * (LDI) + (INDEX_I))); \
        TYPE *const DST = (      TYPE*)(((      char*)(OUT)) + (TYPESIZE) * ((size_t)(INDEX_I) * (LDO) + (INDEX_J)))
/* call JIT-kernel (transpose) */
#define LIBXS_TCOPY_CALL(KERNEL, TYPESIZE, SRC, LDI, DST, LDO) { \
  const unsigned int libxs_tcopy_call_uldi_ = (unsigned int)(LDI); \
  const unsigned int libxs_tcopy_call_uldo_ = (unsigned int)(LDO); \
  (KERNEL)(SRC, &libxs_tcopy_call_uldi_, DST, &libxs_tcopy_call_uldo_); \
}

#define LIBXS_XCOPY_LOOP_UNALIGNED(A)
#define LIBXS_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, HINT_ALIGNED, OUT, IN, LDI, LDO, M0, M1, N0, N1) { \
  /*const*/int libxs_xcopy_loop_generic_ = (sizeof(TYPE) != (TYPESIZE)); /* mute warning (constant conditional) */ \
  libxs_blasint libxs_xcopy_loop_i_, libxs_xcopy_loop_j_; \
  if (0 == libxs_xcopy_loop_generic_) { /* specific type-size */ \
    for (libxs_xcopy_loop_i_ = M0; libxs_xcopy_loop_i_ < (libxs_blasint)(M1); ++libxs_xcopy_loop_i_) { \
      LIBXS_PRAGMA_NONTEMPORAL HINT_ALIGNED(OUT) \
      for (libxs_xcopy_loop_j_ = N0; libxs_xcopy_loop_j_ < (libxs_blasint)(N1); ++libxs_xcopy_loop_j_) { \
        XKERNEL(TYPE, TYPESIZE, OUT, IN, LDI, LDO, libxs_xcopy_loop_i_, libxs_xcopy_loop_j_, \
          libxs_xcopy_loop_src_, libxs_xcopy_loop_dst_); *libxs_xcopy_loop_dst_ = *libxs_xcopy_loop_src_; \
      } \
    } \
  } \
  else { /* generic type-size */ \
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

#define LIBXS_XALIGN_TCOPY(N0, TYPESIZE) (0 == LIBXS_MOD2((N0) * (TYPESIZE), LIBXS_ALIGNMENT))
#define LIBXS_XALIGN_MCOPY(N0, TYPESIZE) (1)

#define LIBXS_XCOPY_XALIGN(TYPE, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN) { \
  if (0 == LIBXS_MOD2((uintptr_t)(OUT), LIBXS_ALIGNMENT) && \
      0 == LIBXS_MOD2((LDO) * (TYPESIZE), LIBXS_ALIGNMENT) && \
      XALIGN(N0, TYPESIZE)) \
  { \
    LIBXS_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXS_PRAGMA_VALIGNED_VAR, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
  } \
  else { /* unaligned store */ \
    LIBXS_XCOPY_LOOP(TYPE, TYPESIZE, XKERNEL, LIBXS_XCOPY_LOOP_UNALIGNED, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
  } \
}

#define LIBXS_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN) { \
  switch(TYPESIZE) { \
    case 2: { \
      LIBXS_XCOPY_XALIGN(short, 2, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
    case 4: { \
      LIBXS_XCOPY_XALIGN(float, 4, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
    case 8: { \
      LIBXS_XCOPY_XALIGN(double, 8, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
    case 16: { \
      typedef struct /*libxs_xcopy_nonjit_elem_t*/ { double value[2]; } libxs_xcopy_nonjit_elem_t; \
      LIBXS_XCOPY_XALIGN(libxs_xcopy_nonjit_elem_t, 16, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
    default: { \
      LIBXS_XCOPY_XALIGN(char, TYPESIZE, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1, XALIGN); \
    } break; \
  } \
}

#if 1
# define LIBXS_XCOPY_PRECOND(COND)
#else
# define LIBXS_XCOPY_PRECOND(COND) COND
#endif

#define LIBXS_XCOPY(XKERNEL, KERNEL_CALL, KERNEL, OUT, IN, TYPESIZE, LDI, LDO, TILE_M, TILE_N, M0, M1, N0, N1, XALIGN) { \
  libxs_blasint libxs_xcopy_i_ = M0, libxs_xcopy_j_ = N0; \
  if (0 != (KERNEL)) { /* inner tiles with JIT */ \
    for (; libxs_xcopy_i_ < (libxs_blasint)((M1) - (TILE_M) + 1); libxs_xcopy_i_ += TILE_M) { \
      for (libxs_xcopy_j_ = N0; libxs_xcopy_j_ < (libxs_blasint)((N1) - (TILE_N) + 1); libxs_xcopy_j_ += TILE_N) { \
        XKERNEL(char, TYPESIZE, OUT, IN, LDI, LDO, libxs_xcopy_i_, libxs_xcopy_j_, libxs_xcopy_src_, libxs_xcopy_dst_); \
        KERNEL_CALL(KERNEL, TYPESIZE, libxs_xcopy_src_, LDI, libxs_xcopy_dst_, LDO); \
      } \
    } \
  } \
  else { /* inner tiles without JIT */ \
    for (; libxs_xcopy_i_ < (libxs_blasint)((M1) - (TILE_M) + 1); libxs_xcopy_i_ += TILE_M) { \
      for (libxs_xcopy_j_ = N0; libxs_xcopy_j_ < (libxs_blasint)((N1) - (TILE_N) + 1); libxs_xcopy_j_ += TILE_N) { \
        LIBXS_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
          libxs_xcopy_i_, libxs_xcopy_i_ + (TILE_M), \
          libxs_xcopy_j_, libxs_xcopy_j_ + (TILE_N), XALIGN); \
      } \
    } \
  } \
  LIBXS_XCOPY_PRECOND(if (libxs_xcopy_j_ < (N1))) { \
    for (libxs_xcopy_i_ = M0; libxs_xcopy_i_ < (libxs_blasint)((M1) - (TILE_M) + 1); libxs_xcopy_i_ += TILE_M) { \
      LIBXS_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
        libxs_xcopy_i_, libxs_xcopy_i_ + (TILE_M), \
        libxs_xcopy_j_, N1, XALIGN); \
    } \
  } \
  LIBXS_XCOPY_PRECOND(if (libxs_xcopy_i_ < (M1))) { \
    for (libxs_xcopy_j_ = N0; libxs_xcopy_j_ < (libxs_blasint)((N1) - (TILE_N)); libxs_xcopy_j_ += TILE_N) { \
      LIBXS_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
        libxs_xcopy_i_, M1, \
        libxs_xcopy_j_, libxs_xcopy_j_ + (TILE_N), XALIGN); \
    } \
  } \
  LIBXS_XCOPY_PRECOND(if (libxs_xcopy_i_ < (M1) && libxs_xcopy_j_ < (N1))) { \
    LIBXS_XCOPY_NONJIT(XKERNEL, TYPESIZE, OUT, IN, LDI, LDO, \
      libxs_xcopy_i_, M1, \
      libxs_xcopy_j_, N1, XALIGN); \
  } \
}


/** Initializes the transpose functionality; NOT thread-safe. */
LIBXS_API_INTERN void libxs_trans_init(int archid);
/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXS_API_INTERN void libxs_trans_finalize(void);

LIBXS_API void libxs_matcopy_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo, const int* prefetch,
  unsigned int tm, unsigned int tn, libxs_xmcopyfunction kernel,
  int tid, int nthreads);
LIBXS_API_INTERN void libxs_matcopy_internal_pf(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxs_xmcopyfunction kernel);
LIBXS_API_INTERN void libxs_matcopy_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxs_xmcopyfunction kernel);

LIBXS_API void libxs_otrans_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int tm, unsigned int tn, libxs_xtransfunction kernel,
  int tid, int nthreads);
LIBXS_API_INTERN void libxs_otrans_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxs_xtransfunction kernel);

/** Determines whether JIT-kernels are used or not (0: none, 1: matcopy, 2: transpose, 3: matcopy+transpose). */
LIBXS_APIVAR_PUBLIC(int libxs_trans_jit);
/** M-factor shaping the N-extent (tile shape). */
LIBXS_APIVAR_PUBLIC(float libxs_trans_tile_stretch);
/** Table of M-extents per type-size (tile shape). */
LIBXS_APIVAR_PUBLIC(unsigned int* libxs_trans_mtile);
/** Determines if OpenMP tasks are used, and scales beyond the number of threads. */
LIBXS_APIVAR_PUBLIC(int libxs_trans_taskscale);

#endif /*LIBXS_TRANS_H*/
