/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
#ifndef LIBXS_BLOCKED_GEMM_TYPES_H
#define LIBXS_BLOCKED_GEMM_TYPES_H

#include "libxs_gemm.h"

#if !defined(LIBXS_BLOCKED_GEMM_CHECKS) && !defined(NDEBUG)
# define LIBXS_BLOCKED_GEMM_CHECKS
#endif


LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_blocked_gemm_lock {
  char pad[LIBXS_CACHELINE];
  volatile LIBXS_ATOMIC_LOCKTYPE state;
} libxs_blocked_gemm_lock;


LIBXS_EXTERN_C struct LIBXS_RETARGETABLE libxs_blocked_gemm_handle {
  union { double d; float s; int w; } alpha, beta;
  libxs_gemm_precision iprec, oprec;
  libxs_xmmfunction kernel_pf;
  libxs_xmmfunction kernel;
  libxs_barrier* barrier;
  libxs_blocked_gemm_lock* locks;
  libxs_blocked_gemm_order order;
  libxs_blasint m, n, k, bm, bn, bk;
  libxs_blasint b_m1, b_n1, b_k1, b_k2;
  libxs_blasint mb, nb, kb;
  void* buffer;
  int nthreads;
};

#endif /*LIBXS_BLOCKED_GEMM_TYPES_H*/
