/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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
