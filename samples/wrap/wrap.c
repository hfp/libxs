/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"
#include <libxs_sync.h>


LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void GEMM_REAL(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  if (NULL == gemm_original) {
    union { const void* pfin; gemm_function_t pfout; } wrapper;
    static volatile LIBXS_ATOMIC_LOCKTYPE lock = 0;
    LIBXS_ATOMIC_ACQUIRE(&lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_SEQ_CST);
    if (NULL == gemm_original) {
      dlerror(); /* clear an eventual error status */
      wrapper.pfin = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(GEMM));
      if (NULL == dlerror() && NULL != wrapper.pfout) {
        gemm_original = wrapper.pfout;
      }
    }
    LIBXS_ATOMIC_RELEASE(&lock, LIBXS_ATOMIC_SEQ_CST);
  }
  if (NULL != gemm_original) {
    gemm_original(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else {
    fprintf(stderr, "ERROR: incorrect linkage against libwrap discovered!\n"
                    "       link statically with -Wl,--wrap=" LIBXS_STRINGIFY(GEMM) ",\n"
                    "       or use LD_PRELOAD=/path/to/libwrap.so\n");
  }
}


LIBXS_API LIBXS_ATTRIBUTE_USED void GEMM(const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  GEMM_WRAP(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
