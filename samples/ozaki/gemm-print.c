/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "gemm.h"


LIBXS_API void print_gemm(FILE* ostream, const char* transa, const char* transb,
  const GEMM_INT_TYPE* m, const GEMM_INT_TYPE* n, const GEMM_INT_TYPE* k,
  const GEMM_REAL_TYPE* alpha, const GEMM_REAL_TYPE* a, const GEMM_INT_TYPE* lda,
                               const GEMM_REAL_TYPE* b, const GEMM_INT_TYPE* ldb,
  const GEMM_REAL_TYPE*  beta, GEMM_REAL_TYPE* c, const GEMM_INT_TYPE* ldc)
{
  const char *const fname = LIBXS_STRINGIFY(LIBXS_TPREFIX(GEMM_REAL_TYPE, gemm));
  fprintf(ostream, "%s('%c', '%c', %lli/*m*/, %lli/*n*/, %lli/*k*/,\n"
                   "  %g/*alpha*/, %p/*a*/, %lli/*lda*/,\n"
                   "              %p/*b*/, %lli/*ldb*/,\n"
                   "   %g/*beta*/, %p/*c*/, %lli/*ldc*/)\n",
    fname, *transa, *transb, (long long int)*m, (long long int)*n, (long long int)*k,
    *alpha, (const void*)a, (long long int)*lda, (const void*)b, (long long int)*ldb,
    *beta, (const void*)c, (long long int)*ldc);
}


LIBXS_API void print_diff(FILE* ostream, const libxs_matdiff_info_t* diff)
{
  fprintf(ostream, "GEMM: ncalls=%i linf=%f linf_rel=%f l2_rel=%f ref=%f val=%f eps=%f rsq=%f\n",
    diff->r, diff->linf_abs, diff->linf_rel, diff->l2_rel, diff->v_ref, diff->v_tst,
    libxs_matdiff_epsilon(diff), diff->rsq);
}
