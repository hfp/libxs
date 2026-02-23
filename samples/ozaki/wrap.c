/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "ozaki.h"
#include <libxs_sync.h>


/**
 * Helper macro: define dlsym-resolved fallback function.
 * Used for both GEMM_REAL and ZGEMM_REAL to avoid code duplication.
 * The generated function resolves the original BLAS symbol via dlsym
 * on first call (LD_PRELOAD path) and caches the function pointer.
 */
#define GEMM_DEFINE_DLSYM(FUNC, SYMBOL, FTYPE, ORIGPTR) \
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void FUNC(GEMM_ARGDECL) \
{ \
  if (NULL == ORIGPTR) { \
    union { const void* pfin; FTYPE pfout; } wrapper; \
    static volatile LIBXS_ATOMIC_LOCKTYPE lock = 0; \
    LIBXS_ATOMIC_ACQUIRE(&lock, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_LOCKORDER); \
    if (NULL == ORIGPTR) { \
      dlerror(); \
      wrapper.pfin = dlsym(LIBXS_RTLD_NEXT, LIBXS_STRINGIFY(SYMBOL)); \
      if (NULL == dlerror() && NULL != wrapper.pfout) { \
        ORIGPTR = wrapper.pfout; \
      } \
    } \
    LIBXS_ATOMIC_RELEASE(&lock, LIBXS_ATOMIC_LOCKORDER); \
  } \
  if (NULL != ORIGPTR) { \
    ORIGPTR(GEMM_ARGPASS); \
  } \
  else { \
    fprintf(stderr, "ERROR: incorrect linkage against libwrap discovered!\n" \
                    "       link statically with -Wl,--wrap=" \
                    LIBXS_STRINGIFY(SYMBOL) ",\n" \
                    "       or use LD_PRELOAD=/path/to/libwrap.so\n"); \
  } \
}

/** Resolve original real GEMM via dlsym (dgemm_ or sgemm_). */
GEMM_DEFINE_DLSYM(GEMM_REAL, GEMM, gemm_function_t, gemm_original)
/** Resolve original complex GEMM via dlsym (zgemm_ or cgemm_). */
GEMM_DEFINE_DLSYM(ZGEMM_REAL, ZGEMM, zgemm_function_t, zgemm_original)


/** Real GEMM entry point: delegates to GEMM_WRAP (Ozaki implementation). */
LIBXS_API LIBXS_ATTRIBUTE_USED void GEMM(GEMM_ARGDECL)
{
  GEMM_WRAP(GEMM_ARGPASS);
}

/** Complex GEMM entry point: delegates to ZGEMM_WRAP (3M implementation). */
LIBXS_API LIBXS_ATTRIBUTE_USED void ZGEMM(GEMM_ARGDECL)
{
  ZGEMM_WRAP(GEMM_ARGPASS);
}
