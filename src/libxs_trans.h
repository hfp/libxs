/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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

#if !defined(LIBXS_TRANS_MIN_CHUNKSIZE)
# define LIBXS_TRANS_MIN_CHUNKSIZE 8
#endif
#if !defined(LIBXS_TRANS_MAX_CHUNKSIZE)
# define LIBXS_TRANS_MAX_CHUNKSIZE 32
#endif
#if !defined(LIBXS_TRANS_TYPEOPT)
# define LIBXS_TRANS_TYPEOPT
#endif

#define LIBXS_OTRANS_GENERIC(TYPESIZE, OUT, IN, M0, M1, N0, N1, N, LD, LDO) { \
  const char *const a = (const char*)(IN); \
  char *const b = (char*)(OUT); \
  libxs_blasint i, j; \
  unsigned int k; \
  for (i = M0; i < (M1); ++i) { \
    LIBXS_PRAGMA_NONTEMPORAL \
    for (j = N0; j < (N1); ++j) { \
      const char *const aji = a + (TYPESIZE) * (j * (LD) + i); \
      char *const bij = b + (TYPESIZE) * (i * (LDO) + j); \
      for (k = 0; k < (TYPESIZE); ++k) { \
        bij[k] = aji[k]; \
      } \
    } \
  } \
}

#define LIBXS_OTRANS(TYPE, OUT, IN, M0, M1, N0, N1, N, LD, LDO) { \
  if (libxs_trans_chunksize == (N) && 0 == LIBXS_MOD2((uintptr_t)(IN), LIBXS_ALIGNMENT)) { \
    const TYPE *const a = (const TYPE*)(IN); \
    TYPE *const b = (TYPE*)(OUT); \
    libxs_blasint i, j; \
    if (LIBXS_TRANS_MAX_CHUNKSIZE == (N)) { \
      for (i = M0; i < (M1); ++i) { \
        LIBXS_PRAGMA_NONTEMPORAL \
        LIBXS_PRAGMA_VALIGNED_VARS(b) \
        for (j = N0; j < (N0) + (LIBXS_TRANS_MAX_CHUNKSIZE); ++j) { \
          /* use consecutive stores and strided loads */ \
          b[i*(LDO)+j] = a[j*(LD)+i]; \
        } \
      } \
    } \
    else { \
      assert(LIBXS_TRANS_MIN_CHUNKSIZE == (N)); \
      for (i = M0; i < (M1); ++i) { \
        LIBXS_PRAGMA_NONTEMPORAL \
        LIBXS_PRAGMA_VALIGNED_VARS(b) \
        for (j = N0; j < (N0) + (LIBXS_TRANS_MIN_CHUNKSIZE); ++j) { \
          /* use consecutive stores and strided loads */ \
          b[i*(LDO)+j] = a[j*(LD)+i]; \
        } \
      } \
    } \
  } \
  else { /* remainder tile */ \
    LIBXS_OTRANS_GENERIC(sizeof(TYPE), OUT, IN, M0, M1, N0, N1, N, LD, LDO); \
  } \
}

#if defined(LIBXS_TRANS_TYPEOPT)
# define LIBXS_OTRANS_TYPEOPT_BEGIN(OUT, IN, TYPESIZE, M0, M1, N0, N1, N, LD, LDO) \
    switch(TYPESIZE) { \
      case 1: { \
        LIBXS_OTRANS(char, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      case 2: { \
        LIBXS_OTRANS(short, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      case 4: { \
        LIBXS_OTRANS(float, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      case 8: { \
        LIBXS_OTRANS(double, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      case 16: { \
        typedef struct dvec2_t { double value[2]; } dvec2_t; \
        LIBXS_OTRANS(dvec2_t, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      default:
#else
# define LIBXS_OTRANS_TYPEOPT_BEGIN(OUT, IN, TYPESIZE, M0, M1, N0, N1, N, LD, LDO) {
#endif
#define LIBXS_OTRANS_TYPEOPT_END }

/**
 * Based on the cache-oblivious transpose by Frigo et.al. with some additional
 * optimization such as using a loop with bounds which are known at compile-time
 * due to splitting up tiles with one fixed-size extent (chunk).
 */
#define LIBXS_OTRANS_MAIN(KERNEL_START, FN, OUT, IN, TYPESIZE, M0, M1, N0, N1, LD, LDO) { \
  const libxs_blasint m = (M1) - (M0), n = (N1) - (N0); \
  if (m * n * (TYPESIZE) <= ((LIBXS_CPU_DCACHESIZE) / 2)) { \
    KERNEL_START(n) \
    { \
      LIBXS_OTRANS_TYPEOPT_BEGIN(OUT, IN, TYPESIZE, M0, M1, N0, N1, n, LD, LDO) \
      /* fall-back code path which is generic with respect to the typesize */ \
      LIBXS_OTRANS_GENERIC(TYPESIZE, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      LIBXS_OTRANS_TYPEOPT_END \
    } \
  } \
  else if (m >= n) { \
    const libxs_blasint mi = ((M0) + (M1)) / 2; \
    (FN)(OUT, IN, TYPESIZE, M0, mi, N0, N1, LD, LDO); \
    (FN)(OUT, IN, TYPESIZE, mi, M1, N0, N1, LD, LDO); \
  } \
  else { \
    if (libxs_trans_chunksize < n) { \
      const libxs_blasint ni = (N0) + libxs_trans_chunksize; \
      (FN)(OUT, IN, TYPESIZE, M0, M1, N0, ni, LD, LDO); \
      (FN)(OUT, IN, TYPESIZE, M0, M1, ni, N1, LD, LDO); \
    } \
    else \
    { \
      const libxs_blasint ni = ((N0) + (N1)) / 2; \
      (FN)(OUT, IN, TYPESIZE, M0, M1, N0, ni, LD, LDO); \
      (FN)(OUT, IN, TYPESIZE, M0, M1, ni, N1, LD, LDO); \
    } \
  } \
}


/** Initializes the tranpose functionality; NOT thread-safe. */
LIBXS_API void libxs_trans_init(int archid);

/** Finalizes the transpose functionality; NOT thread-safe. */
LIBXS_API void libxs_trans_finalize(void);


/** Size of peeled chunks during transposing inner tiles. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_trans_chunksize;

#endif /*LIBXS_TRANS_H*/
