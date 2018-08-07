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
#include "libxs_trans.h"
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_TRANS_JIT)
# if defined(_WIN32) || defined(__CYGWIN__)
/* only enable matcopy code generation (workaround issue with taking GP registers correctly) */
#   define LIBXS_TRANS_JIT 1
# else
#   define LIBXS_TRANS_JIT 3
# endif
#endif

/* min. tile-size is 3x3 rather than 2x2 to avoid remainder tiles of 1x1 */
#if !defined(LIBXS_TRANS_TMIN)
# define LIBXS_TRANS_TMIN 3
#endif


LIBXS_API_INTERN void libxs_trans_init(int archid)
{
  /* setup tile sizes according to CPUID or environment (LIBXS_TRANS_M, LIBXS_TRANS_N) */
  static unsigned int config_tm[/*config*/][2/*DP/SP*/] = {
    /* generic (hsw) */ { 2, 2 },
    /* mic (knl/knm) */ { 2, 2 },
    /* core (skx)    */ { 2, 2 }
  };
  { /* check if JIT-code generation is permitted */
    const char *const env_jit = getenv("LIBXS_TRANS_JIT");
    /* determine if JIT-kernels are used (0: none, 1: matcopy, 2: transpose, 3: matcopy+transpose). */
    libxs_trans_jit = ((0 == env_jit || 0 == *env_jit) ? (LIBXS_TRANS_JIT) : atoi(env_jit));
  }
  { /* load/adjust tile sizes */
    const char *const env_m = getenv("LIBXS_TRANS_M"), *const env_n = getenv("LIBXS_TRANS_N");
    const int m = ((0 == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((0 == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    int i;
    if (LIBXS_X86_AVX512_CORE <= archid) {
      libxs_trans_mtile = config_tm[2];
      libxs_trans_tile_stretch = 32.f;
    }
    else if (LIBXS_X86_AVX512_MIC <= archid && LIBXS_X86_AVX512_CORE > archid) {
      libxs_trans_mtile = config_tm[1];
      libxs_trans_tile_stretch = 32.f;
    }
    else {
      libxs_trans_mtile = config_tm[0];
      libxs_trans_tile_stretch = 32.f;
    }
    for (i = 0; i < 2/*DP/SP*/; ++i) {
      if (0 < m) libxs_trans_mtile[i] = LIBXS_MAX(m, LIBXS_TRANS_TMIN);
      if (0 < n) libxs_trans_tile_stretch = ((float)n) / libxs_trans_mtile[i];
      if (LIBXS_TRANS_TMIN > (libxs_trans_tile_stretch * libxs_trans_mtile[i])) {
        const float stretch = ((float)(LIBXS_TRANS_TMIN)) / libxs_trans_mtile[i];
        libxs_trans_tile_stretch = LIBXS_MAX(stretch, libxs_trans_tile_stretch);
      }
    }
  }
  { /* determines if OpenMP tasks are used (when available) */
    const char *const env_t = getenv("LIBXS_TRANS_TASKS");
    libxs_trans_taskscale = ((0 == env_t || 0 == *env_t)
      ? 0/*disabled*/ : (LIBXS_TRANS_TASKSCALE * atoi(env_t)));
  }
}


LIBXS_API_INTERN void libxs_trans_finalize(void)
{
}


LIBXS_API void libxs_matcopy_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo, const int* prefetch,
  unsigned int tm, unsigned int tn, libxs_xmcopyfunction kernel,
  int tid, int nthreads)
{
  const int mtasks = (m + tm - 1) / tm;
  unsigned int m0, m1, n0, n1;

  LIBXS_ASSERT_MSG(tid < nthreads && 0 < nthreads, "Invalid task setup!");
  LIBXS_ASSERT_MSG(tm <= m && tn <= n, "Invalid problem size!");
  LIBXS_ASSERT_MSG(0 < tm && 0 < tn, "Invalid tile size!");
  LIBXS_ASSERT_MSG(typesize <= 255, "Invalid type-size!");

  if (nthreads <= mtasks) { /* parallelized over M */
    const unsigned int mt = (m + nthreads - 1) / nthreads;
    m0 = LIBXS_MIN(tid * mt, m); m1 = LIBXS_MIN(m0 + mt, m);
    n0 = 0; n1 = n;
  }
  else { /* parallelized over M and N */
    const int ntasks = nthreads / mtasks;
    const int mtid = tid / ntasks, ntid = tid - mtid * ntasks;
    const unsigned int nt = (((n + ntasks - 1) / ntasks + tn - 1) / tn) * tn;
    m0 = LIBXS_MIN(mtid * tm, m); m1 = LIBXS_MIN(m0 + tm, m);
    n0 = LIBXS_MIN(ntid * nt, n); n1 = LIBXS_MIN(n0 + nt, n);
  }

  LIBXS_ASSERT_MSG(m0 <= m1 && m1 <= m, "Invalid task size!");
  LIBXS_ASSERT_MSG(n0 <= n1 && n1 <= n, "Invalid task size!");

  if (0 != prefetch && 0 != *prefetch) { /* prefetch */
    libxs_matcopy_internal_pf(out, in, typesize, ldi, ldo,
      m0, m1, n0, n1, tm, tn, kernel);
  }
  else { /* no prefetch */
    libxs_matcopy_internal(out, in, typesize, ldi, ldo,
      m0, m1, n0, n1, tm, tn, kernel);
  }
}


LIBXS_API_INTERN void libxs_matcopy_internal_pf(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxs_xmcopyfunction kernel)
{
  LIBXS_XCOPY(LIBXS_MCOPY_KERNEL, LIBXS_MCOPY_CALL, kernel,
    out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1,
    LIBXS_XALIGN_MCOPY);
}


LIBXS_API_INTERN void libxs_matcopy_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxs_xmcopyfunction kernel)
{
  LIBXS_XCOPY(LIBXS_MCOPY_KERNEL, LIBXS_MCOPY_CALL_NOPF, kernel,
    out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1,
    LIBXS_XALIGN_MCOPY);
}


LIBXS_API void libxs_matcopy_thread(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo,
  const int* prefetch, int tid, int nthreads)
{
  LIBXS_INIT
  if (0 < typesize && m <= ldi && m <= ldo && out != in &&
    ((0 != out && 0 < m && 0 < n) || (0 == m && 0 == n)) &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < nthreads)
  {
    unsigned int tm = libxs_trans_mtile[4 < typesize ? 0 : 1];
    unsigned int tn = (unsigned int)(libxs_trans_tile_stretch * tm);
    libxs_xmcopyfunction kernel = NULL;
    if (m < (libxs_blasint)tm || n < (libxs_blasint)tn) {
      if (1 < nthreads) {
        const unsigned int tasksize = (((unsigned int)m) * n) / ((unsigned int)(nthreads * libxs_trans_tile_stretch));
        const unsigned int nn = libxs_isqrt_u32(tasksize);
        const unsigned int mm = (unsigned int)(libxs_trans_tile_stretch * nn);
        tn = LIBXS_CLMP((unsigned int)n, 1, nn);
        tm = LIBXS_CLMP((unsigned int)m, 1, mm);
      }
      else {
        tm = m; tn = n;
      }
    }
    else {
      const int iprefetch = (0 == prefetch ? 0 : *prefetch);
      const libxs_mcopy_descriptor* desc;
      libxs_descriptor_blob blob;
      if (0 != (1 & libxs_trans_jit) /* JIT'ted matrix-copy permitted? */
        && NULL != (desc = libxs_mcopy_descriptor_init(&blob, typesize,
        (unsigned int)tm, (unsigned int)tn, (unsigned int)ldo, (unsigned int)ldi,
          0 != in ? 0 : LIBXS_MATCOPY_FLAG_ZERO_SOURCE, iprefetch, NULL/*default unroll*/)))
      {
        kernel = libxs_dispatch_mcopy(desc);
      }
    }
    libxs_matcopy_thread_internal(out, in, typesize,
      (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
      prefetch, tm, tn, kernel, tid, nthreads);
  }
  else {
    static int error_once = 0;
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      if (0 > tid || tid >= nthreads) {
        fprintf(stderr, "LIBXS ERROR: the matrix-copy thread-id or number of threads is incorrect!\n");
      }
      else if (0 == out) {
        fprintf(stderr, "LIBXS ERROR: the matrix-copy input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXS ERROR: output and input of the matrix-copy must be different!\n");
      }
      else if (0 == typesize) {
        fprintf(stderr, "LIBXS ERROR: the type-size of the matrix-copy is zero!\n");
      }
      else if (0 >= m || 0 >= n) {
        fprintf(stderr, "LIBXS ERROR: the matrix extent(s) of the matrix-copy is/are zero or negative!\n");
      }
      else {
        LIBXS_ASSERT(ldi < m || ldo < m);
        fprintf(stderr, "LIBXS ERROR: the leading dimension(s) of the matrix-copy is/are too small!\n");
      }
    }
  }
}


LIBXS_API void libxs_matcopy(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo,
  const int* prefetch)
{
  libxs_matcopy_thread(out, in, typesize, m, n, ldi, ldo, prefetch, 0/*tid*/, 1/*nthreads*/);
}


LIBXS_API void libxs_otrans_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int tm, unsigned int tn, libxs_xtransfunction kernel,
  int tid, int nthreads)
{
  const int mtasks = (m + tm - 1) / tm;
  unsigned int m0, m1, n0, n1;

  LIBXS_ASSERT_MSG(tid < nthreads && 0 < nthreads, "Invalid task setup!");
  LIBXS_ASSERT_MSG(tm <= m && tn <= n, "Invalid problem size!");
  LIBXS_ASSERT_MSG(0 < tm && 0 < tn, "Invalid tile size!");
  LIBXS_ASSERT_MSG(typesize <= 255, "Invalid type-size!");

  if (nthreads <= mtasks) { /* parallelized over M */
    const unsigned int mt = (m + nthreads - 1) / nthreads;
    m0 = LIBXS_MIN(tid * mt, m); m1 = LIBXS_MIN(m0 + mt, m);
    n0 = 0; n1 = n;
  }
  else { /* parallelized over M and N */
    const int ntasks = nthreads / mtasks;
    const int mtid = tid / ntasks, ntid = tid - mtid * ntasks;
    const unsigned int nt = (((n + ntasks - 1) / ntasks + tn - 1) / tn) * tn;
    m0 = LIBXS_MIN(mtid * tm, m); m1 = LIBXS_MIN(m0 + tm, m);
    n0 = LIBXS_MIN(ntid * nt, n); n1 = LIBXS_MIN(n0 + nt, n);
  }

  LIBXS_ASSERT_MSG(m0 <= m1 && m1 <= m, "Invalid task size!");
  LIBXS_ASSERT_MSG(n0 <= n1 && n1 <= n, "Invalid task size!");

  libxs_otrans_internal(out, in, typesize, ldi, ldo, m0, m1, n0, n1, tm, tn, kernel);
}


LIBXS_API_INTERN void libxs_otrans_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxs_xtransfunction kernel)
{
  LIBXS_XCOPY(LIBXS_TCOPY_KERNEL, LIBXS_TCOPY_CALL, kernel,
    out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1,
    LIBXS_XALIGN_TCOPY);
}


LIBXS_API void libxs_otrans_thread(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo,
  int tid, int nthreads)
{
  static int error_once = 0;
  LIBXS_INIT
  if (0 < typesize && m <= ldi && n <= ldo &&
    ((0 != out && 0 != in && 0 < m && 0 < n) || (0 == m && 0 == n)) &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < nthreads)
  {
    if (out != in) {
      unsigned int tm = libxs_trans_mtile[4 < typesize ? 0 : 1];
      unsigned int tn = (unsigned int)(libxs_trans_tile_stretch * tm);
      libxs_xtransfunction kernel = NULL;
      if ((unsigned int)m < tm || (unsigned int)n < tn) {
        const libxs_trans_descriptor* desc;
        libxs_descriptor_blob blob;
        if (1 < nthreads) {
          const unsigned int tasksize = (((unsigned int)m) * n) / ((unsigned int)(nthreads * libxs_trans_tile_stretch));
          const unsigned int nn = libxs_isqrt_u32(tasksize);
          const unsigned int mm = (unsigned int)(libxs_trans_tile_stretch * nn);
          tn = LIBXS_CLMP((unsigned int)n, 1, nn);
          tm = LIBXS_CLMP((unsigned int)m, 1, mm);
          if (0 != (2 & libxs_trans_jit) /* JIT'ted transpose permitted? */
            && NULL != (desc = libxs_trans_descriptor_init(&blob, typesize, tm, tn, (unsigned int)ldo)))
          {
            kernel = libxs_dispatch_trans(desc);
          }
        }
        else {
          if (0 != (2 & libxs_trans_jit) /* JIT'ted transpose permitted? */
            && NULL != (desc = libxs_trans_descriptor_init(&blob, typesize, (unsigned int)m, (unsigned int)n, (unsigned int)ldo))
            && NULL != (kernel = libxs_dispatch_trans(desc))) /* JIT-kernel available */
          {
            LIBXS_TCOPY_CALL(kernel, typesize, in, ldi, out, ldo);
            return; /* fast path */
          }
          tm = m; tn = n;
        }
      }
      libxs_otrans_thread_internal(out, in, typesize,
        (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
        tm, tn, kernel, tid, nthreads);
    }
    else if (ldi == ldo) {
      libxs_itrans(out, typesize, m, n, ldi);
    }
    else if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: output and input of the transpose must be different!\n");
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      if (0 > tid || tid >= nthreads) {
        fprintf(stderr, "LIBXS ERROR: the transpose thread-id or number of threads is incorrect!\n");
      }
      else if (0 == out || 0 == in) {
        fprintf(stderr, "LIBXS ERROR: the transpose input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXS ERROR: output and input of the transpose must be different!\n");
      }
      else if (0 == typesize) {
        fprintf(stderr, "LIBXS ERROR: the type-size of the transpose is zero!\n");
      }
      else if (0 >= m || 0 >= n) {
        fprintf(stderr, "LIBXS ERROR: the matrix extent(s) of the transpose is/are zero or negative!\n");
      }
      else {
        LIBXS_ASSERT(ldi < m || ldo < n);
        fprintf(stderr, "LIBXS ERROR: the leading dimension(s) of the transpose is/are too small!\n");
      }
    }
  }
}


LIBXS_API void libxs_otrans(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  libxs_otrans_thread(out, in, typesize, m, n, ldi, ldo, 0/*tid*/, 1/*nthreads*/);
}


LIBXS_API void libxs_itrans(void* inout, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ld)
{
  static int error_once = 0;
  LIBXS_INIT
  if (0 != inout) {
    if (m == n) { /* some fall-back; still warned as "not implemented" */
      libxs_blasint i, j;
      for (i = 0; i < m; ++i) {
        for (j = 0; j < i; ++j) {
          char *const a = (char*)inout + ((size_t)i * ld + j) * typesize;
          char *const b = (char*)inout + ((size_t)j * ld + i) * typesize;
          unsigned int k;
          for (k = 0; k < typesize; ++k) {
            const char tmp = a[k];
            a[k] = b[k];
            b[k] = tmp;
          }
        }
      }
#if defined(LIBXS_TRANS_CHECK)
      if ((1 < libxs_verbosity || 0 > libxs_verbosity) /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS WARNING: in-place transpose is not fully implemented!\n");
      }
#endif
    }
    else {
      if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: in-place transpose is not fully implemented!\n");
      }
      LIBXS_ASSERT(0/*TODO: proper implementation is pending*/);
    }
  }
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: the transpose input/output cannot be NULL!\n");
  }
}


#if defined(LIBXS_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_matcopy)(void* /*out*/, const void* /*in*/, const unsigned int* /*typesize*/,
  const libxs_blasint* /*m*/, const libxs_blasint* /*n*/, const libxs_blasint* /*ldi*/, const libxs_blasint* /*ldo*/,
  const int* /*prefetch*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_matcopy)(void* out, const void* in, const unsigned int* typesize,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* ldi, const libxs_blasint* ldo,
  const int* prefetch)
{
  libxs_blasint ldx;
  LIBXS_ASSERT(0 != typesize && 0 != m);
  ldx = *(0 != ldi ? ldi : m);
  libxs_matcopy(out, in, *typesize, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx, prefetch);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_otrans)(void* /*out*/, const void* /*in*/, const unsigned int* /*typesize*/,
  const libxs_blasint* /*m*/, const libxs_blasint* /*n*/, const libxs_blasint* /*ldi*/, const libxs_blasint* /*ldo*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_otrans)(void* out, const void* in, const unsigned int* typesize,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* ldi, const libxs_blasint* ldo)
{
  libxs_blasint ldx;
  LIBXS_ASSERT(0 != typesize && 0 != m);
  ldx = *(0 != ldi ? ldi : m);
  libxs_otrans(out, in, *typesize, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXS_API void LIBXS_FSYMBOL(libxs_itrans)(void* /*inout*/, const unsigned int* /*typesize*/,
  const libxs_blasint* /*m*/, const libxs_blasint* /*n*/, const libxs_blasint* /*ld*/);
LIBXS_API void LIBXS_FSYMBOL(libxs_itrans)(void* inout, const unsigned int* typesize,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* ld)
{
  LIBXS_ASSERT(0 != typesize && 0 != m);
  libxs_itrans(inout, *typesize, *m, *(n ? n : m), *(0 != ld ? ld : m));
}

#endif /*defined(LIBXS_BUILD)*/
