/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
#include "libxs_crc32.h"
#include <libxs_generator.h>
#include <libxs.h>

#if defined(LIBXS_OFFLOAD_BUILD)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_WIN32)
# include <Windows.h>
#else
# include <fcntl.h>
# include <unistd.h>
# include <sys/mman.h>
#endif
#if !defined(NDEBUG)
#include <errno.h>
#endif
#if defined(LIBXS_OFFLOAD_BUILD)
# pragma offload_attribute(pop)
#endif

/* rely on a "pseudo prime" number (Mersenne) to improve cache spread */
#define LIBXS_DISPATCH_CACHESIZE ((2U << LIBXS_NBITS(LIBXS_MAX_MNK * (0 != LIBXS_JIT ? 2 : 5))) - 1)
#define LIBXS_DISPATCH_HASH_SEED 0


typedef union LIBXS_RETARGETABLE libxs_dispatch_entry {
  libxs_sfunction smm;
  libxs_dfunction dmm;
  libxs_sxfunction sxmm;
  libxs_dxfunction dxmm;
  const void* pv;
} libxs_dispatch_entry;
LIBXS_RETARGETABLE libxs_dispatch_entry* libxs_dispatch_cache = 0;

#if !defined(_OPENMP)
LIBXS_RETARGETABLE LIBXS_LOCK_TYPE libxs_dispatch_lock[] = {
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT
};
#define LIBXS_DISPATCH_LOCKMASTER 0
#endif


LIBXS_INLINE LIBXS_RETARGETABLE void internal_init(void)
{
#if !defined(_OPENMP)
  /* acquire one of the locks as the master lock */
  LIBXS_LOCK_ACQUIRE(libxs_dispatch_lock[LIBXS_DISPATCH_LOCKMASTER]);
#else
# pragma omp critical(libxs_dispatch_lock)
#endif
  if (0 == libxs_dispatch_cache) {
    libxs_dispatch_entry *const buffer = (libxs_dispatch_entry*)malloc(
      LIBXS_DISPATCH_CACHESIZE * sizeof(libxs_dispatch_entry));
    assert(buffer);
    if (buffer) {
      int i;
      for (i = 0; i < LIBXS_DISPATCH_CACHESIZE; ++i) buffer[i].pv = 0;
      { /* open scope for variable declarations */
        /* setup the dispatch table for the statically generated code */
#       include <libxs_dispatch.h>
      }
      { /* acquire and release remaining locks to shortcut any lazy initialization later on */
        const int nlocks = sizeof(libxs_dispatch_lock) / sizeof(*libxs_dispatch_lock);
        for (i = 1; i < nlocks; ++i) {
          LIBXS_LOCK_ACQUIRE(libxs_dispatch_lock[i]);
          LIBXS_LOCK_RELEASE(libxs_dispatch_lock[i]);
        }
      }
      libxs_dispatch_cache = buffer;
    }
  }
#if !defined(_OPENMP)
  /* release the master lock */
  LIBXS_LOCK_RELEASE(libxs_dispatch_lock[LIBXS_DISPATCH_LOCKMASTER]);
#endif
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_init(void)
{
  if (0 == libxs_dispatch_cache) {
    internal_init();
  }
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_finalize(void)
{
  if (0 != libxs_dispatch_cache) {
#if !defined(_OPENMP)
    /* acquire one of the locks as the master lock */
    LIBXS_LOCK_ACQUIRE(libxs_dispatch_lock[LIBXS_DISPATCH_LOCKMASTER]);
#else
#   pragma omp critical(libxs_dispatch_lock)
#endif
    if (0 != libxs_dispatch_cache) {
      void *const buffer = libxs_dispatch_cache;
      libxs_dispatch_cache = 0;
      free(buffer);
    }
#if !defined(_OPENMP)
    /* release the master lock */
    LIBXS_LOCK_RELEASE(libxs_dispatch_lock[LIBXS_DISPATCH_LOCKMASTER]);
#endif
  }
}


LIBXS_INLINE LIBXS_RETARGETABLE libxs_dispatch_entry internal_build(const libxs_gemm_descriptor* desc)
{
  libxs_dispatch_entry result;
  unsigned int hash, indx;
  assert(0 != desc);

  /* lazy initialization */
  if (0 == libxs_dispatch_cache) {
    internal_init();
  }

  /* check if the requested xGEMM is already JITted */
  LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */
  hash = libxs_crc32(desc, LIBXS_GEMM_DESCRIPTOR_SIZE, LIBXS_DISPATCH_HASH_SEED);

  indx = hash % LIBXS_DISPATCH_CACHESIZE;
  result = libxs_dispatch_cache[indx]; /* TODO: handle collision */
#if (0 != LIBXS_JIT)
  if (0 == result.pv) {
# if !defined(_WIN32) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*allow code coverage with Cygwin; fails at runtime!*/)
# if !defined(_OPENMP)
    const unsigned int lock = LIBXS_MOD2(indx, sizeof(libxs_dispatch_lock) / sizeof(*libxs_dispatch_lock));
    LIBXS_LOCK_ACQUIRE(libxs_dispatch_lock[lock]);
# else
#   pragma omp critical(libxs_dispatch_lock)
# endif
    {
      result = libxs_dispatch_cache[indx];

      if (0 == result.pv) {
        char l_arch[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; /* empty initial arch string */
        libxs_generated_code l_generated_code;
        void* l_code;

# if defined(__AVX512F__)
        strcpy(l_arch, "knl");
# elif defined(__AVX2__)
        strcpy(l_arch, "hsw");
# elif defined(__AVX__)
        strcpy(l_arch, "snb");
# elif defined(__SSE3__)
#       error "SSE3 instruction set extension is not supported for JIT-code generation!"
# elif defined(__MIC__)
#       error "IMCI architecture (Xeon Phi coprocessor) is not supported for JIT-code generation!"
# else
#       error "No instruction set extension found for JIT-code generation!"
# endif

        /* allocate buffer for code */
        l_generated_code.generated_code = malloc(131072 * sizeof(unsigned char));
        l_generated_code.buffer_size = 0 != l_generated_code.generated_code ? 131072 : 0;
        l_generated_code.code_size = 0;
        l_generated_code.code_type = 2;
        l_generated_code.last_error = 0;

        /* generate kernel */
        libxs_generator_dense_kernel(&l_generated_code, desc, l_arch);

        /* handle an eventual error */
        if (l_generated_code.last_error != 0) {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
          fprintf(stderr, "%s\n", libxs_strerror(l_generated_code.last_error));
# endif /*NDEBUG*/
          free(l_generated_code.generated_code);
          return result;
        }

        { /* create executable buffer */
          const int l_fd = open("/dev/zero", O_RDWR);
          /* must be a superset of what mprotect populates (see below) */
          const int perms = PROT_READ | PROT_WRITE | PROT_EXEC;
          l_code = mmap(0, l_generated_code.code_size, perms, MAP_PRIVATE, l_fd, 0);
          close(l_fd);
        }

        if (MAP_FAILED == l_code) {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
          fprintf(stderr, "LIBXS: mapping memory failed!\n");
# endif /*NDEBUG*/
          free(l_generated_code.generated_code);
          return result;
        }

        /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
# if defined(MADV_NOHUGEPAGE)
        { /* open new scope for variable declaration */
#   if !defined(NDEBUG)
          const int error =
#   endif
          madvise(l_code, l_generated_code.code_size, MADV_NOHUGEPAGE);
#   if !defined(NDEBUG) /* library code is usually expected to be mute */
          if (-1 == error) fprintf(stderr, "LIBXS: failed to advise page size!\n");
#   endif
        }
# endif /*MADV_NOHUGEPAGE*/
        memcpy(l_code, l_generated_code.generated_code, l_generated_code.code_size);
        if (-1 == mprotect(l_code, l_generated_code.code_size, PROT_EXEC | PROT_READ)) {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
          switch (errno) {
            case EINVAL: fprintf(stderr, "LIBXS: protecting memory failed (invalid pointer)!\n"); break;
            case ENOMEM: fprintf(stderr, "LIBXS: protecting memory failed (kernel out of memory)\n"); break;
            case EACCES: fprintf(stderr, "LIBXS: protecting memory failed (permission denied)!\n"); break;
            default: fprintf(stderr, "LIBXS: protecting memory failed (unknown error)!\n");
          }
          { /* open new scope for variable declaration */
            const int error =
# else
          {
# endif /*NDEBUG*/
            munmap(l_code, l_generated_code.code_size);
#   if !defined(NDEBUG) /* library code is usually expected to be mute */
            if (-1 == error) fprintf(stderr, "LIBXS: failed to unmap memory!\n");
#   endif
          }
          free(l_generated_code.generated_code);
          return result;
        }

# if !defined(NDEBUG)
        { /* write buffer for manual decode as binary to a file */
          char l_objdump_name[512];
          FILE* l_byte_code;
          sprintf(l_objdump_name, "kernel_prec%i_m%u_n%u_k%u_lda%u_ldb%u_ldc%u_a%i_b%i_ta%c_tb%c_pf%i.bin",
            0 == (LIBXS_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1,
            desc->m, desc->n, desc->k, desc->lda, desc->ldb, desc->ldc, desc->alpha, desc->beta,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & desc->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & desc->flags) ? 'n' : 't',
            desc->prefetch);
          l_byte_code = fopen(l_objdump_name, "wb");
          if (l_byte_code != NULL) {
            fwrite(l_generated_code.generated_code, 1, l_generated_code.code_size, l_byte_code);
            fclose(l_byte_code);
          }
        }
# endif /*NDEBUG*/
        /* free temporary buffer */
        free(l_generated_code.generated_code);

        /* make function pointer available for dispatch */
        libxs_dispatch_cache[indx].pv = l_code;
        
        /* prepare return value */
        result.pv = l_code;
      }
    }

# if !defined(_OPENMP)
    LIBXS_LOCK_RELEASE(libxs_dispatch_lock[lock]);
# endif
# else
#   error "LIBXS ERROR: JITTING IS NOT SUPPORTED ON WINDOWS RIGHT NOW!"
# endif /*_WIN32*/
  }
#endif /*LIBXS_JIT*/
  return result;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_sfunction libxs_sdispatch(
  int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const float* alpha, const float* beta)
{
  LIBXS_GEMM_DESCRIPTOR_TYPE(desc, LIBXS_ALIGNMENT, flags | LIBXS_GEMM_FLAG_F32PREC, m, n, k, lda, ldb, ldc,
    0 == alpha ? LIBXS_ALPHA : *alpha, 0 == beta ? LIBXS_BETA : *beta, LIBXS_PREFETCH);
  return internal_build(&desc).smm;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dfunction libxs_ddispatch(
  int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const double* alpha, const double* beta)
{
  LIBXS_GEMM_DESCRIPTOR_TYPE(desc, LIBXS_ALIGNMENT, flags, m, n, k, lda, ldb, ldc,
    0 == alpha ? LIBXS_ALPHA : *alpha, 0 == beta ? LIBXS_BETA : *beta, LIBXS_PREFETCH);
  return internal_build(&desc).dmm;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_sxfunction libxs_sxdispatch(
  int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const float* alpha, const float* beta, int prefetch)
{
  LIBXS_GEMM_DESCRIPTOR_TYPE(desc, LIBXS_ALIGNMENT, flags | LIBXS_GEMM_FLAG_F32PREC, m, n, k, lda, ldb, ldc,
    0 == alpha ? LIBXS_ALPHA : *alpha, 0 == beta ? LIBXS_BETA : *beta, prefetch);
  return internal_build(&desc).sxmm;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dxfunction libxs_dxdispatch(
  int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const double* alpha, const double* beta, int prefetch)
{
  LIBXS_GEMM_DESCRIPTOR_TYPE(desc, LIBXS_ALIGNMENT, flags, m, n, k, lda, ldb, ldc,
    0 == alpha ? LIBXS_ALPHA : *alpha, 0 == beta ? LIBXS_BETA : *beta, prefetch);
  return internal_build(&desc).dxmm;
}
