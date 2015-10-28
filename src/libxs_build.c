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
#include "generator_extern_typedefs.h"
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

#define LIBXS_BUILD_CACHESIZE ((LIBXS_MAX_MNK) * 8)
#if !defined(_WIN32)
#define LIBXS_BUILD_PAGESIZE sysconf(_SC_PAGESIZE)
#else
#define LIBXS_BUILD_PAGESIZE 4096
#endif
#define LIBXS_BUILD_SEED 0


/** Filled with zeros due to C language rule. */
LIBXS_RETARGETABLE libxs_function libxs_cache[2][(LIBXS_BUILD_CACHESIZE)];

#if !defined(_OPENMP)
LIBXS_RETARGETABLE LIBXS_LOCK_TYPE libxs_build_lock[] = {
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT
};
#endif


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_build_static(void)
{
  static int init = 0;

  if (0 == init) {
#if !defined(_OPENMP)
    int i;
    for (i = 0; i < 0; ++i) {
      LIBXS_LOCK_ACQUIRE(libxs_build_lock[i]);
    }
#else
#   pragma omp critical(libxs_build_lock)
#endif
    if (0 == init) {
#     include <libxs_build.h>
      init = 1;
    }
#if !defined(_OPENMP)
    for (i = 0; i < 0; ++i) {
      LIBXS_LOCK_RELEASE(libxs_build_lock[i]);
    }
#endif
  }
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_function libxs_build_jit(int single_precision, int m, int n, int k)
{
  libxs_function result = 0;

  /* calling libxs_build_jit shall imply an early/explicit initialization of the library, this is lazy initialization */
  libxs_build_static();

#if (0 != (LIBXS_JIT))
  {
    libxs_function *const cache = libxs_cache[single_precision&1];
    unsigned int hash, indx;

    /* build xgemm descriptor: LIBXS_XGEMM_DESCRIPTOR(DESCRIPTOR, M, N, K, LDA, LDB, LDC, PREFETCH, FLAGS, FALPHA, FBETA) */
    LIBXS_XGEMM_DESCRIPTOR_TYPE(l_xgemm_desc,
      m, n, k, m, k, LIBXS_ALIGN_STORES(m, 0 != single_precision ? sizeof(float) : sizeof(double)), LIBXS_PREFETCH,
      (0 == single_precision ? 0 : LIBXS_XGEMM_FLAG_F32PREC)
        | (1 < (LIBXS_ALIGNED_LOADS) ? LIBXS_XGEMM_FLAG_ALIGN_A : 0)
        | (1 < (LIBXS_ALIGNED_STORES) ? LIBXS_XGEMM_FLAG_ALIGN_C : 0),
      1/*alpha*/, LIBXS_BETA);

    /* check if the requested xGEMM is already JITted */
    LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */
    hash = libxs_crc32(&l_xgemm_desc, LIBXS_XGEMM_DESCRIPTOR_SIZE, LIBXS_BUILD_SEED);

    indx = hash % (LIBXS_BUILD_CACHESIZE);
    /* TODO: handle collision */
    result = cache[indx];

    if (0 == result) {
# if !defined(_WIN32)
# if !defined(_OPENMP)
      const unsigned int lock = LIBXS_MOD2(indx, sizeof(libxs_build_lock) / sizeof(*libxs_build_lock));
      LIBXS_LOCK_ACQUIRE(libxs_build_lock[lock]);
# else
#     pragma omp critical(libxs_build_lock)
# endif
      {
        result = cache[indx];

        if (0 == result) {
          int l_code_pages, l_code_page_size, l_fd;
          libxs_generated_code l_generated_code;
          char l_arch[14]; /* set arch string */
          union { /* used to avoid conversion warning */
            libxs_function pf;
            void* pv;
          } l_code;

# ifdef __SSE3__
#   ifndef __AVX__
#         error "SSE3 instructions set extensions have no jitting support!"
#   endif
# endif
# ifdef __MIC__
#         error "Xeon Phi coprocessors (IMCI architecture) have no jitting support!"
# endif
# ifdef __AVX__
          strcpy(l_arch, "snb");
# endif
# ifdef __AVX2__
          strcpy(l_arch, "hsw");
# endif
# ifdef __AVX512F__
          strcpy(l_arch, "knl");
# endif
          /* allocate buffer for code */
          l_generated_code.generated_code = malloc(131072 * sizeof(unsigned char));
          l_generated_code.buffer_size = 0 != l_generated_code.generated_code ? 131072 : 0;
          l_generated_code.code_size = 0;
          l_generated_code.code_type = 2;
          l_generated_code.last_error = 0;

          /* generate kernel */
          libxs_generator_dense_kernel(&l_generated_code, &l_xgemm_desc, l_arch);

          /* handle an eventual error */
          if (l_generated_code.last_error != 0) {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
            fprintf(stderr, "%s\n", libxs_strerror(l_generated_code.last_error));
# endif /*NDEBUG*/
            free(l_generated_code.generated_code);
            return 0;
          }

          /* create executable buffer */
          l_code_pages = (((l_generated_code.code_size-1)*sizeof(unsigned char))/(LIBXS_BUILD_PAGESIZE))+1;
          l_code_page_size = (LIBXS_BUILD_PAGESIZE)*l_code_pages;
          l_fd = open("/dev/zero", O_RDWR);
          l_code.pv = mmap(0, l_code_page_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, l_fd, 0);
          close(l_fd);

          /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
# if defined(MADV_NOHUGEPAGE)
          madvise(l_code.pv, l_code_page_size, MADV_NOHUGEPAGE);
# endif /*MADV_NOHUGEPAGE*/
          if (l_code.pv == MAP_FAILED) {
# if !defined(NDEBUG) /* library code is usually expected to be mute */
            fprintf(stderr, "LIBXS: something bad happend in mmap, couldn't allocate code buffer!\n");
# endif /*NDEBUG*/
            free(l_generated_code.generated_code);
            return 0;
          }

          memcpy( l_code.pv, l_generated_code.generated_code, l_generated_code.code_size );
          if (-1 == mprotect(l_code.pv, l_code_page_size, PROT_EXEC | PROT_READ)) {
# if !defined(NDEBUG)
            int errsv = errno;
            if (errsv == EINVAL) {
              fprintf(stderr, "LIBXS: mprotect failed: addr is not a valid pointer, or not a multiple of the system page size!\n");
            } else if (errsv == ENOMEM) {
              fprintf(stderr, "LIBXS: mprotect failed: Internal kernel structures could not be allocated!\n");
            } else if (errsv == EACCES) {
              fprintf(stderr, "LIBXS: mprotect failed: The memory cannot be given the specified access!\n");
            } else {
              fprintf(stderr, "LIBXS: mprotect failed: Unknown Error!\n");
            }
# endif /*NDEBUG*/
            free(l_generated_code.generated_code);
            return 0;
          }

# if !defined(NDEBUG)
          { /* write buffer for manual decode as binary to a file */
            char l_objdump_name[512];
            FILE* l_byte_code;
            sprintf(l_objdump_name, "kernel_prec%i_m%u_n%u_k%u_lda%u_ldb%u_ldc%u_a%i_b%i_ta%c_tb%c_pf%i.bin",
              0 == (LIBXS_XGEMM_FLAG_F32PREC & l_xgemm_desc.flags) ? 0 : 1,
              l_xgemm_desc.m, l_xgemm_desc.n, l_xgemm_desc.k, l_xgemm_desc.lda, l_xgemm_desc.ldb, l_xgemm_desc.ldc,
              l_xgemm_desc.alpha, l_xgemm_desc.beta,
              0 == (LIBXS_XGEMM_FLAG_TRANS_A & l_xgemm_desc.flags) ? 'n' : 't',
              0 == (LIBXS_XGEMM_FLAG_TRANS_B & l_xgemm_desc.flags) ? 'n' : 't',
              l_xgemm_desc.prefetch);
            l_byte_code = fopen(l_objdump_name, "wb");
            if (l_byte_code != NULL) {
              fwrite(l_generated_code.generated_code, 1, l_generated_code.code_size, l_byte_code);
              fclose(l_byte_code);
            }
          }
# endif /*NDEBUG*/
          /* free temporary buffer, and prepare return value */
          free(l_generated_code.generated_code);
          result = l_code.pf;

          /* make function pointer available for dispatch */
          cache[indx] = result;
        }
      }

# if !defined(_OPENMP)
      LIBXS_LOCK_RELEASE(libxs_build_lock[lock]);
# endif
# else
#     error "LIBXS ERROR: JITTING IS NOT SUPPORTED ON WINDOWS RIGHT NOW!"
# endif /*_WIN32*/
    }
  }
#else
  LIBXS_UNUSED(single_precision); LIBXS_UNUSED(m); LIBXS_UNUSED(n); LIBXS_UNUSED(k);
#endif /*LIBXS_JIT*/
  return result;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_smm_function libxs_smm_dispatch(int m, int n, int k)
{
#if 1 == (LIBXS_JIT) || 0 > (LIBXS_JIT) /* automatic JITting */
  return (libxs_smm_function)libxs_build_jit(1/*single precision*/, m, n, k);
#else /* explicit JITting and static code generation */
  unsigned int hash, indx;
  /* calling libxs_build_jit shall imply an early/explicit initialization of the librar, this is lazy initializationy */
  libxs_build_static();
  {
    /* build xgemm descriptor: LIBXS_XGEMM_DESCRIPTOR(DESCRIPTOR, M, N, K, LDA, LDB, LDC, PREFETCH, FLAGS, FALPHA, FBETA) */
    LIBXS_XGEMM_DESCRIPTOR_TYPE(desc,
      m, n, k, m, k, LIBXS_ALIGN_STORES(m, sizeof(float)), LIBXS_PREFETCH, LIBXS_XGEMM_FLAG_F32PREC
        | (1 < (LIBXS_ALIGNED_LOADS) ? LIBXS_XGEMM_FLAG_ALIGN_A : 0)
        | (1 < (LIBXS_ALIGNED_STORES) ? LIBXS_XGEMM_FLAG_ALIGN_C : 0),
      1/*alpha*/, LIBXS_BETA);
    LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */
    hash = libxs_crc32(&desc, LIBXS_XGEMM_DESCRIPTOR_SIZE, LIBXS_BUILD_SEED);
    indx = hash % (LIBXS_BUILD_CACHESIZE);
  }
  return (libxs_smm_function)libxs_cache[1/*single precision*/][indx];
#endif
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dmm_function libxs_dmm_dispatch(int m, int n, int k)
{
#if 1 == (LIBXS_JIT) || 0 > (LIBXS_JIT) /* automatic JITting */
  return (libxs_dmm_function)libxs_build_jit(0/*double precision*/, m, n, k);
#else /* explicit JITting and static code generation */
  unsigned int hash, indx;
  /* calling libxs_build_jit shall imply an early/explicit initialization of the library, this is lazy initialization */
  libxs_build_static();
  {
    /* build xgemm descriptor: LIBXS_XGEMM_DESCRIPTOR(DESCRIPTOR, M, N, K, LDA, LDB, LDC, PREFETCH, FLAGS, FALPHA, FBETA) */
    LIBXS_XGEMM_DESCRIPTOR_TYPE(desc,
      m, n, k, m, k, LIBXS_ALIGN_STORES(m, sizeof(double)), LIBXS_PREFETCH, 0/*double-precision*/
        | (1 < (LIBXS_ALIGNED_LOADS) ? LIBXS_XGEMM_FLAG_ALIGN_A : 0)
        | (1 < (LIBXS_ALIGNED_STORES) ? LIBXS_XGEMM_FLAG_ALIGN_C : 0),
      1/*alpha*/, LIBXS_BETA);
    LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */
    hash = libxs_crc32(&desc, LIBXS_XGEMM_DESCRIPTOR_SIZE, LIBXS_BUILD_SEED);
    indx = hash % (LIBXS_BUILD_CACHESIZE);
  }
  return (libxs_dmm_function)libxs_cache[0/*double precision*/][indx];
#endif
}
