/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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
#include "libxs_gemm_diff.h"
#include "libxs_hash.h"
#include "libxs_cpuid.h"
#include "libxs_gemm.h"

#if defined(__TRACE)
# include "libxs_trace.h"
#endif

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
/* mute warning about target attribute; KNC/native plus JIT is disabled below! */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if !defined(NDEBUG)
# include <assert.h>
# include <errno.h>
#endif
#if defined(_WIN32)
# include <Windows.h>
#else
# include <sys/mman.h>
# include <pthread.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
#if defined(__GNUC__)
# if !defined(LIBXS_GCCATOMICS)
#   if (LIBXS_VERSION3(4, 7, 4) <= LIBXS_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#     define LIBXS_GCCATOMICS 1
#   else
#     define LIBXS_GCCATOMICS 0
#   endif
# endif
#endif

/**
 * LIBXS is agnostic with respect to the threading runtime!
 * LIBXS_OPENMP suppresses using OS primitives (PThreads)
 */
#if defined(_OPENMP) && !defined(LIBXS_OPENMP)
/*# define LIBXS_OPENMP*/
#endif

/* enable generic variant of libxs_gemm_diff */
#if !defined(LIBXS_GEMM_DIFF_SW) /*&& defined(__MIC__)*/
# define LIBXS_GEMM_DIFF_SW
#endif

/* alternative hash algorithm (instead of CRC32) */
#if !defined(LIBXS_HASH_BASIC) && !defined(LIBXS_REGSIZE)
# if !defined(LIBXS_SSE_MAX) || (4 > (LIBXS_SSE_MAX))
#   define LIBXS_HASH_BASIC
# endif
#endif

/* allow external definition to enable testing */
#if !defined(LIBXS_REGSIZE)
# define LIBXS_REGSIZE 524288 /* 524287: Mersenne Prime number */
# define LIBXS_HASH_MOD(N, NPOT) LIBXS_MOD2(N, NPOT)
#else
# define LIBXS_HASH_MOD(N, NGEN) ((N) % (NGEN))
#endif

#if defined(LIBXS_HASH_BASIC)
# define LIBXS_HASH_FUNCTION libxs_hash_npot
# define LIBXS_HASH_FUNCTION_CALL(HASH, INDX, HASH_FUNCTION, DESCRIPTOR) \
    HASH = (HASH_FUNCTION)(&(DESCRIPTOR), LIBXS_GEMM_DESCRIPTOR_SIZE, LIBXS_REGSIZE); \
    assert((LIBXS_REGSIZE) > (HASH)); \
    INDX = (HASH)
#else
# define LIBXS_HASH_FUNCTION libxs_crc32
# define LIBXS_HASH_FUNCTION_CALL(HASH, INDX, HASH_FUNCTION, DESCRIPTOR) \
    HASH = (HASH_FUNCTION)(&(DESCRIPTOR), LIBXS_GEMM_DESCRIPTOR_SIZE, 25071975/*seed*/); \
    INDX = LIBXS_HASH_MOD(HASH, LIBXS_REGSIZE)
#endif

/* flag fused into the memory address of a code version in case of collision */
#define LIBXS_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 1))

typedef union LIBXS_RETARGETABLE internal_code {
  libxs_xmmfunction xmm;
  /*const*/void* pmm;
  uintptr_t imm;
} internal_code;
typedef struct LIBXS_RETARGETABLE internal_regentry {
  libxs_gemm_descriptor descriptor;
  internal_code code;
  /* needed to distinct statically generated code and for munmap */
  unsigned int code_size;
} internal_regentry;

LIBXS_DEBUG(LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL unsigned int internal_ncollisions = 0;)
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL internal_regentry* internal_registry = 0;
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL const char* internal_arch_name = 0;
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL const char* internal_jit = 0;
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL int internal_has_crc32 = 0;

#if !defined(LIBXS_OPENMP)
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL LIBXS_LOCK_TYPE internal_reglock[] = {
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT
};
#endif

#if defined(__GNUC__)
  /* libxs_init already executed via GCC constructor attribute */
# define INTERNAL_FIND_CODE_INIT(VARIABLE) assert(0 != (VARIABLE))
#else /* lazy initialization */
  /* use return value of internal_init to refresh local representation */
# define INTERNAL_FIND_CODE_INIT(VARIABLE) if (0 == (VARIABLE)) VARIABLE = internal_init()
#endif

#if defined(LIBXS_OPENMP)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX) LIBXS_PRAGMA(omp critical(internal_reglock)) { \
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) }
#else
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX) { \
    const unsigned int LOCKINDEX = LIBXS_MOD2(INDEX, sizeof(internal_reglock) / sizeof(*internal_reglock)); \
    LIBXS_LOCK_ACQUIRE(internal_reglock[LOCKINDEX])
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(internal_reglock[LOCKINDEX]); }
#endif

#if (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(LIBXS_GCCATOMICS)
# if (0 != LIBXS_GCCATOMICS)
#   define INTERNAL_FIND_CODE_DECLARE(ENTRY) internal_regentry* ENTRY = __atomic_load_n(&internal_registry, __ATOMIC_RELAXED); unsigned int i
#   define INTERNAL_FIND_CODE_READ(ENTRY, DST) DST = __atomic_load_n(&((ENTRY)->code.pmm), __ATOMIC_SEQ_CST)
#   define INTERNAL_FIND_CODE_WRITE(ENTRY, SRC) __atomic_store_n(&((ENTRY)->code.pmm), SRC, __ATOMIC_SEQ_CST)
# else
#   define INTERNAL_FIND_CODE_DECLARE(ENTRY) internal_regentry* ENTRY = __sync_or_and_fetch(&internal_registry, 0); unsigned int i
#   define INTERNAL_FIND_CODE_READ(ENTRY, DST) DST = __sync_or_and_fetch(&((ENTRY)->code.pmm), 0)
#   define INTERNAL_FIND_CODE_WRITE(ENTRY, SRC) { \
      /*const*/void* old = (ENTRY)->code.pmm; \
      while (!__sync_bool_compare_and_swap(&((ENTRY)->code.pmm), old, SRC)) old = (ENTRY)->code.pmm; \
    }
# endif
#elif (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(_WIN32) /*TODO*/
# define INTERNAL_FIND_CODE_DECLARE(ENTRY) internal_regentry* ENTRY = internal_registry; unsigned int i
# define INTERNAL_FIND_CODE_READ(ENTRY, DST) DST = (ENTRY)->code.pmm
# define INTERNAL_FIND_CODE_WRITE(ENTRY, SRC) (ENTRY)->code.pmm = (SRC)
#else
# define INTERNAL_FIND_CODE_DECLARE(ENTRY) internal_regentry* ENTRY = internal_registry; unsigned int i
# define INTERNAL_FIND_CODE_READ(ENTRY, DST) DST = (ENTRY)->code.pmm
# define INTERNAL_FIND_CODE_WRITE(ENTRY, SRC) (ENTRY)->code.pmm = (SRC)
#endif

#define INTERNAL_FIND_CODE(DESCRIPTOR, ENTRY, HASH_FUNCTION, DIFF_FUNCTION) \
  internal_code internal_find_code_result; \
{ \
  unsigned int hash, diff = 0, diff0 = 0, i0; \
  INTERNAL_FIND_CODE_INIT(ENTRY); \
  /* check if the requested xGEMM is already JITted */ \
  LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */ \
  LIBXS_HASH_FUNCTION_CALL(hash, i = i0, HASH_FUNCTION, DESCRIPTOR); \
  (ENTRY) += i; /* actual entry */ \
  do { \
    INTERNAL_FIND_CODE_READ(ENTRY, internal_find_code_result.pmm); /* read registered code */ \
    /* entire block is conditional wrt LIBXS_JIT; static code currently does not have collisions */ \
    if (0 != internal_find_code_result.pmm) { \
      if (0 == diff0) { \
        if (0 == (LIBXS_HASH_COLLISION & internal_find_code_result.imm)) { /* check for no collision */ \
          /* calculate bitwise difference (deep check) */ \
          LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */ \
          diff = (DIFF_FUNCTION)(&(DESCRIPTOR), &((ENTRY)->descriptor)); \
          if (0 != diff) { /* new collision discovered (but no code version yet) */ \
            /* allow to fix-up current entry inside of the guarded/locked region */ \
            internal_find_code_result.pmm = 0; \
          } \
        } \
        /* collision discovered but code version exists; perform initial deep check */ \
        else if (0 != (DIFF_FUNCTION)(&(DESCRIPTOR), &((ENTRY)->descriptor))) { \
          /* continue linearly searching code starting at re-hashed index position */ \
          const unsigned int index = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE); \
          unsigned int next; \
          internal_regentry *const registry = (ENTRY) - i; /* recalculate base address */ \
          for (i0 = (index != i ? index : LIBXS_HASH_MOD(index + 1, LIBXS_REGSIZE)), \
            i = i0, next = LIBXS_HASH_MOD(i0 + 1, LIBXS_REGSIZE); next != i0/*no code found*/ && \
            /* skip any (still invalid) descriptor which corresponds to no code, or continue on difference */ \
            (0 == (ENTRY = (registry + i))->code.pmm || 0 != (diff = (DIFF_FUNCTION)(&(DESCRIPTOR), &((ENTRY)->descriptor)))); \
            i = next, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE)); \
          if (0 == diff) { /* found exact code version; continue with atomic load */ \
            continue; \
          } \
          else { /* no code found */ \
            internal_find_code_result.pmm = 0; \
            break; \
          } \
        } \
        else { /* clear the uppermost bit of the address */ \
          internal_find_code_result.imm &= ~LIBXS_HASH_COLLISION; \
        } \
      } \
      else { /* new collision discovered (but no code version yet) */ \
        internal_find_code_result.pmm = 0; \
      } \
    } \
    /* check if code generation or fix-up is needed, also check whether JIT is supported (CPUID) */ \
    if (0 == internal_find_code_result.pmm && 0 != internal_jit) { \
      INTERNAL_FIND_CODE_LOCK(lock, i); /* lock the registry entry */ \
      /* re-read registry entry after acquiring the lock */ \
      if (0 == diff) { \
        internal_find_code_result = (ENTRY)->code; \
        internal_find_code_result.imm &= ~LIBXS_HASH_COLLISION; \
      } \
      if (0 == internal_find_code_result.pmm) { /* double-check after acquiring the lock */ \
        if (0 == diff) { \
          /* found a conflict-free registry-slot, and attempt to build the kernel */ \
          internal_build(&(DESCRIPTOR), &internal_find_code_result, &((ENTRY)->code_size)); \
          if (0 != internal_find_code_result.pmm) { /* synchronize registry entry */ \
            (ENTRY)->descriptor = (DESCRIPTOR); \
            INTERNAL_FIND_CODE_WRITE(ENTRY, internal_find_code_result.pmm); \
          } \
        } \
        else { /* 0 != diff */ \
          const unsigned int base = i; \
          if (0 == diff0) { \
            /* flag existing entry as collision */ \
            /*const*/ void * /*const*/ code = (void*)((ENTRY)->code.imm | LIBXS_HASH_COLLISION); \
            /* find new slot to store the code version */ \
            const unsigned int index = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE); \
            i = (index != i ? index : LIBXS_HASH_MOD(index + 1, LIBXS_REGSIZE)); \
            i0 = i; /* keep starting point of free-slot-search in mind */ \
            LIBXS_DEBUG(++internal_ncollisions;) \
            INTERNAL_FIND_CODE_WRITE(ENTRY, code); /* fix-up existing entry */ \
            diff0 = diff; /* no more fix-up */ \
          } \
          else { \
            const unsigned int next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE); \
            if (next != i0) { /* linear search for free slot */ \
              i = next; \
            } \
            else { /* out of registry capacity (no free slot found) */ \
              diff = 0; \
            } \
          } \
          (ENTRY) -= base; /* recalculate base address */ \
          (ENTRY) += i; \
        } \
      } \
      INTERNAL_FIND_CODE_UNLOCK(lock); \
    } \
    else { \
      diff = 0; \
    } \
  } \
  while (0 != diff); \
} \
return internal_find_code_result.xmm

#define INTERNAL_DISPATCH(VECTOR_WIDTH, FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION) { \
  INTERNAL_FIND_CODE_DECLARE(entry); \
  union { libxs_gemm_descriptor descriptor; char simd[0!=(VECTOR_WIDTH)?(VECTOR_WIDTH):(LIBXS_GEMM_DESCRIPTOR_SIZE)]; } simd_descriptor; \
  LIBXS_GEMM_DESCRIPTOR(simd_descriptor.descriptor, 0 != (VECTOR_WIDTH) ? (VECTOR_WIDTH): LIBXS_ALIGNMENT, FLAGS, LIBXS_LD(M, N), LIBXS_LD(N, M), K, \
    0 == LIBXS_LD(PLDA, PLDB) ? LIBXS_LD(M, N) : *LIBXS_LD(PLDA, PLDB), \
    0 == LIBXS_LD(PLDB, PLDA) ? (K) : *LIBXS_LD(PLDB, PLDA), \
    0 == (PLDC) ? LIBXS_LD(M, N) : *(PLDC), \
    0 == (PALPHA) ? LIBXS_ALPHA : *(PALPHA), \
    0 == (PBETA) ? LIBXS_BETA : *(PBETA), \
    0 == (PREFETCH) ? LIBXS_PREFETCH : *(PREFETCH)); \
  for (i = LIBXS_GEMM_DESCRIPTOR_SIZE; i < sizeof(simd_descriptor.simd); ++i) simd_descriptor.simd[i] = 0; \
  { \
    INTERNAL_FIND_CODE(simd_descriptor.descriptor, entry, HASH_FUNCTION, DIFF_FUNCTION).SELECTOR; \
  } \
}

#define INTERNAL_SMMDISPATCH(VECTOR_WIDTH, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, HASH_FUNCTION, DIFF_FUNCTION) \
  INTERNAL_DISPATCH(VECTOR_WIDTH, (0 == (PFLAGS) ? LIBXS_FLAGS : *(PFLAGS)) | LIBXS_GEMM_FLAG_F32PREC, \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, smm, HASH_FUNCTION, DIFF_FUNCTION)

#define INTERNAL_DMMDISPATCH(VECTOR_WIDTH, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, HASH_FUNCTION, DIFF_FUNCTION) \
  INTERNAL_DISPATCH(VECTOR_WIDTH, (0 == (PFLAGS) ? LIBXS_FLAGS : *(PFLAGS)), \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, dmm, HASH_FUNCTION, DIFF_FUNCTION)

#if defined(LIBXS_GEMM_DIFF_MASK_A)
# define LIBXS_GEMM_DESCRIPTOR_XSIZE 0/*LIBXS_GEMM_DESCRIPTOR_SIZE*/
#else
# define LIBXS_GEMM_DESCRIPTOR_XSIZE 32
#endif


LIBXS_INLINE LIBXS_RETARGETABLE internal_regentry* internal_init(void)
{
  /*const*/internal_regentry* result;
  int i;

#if !defined(LIBXS_OPENMP)
  /* acquire locks and thereby shortcut lazy initialization later on */
  const int nlocks = sizeof(internal_reglock) / sizeof(*internal_reglock);
  for (i = 0; i < nlocks; ++i) LIBXS_LOCK_ACQUIRE(internal_reglock[i]);
#else
# pragma omp critical(internal_reglock)
#endif
  {
#if (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(LIBXS_GCCATOMICS)
# if (0 != LIBXS_GCCATOMICS)
    result = __atomic_load_n(&internal_registry, __ATOMIC_SEQ_CST);
# else
    result = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(_WIN32)
    result = internal_registry; /*TODO*/
#else
    result = internal_registry;
#endif
    if (0 == result) {
      int is_static = 0, init_code;
      /* decide using internal_has_crc32 instead of relying on a libxs_hash_function pointer
       * which will allow to inline the call instead of using an indirection (via fn. pointer)
       */
      internal_arch_name = libxs_cpuid(&is_static, &internal_has_crc32);
      init_code = libxs_gemm_init(internal_arch_name, 0/*auto-discovered*/, 0/*auto-discovered*/);
#if defined(__TRACE)
      const char *const env_trace_init = getenv("LIBXS_TRACE");
      if (EXIT_SUCCESS == init_code && 0 != env_trace_init) {
        int match[] = { 0, 0 }, filter_threadid = 0, filter_mindepth = 1, filter_maxnsyms = -1;
        char buffer[32];

        if (1 == sscanf(env_trace_init, "%32[^,],", buffer)) {
          sscanf(buffer, "%i", &filter_threadid);
        }
        if (1 == sscanf(env_trace_init, "%*[^,],%32[^,],", buffer)) {
          match[0] = sscanf(buffer, "%i", &filter_mindepth);
        }
        if (1 == sscanf(env_trace_init, "%*[^,],%*[^,],%32s", buffer)) {
          match[1] = sscanf(buffer, "%i", &filter_maxnsyms);
        }
        init_code = (0 == filter_threadid && 0 == match[0] && 0 == match[1]) ? EXIT_SUCCESS
          : libxs_trace_init(filter_threadid - 1, filter_mindepth, filter_maxnsyms);
      }
#endif
      if (EXIT_SUCCESS == init_code) {
        result = (internal_regentry*)malloc((LIBXS_REGSIZE + 1/*padding*/) * sizeof(internal_regentry));

        if (result) {
          if (0 != internal_has_crc32) {
#if !defined(LIBXS_SSE_MAX) || (4 > (LIBXS_SSE_MAX))
            internal_has_crc32 = 0;
# if !defined(NDEBUG) /* library code is expected to be mute */ && !defined(LIBXS_HASH_BASIC)
            fprintf(stderr, "LIBXS: CRC32 instructions are not accessible due to the compiler used!\n");
# endif
#endif
          }
#if !defined(NDEBUG) /* library code is expected to be mute */ && !defined(LIBXS_HASH_BASIC)
          else {
            fprintf(stderr, "LIBXS: CRC32 instructions are not available!\n");
          }
#endif
          for (i = 0; i < LIBXS_REGSIZE; ++i) result[i].code.pmm = 0;
          { /* omit registering code if JIT is enabled and if an ISA extension is found
             * which is beyond the static code path used to compile the library
             */
#if (0 != LIBXS_JIT) && !defined(__MIC__)
            const char *const env_jit = getenv("LIBXS_JIT");
            internal_jit = (0 == env_jit || 0 == *env_jit || '1' == *env_jit) ? internal_arch_name : ('0' != *env_jit ? env_jit : 0);
            if (0 == internal_jit || 0 != is_static)
#endif
            { /* open scope for variable declarations */
              LIBXS_DEBUG(unsigned int csp = 0, cdp = 0;)
              /* setup the dispatch table for the statically generated code */
#             include <libxs_dispatch.h>
#if !defined(NDEBUG) /* library code is expected to be mute */ && (0 != LIBXS_JIT)
# if defined(__MIC__)
                if (0 == internal_arch_name)
# else
                if (0 == internal_arch_name && (0 == env_jit || '1' == *env_jit))
# endif
                {
# if defined(LIBXS_SSE) && (3 <= (LIBXS_SSE))
                fprintf(stderr, "LIBXS: SSE instruction set extension is not supported for JIT-code generation!\n");
# elif defined(__MIC__)
                fprintf(stderr, "LIBXS: IMCI architecture (Xeon Phi coprocessor) is not supported for JIT-code generation!\n");
# else
                fprintf(stderr, "LIBXS: no instruction set extension found for JIT-code generation!\n");
# endif
              }
              if (0 < csp) {
                fprintf(stderr, "LIBXS: %u SP-kernels are not registered due to hash key collisions!\n", csp);
              }
              if (0 < cdp) {
                fprintf(stderr, "LIBXS: %u DP-kernels are not registered due to hash key collisions!\n", cdp);
              }
#endif
            }
          }
          atexit(libxs_finalize);
#if (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(LIBXS_GCCATOMICS)
# if (0 != LIBXS_GCCATOMICS)
          __atomic_store_n(&internal_registry, result, __ATOMIC_SEQ_CST);
# else
          {
            internal_regentry* old = internal_registry;
            while (!__sync_bool_compare_and_swap(&internal_registry, old, result)) old = internal_registry;
          }
# endif
#elif (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(_WIN32)
          internal_registry = result; /*TODO*/
#else
          internal_registry = result;
#endif
        }
      }
#if !defined(NDEBUG) && defined(__TRACE) /* library code is expected to be mute */
      else {
        fprintf(stderr, "LIBXS: failed to initialize sub-component (error #%i)!\n", init_code);
      }
#endif
    }
  }
#if !defined(LIBXS_OPENMP) /* release locks */
  for (i = 0; i < nlocks; ++i) LIBXS_LOCK_RELEASE(internal_reglock[i]);
#endif
  assert(result);
  return result;
}


LIBXS_EXTERN_C
#if defined(__GNUC__)
LIBXS_ATTRIBUTE(constructor)
#endif
LIBXS_RETARGETABLE void libxs_init(void)
{
#if (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(LIBXS_GCCATOMICS)
# if (0 != LIBXS_GCCATOMICS)
  const void *const registry = __atomic_load_n(&internal_registry, __ATOMIC_RELAXED);
# else
  const void *const registry = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(_WIN32)
  const void *const registry = internal_registry; /*TODO*/
#else
  const void *const registry = internal_registry;
#endif
  if (0 == registry) {
    internal_init();
  }
}


LIBXS_EXTERN_C
#if defined(__GNUC__)
LIBXS_ATTRIBUTE(destructor)
LIBXS_ATTRIBUTE(no_instrument_function)
#endif
LIBXS_RETARGETABLE void libxs_finalize(void)
{
#if (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(LIBXS_GCCATOMICS)
# if (0 != LIBXS_GCCATOMICS)
  internal_regentry* registry = __atomic_load_n(&internal_registry, __ATOMIC_SEQ_CST);
# else
  internal_regentry* registry = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(_WIN32)
  internal_regentry* registry = internal_registry; /*TODO*/
#else
  internal_regentry* registry = internal_registry;
#endif

  if (0 != registry) {
    int i;
#if !defined(LIBXS_OPENMP)
    /* acquire locks and thereby shortcut lazy initialization later on */
    const int nlocks = sizeof(internal_reglock) / sizeof(*internal_reglock);
    for (i = 0; i < nlocks; ++i) LIBXS_LOCK_ACQUIRE(internal_reglock[i]);
#else
#   pragma omp critical(internal_reglock)
#endif
    {
      registry = internal_registry;

      if (0 != registry) {
#if defined(__TRACE)
        i = libxs_trace_finalize();
# if !defined(NDEBUG) /* library code is expected to be mute */
        if (EXIT_SUCCESS != i) {
          fprintf(stderr, "LIBXS: failed to finalize trace (error #%i)!\n", i);
        }
# endif
#endif
        i = libxs_gemm_finalize();
# if !defined(NDEBUG) /* library code is expected to be mute */
        if (EXIT_SUCCESS != i) {
          fprintf(stderr, "LIBXS: failed to finalize (error #%i)!\n", i);
        }
# endif
#if (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(LIBXS_GCCATOMICS)
# if (0 != LIBXS_GCCATOMICS)
        __atomic_store_n(&internal_registry, 0, __ATOMIC_SEQ_CST);
# else
        { /* use store side-effect of built-in (dummy assignment to mute warning) */
          internal_regentry *const dummy = __sync_and_and_fetch(&internal_registry, 0);
          LIBXS_UNUSED(dummy);
        }
# endif
#elif (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(_WIN32)
        internal_registry = 0; /*TODO*/
#else
        internal_registry = 0;
#endif
#if defined(_WIN32)
        /* TODO: to be implemented */
        LIBXS_UNUSED(i);
#else
        for (i = 0; i < LIBXS_REGSIZE; ++i) {
          const unsigned int code_size = registry[i].code_size;
          internal_code code = registry[i].code;
          if (0 != code.pmm/*allocated*/ && 0 != code_size/*JIT*/) {
            /* make address valid by clearing an eventual collision flag */
            code.imm &= ~LIBXS_HASH_COLLISION;
# if defined(NDEBUG)
            munmap(code.pmm, code_size);
# else /* library code is expected to be mute */
            if (0 != munmap(code.pmm, code_size)) {
              const int error = errno;
              fprintf(stderr, "LIBXS: %s (munmap error #%i at %p+%u)!\n",
                strerror(error), error, code.pmm, code_size);
            }
# endif
          }
        }
#endif /*defined(__GNUC__)*/
        free((void*)registry);
#if !defined(NDEBUG) /* library code is expected to be mute */
        if (0 != internal_ncollisions) {
          fprintf(stderr, "LIBXS: %u hash key collisions found in the registry!\n", internal_ncollisions);
        }
#endif
      }
    }
#if !defined(LIBXS_OPENMP) /* release locks */
  for (i = 0; i < nlocks; ++i) LIBXS_LOCK_RELEASE(internal_reglock[i]);
#endif
  }
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_build(const libxs_gemm_descriptor* desc, internal_code* code, unsigned int* code_size)
{
#if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  libxs_generated_code generated_code;
  assert(0 != desc && 0 != code && 0 != code_size);
  assert(0 != internal_jit);
  assert(0 == code->pmm);

  /* allocate temporary buffer which is large enough to cover the generated code */
  generated_code.generated_code = malloc(131072 * sizeof(unsigned char));
  generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
  generated_code.code_size = 0;
  generated_code.code_type = 2;
  generated_code.last_error = 0;

  /* generate kernel */
  libxs_generator_dense_kernel(&generated_code, desc, internal_jit);

  /* handle an eventual error in the else-branch */
  if (0 == generated_code.last_error) {
#if defined(__APPLE__) && defined(__MACH__)
    const int fd = 0;
#else
    const int fd = open("/dev/zero", O_RDWR);
#endif
    if (0 <= fd) {
      /* create executable buffer */
      code->pmm = mmap(0, generated_code.code_size,
        /* must be a superset of what mprotect populates (see below) */
        PROT_READ | PROT_WRITE | PROT_EXEC,
#if defined(__APPLE__) && defined(__MACH__)
        MAP_ANON | MAP_PRIVATE, fd, 0);
#elif !defined(__CYGWIN__)
        MAP_PRIVATE | MAP_32BIT, fd, 0);
      close(fd);
#else
        MAP_PRIVATE, fd, 0);
      close(fd);
#endif
      if (MAP_FAILED != code->pmm) {
        /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
#if defined(MADV_NOHUGEPAGE)
# if defined(NDEBUG)
        madvise(code->pmm, generated_code.code_size, MADV_NOHUGEPAGE);
# else /* library code is expected to be mute */
        /* proceed even in case of an error, we then just take what we got (THP) */
        if (0 != madvise(code->pmm, generated_code.code_size, MADV_NOHUGEPAGE)) {
          static LIBXS_TLS int once = 0;
          if (0 == once) {
            const int error = errno;
            fprintf(stderr, "LIBXS: %s (madvise error #%i at %p)!\n",
              strerror(error), error, code->pmm);
            once = 1;
          }
        }
# endif /*defined(NDEBUG)*/
#elif !(defined(__APPLE__) && defined(__MACH__)) && !defined(__CYGWIN__)
        LIBXS_MESSAGE("================================================================================")
        LIBXS_MESSAGE("LIBXS: Adjusting THP is unavailable due to C89 or kernel older than 2.6.38!")
        LIBXS_MESSAGE("================================================================================")
#endif /*MADV_NOHUGEPAGE*/
        /* copy temporary buffer into the prepared executable buffer */
        memcpy(code->pmm, generated_code.generated_code, generated_code.code_size);

        if (0/*ok*/ == mprotect(code->pmm, generated_code.code_size, PROT_EXEC | PROT_READ)) {
#if !defined(NDEBUG) && defined(_DEBUG)
          /* write buffer for manual decode as binary to a file */
          char objdump_name[512];
          FILE* byte_code;
          sprintf(objdump_name, "kernel_%s_f%i_%c%c_m%u_n%u_k%u_lda%u_ldb%u_ldc%u_a%i_b%i_pf%i.bin",
            internal_jit /* best available/supported code path */,
            0 == (LIBXS_GEMM_FLAG_F32PREC & desc->flags) ? 64 : 32,
            0 == (LIBXS_GEMM_FLAG_TRANS_A & desc->flags) ? 'n' : 't',
            0 == (LIBXS_GEMM_FLAG_TRANS_B & desc->flags) ? 'n' : 't',
            desc->m, desc->n, desc->k, desc->lda, desc->ldb, desc->ldc,
            desc->alpha, desc->beta, desc->prefetch);
          byte_code = fopen(objdump_name, "wb");
          if (0 != byte_code) {
            fwrite(generated_code.generated_code, 1, generated_code.code_size, byte_code);
            fclose(byte_code);
          }
#endif /*!defined(NDEBUG) && defined(_DEBUG)*/
          /* free temporary/initial code buffer */
          free(generated_code.generated_code);
          /* finalize code generation */
          *code_size = generated_code.code_size;
        }
        else { /* there was an error with mprotect */
#if defined(NDEBUG)
          munmap(code->pmm, generated_code.code_size);
#else /* library code is expected to be mute */
          static LIBXS_TLS int once = 0;
          if (0 == once) {
            const int error = errno;
            fprintf(stderr, "LIBXS: %s (mprotect error #%i at %p+%u)!\n",
              strerror(error), error, code->pmm, generated_code.code_size);
            once = 1;
          }
          if (0 != munmap(code->pmm, generated_code.code_size)) {
            static LIBXS_TLS int once_mmap_error = 0;
            if (0 == once_mmap_error) {
              const int error = errno;
              fprintf(stderr, "LIBXS: %s (munmap error #%i at %p+%u)!\n",
                strerror(error), error, code->pmm, generated_code.code_size);
              once_mmap_error = 1;
            }
          }
#endif
          free(generated_code.generated_code);
        }
      }
      else {
#if !defined(NDEBUG) /* library code is expected to be mute */
        static LIBXS_TLS int once = 0;
        if (0 == once) {
          const int error = errno;
          fprintf(stderr, "LIBXS: %s (mmap allocation error #%i)!\n",
            strerror(error), error);
          once = 1;
        }
#endif
        free(generated_code.generated_code);
        /* clear MAP_FAILED value */
        code->pmm = 0;
      }
    }
#if !defined(NDEBUG)/* library code is expected to be mute */
    else {
      static LIBXS_TLS int once = 0;
      if (0 == once) {
        fprintf(stderr, "LIBXS: invalid file descriptor (%i)\n", fd);
        once = 1;
      }
    }
#endif
  }
  else {
#if !defined(NDEBUG) /* library code is expected to be mute */
    static LIBXS_TLS int once = 0;
    if (0 == once) {
      fprintf(stderr, "%s (error #%u)\n", libxs_strerror(generated_code.last_error),
        generated_code.last_error);
      once = 1;
    }
#endif
    free(generated_code.generated_code);
  }
#elif !defined(__MIC__)
  LIBXS_UNUSED(desc); LIBXS_UNUSED(code); LIBXS_UNUSED(code_size);
  LIBXS_MESSAGE("================================================================================")
  LIBXS_MESSAGE("LIBXS: The JIT BACKEND is currently not supported under Microsoft Windows!")
  LIBXS_MESSAGE("================================================================================")
#else
  LIBXS_UNUSED(desc); LIBXS_UNUSED(code); LIBXS_UNUSED(code_size);
#endif /*_WIN32*/
}


LIBXS_INLINE LIBXS_RETARGETABLE libxs_xmmfunction internal_xmmdispatch(const libxs_gemm_descriptor* descriptor)
{
  assert(descriptor);
  {
    INTERNAL_FIND_CODE_DECLARE(entry);
    {
#if defined(LIBXS_GEMM_DIFF_SW)
# if defined(LIBXS_HASH_BASIC)
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_hash_npot, libxs_gemm_diff);
# else
      const libxs_hash_function crc32_fun = 0 != internal_has_crc32 ? libxs_crc32_sse42 : libxs_crc32;
      INTERNAL_FIND_CODE(*descriptor, entry, crc32_fun, libxs_gemm_diff);
# endif
#elif defined(__MIC__)
# if defined(LIBXS_HASH_BASIC)
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_hash_npot, libxs_gemm_diff_imci);
# else
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_crc32, libxs_gemm_diff_imci);
# endif
#elif defined(LIBXS_AVX) && (2 <= (LIBXS_AVX))
# if defined(LIBXS_HASH_BASIC)
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_hash_npot, libxs_gemm_diff_avx2);
# else
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_crc32_sse42, libxs_gemm_diff_avx2);
# endif
#elif defined(LIBXS_AVX) && (1 <= (LIBXS_AVX))
# if defined(LIBXS_HASH_BASIC)
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_hash_npot, libxs_gemm_diff_avx);
# else
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_crc32_sse42, libxs_gemm_diff_avx);
# endif
#elif defined(LIBXS_SSE) && (4 <= (LIBXS_SSE))
# if defined(LIBXS_HASH_BASIC)
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_hash_npot, libxs_gemm_diff_sse);
# else
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_crc32_sse42, libxs_gemm_diff_sse);
# endif
#else
      const libxs_gemm_diff_function diff_fun = 0 != internal_arch_name ? libxs_gemm_diff_avx : (0 != internal_has_crc32 ? libxs_gemm_diff_sse : libxs_gemm_diff);
# if defined(LIBXS_HASH_BASIC)
      INTERNAL_FIND_CODE(*descriptor, entry, libxs_hash_npot, diff_fun);
# else
      const libxs_hash_function crc32_fun = 0 != internal_has_crc32 ? libxs_crc32_sse42 : libxs_crc32;
      INTERNAL_FIND_CODE(*descriptor, entry, crc32_fun, diff_fun);
# endif
#endif
    }
  }
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_xmmfunction libxs_xmmdispatch(const libxs_gemm_descriptor* descriptor)
{
  const libxs_xmmfunction null_mmfunction = { 0 };
  return 0 != descriptor ? internal_xmmdispatch(descriptor) : null_mmfunction;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_smmfunction libxs_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch)
{
#if defined(LIBXS_GEMM_DIFF_SW)
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_SMMDISPATCH(0, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff);
# else
  const libxs_hash_function crc32_fun = 0 != internal_has_crc32 ? libxs_crc32_sse42 : libxs_crc32;
  INTERNAL_SMMDISPATCH(0, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, crc32_fun, libxs_gemm_diff);
# endif
#elif defined(__MIC__)
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_SMMDISPATCH(2 * LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff_imci);
# else
  INTERNAL_SMMDISPATCH(2 * LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32, libxs_gemm_diff_imci);
# endif
#elif defined(LIBXS_AVX) && (2 <= (LIBXS_AVX))
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_SMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff_avx2);
# else
  INTERNAL_SMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32_sse42, libxs_gemm_diff_avx2);
# endif
#elif defined(LIBXS_AVX) && (1 <= (LIBXS_AVX))
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_SMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff_avx);
# else
  INTERNAL_SMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32_sse42, libxs_gemm_diff_avx);
# endif
#elif defined(LIBXS_SSE) && (4 <= (LIBXS_SSE))
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_SMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff_sse);
# else
  INTERNAL_SMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32_sse42, libxs_gemm_diff_sse);
# endif
#else
  const libxs_gemm_diff_function diff_fun = 0 != internal_arch_name ? libxs_gemm_diff_avx : (0 != internal_has_crc32 ? libxs_gemm_diff_sse : libxs_gemm_diff);
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_SMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, diff_fun);
# else
  const libxs_hash_function crc32_fun = 0 != internal_has_crc32 ? libxs_crc32_sse42 : libxs_crc32;
  INTERNAL_SMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, crc32_fun, diff_fun);
# endif
#endif
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dmmfunction libxs_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
#if defined(LIBXS_GEMM_DIFF_SW)
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_DMMDISPATCH(0, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff);
# else
  const libxs_hash_function crc32_fun = 0 != internal_has_crc32 ? libxs_crc32_sse42 : libxs_crc32;
  INTERNAL_DMMDISPATCH(0, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, crc32_fun, libxs_gemm_diff);
# endif
#elif defined(__MIC__)
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_DMMDISPATCH(2 * LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff_imci);
# else
  INTERNAL_DMMDISPATCH(2 * LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32, libxs_gemm_diff_imci);
# endif
#elif defined(LIBXS_AVX) && (2 <= (LIBXS_AVX))
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_DMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff_avx2);
# else
  INTERNAL_DMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32_sse42, libxs_gemm_diff_avx2);
# endif
#elif defined(LIBXS_AVX) && (1 <= (LIBXS_AVX))
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_DMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff_avx);
# else
  INTERNAL_DMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32_sse42, libxs_gemm_diff_avx);
# endif
#elif defined(LIBXS_SSE) && (4 <= (LIBXS_SSE))
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_DMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff_sse);
# else
  INTERNAL_DMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32_sse42, libxs_gemm_diff_sse);
# endif
#else
  const libxs_gemm_diff_function diff_fun = 0 != internal_arch_name ? libxs_gemm_diff_avx : (0 != internal_has_crc32 ? libxs_gemm_diff_sse : libxs_gemm_diff);
# if defined(LIBXS_HASH_BASIC)
  INTERNAL_DMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, diff_fun);
# else
  const libxs_hash_function crc32_fun = 0 != internal_has_crc32 ? libxs_crc32_sse42 : libxs_crc32;
  INTERNAL_DMMDISPATCH(LIBXS_GEMM_DESCRIPTOR_XSIZE, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, crc32_fun, diff_fun);
# endif
#endif
}

