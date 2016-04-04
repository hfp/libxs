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
#include "libxs_intrinsics_x86.h"
#include "libxs_cpuid_x86.h"
#include "libxs_gemm_diff.h"
#include "libxs_hash.h"
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
# if !defined(LIBXS_INTERNAL_MAP)
#   define LIBXS_INTERNAL_MAP MAP_PRIVATE
# endif
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

/* alternative hash algorithm (instead of CRC32) */
#if !defined(LIBXS_HASH_BASIC) && !defined(LIBXS_REGSIZE)
# if !defined(LIBXS_MAX_STATIC_TARGET_ARCH) || (LIBXS_X86_SSE4_2 > LIBXS_MAX_STATIC_TARGET_ARCH)
#   define LIBXS_HASH_BASIC
# endif
#endif

/* allow external definition to enable testing corner cases (exhausted registry space) */
#if !defined(LIBXS_REGSIZE)
# if defined(LIBXS_HASH_BASIC) /* consider larger registry to better deal with low-quality hash */
#   define LIBXS_REGSIZE /*1048576*/524288 /* no Mersenne Prime number required, but POT number */
# else
#   define LIBXS_REGSIZE 524288 /* 524287: Mersenne Prime number (2^19-1) */
# endif
# define LIBXS_HASH_MOD(N, NPOT) LIBXS_MOD2(N, NPOT)
#else
# define LIBXS_HASH_MOD(N, NGEN) ((N) % (NGEN))
#endif

#if !defined(LIBXS_CACHESIZE)
# define LIBXS_CACHESIZE 4
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

/** Helper macro determining the default prefetch strategy which is used for statically generated kernels. */
#if defined(_WIN32) || defined(__CYGWIN__) /*TODO: account for calling convention; avoid passing six arguments*/
# define INTERNAL_PREFETCH LIBXS_PREFETCH_NONE
#elif defined(__MIC__) && (0 > LIBXS_PREFETCH) /* auto-prefetch (frontend) */
# define INTERNAL_PREFETCH LIBXS_PREFETCH_AL2BL2_VIA_C
#elif (0 > LIBXS_PREFETCH) /* auto-prefetch (frontend) */
# define INTERNAL_PREFETCH LIBXS_PREFETCH_SIGONLY
#endif
#if !defined(INTERNAL_PREFETCH)
# define INTERNAL_PREFETCH LIBXS_PREFETCH
#endif

LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL int internal_prefetch = LIBXS_MAX(INTERNAL_PREFETCH, 0);
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL int internal_target_arch = LIBXS_TARGET_ARCH_GENERIC;
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL const char* internal_target_archid = 0;

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

#if defined(LIBXS_CACHESIZE) && (0 < LIBXS_CACHESIZE)
# define INTERNAL_FIND_CODE_CACHE_DECL \
  static LIBXS_TLS union { char padding[32]; libxs_gemm_descriptor desc; } cache[LIBXS_CACHESIZE]; \
  static LIBXS_TLS internal_code cache_code[LIBXS_CACHESIZE]; \
  static LIBXS_TLS unsigned int cache_hit = LIBXS_CACHESIZE
# define INTERNAL_FIND_CODE_CACHE_BEGIN(DESCRIPTOR, RESULT) \
  assert(32 >= LIBXS_GEMM_DESCRIPTOR_SIZE); \
  /* search small cache starting with the last hit on record */ \
  i = libxs_gemm_diffn(DESCRIPTOR, &cache[0].desc, cache_hit, LIBXS_CACHESIZE, 32); \
  if (LIBXS_CACHESIZE > i) { /* cache hit */ \
    RESULT = cache_code[i]; \
    cache_hit = i; \
  } \
  else
# if defined(LIBXS_GEMM_DIFF_SW) && (2 == (LIBXS_GEMM_DIFF_SW)) /* most general implementation */
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(DESCRIPTOR, RESULT) \
    i = (cache_hit + LIBXS_CACHESIZE - 1) % LIBXS_CACHESIZE; \
    cache_code[i] = internal_find_code_result; \
    cache[i].desc = *(DESCRIPTOR); \
    cache_hit = i
# else
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(DESCRIPTOR, RESULT) \
    assert(/*is pot*/LIBXS_CACHESIZE == (1 << LIBXS_LOG2(LIBXS_CACHESIZE))); \
    i = LIBXS_MOD2(cache_hit + LIBXS_CACHESIZE - 1, LIBXS_CACHESIZE); \
    cache_code[i] = internal_find_code_result; \
    cache[i].desc = *(DESCRIPTOR); \
    cache_hit = i
# endif
#else
# define INTERNAL_FIND_CODE_CACHE_DECL
# define INTERNAL_FIND_CODE_CACHE_BEGIN(DESCRIPTOR, RESULT)
# define INTERNAL_FIND_CODE_CACHE_FINALIZE(DESCRIPTOR, RESULT)
#endif

#if (0 != LIBXS_JIT)
# define INTERNAL_FIND_CODE_JIT(DESCRIPTOR, ENTRY) \
  /* check if code generation or fix-up is needed, also check whether JIT is supported (CPUID) */ \
  if (0 == internal_find_code_result.pmm && LIBXS_X86_AVX <= internal_target_arch) { \
    /* instead of blocking others, a try-lock would allow to let others to fallback to BLAS (return 0) during lock-time */ \
    INTERNAL_FIND_CODE_LOCK(lock, i); /* lock the registry entry */ \
    /* re-read registry entry after acquiring the lock */ \
    if (0 == diff) { \
      internal_find_code_result = (ENTRY)->code; \
      internal_find_code_result.imm &= ~LIBXS_HASH_COLLISION; \
    } \
    if (0 == internal_find_code_result.pmm) { /* double-check after acquiring the lock */ \
      if (0 == diff) { \
        /* found a conflict-free registry-slot, and attempt to build the kernel */ \
        internal_build(DESCRIPTOR, &internal_find_code_result, &((ENTRY)->code_size)); \
        if (0 != internal_find_code_result.pmm) { /* synchronize registry entry */ \
          (ENTRY)->descriptor = *(DESCRIPTOR); \
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
  else
#else
# define INTERNAL_FIND_CODE_JIT(DESCRIPTOR, ENTRY)
#endif

#define INTERNAL_FIND_CODE(DESCRIPTOR, ENTRY, HASH_FUNCTION, DIFF_FUNCTION) \
  internal_code internal_find_code_result; \
{ \
  INTERNAL_FIND_CODE_CACHE_DECL; \
  unsigned int hash, diff = 0, diff0 = 0, i0; \
  INTERNAL_FIND_CODE_INIT(ENTRY); \
  INTERNAL_FIND_CODE_CACHE_BEGIN(DESCRIPTOR, internal_find_code_result) { \
    /* check if the requested xGEMM is already JITted */ \
    LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */ \
    LIBXS_HASH_FUNCTION_CALL(hash, i = i0, HASH_FUNCTION, *(DESCRIPTOR)); \
    (ENTRY) += i; /* actual entry */ \
    do { \
      INTERNAL_FIND_CODE_READ(ENTRY, internal_find_code_result.pmm); /* read registered code */ \
      if (0 != internal_find_code_result.pmm) { \
        if (0 == diff0) { \
          if (0 == (LIBXS_HASH_COLLISION & internal_find_code_result.imm)) { /* check for no collision */ \
            /* calculate bitwise difference (deep check) */ \
            LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */ \
            diff = (DIFF_FUNCTION)(DESCRIPTOR, &((ENTRY)->descriptor)); \
            if (0 != diff) { /* new collision discovered (but no code version yet) */ \
              /* allow to fix-up current entry inside of the guarded/locked region */ \
              internal_find_code_result.pmm = 0; \
            } \
          } \
          /* collision discovered but code version exists; perform deep check */ \
          else if (0 != (DIFF_FUNCTION)(DESCRIPTOR, &((ENTRY)->descriptor))) { \
            /* continue linearly searching code starting at re-hashed index position */ \
            const unsigned int index = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE); \
            unsigned int next; \
            internal_regentry *const registry = (ENTRY) - i; /* recalculate base address */ \
            for (i0 = (index != i ? index : LIBXS_HASH_MOD(index + 1, LIBXS_REGSIZE)), \
              i = i0, next = LIBXS_HASH_MOD(i0 + 1, LIBXS_REGSIZE); \
              /* skip any (still invalid) descriptor which corresponds to no code, or continue on difference */ \
              (0 == (ENTRY = (registry + i))->code.pmm || 0 != (diff = (DIFF_FUNCTION)(DESCRIPTOR, &((ENTRY)->descriptor)))) \
                /* entire registry was searched and no code version was found */ \
                && next != i0; \
              i = next, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE)); \
            if (0 == diff) { /* found exact code version; continue with atomic load */ \
              internal_find_code_result.pmm = (ENTRY)->code.pmm; \
              /* clear the uppermost bit of the address */ \
              internal_find_code_result.imm &= ~LIBXS_HASH_COLLISION; \
            } \
            else { /* no code found */ \
              internal_find_code_result.pmm = 0; \
            } \
            break; \
          } \
          else { /* clear the uppermost bit of the address */ \
            internal_find_code_result.imm &= ~LIBXS_HASH_COLLISION; \
          } \
        } \
        else { /* new collision discovered (but no code version yet) */ \
          internal_find_code_result.pmm = 0; \
        } \
      } \
      INTERNAL_FIND_CODE_JIT(DESCRIPTOR, ENTRY) \
      { \
        diff = 0; \
      } \
    } \
    while (0 != diff); \
    INTERNAL_FIND_CODE_CACHE_FINALIZE(DESCRIPTOR, internal_find_code_result); \
  } \
} \
return internal_find_code_result.xmm

#define INTERNAL_DISPATCH_MAIN(DESCRIPTOR_DECL, DESC, FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION) { \
  INTERNAL_FIND_CODE_DECLARE(entry); \
  const signed char scalpha = (signed char)(0 == (PALPHA) ? LIBXS_ALPHA : *(PALPHA)), scbeta = (signed char)(0 == (PBETA) ? LIBXS_BETA : *(PBETA)); \
  if (0 == ((FLAGS) & (LIBXS_GEMM_FLAG_TRANS_A | LIBXS_GEMM_FLAG_TRANS_B)) && 1 == scalpha && (1 == scbeta || 0 == scbeta)) { \
    const int internal_dispatch_main_prefetch = (0 == (PREFETCH) ? INTERNAL_PREFETCH : *(PREFETCH)); \
    DESCRIPTOR_DECL; LIBXS_GEMM_DESCRIPTOR(*(DESC), 0 != (VECTOR_WIDTH) ? (VECTOR_WIDTH): LIBXS_ALIGNMENT, FLAGS, LIBXS_LD(M, N), LIBXS_LD(N, M), K, \
      0 == LIBXS_LD(PLDA, PLDB) ? LIBXS_LD(M, N) : *LIBXS_LD(PLDA, PLDB), \
      0 == LIBXS_LD(PLDB, PLDA) ? (K) : *LIBXS_LD(PLDB, PLDA), \
      0 == (PLDC) ? LIBXS_LD(M, N) : *(PLDC), scalpha, scbeta, \
      0 > internal_dispatch_main_prefetch ? internal_prefetch : internal_dispatch_main_prefetch); \
    { \
      INTERNAL_FIND_CODE(DESC, entry, HASH_FUNCTION, DIFF_FUNCTION).SELECTOR; \
    } \
  } \
  else { /* TODO: not supported (bypass) */ \
    return 0; \
  } \
}

#if defined(LIBXS_GEMM_DIFF_MASK_A) /* no padding i.e., LIBXS_GEMM_DESCRIPTOR_SIZE */
# define INTERNAL_DISPATCH(FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION) \
    INTERNAL_DISPATCH_MAIN(libxs_gemm_descriptor descriptor, &descriptor, \
    FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION)
#else /* padding: LIBXS_GEMM_DESCRIPTOR_SIZE -> LIBXS_ALIGNMENT */
# define INTERNAL_DISPATCH(FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION) { \
    INTERNAL_DISPATCH_MAIN(union { libxs_gemm_descriptor desc; char simd[LIBXS_ALIGNMENT]; } simd_descriptor; \
      for (i = LIBXS_GEMM_DESCRIPTOR_SIZE; i < sizeof(simd_descriptor.simd); ++i) simd_descriptor.simd[i] = 0, &simd_descriptor.desc, \
    FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION)
#endif

#define INTERNAL_SMMDISPATCH(PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, HASH_FUNCTION, DIFF_FUNCTION) \
  INTERNAL_DISPATCH((0 == (PFLAGS) ? LIBXS_FLAGS : *(PFLAGS)) | LIBXS_GEMM_FLAG_F32PREC, \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, smm, HASH_FUNCTION, DIFF_FUNCTION)

#define INTERNAL_DMMDISPATCH(PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, HASH_FUNCTION, DIFF_FUNCTION) \
  INTERNAL_DISPATCH((0 == (PFLAGS) ? LIBXS_FLAGS : *(PFLAGS)), \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, dmm, HASH_FUNCTION, DIFF_FUNCTION)


LIBXS_INLINE LIBXS_RETARGETABLE void internal_register_static_code(
  const libxs_gemm_descriptor* desc, unsigned int index, unsigned int hash, libxs_xmmfunction src,
  internal_regentry* dst, unsigned int* registered, unsigned int* total)
{
  assert(0 != desc && 0 != src.dmm && 0 != dst && 0 != registered && 0 != total);
  if (0 != dst->code.pmm) { /* collision? */
    /* start at a re-hashed index position */
    const unsigned int start = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE);
    internal_regentry *const registry = dst - index;
    unsigned int i0, i, next;

    /* mark current entry as a collision (this might be already the case) */
    dst->code.imm |= LIBXS_HASH_COLLISION;

    /* start linearly searching for an available slot */
    for (i = (start != index) ? start : LIBXS_HASH_MOD(start + 1, LIBXS_REGSIZE), i0 = i, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE);
      0 != (dst = registry + i)->code.pmm && next != i0; i = next, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE));
  }
  if (0 == dst->code.pmm) { /* registry not (yet) exhausted */
    dst->code.xmm = src;
    dst->code_size = 0; /* statically generated code */
    dst->descriptor = *desc;
    ++(*registered);
  }
  ++(*total);
}


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
      int init_code;
      const char *const env_jit = getenv("LIBXS_JIT");
      if (env_jit && *env_jit) {
        const int jit = atoi(env_jit);
        if (0 == strcmp("0", env_jit)) { /* suppress running libxs_cpuid_x86 */
          internal_target_archid = "generic";
        }
        else if (1 < jit) { /* suppress libxs_cpuid_x86 and override archid */
          switch (LIBXS_X86_GENERIC + jit) {
            case LIBXS_X86_AVX512: {
              internal_target_arch = LIBXS_X86_AVX512;
              internal_target_archid = "knl"; /* "skx" is fine too */
            } break;
            case LIBXS_X86_AVX2: {
              internal_target_arch = LIBXS_X86_AVX2;
              internal_target_archid = "hsw";
            } break;
            case LIBXS_X86_AVX: {
              internal_target_arch = LIBXS_X86_AVX;
              internal_target_archid = "snb";
            } break;
            default: if (LIBXS_X86_SSE3 <= (LIBXS_X86_GENERIC + jit)) {
              internal_target_arch = LIBXS_X86_GENERIC + jit;
              internal_target_archid = "sse";
            }
          }
        }
        else if (0 == strcmp("knl", env_jit) || 0 == strcmp("skx", env_jit)) {
          internal_target_arch = LIBXS_X86_AVX512;
          internal_target_archid = env_jit;
        }
        else if (0 == strcmp("hsw", env_jit)) {
          internal_target_arch = LIBXS_X86_AVX2;
          internal_target_archid = env_jit;
        }
        else if (0 == strcmp("snb", env_jit)) {
          internal_target_arch = LIBXS_X86_AVX;
          internal_target_archid = env_jit;
        }
      }
      if (0 == internal_target_archid) {
        internal_target_arch = libxs_cpuid_x86(&internal_target_archid);
        assert(0 != internal_target_archid);
      }
      { /* select prefetch strategy for JIT */
        const char *const env_prefetch = getenv("LIBXS_PREFETCH");
        if (0 == env_prefetch || 0 == *env_prefetch) {
          if (0 > LIBXS_PREFETCH) { /* permitted by LIBXS_PREFETCH_AUTO */
            assert(0 != internal_target_archid);
            internal_prefetch = 0 != strcmp("knl", internal_target_archid)
              ? LIBXS_PREFETCH_NONE : LIBXS_PREFETCH_AL2BL2_VIA_C;
          }
        }
        else { /* user input considered even if LIBXS_PREFETCH_AUTO is disabled */
          switch (atoi(env_prefetch)) {
            case 2: internal_prefetch = LIBXS_PREFETCH_SIGONLY; break;
            case 3: internal_prefetch = LIBXS_PREFETCH_BL2_VIA_C; break;
            case 4: internal_prefetch = LIBXS_PREFETCH_AL2; break;
            case 5: internal_prefetch = LIBXS_PREFETCH_AL2_AHEAD; break;
            case 6: internal_prefetch = LIBXS_PREFETCH_AL2BL2_VIA_C; break;
            case 7: internal_prefetch = LIBXS_PREFETCH_AL2BL2_VIA_C_AHEAD; break;
            case 8: internal_prefetch = LIBXS_PREFETCH_AL2_JPST; break;
            case 9: internal_prefetch = LIBXS_PREFETCH_AL2BL2_VIA_C_JPST; break;
            default: internal_prefetch = LIBXS_PREFETCH_NONE;
          }
        }
      }
      libxs_hash_init(internal_target_arch);
      libxs_gemm_diff_init(internal_target_arch);
      init_code = libxs_gemm_init(internal_target_archid, internal_prefetch);
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
          for (i = 0; i < LIBXS_REGSIZE; ++i) result[i].code.pmm = 0;
          /* omit registering code if JIT is enabled and if an ISA extension is found
           * which is beyond the static code path used to compile the library
           */
#if (0 != LIBXS_JIT) && !defined(__MIC__)
          if (LIBXS_STATIC_TARGET_ARCH >= internal_target_arch)
#endif
          { /* opening a scope for eventually declaring variables */
            unsigned int csp_tot = 0, csp_reg = 0, cdp_tot = 0, cdp_reg = 0;
            /* setup the dispatch table for the statically generated code */
#           include <libxs_dispatch.h>
#if !defined(NDEBUG) /* library code is expected to be mute */
            if (csp_reg < csp_tot) {
              fprintf(stderr, "LIBXS: %u of %u SP-kernels are not registered due to hash key collisions!\n", csp_tot - csp_reg, csp_tot);
            }
            if (cdp_reg < cdp_tot) {
              fprintf(stderr, "LIBXS: %u of %u DP-kernels are not registered due to hash key collisions!\n", cdp_tot - cdp_reg, cdp_tot);
            }
#endif
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
        libxs_gemm_finalize();
        libxs_gemm_diff_finalize();
        libxs_hash_finalize();
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
        { /* open scope to allocate variables */
          LIBXS_DEBUG(unsigned int njit = 0, nstatic = 0;)
          for (i = 0; i < LIBXS_REGSIZE; ++i) {
            const unsigned int code_size = registry[i].code_size;
            internal_code code = registry[i].code;
            if (0 != code.pmm/*potentially allocated*/) {
              if (0 != code_size/*JIT: actually allocated*/) {
                /* make address valid by clearing an eventual collision flag */
                code.imm &= ~LIBXS_HASH_COLLISION;
#if defined(_WIN32)
                /* TODO: executable memory buffer under Windows */
#else
# if defined(NDEBUG)
                munmap(code.pmm, code_size);
# else /* library code is expected to be mute */
                if (0 != munmap(code.pmm, code_size)) {
                  const int error = errno;
                  fprintf(stderr, "LIBXS: %s (munmap error #%i at %p+%u)!\n",
                    strerror(error), error, code.pmm, code_size);
                }
# endif
#endif
                LIBXS_DEBUG(++njit;)
              }
              else {
                LIBXS_DEBUG(++nstatic;)
              }
            }
          }
#if !defined(NDEBUG) /* library code is expected to be mute */
          fprintf(stderr, "LIBXS_JIT=%s NJIT=%u NSTATIC=%u", 0 != internal_target_archid ? internal_target_archid : "0", njit, nstatic);
          if (0 != internal_ncollisions) {
            fprintf(stderr, ": %u hash key collisions handled!\n", internal_ncollisions);
          }
          else {
            fprintf(stderr, "\n");
          }
#endif
        }
        free((void*)registry);
      }
    }
#if !defined(LIBXS_OPENMP) /* release locks */
  for (i = 0; i < nlocks; ++i) LIBXS_LOCK_RELEASE(internal_reglock[i]);
#endif
  }
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_get_target_arch()
{
#if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  return internal_target_arch;
#else /* no JIT support */
  return LIBXS_TARGET_ARCH_GENERIC;
#endif
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE const char* libxs_get_target_archid()
{
  return internal_target_archid;
}


/* function serves as a helper for implementing the Fortran interface */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void get_target_archid(char* name, int length)
{
  const char* c = internal_target_archid ? internal_target_archid : "";
  int i;
  assert(0 != name); /* valid here since function is not in the public interface */
  for (i = 0; i < length && 0 != *c; ++i, ++c) name[i] = *c;
  for (; i < length; ++i) name[i] = ' ';
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_build(const libxs_gemm_descriptor* desc, internal_code* code, unsigned int* code_size)
{
#if (0 != LIBXS_JIT)
# if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  libxs_generated_code generated_code;
  assert(0 != desc && 0 != code && 0 != code_size);
  assert(0 != internal_target_archid);
  assert(0 == code->pmm);

  /* allocate temporary buffer which is large enough to cover the generated code */
  generated_code.generated_code = malloc(131072);
  generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
  generated_code.code_size = 0;
  generated_code.code_type = 2;
  generated_code.last_error = 0;

  /* generate kernel */
  libxs_generator_gemm_kernel(&generated_code, desc, internal_target_archid);

  /* handle an eventual error in the else-branch */
  if (0 == generated_code.last_error) {
# if defined(__APPLE__) && defined(__MACH__)
    const int fd = 0;
# else
    const int fd = open("/dev/zero", O_RDWR);
# endif
    if (0 <= fd) {
      /* create executable buffer */
      code->pmm = mmap(0, generated_code.code_size,
        /* must be a superset of what mprotect populates (see below) */
        PROT_READ | PROT_WRITE | PROT_EXEC,
# if defined(__APPLE__) && defined(__MACH__)
        LIBXS_INTERNAL_MAP | MAP_ANON, fd, 0);
# elif !defined(__CYGWIN__)
        LIBXS_INTERNAL_MAP | MAP_32BIT, fd, 0);
      close(fd);
# else
        LIBXS_INTERNAL_MAP, fd, 0);
      close(fd);
# endif
      if (MAP_FAILED != code->pmm) {
        /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
# if defined(MADV_NOHUGEPAGE)
#  if defined(NDEBUG)
        madvise(code->pmm, generated_code.code_size, MADV_NOHUGEPAGE);
#  else /* library code is expected to be mute */
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
#  endif /*defined(NDEBUG)*/
# elif !(defined(__APPLE__) && defined(__MACH__)) && !defined(__CYGWIN__)
        LIBXS_MESSAGE("================================================================================")
        LIBXS_MESSAGE("LIBXS: Adjusting THP is unavailable due to C89 or kernel older than 2.6.38!")
        LIBXS_MESSAGE("================================================================================")
# endif /*MADV_NOHUGEPAGE*/
        /* copy temporary buffer into the prepared executable buffer */
        memcpy(code->pmm, generated_code.generated_code, generated_code.code_size);

        if (0/*ok*/ == mprotect(code->pmm, generated_code.code_size, PROT_EXEC | PROT_READ)) {
# if !defined(NDEBUG) && defined(_DEBUG)
          /* write buffer for manual decode as binary to a file */
          char objdump_name[512];
          FILE* byte_code;
          sprintf(objdump_name, "kernel_%s_f%i_%c%c_m%u_n%u_k%u_lda%u_ldb%u_ldc%u_a%i_b%i_pf%i.bin",
            internal_target_archid /* best available/supported code path */,
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
# endif /*!defined(NDEBUG) && defined(_DEBUG)*/
          /* free temporary/initial code buffer */
          free(generated_code.generated_code);
          /* finalize code generation */
          *code_size = generated_code.code_size;
        }
        else { /* there was an error with mprotect */
# if defined(NDEBUG)
          munmap(code->pmm, generated_code.code_size);
# else /* library code is expected to be mute */
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
# endif
          free(generated_code.generated_code);
        }
      }
      else {
# if !defined(NDEBUG) /* library code is expected to be mute */
        static LIBXS_TLS int once = 0;
        if (0 == once) {
          const int error = errno;
          fprintf(stderr, "LIBXS: %s (mmap allocation error #%i)!\n",
            strerror(error), error);
          once = 1;
        }
# endif
        free(generated_code.generated_code);
        /* clear MAP_FAILED value */
        code->pmm = 0;
      }
    }
# if !defined(NDEBUG)/* library code is expected to be mute */
    else {
      static LIBXS_TLS int once = 0;
      if (0 == once) {
        fprintf(stderr, "LIBXS: invalid file descriptor (%i)\n", fd);
        once = 1;
      }
    }
# endif
  }
  else {
# if !defined(NDEBUG) /* library code is expected to be mute */
    static LIBXS_TLS int once = 0;
    if (0 == once) {
      fprintf(stderr, "%s (error #%u)\n", libxs_strerror(generated_code.last_error),
        generated_code.last_error);
      once = 1;
    }
# endif
    free(generated_code.generated_code);
  }
# else
#   if !defined(__MIC__)
  LIBXS_MESSAGE("================================================================================")
  LIBXS_MESSAGE("LIBXS: The JIT BACKEND is currently not supported under Microsoft Windows!")
  LIBXS_MESSAGE("================================================================================")
#   endif
  LIBXS_UNUSED(desc); LIBXS_UNUSED(code); LIBXS_UNUSED(code_size);
  /* libxs_get_target_arch also serves as a runtime check whether JIT is available or not */
  assert(LIBXS_X86_AVX > libxs_get_target_arch());
# endif /*_WIN32*/
#endif /*LIBXS_JIT*/
}


LIBXS_INLINE LIBXS_RETARGETABLE libxs_xmmfunction internal_xmmdispatch(const libxs_gemm_descriptor* descriptor)
{
  INTERNAL_FIND_CODE_DECLARE(entry);
  assert(descriptor);
  {
#if defined(LIBXS_HASH_BASIC)
    INTERNAL_FIND_CODE(descriptor, entry, libxs_hash_npot, libxs_gemm_diff);
#else
    INTERNAL_FIND_CODE(descriptor, entry, libxs_crc32, libxs_gemm_diff);
#endif
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
#if defined(LIBXS_HASH_BASIC)
  INTERNAL_SMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff);
#else
  INTERNAL_SMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32, libxs_gemm_diff);
#endif
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dmmfunction libxs_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
#if defined(LIBXS_HASH_BASIC)
  INTERNAL_DMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_hash_npot, libxs_gemm_diff);
#else
  INTERNAL_DMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxs_crc32, libxs_gemm_diff);
#endif
}

