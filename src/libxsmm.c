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
#include "libxs_gemm_ext.h"
#include "libxs_hash.h"

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
#include <math.h>
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
#if defined(LIBXS_VTUNE)
# include <jitprofiling.h>
# define LIBXS_VTUNE_JITVERSION 2
# if (2 == LIBXS_VTUNE_JITVERSION)
#   define LIBXS_VTUNE_JIT_DESC_TYPE iJIT_Method_Load_V2
#   define LIBXS_VTUNE_JIT_LOAD iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V2
# else
#   define LIBXS_VTUNE_JIT_DESC_TYPE iJIT_Method_Load
#   define LIBXS_VTUNE_JIT_LOAD iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED
# endif
# define LIBXS_VTUNE_JIT_UNLOAD iJVM_EVENT_TYPE_METHOD_UNLOAD_START
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
/*#   define LIBXS_HASH_BASIC*/
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
# define LIBXS_HASH_FUNCTION_CALL(HASH, INDX, DESCRIPTOR) \
    HASH = libxs_hash_npot(&(DESCRIPTOR), LIBXS_GEMM_DESCRIPTOR_SIZE, LIBXS_REGSIZE); \
    assert((LIBXS_REGSIZE) > (HASH)); \
    INDX = (HASH)
#else
# define LIBXS_HASH_FUNCTION_CALL(HASH, INDX, DESCRIPTOR) \
    HASH = libxs_crc32(&(DESCRIPTOR), LIBXS_GEMM_DESCRIPTOR_SIZE, 25071975/*seed*/); \
    INDX = LIBXS_HASH_MOD(HASH, LIBXS_REGSIZE)
#endif

/* flag fused into the memory address of a code version in case of collision */
#define LIBXS_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 1))

#if 16 >= (LIBXS_GEMM_DESCRIPTOR_SIZE)
# define LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE 16
#elif 32 >= (LIBXS_GEMM_DESCRIPTOR_SIZE)
# define LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE 32
#else
# define LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE LIBXS_GEMM_DESCRIPTOR_SIZE
#endif

typedef union LIBXS_RETARGETABLE internal_regkey {
  char simd[LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE];
  libxs_gemm_descriptor descriptor;
} internal_regkey;

typedef struct LIBXS_RETARGETABLE internal_regentry {
  union {
    libxs_xmmfunction xmm;
    /*const*/void* pmm;
    uintptr_t imm;
  } function;
  /* statically generated code (=0), dynamically generated code (>0). */
  unsigned int size;
#if defined(LIBXS_VTUNE)
  unsigned int id;
#endif
} internal_regentry;

LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL struct LIBXS_RETARGETABLE {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic[2/*DP/SP*/][3/*sml/med/big*/];

LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL unsigned int internal_statistic_sml = 13;
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL unsigned int internal_statistic_med = 23;
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL unsigned int internal_statistic_mnk = LIBXS_MAX_M;

#if defined(NDEBUG)
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL int internal_verbose = 0; /* quiet */
#else
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL int internal_verbose = 1; /* verbose */
#endif

LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL internal_regkey* internal_registry_keys = 0;
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL internal_regentry* internal_registry = 0;
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL unsigned int internal_teardown = 0;

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
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL int internal_target_archid = LIBXS_TARGET_ARCH_GENERIC;

#if !defined(LIBXS_OPENMP)
LIBXS_RETARGETABLE LIBXS_VISIBILITY_INTERNAL LIBXS_LOCK_TYPE internal_reglock[] = {
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
  LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT
};
#endif

#if defined(__GNUC__)
# define LIBXS_INIT
  /* libxs_init already executed via GCC constructor attribute */
# define INTERNAL_FIND_CODE_INIT(VARIABLE) assert(0 != (VARIABLE))
#else /* lazy initialization */
# define LIBXS_INIT libxs_init();
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
#   define INTERNAL_FIND_CODE_DECLARE(CODE) internal_regentry* CODE = __atomic_load_n(&internal_registry, __ATOMIC_RELAXED); unsigned int i
#   define INTERNAL_FIND_CODE_READ(CODE, DST) DST = __atomic_load_n(&(CODE)->function.pmm, __ATOMIC_SEQ_CST)
#   define INTERNAL_FIND_CODE_WRITE(CODE, SRC) __atomic_store_n(&(CODE)->function.pmm, SRC, __ATOMIC_SEQ_CST);
# else
#   define INTERNAL_FIND_CODE_DECLARE(CODE) internal_regentry* CODE = __sync_or_and_fetch(&internal_registry, 0); unsigned int i
#   define INTERNAL_FIND_CODE_READ(CODE, DST) DST = __sync_or_and_fetch(&(CODE)->function.pmm, 0)
#   define INTERNAL_FIND_CODE_WRITE(CODE, SRC) { \
      /*const*/void* old = (CODE)->function.pmm; \
      while (!__sync_bool_compare_and_swap(&(CODE)->function.pmm, old, SRC)) { \
        old = (CODE)->function.pmm; \
      } \
    }
# endif
#elif (defined(_REENTRANT) || defined(LIBXS_OPENMP)) && defined(_WIN32) /*TODO*/
# define INTERNAL_FIND_CODE_DECLARE(CODE) internal_regentry* CODE = internal_registry; unsigned int i
# define INTERNAL_FIND_CODE_READ(CODE, DST) DST = (CODE)->function.pmm
# define INTERNAL_FIND_CODE_WRITE(CODE, SRC) (CODE)->function.pmm = (SRC)
#else
# define INTERNAL_FIND_CODE_DECLARE(CODE) internal_regentry* CODE = internal_registry; unsigned int i
# define INTERNAL_FIND_CODE_READ(CODE, DST) DST = (CODE)->function.pmm
# define INTERNAL_FIND_CODE_WRITE(CODE, SRC) (CODE)->function.pmm = (SRC)
#endif

#if defined(LIBXS_CACHESIZE) && (0 < (LIBXS_CACHESIZE))
# define INTERNAL_FIND_CODE_CACHE_DECL(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT) \
  static LIBXS_TLS union { libxs_gemm_descriptor desc; char padding[LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE]; } CACHE_KEYS[LIBXS_CACHESIZE]; \
  static LIBXS_TLS libxs_xmmfunction CACHE[LIBXS_CACHESIZE]; \
  static LIBXS_TLS unsigned int CACHE_ID = (unsigned int)(-1); \
  static LIBXS_TLS unsigned int CACHE_HIT = LIBXS_CACHESIZE
# define INTERNAL_FIND_CODE_CACHE_BEGIN(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR) \
  assert(LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE >= LIBXS_GEMM_DESCRIPTOR_SIZE); \
  /* search small cache starting with the last hit on record */ \
  i = libxs_gemm_diffn(DESCRIPTOR, &(CACHE_KEYS)->desc, CACHE_HIT, LIBXS_CACHESIZE, LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE); \
  if ((LIBXS_CACHESIZE) > i && (CACHE_ID) == internal_teardown) { /* cache hit, and valid */ \
    (RESULT).function.xmm = (CACHE)[i]; \
    CACHE_HIT = i; \
  } \
  else
# if defined(LIBXS_GEMM_DIFF_SW) && (2 == (LIBXS_GEMM_DIFF_SW)) /* most general implementation */
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR) \
    if ((CACHE_ID) != internal_teardown) { \
      memset(CACHE_KEYS, -1, sizeof(CACHE_KEYS)); \
      CACHE_ID = internal_teardown; \
    } \
    i = ((CACHE_HIT) + ((LIBXS_CACHESIZE) - 1)) % (LIBXS_CACHESIZE); \
    ((CACHE_KEYS)[i]).desc = *(DESCRIPTOR); \
    (CACHE)[i] = (RESULT).function.xmm; \
    CACHE_HIT = i
# else
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR) \
    assert(/*is pot*/(LIBXS_CACHESIZE) == (1 << LIBXS_LOG2(LIBXS_CACHESIZE))); \
    if ((CACHE_ID) != internal_teardown) { \
      memset(CACHE_KEYS, -1, sizeof(CACHE_KEYS)); \
      CACHE_ID = internal_teardown; \
    } \
    i = LIBXS_MOD2((CACHE_HIT) + ((LIBXS_CACHESIZE) - 1), LIBXS_CACHESIZE); \
    (CACHE_KEYS)[i].desc = *(DESCRIPTOR); \
    (CACHE)[i] = (RESULT).function.xmm; \
    CACHE_HIT = i
# endif
#else
# define INTERNAL_FIND_CODE_CACHE_DECL(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT)
# define INTERNAL_FIND_CODE_CACHE_BEGIN(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR)
# define INTERNAL_FIND_CODE_CACHE_FINALIZE(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR)
#endif

#if (0 != LIBXS_JIT)
# define INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, RESULT) \
  /* check if code generation or fix-up is needed, also check whether JIT is supported (CPUID) */ \
  if (0 == (RESULT).function.pmm /* code version does not exist */ && LIBXS_X86_AVX <= internal_target_archid) { \
    /* instead of blocking others, a try-lock would allow to let others to fallback to BLAS (return 0) during lock-time */ \
    INTERNAL_FIND_CODE_LOCK(lock, i); /* lock the registry entry */ \
    /* re-read registry entry after acquiring the lock */ \
    if (0 == diff) { \
      RESULT = *(CODE); \
      (RESULT).function.imm &= ~LIBXS_HASH_COLLISION; \
    } \
    if (0 == (RESULT).function.pmm) { /* double-check after acquiring the lock */ \
      if (0 == diff) { \
        /* found a conflict-free registry-slot, and attempt to build the kernel */ \
        internal_build(DESCRIPTOR, &(RESULT)); \
        internal_update_statistic(DESCRIPTOR, 1, 0); \
        if (0 != (RESULT).function.pmm) { /* synchronize registry entry */ \
          internal_registry_keys[i].descriptor = *(DESCRIPTOR); \
          *(CODE) = RESULT; \
          INTERNAL_FIND_CODE_WRITE(CODE, (RESULT).function.pmm); \
        } \
      } \
      else { /* 0 != diff */ \
        if (0 == diff0) { \
          /* flag existing entry as collision */ \
          /*const*/ void * /*const*/ collision = (void*)((CODE)->function.imm | LIBXS_HASH_COLLISION); \
          /* find new slot to store the code version */ \
          const unsigned int index = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE); \
          i = (index != i ? index : LIBXS_HASH_MOD(index + 1, LIBXS_REGSIZE)); \
          i0 = i; /* keep starting point of free-slot-search in mind */ \
          internal_update_statistic(DESCRIPTOR, 0, 1); \
          INTERNAL_FIND_CODE_WRITE(CODE, collision); /* fix-up existing entry */ \
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
        (CODE) = internal_registry + i; \
      } \
    } \
    INTERNAL_FIND_CODE_UNLOCK(lock); \
  } \
  else
#else
# define INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, RESULT)
#endif

#define INTERNAL_FIND_CODE(DESCRIPTOR, CODE) \
  internal_regentry flux_entry; \
{ \
  INTERNAL_FIND_CODE_CACHE_DECL(cache_id, cache_keys, cache, cache_hit); \
  unsigned int hash, diff = 0, diff0 = 0, i0; \
  INTERNAL_FIND_CODE_INIT(CODE); \
  INTERNAL_FIND_CODE_CACHE_BEGIN(cache_id, cache_keys, cache, cache_hit, flux_entry, DESCRIPTOR) { \
    /* check if the requested xGEMM is already JITted */ \
    LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */ \
    LIBXS_HASH_FUNCTION_CALL(hash, i = i0, *(DESCRIPTOR)); \
    (CODE) += i; /* actual entry */ \
    do { \
      INTERNAL_FIND_CODE_READ(CODE, flux_entry.function.pmm); /* read registered code */ \
      if (0 != flux_entry.function.pmm) { /* code version exists */ \
        if (0 == diff0) { \
          if (0 == (LIBXS_HASH_COLLISION & flux_entry.function.imm)) { /* check for no collision */ \
            /* calculate bitwise difference (deep check) */ \
            LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */ \
            diff = libxs_gemm_diff(DESCRIPTOR, &internal_registry_keys[i].descriptor); \
            if (0 != diff) { /* new collision discovered (but no code version yet) */ \
              /* allow to fix-up current entry inside of the guarded/locked region */ \
              flux_entry.function.pmm = 0; \
            } \
          } \
          /* collision discovered but code version exists; perform deep check */ \
          else if (0 != libxs_gemm_diff(DESCRIPTOR, &internal_registry_keys[i].descriptor)) { \
            /* continue linearly searching code starting at re-hashed index position */ \
            const unsigned int index = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE); \
            unsigned int next; \
            for (i0 = (index != i ? index : LIBXS_HASH_MOD(index + 1, LIBXS_REGSIZE)), \
              i = i0, next = LIBXS_HASH_MOD(i0 + 1, LIBXS_REGSIZE); \
              /* skip any (still invalid) descriptor which corresponds to no code, or continue on difference */ \
              (0 == (CODE = (internal_registry + i))->function.pmm || \
                0 != (diff = libxs_gemm_diff(DESCRIPTOR, &internal_registry_keys[i].descriptor))) \
                /* entire registry was searched and no code version was found */ \
                && next != i0; \
              i = next, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE)); \
            if (0 == diff) { /* found exact code version; continue with atomic load */ \
              flux_entry.function.pmm = (CODE)->function.pmm; \
              /* clear the uppermost bit of the address */ \
              flux_entry.function.imm &= ~LIBXS_HASH_COLLISION; \
            } \
            else { /* no code found */ \
              flux_entry.function.pmm = 0; \
            } \
            break; \
          } \
          else { /* clear the uppermost bit of the address */ \
            flux_entry.function.imm &= ~LIBXS_HASH_COLLISION; \
          } \
        } \
        else { /* new collision discovered (but no code version yet) */ \
          flux_entry.function.pmm = 0; \
        } \
      } \
      INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, flux_entry) \
      { \
        diff = 0; \
      } \
    } \
    while (0 != diff); \
    assert(0 == diff || 0 == flux_entry.function.pmm); \
    INTERNAL_FIND_CODE_CACHE_FINALIZE(cache_id, cache_keys, cache, cache_hit, flux_entry, DESCRIPTOR); \
  } \
} \
return flux_entry.function.xmm

#define INTERNAL_DISPATCH_MAIN(DESCRIPTOR_DECL, DESC, FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/) { \
  INTERNAL_FIND_CODE_DECLARE(code); \
  const signed char scalpha = (signed char)(0 == (PALPHA) ? LIBXS_ALPHA : *(PALPHA)), scbeta = (signed char)(0 == (PBETA) ? LIBXS_BETA : *(PBETA)); \
  if (0 == ((FLAGS) & (LIBXS_GEMM_FLAG_TRANS_A | LIBXS_GEMM_FLAG_TRANS_B)) && 1 == scalpha && (1 == scbeta || 0 == scbeta)) { \
    const int internal_dispatch_main_prefetch = (0 == (PREFETCH) ? INTERNAL_PREFETCH : *(PREFETCH)); \
    DESCRIPTOR_DECL; LIBXS_GEMM_DESCRIPTOR(*(DESC), 0 != (VECTOR_WIDTH) ? (VECTOR_WIDTH): LIBXS_ALIGNMENT, FLAGS, LIBXS_LD(M, N), LIBXS_LD(N, M), K, \
      0 == LIBXS_LD(PLDA, PLDB) ? LIBXS_LD(M, N) : *LIBXS_LD(PLDA, PLDB), \
      0 == LIBXS_LD(PLDB, PLDA) ? (K) : *LIBXS_LD(PLDB, PLDA), \
      0 == (PLDC) ? LIBXS_LD(M, N) : *(PLDC), scalpha, scbeta, \
      0 > internal_dispatch_main_prefetch ? internal_prefetch : internal_dispatch_main_prefetch); \
    { \
      INTERNAL_FIND_CODE(DESC, code).SELECTOR; \
    } \
  } \
  else { /* TODO: not supported (bypass) */ \
    return 0; \
  } \
}

#if defined(LIBXS_GEMM_DIFF_MASK_A) /* no padding i.e., LIBXS_GEMM_DESCRIPTOR_SIZE */
# define INTERNAL_DISPATCH(FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/) \
    INTERNAL_DISPATCH_MAIN(libxs_gemm_descriptor descriptor, &descriptor, \
    FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/)
#else /* padding: LIBXS_GEMM_DESCRIPTOR_SIZE -> LIBXS_ALIGNMENT */
# define INTERNAL_DISPATCH(FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/) { \
    INTERNAL_DISPATCH_MAIN(union { libxs_gemm_descriptor desc; char simd[LIBXS_ALIGNMENT]; } simd_descriptor; \
      for (i = LIBXS_GEMM_DESCRIPTOR_SIZE; i < sizeof(simd_descriptor.simd); ++i) simd_descriptor.simd[i] = 0, &simd_descriptor.desc, \
    FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/)
#endif

#define INTERNAL_SMMDISPATCH(PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) \
  INTERNAL_DISPATCH((0 == (PFLAGS) ? LIBXS_FLAGS : *(PFLAGS)) | LIBXS_GEMM_FLAG_F32PREC, \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, smm)
#define INTERNAL_DMMDISPATCH(PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) \
  INTERNAL_DISPATCH((0 == (PFLAGS) ? LIBXS_FLAGS : *(PFLAGS)), \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, dmm)


LIBXS_INLINE LIBXS_RETARGETABLE void internal_update_statistic(const libxs_gemm_descriptor* desc,
  unsigned ntry, unsigned ncol)
{
  assert(0 != desc);
  {
    const unsigned long long size = LIBXS_MNK_SIZE(desc->m, desc->n, desc->k);
    const int precision = (0 == (LIBXS_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1);
    int bucket = 2/*big*/;

    if (LIBXS_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= size) {
      bucket = 0;
    }
    else if (LIBXS_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= size) {
      bucket = 1;
    }

    /* TODO: use atomic updates */
    internal_statistic[precision][bucket].ncol += ncol;
    internal_statistic[precision][bucket].ntry += ntry;
  }
}


LIBXS_INLINE LIBXS_RETARGETABLE const char* internal_get_target_arch(int archid);
LIBXS_INLINE LIBXS_RETARGETABLE const char* internal_get_target_arch(int archid)
{
  const char* target_arch = 0;
  switch (archid) {
    case LIBXS_X86_AVX512_CORE: {
      target_arch = "skx";
    } break;
    case LIBXS_X86_AVX512_MIC: {
      target_arch = "knl";
    } break;
    case LIBXS_X86_AVX2: {
      target_arch = "hsw";
    } break;
    case LIBXS_X86_AVX: {
      target_arch = "snb";
    } break;
    case LIBXS_X86_SSE4_2: {
      target_arch = "wsm";
    } break;
    case LIBXS_X86_SSE4_1: {
      target_arch = "sse4";
    } break;
    case LIBXS_X86_SSE3: {
      target_arch = "sse3";
    } break;
    case LIBXS_TARGET_ARCH_GENERIC: {
      target_arch = "generic";
    } break;
    default: if (LIBXS_X86_GENERIC <= archid) {
      target_arch = "x86";
    }
    else {
      target_arch = "unknown";
    }
  }

  assert(0 != target_arch);
  return target_arch;
}


LIBXS_INLINE LIBXS_RETARGETABLE unsigned int internal_print_statistic(FILE* ostream, const char* target_arch, int precision, unsigned int linebreaks, unsigned int indent)
{
  int printed = 0;
  assert(0 != ostream && 0 != target_arch && (0 <= precision && precision < 2));

  if (/* omit to print anything if it is superfluous */
    0 != internal_statistic[precision][0/*SML*/].ntry ||
    0 != internal_statistic[precision][0/*SML*/].njit ||
    0 != internal_statistic[precision][0/*SML*/].nsta ||
    0 != internal_statistic[precision][0/*SML*/].ncol ||
    0 != internal_statistic[precision][1/*MED*/].ntry ||
    0 != internal_statistic[precision][1/*MED*/].njit ||
    0 != internal_statistic[precision][1/*MED*/].nsta ||
    0 != internal_statistic[precision][1/*MED*/].ncol ||
    0 != internal_statistic[precision][2/*BIG*/].ntry ||
    0 != internal_statistic[precision][2/*BIG*/].njit ||
    0 != internal_statistic[precision][2/*BIG*/].nsta ||
    0 != internal_statistic[precision][2/*BIG*/].ncol)
  {
    char title[256], sml[256], med[256], big[256];

    LIBXS_SNPRINTF(sml, sizeof(sml), "%u..%u",                     0u, internal_statistic_sml);
    LIBXS_SNPRINTF(med, sizeof(sml), "%u..%u", internal_statistic_sml, internal_statistic_med);
    LIBXS_SNPRINTF(big, sizeof(sml), "%u..%u", internal_statistic_med, internal_statistic_mnk);
    {
      unsigned int n = 0;
      for (n = 0; 0 != target_arch[n] && n < sizeof(title); ++n) { /* toupper */
        const char c = target_arch[n];
        title[n] = (char)(('a' <= c && c <= 'z') ? (c - 32) : c);
      }
      LIBXS_SNPRINTF(title + n, sizeof(title) - n, "/%s", 0 == precision ? "DP" : "SP");
      for (n = 0; n < linebreaks; ++n) fprintf(ostream, "\n");
    }
    fprintf(ostream, "%*s%-10s %6s %6s %6s %6s\n", (int)indent, "", title, "TRY" ,"JIT", "STA", "COL");
    fprintf(ostream,  "%*s%10s %6u %6u %6u %6u\n", (int)indent, "", sml,
      internal_statistic[precision][0/*SML*/].ntry,
      internal_statistic[precision][0/*SML*/].njit,
      internal_statistic[precision][0/*SML*/].nsta,
      internal_statistic[precision][0/*SML*/].ncol);
    fprintf(ostream,  "%*s%10s %6u %6u %6u %6u\n", (int)indent, "", med,
      internal_statistic[precision][1/*MED*/].ntry,
      internal_statistic[precision][1/*MED*/].njit,
      internal_statistic[precision][1/*MED*/].nsta,
      internal_statistic[precision][1/*MED*/].ncol);
    fprintf(ostream,  "%*s%10s %6u %6u %6u %6u\n", (int)indent, "", big,
      internal_statistic[precision][2/*BIG*/].ntry,
      internal_statistic[precision][2/*BIG*/].njit,
      internal_statistic[precision][2/*BIG*/].nsta,
      internal_statistic[precision][2/*BIG*/].ncol);
    printed = 1;
  }

  return printed;
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_register_static_code(const libxs_gemm_descriptor* desc,
  unsigned int index, unsigned int hash, libxs_xmmfunction src, internal_regentry* registry)
{
  internal_regkey* dst_key = internal_registry_keys + index;
  internal_regentry* dst_entry = registry + index;
  assert(0 != desc && 0 != src.dmm && 0 != dst_key && 0 != registry);

  if (0 != dst_entry->function.pmm) { /* collision? */
    /* start at a re-hashed index position */
    const unsigned int start = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE);
    unsigned int i0, i, next;

    /* mark current entry as a collision (this might be already the case) */
    dst_entry->function.imm |= LIBXS_HASH_COLLISION;

    /* start linearly searching for an available slot */
    for (i = (start != index) ? start : LIBXS_HASH_MOD(start + 1, LIBXS_REGSIZE), i0 = i, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE);
      0 != (dst_entry = registry + i)->function.pmm && next != i0; i = next, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE));

    /* corresponding key position */
    dst_key = internal_registry_keys + i;

    internal_update_statistic(desc, 0, 1);
  }

  if (0 == dst_entry->function.pmm) { /* registry not (yet) exhausted */
    dst_entry->function.xmm = src;
    dst_entry->size = 0; /* statically generated code */
    dst_key->descriptor = *desc;
  }

  internal_update_statistic(desc, 1, 0);
}


LIBXS_INLINE LIBXS_RETARGETABLE int internal_get_prefetch(const libxs_gemm_descriptor* desc)
{
  assert(0 != desc);
  switch (desc->prefetch) {
    case LIBXS_PREFETCH_SIGONLY:            return 2;
    case LIBXS_PREFETCH_BL2_VIA_C:          return 3;
    case LIBXS_PREFETCH_AL2:                return 4;
    case LIBXS_PREFETCH_AL2_AHEAD:          return 5;
    case LIBXS_PREFETCH_AL2BL2_VIA_C:       return 6;
    case LIBXS_PREFETCH_AL2BL2_VIA_C_AHEAD: return 7;
    case LIBXS_PREFETCH_AL2_JPST:           return 8;
    case LIBXS_PREFETCH_AL2BL2_VIA_C_JPST:  return 9;
    default: {
      assert(LIBXS_PREFETCH_NONE == desc->prefetch);
      return 0;
    }
  }
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_get_code_name(const char* target_arch,
  const libxs_gemm_descriptor* desc, unsigned int buffer_size, char* name)
{
  assert((0 != desc && 0 != name) || 0 == buffer_size);
  LIBXS_SNPRINTF(name, buffer_size, "libxs_%s_%c%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.jit",
    target_arch /* code path name */,
    0 == (LIBXS_GEMM_FLAG_F32PREC & desc->flags) ? 'd' : 's',
    0 == (LIBXS_GEMM_FLAG_TRANS_A & desc->flags) ? 'n' : 't',
    0 == (LIBXS_GEMM_FLAG_TRANS_B & desc->flags) ? 'n' : 't',
    (unsigned int)desc->m, (unsigned int)desc->n, (unsigned int)desc->k,
    (unsigned int)desc->lda, (unsigned int)desc->ldb, (unsigned int)desc->ldc,
    desc->alpha, desc->beta, internal_get_prefetch(desc));
}


#if defined(LIBXS_VTUNE)
LIBXS_INLINE LIBXS_RETARGETABLE void internal_get_vtune_jitdesc(const internal_regentry* code, const char* name, LIBXS_VTUNE_JIT_DESC_TYPE* desc)
{
  assert(0 != code && 0 != code->id && 0 != code->size && 0 != desc);
  desc->method_id = code->id;
  /* incorrect constness (method_name) */
  desc->method_name = (char*)name;
  desc->method_load_address = code->function.pmm;
  desc->method_size = code->size;
  desc->line_number_size = 0;
  desc->line_number_table = NULL;
  desc->class_file_name = NULL;
  desc->source_file_name = NULL;
# if (2 == LIBXS_VTUNE_JITVERSION)
  desc->module_name = "libxs.jit";
# endif
}
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
      int init_code;
      /* set internal_target_archid */
      libxs_set_target_arch(getenv("LIBXS_JIT"));
      { /* select prefetch strategy for JIT */
        const char *const env_prefetch = getenv("LIBXS_PREFETCH");
        if (0 == env_prefetch || 0 == *env_prefetch) {
          if (0 > LIBXS_PREFETCH) { /* permitted by LIBXS_PREFETCH_AUTO */
            internal_prefetch = LIBXS_X86_AVX512_MIC != internal_target_archid
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
      libxs_hash_init(internal_target_archid);
      libxs_gemm_diff_init(internal_target_archid);
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
        assert(0 == internal_registry_keys && 0 == internal_registry/*should never happen*/);
        result = (internal_regentry*)malloc(LIBXS_REGSIZE * sizeof(internal_regentry));
        internal_registry_keys = (internal_regkey*)malloc(LIBXS_REGSIZE * sizeof(internal_regkey));
        if (result && internal_registry_keys) {
          const char *const env_verbose = getenv("LIBXS_VERBOSE");
          internal_statistic_mnk = (unsigned int)(pow((double)(LIBXS_MAX_MNK), 0.3333333333333333) + 0.5);
          if (0 != env_verbose && 0 != *env_verbose) {
            internal_verbose = atoi(env_verbose);
          }
          for (i = 0; i < LIBXS_REGSIZE; ++i) result[i].function.pmm = 0;
          /* omit registering code if JIT is enabled and if an ISA extension is found
           * which is beyond the static code path used to compile the library
           */
#if (0 != LIBXS_JIT) && !defined(__MIC__)
          if (LIBXS_STATIC_TARGET_ARCH <= internal_target_archid && LIBXS_X86_AVX > internal_target_archid)
#endif
          { /* opening a scope for eventually declaring variables */
            /* setup the dispatch table for the statically generated code */
#           include <libxs_dispatch.h>
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
        else {
#if !defined(NDEBUG) && defined(__TRACE) /* library code is expected to be mute */
          fprintf(stderr, "LIBXS: failed to allocate code registry!\n");
#endif
          free(internal_registry_keys);
          free(result);
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
        internal_regkey *const registry_keys = internal_registry_keys;
        const char *const target_arch = internal_get_target_arch(internal_target_archid);
        /* serves as an id to invalidate the thread-local cache; never decremented */
        ++internal_teardown;
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
        internal_registry_keys = 0;
        for (i = 0; i < LIBXS_REGSIZE; ++i) {
          internal_regentry code = registry[i];
          if (0 != code.function.pmm/*potentially allocated*/) {
            const libxs_gemm_descriptor *const desc = &registry_keys[i].descriptor;
            const unsigned long long size = LIBXS_MNK_SIZE(desc->m, desc->n, desc->k);
            const int precision = (0 == (LIBXS_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1);
            int bucket = 2;
            if (LIBXS_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= size) {
              bucket = 0;
            }
            else if (LIBXS_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= size) {
              bucket = 1;
            }
            if (0 != code.size/*JIT: actually allocated*/) {
              /* make address valid by clearing an eventual collision flag */
              code.function.imm &= ~LIBXS_HASH_COLLISION;
#if defined(LIBXS_VTUNE)
              if (0 != code.id && iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
                char jit_code_name[256];
                LIBXS_VTUNE_JIT_DESC_TYPE vtune_jit_desc;
                internal_get_code_name(target_arch, desc,
                  sizeof(jit_code_name), jit_code_name);
                internal_get_vtune_jitdesc(&code, jit_code_name, &vtune_jit_desc);
                iJIT_NotifyEvent(LIBXS_VTUNE_JIT_UNLOAD, &vtune_jit_desc);
              }
#endif
#if defined(_WIN32)
              /* TODO: executable memory buffer under Windows */
#else
# if defined(NDEBUG)
              munmap(code.function.pmm, code.size);
# else /* library code is expected to be mute */
              if (0 != munmap(code.function.pmm, code.size)) {
                const int error = errno;
                fprintf(stderr, "LIBXS: %s (munmap error #%i at %p+%u)!\n",
                  strerror(error), error, code.function.pmm, code.size);
              }
# endif
#endif
              ++internal_statistic[precision][bucket].njit;
            }
            else {
              ++internal_statistic[precision][bucket].nsta;
            }
          }
        }
        if (0 != internal_verbose) { /* print statistic on termination */
          fflush(stdout); /* synchronize with standard output */
          {
            const unsigned int linebreak = 0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0) ? 1 : 0;
            if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak) {
              fprintf(stderr, "LIBXS_JIT=%s\n", target_arch);
            }
          }
        }
        free((void*)registry_keys);
        free((void*)registry);
      }
    }
#if !defined(LIBXS_OPENMP) /* release locks */
  for (i = 0; i < nlocks; ++i) LIBXS_LOCK_RELEASE(internal_reglock[i]);
#endif
  }
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_get_target_archid(void)
{
  LIBXS_INIT
#if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  return internal_target_archid;
#else /* no JIT support */
  return LIBXS_TARGET_ARCH_GENERIC;
#endif
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_set_target_archid(int archid)
{
  switch (archid) {
    case LIBXS_X86_AVX512_CORE:
    case LIBXS_X86_AVX512_MIC:
    case LIBXS_X86_AVX2:
    case LIBXS_X86_AVX:
    case LIBXS_X86_SSE4_2:
    case LIBXS_X86_SSE4_1:
    case LIBXS_X86_SSE3:
    case LIBXS_TARGET_ARCH_GENERIC: {
      internal_target_archid = archid;
    } break;
    default: if (LIBXS_X86_GENERIC <= archid) {
      internal_target_archid = LIBXS_X86_GENERIC;
    }
    else {
      internal_target_archid = LIBXS_TARGET_ARCH_UNKNOWN;
    }
  }

#if !defined(NDEBUG) /* library code is expected to be mute */
  {
    const int cpuid_archid = libxs_cpuid_x86();
    if (cpuid_archid < internal_target_archid) {
      fprintf(stderr, "LIBXS: \"%s\" code will fail to run on \"%s\"!\n",
        internal_get_target_arch(internal_target_archid),
        internal_get_target_arch(cpuid_archid));
    }
  }
#endif
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE const char* libxs_get_target_arch(void)
{
  LIBXS_INIT
  return internal_get_target_arch(internal_target_archid);
}


/* function serves as a helper for implementing the Fortran interface */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void get_target_arch(char* target_arch, int length);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void get_target_arch(char* target_arch, int length)
{
  const char* c = libxs_get_target_arch();
  int i;
  assert(0 != target_arch); /* valid here since function is not in the public interface */
  for (i = 0; i < length && 0 != *c; ++i, ++c) target_arch[i] = *c;
  for (; i < length; ++i) target_arch[i] = ' ';
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_set_target_arch(const char* name)
{
  int target_archid = LIBXS_TARGET_ARCH_UNKNOWN;

  if (name && *name) {
    const int jit = atoi(name);
    if (0 == strcmp("0", name)) {
      target_archid = LIBXS_TARGET_ARCH_GENERIC;
    }
    else if (1 < jit) {
      target_archid = LIBXS_X86_GENERIC + jit;
    }
    else if (0 == strcmp("skx", name) || 0 == strcmp("avx3", name) || 0 == strcmp("avx512", name)) {
      target_archid = LIBXS_X86_AVX512_CORE;
    }
    else if (0 == strcmp("knl", name) || 0 == strcmp("mic2", name)) {
      target_archid = LIBXS_X86_AVX512_MIC;
    }
    else if (0 == strcmp("hsw", name) || 0 == strcmp("avx2", name)) {
      target_archid = LIBXS_X86_AVX2;
    }
    else if (0 == strcmp("snb", name) || 0 == strcmp("avx", name)) {
      target_archid = LIBXS_X86_AVX;
    }
    else if (0 == strcmp("wsm", name) || 0 == strcmp("nhm", name) || 0 == strcmp("sse4", name) || 0 == strcmp("sse4_2", name) || 0 == strcmp("sse4.2", name)) {
      target_archid = LIBXS_X86_SSE4_2;
    }
    else if (0 == strcmp("sse4_1", name) || 0 == strcmp("sse4.1", name)) {
      target_archid = LIBXS_X86_SSE4_1;
    }
    else if (0 == strcmp("sse3", name) || 0 == strcmp("sse", name)) {
      target_archid = LIBXS_X86_SSE3;
    }
    else if (0 == strcmp("x86", name) || 0 == strcmp("sse2", name)) {
      target_archid = LIBXS_X86_GENERIC;
    }
    else if (0 == strcmp("generic", name) || 0 == strcmp("none", name)) {
      target_archid = LIBXS_TARGET_ARCH_GENERIC;
    }
  }

  if (LIBXS_TARGET_ARCH_UNKNOWN == target_archid || LIBXS_X86_AVX512_CORE < target_archid) {
    target_archid = libxs_cpuid_x86();
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    const int cpuid_archid = libxs_cpuid_x86();
    if (cpuid_archid < target_archid) {
      fprintf(stderr, "LIBXS: \"%s\" code will fail to run on \"%s\"!\n",
        internal_get_target_arch(target_archid),
        internal_get_target_arch(cpuid_archid));
    }
  }
#endif
  internal_target_archid = target_archid;
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_build(const libxs_gemm_descriptor* desc, internal_regentry* code)
{
#if (0 != LIBXS_JIT)
# if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  libxs_generated_code generated_code;
  assert(0 != desc && 0 != code);
  assert(0 != internal_target_archid);
  assert(0 == code->function.pmm);

  /* allocate temporary buffer which is large enough to cover the generated code */
  generated_code.generated_code = malloc(131072);
  generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
  generated_code.code_size = 0;
  generated_code.code_type = 2;
  generated_code.last_error = 0;

  /* generate kernel */
  libxs_generator_gemm_kernel(&generated_code, desc, internal_get_target_arch(internal_target_archid));

  /* handle an eventual error in the else-branch */
  if (0 == generated_code.last_error) {
# if defined(__APPLE__) && defined(__MACH__)
    const int fd = 0;
# else
    const int fd = open("/dev/zero", O_RDWR);
# endif
    if (0 <= fd) {
      /* create executable buffer */
      code->function.pmm = mmap(0, generated_code.code_size,
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
      if (MAP_FAILED != code->function.pmm) {
        /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
# if defined(MADV_NOHUGEPAGE)
#  if defined(NDEBUG)
        madvise(code->function.pmm, generated_code.code_size, MADV_NOHUGEPAGE);
#  else /* library code is expected to be mute */
        /* proceed even in case of an error, we then just take what we got (THP) */
        if (0 != madvise(code->function.pmm, generated_code.code_size, MADV_NOHUGEPAGE)) {
          static LIBXS_TLS int once = 0;
          if (0 == once) {
            const int error = errno;
            fprintf(stderr, "LIBXS: %s (madvise error #%i at %p)!\n",
              strerror(error), error, code->function.pmm);
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
        memcpy(code->function.pmm, generated_code.generated_code, generated_code.code_size);

        if (0/*ok*/ == mprotect(code->function.pmm, generated_code.code_size, PROT_EXEC | PROT_READ)) {
# if (!defined(NDEBUG) && defined(_DEBUG)) || defined(LIBXS_VTUNE)
          char jit_code_name[256];
          internal_get_code_name(internal_target_archid, desc, sizeof(jit_code_name), jit_code_name);
# endif
          /* finalize code generation */
          code->size = generated_code.code_size;
          /* free temporary/initial code buffer */
          free(generated_code.generated_code);
# if !defined(NDEBUG) && defined(_DEBUG)
          { /* dump byte-code into file */
            FILE *const byte_code = fopen(jit_code_name, "wb");
            if (0 != byte_code) {
              fwrite(code->function.pmm, 1, code->size, byte_code);
              fclose(byte_code);
            }
          }
# endif /*!defined(NDEBUG) && defined(_DEBUG)*/
# if defined(LIBXS_VTUNE)
          if (iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
            LIBXS_VTUNE_JIT_DESC_TYPE vtune_jit_desc;
            code->id = iJIT_GetNewMethodID();
            internal_get_vtune_jitdesc(code, jit_code_name, &vtune_jit_desc);
            iJIT_NotifyEvent(LIBXS_VTUNE_JIT_LOAD, &vtune_jit_desc);
          }
          else {
            code->id = 0;
          }
# endif
        }
        else { /* there was an error with mprotect */
# if defined(NDEBUG)
          munmap(code->function.pmm, generated_code.code_size);
# else /* library code is expected to be mute */
          static LIBXS_TLS int once = 0;
          if (0 == once) {
            const int error = errno;
            fprintf(stderr, "LIBXS: %s (mprotect error #%i at %p+%u)!\n",
              strerror(error), error, code->function.pmm, generated_code.code_size);
            once = 1;
          }
          if (0 != munmap(code->function.pmm, generated_code.code_size)) {
            static LIBXS_TLS int once_mmap_error = 0;
            if (0 == once_mmap_error) {
              const int error = errno;
              fprintf(stderr, "LIBXS: %s (munmap error #%i at %p+%u)!\n",
                strerror(error), error, code->function.pmm, generated_code.code_size);
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
        code->function.pmm = 0;
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
  LIBXS_UNUSED(desc); LIBXS_UNUSED(code);
  /* libxs_get_target_arch also serves as a runtime check whether JIT is available or not */
  assert(LIBXS_X86_AVX > internal_target_archid);
# endif /*_WIN32*/
#endif /*LIBXS_JIT*/
}


LIBXS_INLINE LIBXS_RETARGETABLE libxs_xmmfunction internal_xmmdispatch(const libxs_gemm_descriptor* descriptor)
{
  INTERNAL_FIND_CODE_DECLARE(code);
  assert(descriptor);
  {
    INTERNAL_FIND_CODE(descriptor, code);
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
  INTERNAL_SMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dmmfunction libxs_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
  INTERNAL_DMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


#if defined(LIBXS_GEMM_EXTWRAP)
#if defined(__STATIC)

LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(__real_sgemm)(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void LIBXS_FSYMBOL(__real_dgemm)(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc)
{
  dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*defined(__STATIC)*/
#endif /*defined(LIBXS_GEMM_EXTWRAP)*/
