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
#include "libxs_alloc.h"
#include "libxs_hash.h"
#include "libxs_sync.h"

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
# include <sys/mman.h>
# include <pthread.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
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
/* flag fused into the memory address of a code version in case of non-JIT */
#define LIBXS_CODE_STATIC (1ULL << (8 * sizeof(void*) - 2))

#if 16 >= (LIBXS_GEMM_DESCRIPTOR_SIZE)
# define LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE 16
#elif 32 >= (LIBXS_GEMM_DESCRIPTOR_SIZE)
# define LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE 32
#else
# define LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE LIBXS_GEMM_DESCRIPTOR_SIZE
#endif

typedef union LIBXS_RETARGETABLE internal_regkey_type {
  char simd[LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE];
  libxs_gemm_descriptor descriptor;
} internal_regkey_type;

typedef union LIBXS_RETARGETABLE internal_code_type {
  libxs_xmmfunction xmm;
  /*const*/void* pmm;
  uintptr_t imm;
} internal_code_type;

typedef struct LIBXS_RETARGETABLE internal_statistic_type {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic_type;

typedef struct LIBXS_RETARGETABLE internal_desc_extra_type {
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} internal_desc_extra_type;

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
    const unsigned int LOCKINDEX = LIBXS_MOD2(INDEX, INTERNAL_REGLOCK_COUNT); \
    LIBXS_LOCK_TRYLOCK(internal_reglock(LOCKINDEX))
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXS_LOCK_RELEASE(internal_reglock(LOCKINDEX)); }
#endif

#define INTERNAL_FIND_CODE_DECLARE(CODE) internal_code_type* CODE = \
  LIBXS_ATOMIC_LOAD(internal_registry(), LIBXS_ATOMIC_RELAXED); unsigned int i

#if defined(LIBXS_CACHESIZE) && (0 < (LIBXS_CACHESIZE))
# define INTERNAL_FIND_CODE_CACHE_DECL(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT) \
  static LIBXS_TLS union { libxs_gemm_descriptor desc; char padding[LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE]; } CACHE_KEYS[LIBXS_CACHESIZE]; \
  static LIBXS_TLS libxs_xmmfunction CACHE[LIBXS_CACHESIZE]; \
  static LIBXS_TLS unsigned int CACHE_ID = (unsigned int)(-1); \
  static LIBXS_TLS unsigned int CACHE_HIT = LIBXS_CACHESIZE;
# define INTERNAL_FIND_CODE_CACHE_BEGIN(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR) \
  assert(LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE >= LIBXS_GEMM_DESCRIPTOR_SIZE); \
  /* search small cache starting with the last hit on record */ \
  i = libxs_gemm_diffn(DESCRIPTOR, &(CACHE_KEYS)->desc, CACHE_HIT, LIBXS_CACHESIZE, LIBXS_GEMM_DESCRIPTOR_SIMD_SIZE); \
  if ((LIBXS_CACHESIZE) > i && (CACHE_ID) == *internal_teardown()) { /* cache hit, and valid */ \
    (RESULT).xmm = (CACHE)[i]; \
    CACHE_HIT = i; \
  } \
  else
# if defined(LIBXS_GEMM_DIFF_SW) && (2 == (LIBXS_GEMM_DIFF_SW)) /* most general implementation */
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR) \
    if ((CACHE_ID) != *internal_teardown()) { \
      memset(CACHE_KEYS, -1, sizeof(CACHE_KEYS)); \
      CACHE_ID = *internal_teardown(); \
    } \
    i = ((CACHE_HIT) + ((LIBXS_CACHESIZE) - 1)) % (LIBXS_CACHESIZE); \
    ((CACHE_KEYS)[i]).desc = *(DESCRIPTOR); \
    (CACHE)[i] = (RESULT).xmm; \
    CACHE_HIT = i
# else
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR) \
    assert(/*is pot*/(LIBXS_CACHESIZE) == (1 << LIBXS_LOG2(LIBXS_CACHESIZE))); \
    if ((CACHE_ID) != *internal_teardown()) { \
      memset(CACHE_KEYS, -1, sizeof(CACHE_KEYS)); \
      CACHE_ID = *internal_teardown(); \
    } \
    i = LIBXS_MOD2((CACHE_HIT) + ((LIBXS_CACHESIZE) - 1), LIBXS_CACHESIZE); \
    (CACHE_KEYS)[i].desc = *(DESCRIPTOR); \
    (CACHE)[i] = (RESULT).xmm; \
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
  if (0 == (RESULT).pmm/*code version does not exist*/ && LIBXS_X86_AVX <= *internal_target_archid()) { \
    /* instead of blocking others, a try-lock allows to let other threads fallback to BLAS during lock-duration */ \
    INTERNAL_FIND_CODE_LOCK(lock, i); /* lock the registry entry */ \
    if (0 == diff) { \
      RESULT = *(CODE); /* deliver code */ \
      /* clear collision flag; can be never static code */ \
      assert(0 == (LIBXS_CODE_STATIC & (RESULT).imm)); \
      (RESULT).imm &= ~LIBXS_HASH_COLLISION; \
    } \
    if (0 == (RESULT).pmm) { /* double-check (re-read registry entry) after acquiring the lock */ \
      if (0 == diff) { \
        /* found a conflict-free registry-slot, and attempt to build the kernel */ \
        internal_build(DESCRIPTOR, "smm", 0/*extra desc*/, &(RESULT)); \
        internal_update_statistic(DESCRIPTOR, 1, 0); \
        if (0 != (RESULT).pmm) { /* synchronize registry entry */ \
          (*internal_registry_keys())[i].descriptor = *(DESCRIPTOR); \
          *(CODE) = RESULT; \
          LIBXS_ATOMIC_STORE(&(CODE)->pmm, (RESULT).pmm, LIBXS_ATOMIC_SEQ_CST); \
        } \
      } \
      else { /* 0 != diff */ \
        if (0 == diff0) { \
          /* flag existing entry as collision */ \
          internal_code_type collision; \
          /* find new slot to store the code version */ \
          const unsigned int index = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE); \
          collision.imm = (CODE)->imm | LIBXS_HASH_COLLISION; \
          i = (index != i ? index : LIBXS_HASH_MOD(index + 1, LIBXS_REGSIZE)); \
          i0 = i; /* keep starting point of free-slot-search in mind */ \
          internal_update_statistic(DESCRIPTOR, 0, 1); \
          LIBXS_ATOMIC_STORE(&(CODE)->pmm, collision.pmm, LIBXS_ATOMIC_SEQ_CST); /* fix-up existing entry */ \
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
        (CODE) = *internal_registry() + i; \
      } \
    } \
    INTERNAL_FIND_CODE_UNLOCK(lock); \
  } \
  else
#else
# define INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, RESULT)
#endif

#define INTERNAL_FIND_CODE(DESCRIPTOR, CODE) \
  internal_code_type flux_entry; \
{ \
  INTERNAL_FIND_CODE_CACHE_DECL(cache_id, cache_keys, cache, cache_hit) \
  unsigned int hash, diff = 0, diff0 = 0, i0; \
  INTERNAL_FIND_CODE_INIT(CODE); \
  INTERNAL_FIND_CODE_CACHE_BEGIN(cache_id, cache_keys, cache, cache_hit, flux_entry, DESCRIPTOR) { \
    /* check if the requested xGEMM is already JITted */ \
    LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */ \
    LIBXS_HASH_FUNCTION_CALL(hash, i = i0, *(DESCRIPTOR)); \
    (CODE) += i; /* actual entry */ \
    do { \
      flux_entry.pmm = LIBXS_ATOMIC_LOAD(&(CODE)->pmm, LIBXS_ATOMIC_SEQ_CST); /* read registered code */ \
      if (0 != flux_entry.pmm) { /* code version exists */ \
        if (0 == diff0) { \
          if (0 == (LIBXS_HASH_COLLISION & flux_entry.imm)) { /* check for no collision */ \
            /* calculate bitwise difference (deep check) */ \
            LIBXS_PRAGMA_FORCEINLINE /* must precede a statement */ \
            diff = libxs_gemm_diff(DESCRIPTOR, &(*internal_registry_keys())[i].descriptor); \
            if (0 != diff) { /* new collision discovered (but no code version yet) */ \
              /* allow to fix-up current entry inside of the guarded/locked region */ \
              flux_entry.pmm = 0; \
            } \
          } \
          /* collision discovered but code version exists; perform deep check */ \
          else if (0 != libxs_gemm_diff(DESCRIPTOR, &(*internal_registry_keys())[i].descriptor)) { \
            /* continue linearly searching code starting at re-hashed index position */ \
            const unsigned int index = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE); \
            unsigned int next; \
            for (i0 = (index != i ? index : LIBXS_HASH_MOD(index + 1, LIBXS_REGSIZE)), \
              i = i0, next = LIBXS_HASH_MOD(i0 + 1, LIBXS_REGSIZE); \
              /* skip any (still invalid) descriptor which corresponds to no code, or continue on difference */ \
              (0 == (CODE = (*internal_registry() + i))->pmm || \
                0 != (diff = libxs_gemm_diff(DESCRIPTOR, &(*internal_registry_keys())[i].descriptor))) \
                /* entire registry was searched and no code version was found */ \
                && next != i0; \
              i = next, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE)); \
            if (0 == diff) { /* found exact code version; continue with atomic load */ \
              flux_entry.pmm = (CODE)->pmm; \
              /* clear the collision and the non-JIT flag */ \
              flux_entry.imm &= ~(LIBXS_HASH_COLLISION | LIBXS_CODE_STATIC); \
            } \
            else { /* no code found */ \
              flux_entry.pmm = 0; \
            } \
            break; \
          } \
          else { /* clear the collision and the non-JIT flag */ \
            flux_entry.imm &= ~(LIBXS_HASH_COLLISION | LIBXS_CODE_STATIC); \
          } \
        } \
        else { /* new collision discovered (but no code version yet) */ \
          flux_entry.pmm = 0; \
        } \
      } \
      INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, flux_entry) \
      { \
        diff = 0; \
      } \
    } \
    while (0 != diff); \
    assert(0 == diff || 0 == flux_entry.pmm); \
    INTERNAL_FIND_CODE_CACHE_FINALIZE(cache_id, cache_keys, cache, cache_hit, flux_entry, DESCRIPTOR); \
  } \
} \
return flux_entry.xmm

#define INTERNAL_DISPATCH_BYPASS_CHECK(FLAGS, ALPHA, BETA) ( \
  0 == ((FLAGS) & (LIBXS_GEMM_FLAG_TRANS_A | LIBXS_GEMM_FLAG_TRANS_B)) && \
  1 == (ALPHA) && (1 == (BETA) || 0 == (BETA)))

#define INTERNAL_DISPATCH_MAIN(DESCRIPTOR_DECL, DESC, FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/) { \
  INTERNAL_FIND_CODE_DECLARE(code); \
  const signed char scalpha = (signed char)(0 == (PALPHA) ? LIBXS_ALPHA : *(PALPHA)), scbeta = (signed char)(0 == (PBETA) ? LIBXS_BETA : *(PBETA)); \
  if (INTERNAL_DISPATCH_BYPASS_CHECK(FLAGS, scalpha, scbeta)) { \
    const int internal_dispatch_main_prefetch = (0 == (PREFETCH) ? INTERNAL_PREFETCH : *(PREFETCH)); \
    DESCRIPTOR_DECL; LIBXS_GEMM_DESCRIPTOR(*(DESC), 0 != (VECTOR_WIDTH) ? (VECTOR_WIDTH): LIBXS_ALIGNMENT, \
      FLAGS, LIBXS_LD(M, N), LIBXS_LD(N, M), K, \
      0 == LIBXS_LD(PLDA, PLDB) ? LIBXS_LD(M, N) : *LIBXS_LD(PLDA, PLDB), \
      0 == LIBXS_LD(PLDB, PLDA) ? (K) : *LIBXS_LD(PLDB, PLDA), \
      0 == (PLDC) ? LIBXS_LD(M, N) : *(PLDC), scalpha, scbeta, \
      0 > internal_dispatch_main_prefetch ? *internal_prefetch() : internal_dispatch_main_prefetch); \
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


#if !defined(LIBXS_OPENMP)
# define INTERNAL_REGLOCK_COUNT 16
static LIBXS_RETARGETABLE LIBXS_LOCK_TYPE* internal_reglock(int i)
{
  static LIBXS_RETARGETABLE LIBXS_LOCK_TYPE instance[] = {
    LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
    LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
    LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT,
    LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT, LIBXS_LOCK_CONSTRUCT
  };
  assert(sizeof(instance) == (INTERNAL_REGLOCK_COUNT * sizeof(*instance)));
  assert(0 <= i && i < INTERNAL_REGLOCK_COUNT);
  return instance + i;
}
#endif


static LIBXS_RETARGETABLE internal_regkey_type** internal_registry_keys(void)
{
  static LIBXS_RETARGETABLE internal_regkey_type* instance = 0;
  return &instance;
}


static LIBXS_RETARGETABLE internal_code_type** internal_registry(void)
{
  static LIBXS_RETARGETABLE internal_code_type* instance = 0;
  return &instance;
}


static LIBXS_RETARGETABLE internal_statistic_type* internal_statistic(int precision)
{
  static LIBXS_RETARGETABLE internal_statistic_type instance[2/*DP/SP*/][3/*sml/med/big*/];
  assert(0 <= precision && precision < 2);
  return instance[precision];
}


static LIBXS_RETARGETABLE unsigned int* internal_statistic_sml(void)
{
  static LIBXS_RETARGETABLE unsigned int instance = 13;
  return &instance;
}


static LIBXS_RETARGETABLE unsigned int* internal_statistic_med(void)
{
  static LIBXS_RETARGETABLE unsigned int instance = 23;
  return &instance;
}


static LIBXS_RETARGETABLE unsigned int* internal_statistic_mnk(void)
{
  static LIBXS_RETARGETABLE unsigned int instance = LIBXS_MAX_M;
  return &instance;
}


static LIBXS_RETARGETABLE unsigned int* internal_teardown(void)
{
  static LIBXS_RETARGETABLE unsigned int instance = 0;
  return &instance;
}


static LIBXS_RETARGETABLE int* internal_target_archid(void)
{
  static LIBXS_RETARGETABLE int instance = LIBXS_TARGET_ARCH_GENERIC;
  return &instance;
}


static LIBXS_RETARGETABLE int* internal_verbose(void)
{
#if defined(NDEBUG)
  static LIBXS_RETARGETABLE int instance = 0; /* quiet */
#else
  static LIBXS_RETARGETABLE int instance = 1; /* verbose */
#endif
  return &instance;
}


static LIBXS_RETARGETABLE int* internal_prefetch(void)
{
  static LIBXS_RETARGETABLE int instance = LIBXS_MAX(INTERNAL_PREFETCH, 0);
  return &instance;
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_update_statistic(const libxs_gemm_descriptor* desc,
  unsigned ntry, unsigned ncol)
{
  assert(0 != desc);
  {
    const unsigned long long size = LIBXS_MNK_SIZE(desc->m, desc->n, desc->k);
    const int precision = (0 == (LIBXS_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1);
    const unsigned int statistic_sml = *internal_statistic_sml();
    int bucket = 2/*big*/;

    if (LIBXS_MNK_SIZE(statistic_sml, statistic_sml, statistic_sml) >= size) {
      bucket = 0;
    }
    else {
      const unsigned int statistic_med = *internal_statistic_med();
      if (LIBXS_MNK_SIZE(statistic_med, statistic_med, statistic_med) >= size) {
        bucket = 1;
      }
    }

    LIBXS_ATOMIC_ADD_FETCH(&internal_statistic(precision)[bucket].ntry, ntry, LIBXS_ATOMIC_RELAXED);
    LIBXS_ATOMIC_ADD_FETCH(&internal_statistic(precision)[bucket].ncol, ncol, LIBXS_ATOMIC_RELAXED);
  }
}


LIBXS_INLINE LIBXS_RETARGETABLE const char* internal_get_target_arch(int id);
LIBXS_INLINE LIBXS_RETARGETABLE const char* internal_get_target_arch(int id)
{
  const char* target_arch = 0;
  switch (id) {
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
    default: if (LIBXS_X86_GENERIC <= id) {
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
  const internal_statistic_type statistic_sml = internal_statistic(precision)[0/*SML*/];
  const internal_statistic_type statistic_med = internal_statistic(precision)[1/*MED*/];
  const internal_statistic_type statistic_big = internal_statistic(precision)[2/*BIG*/];
  int printed = 0;
  assert(0 != ostream && 0 != target_arch && (0 <= precision && precision < 2));

  if (/* omit to print anything if it is superfluous */
    0 != statistic_sml.ntry || 0 != statistic_sml.njit || 0 != statistic_sml.nsta || 0 != statistic_sml.ncol ||
    0 != statistic_med.ntry || 0 != statistic_med.njit || 0 != statistic_med.nsta || 0 != statistic_med.ncol ||
    0 != statistic_big.ntry || 0 != statistic_big.njit || 0 != statistic_big.nsta || 0 != statistic_big.ncol)
  {
    const unsigned int sml = *internal_statistic_sml(), med = *internal_statistic_med(), mnk = *internal_statistic_mnk();
    char title[256], csml[256], cmed[256], cbig[256];
    LIBXS_SNPRINTF(csml, sizeof(csml), "%u..%u",       0u, sml);
    LIBXS_SNPRINTF(cmed, sizeof(cmed), "%u..%u", sml + 1u, med);
    LIBXS_SNPRINTF(cbig, sizeof(cbig), "%u..%u", med + 1u, mnk);
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
    fprintf(ostream,  "%*s%10s %6u %6u %6u %6u\n", (int)indent, "", csml,
      statistic_sml.ntry, statistic_sml.njit, statistic_sml.nsta, statistic_sml.ncol);
    fprintf(ostream,  "%*s%10s %6u %6u %6u %6u\n", (int)indent, "", cmed,
      statistic_med.ntry, statistic_med.njit, statistic_med.nsta, statistic_med.ncol);
    fprintf(ostream,  "%*s%10s %6u %6u %6u %6u\n", (int)indent, "", cbig,
      statistic_big.ntry, statistic_big.njit, statistic_big.nsta, statistic_big.ncol);
    printed = 1;
  }

  return printed;
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_register_static_code(const libxs_gemm_descriptor* desc,
  unsigned int index, unsigned int hash, libxs_xmmfunction src, internal_code_type* registry)
{
  internal_regkey_type* dst_key = *internal_registry_keys() + index;
  internal_code_type* dst_entry = registry + index;
  assert(0 != desc && 0 != src.dmm && 0 != dst_key && 0 != registry);

  if (0 != dst_entry->pmm) { /* collision? */
    /* start at a re-hashed index position */
    const unsigned int start = LIBXS_HASH_MOD(LIBXS_HASH_VALUE(hash), LIBXS_REGSIZE);
    unsigned int i0, i, next;

    /* mark current entry as a collision (this might be already the case) */
    dst_entry->imm |= LIBXS_HASH_COLLISION;

    /* start linearly searching for an available slot */
    for (i = (start != index) ? start : LIBXS_HASH_MOD(start + 1, LIBXS_REGSIZE), i0 = i, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE);
      0 != (dst_entry = registry + i)->pmm && next != i0; i = next, next = LIBXS_HASH_MOD(i + 1, LIBXS_REGSIZE));

    /* corresponding key position */
    dst_key = *internal_registry_keys() + i;

    internal_update_statistic(desc, 0, 1);
  }

  if (0 == dst_entry->pmm) { /* registry not (yet) exhausted */
    dst_key->descriptor = *desc;
    dst_entry->xmm = src;
    /* mark current entry as a static (non-JIT) */
    dst_entry->imm |= LIBXS_CODE_STATIC;
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


LIBXS_INLINE LIBXS_RETARGETABLE void internal_get_code_name(const char* target_arch, const char* jit_kind,
  const libxs_gemm_descriptor* desc, unsigned int buffer_size, char* name)
{
  assert((0 != desc && 0 != name) || 0 == buffer_size);
  LIBXS_SNPRINTF(name, buffer_size, "libxs_%s_%c%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.%s",
    target_arch /* code path name */,
    0 == (LIBXS_GEMM_FLAG_F32PREC & desc->flags) ? 'd' : 's',
    0 == (LIBXS_GEMM_FLAG_TRANS_A & desc->flags) ? 'n' : 't',
    0 == (LIBXS_GEMM_FLAG_TRANS_B & desc->flags) ? 'n' : 't',
    (unsigned int)desc->m, (unsigned int)desc->n, (unsigned int)desc->k,
    (unsigned int)desc->lda, (unsigned int)desc->ldb, (unsigned int)desc->ldc,
    desc->alpha, desc->beta, internal_get_prefetch(desc),
    0 != jit_kind ? jit_kind : "jit");
}


LIBXS_INLINE LIBXS_RETARGETABLE internal_code_type* internal_init(void)
{
  /*const*/internal_code_type* result;
  int i;

#if !defined(LIBXS_OPENMP)
  /* acquire locks and thereby shortcut lazy initialization later on */
  for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXS_LOCK_ACQUIRE(internal_reglock(i));
#else
# pragma omp critical(internal_reglock)
#endif
  {
    result = LIBXS_ATOMIC_LOAD(internal_registry(), LIBXS_ATOMIC_SEQ_CST);
    if (0 == result) {
      int target_archid = LIBXS_TARGET_ARCH_GENERIC;
      int init_code;
      /* set internal_target_archid */
      libxs_set_target_arch(getenv("LIBXS_TARGET"));
      target_archid = *internal_target_archid();
      { /* select prefetch strategy for JIT */
        const char *const env_prefetch = getenv("LIBXS_PREFETCH");
        if (0 == env_prefetch || 0 == *env_prefetch) {
#if (0 > LIBXS_PREFETCH) /* permitted by LIBXS_PREFETCH_AUTO */
          *internal_prefetch() = (LIBXS_X86_AVX512_MIC != target_archid
            ? LIBXS_PREFETCH_NONE : LIBXS_PREFETCH_AL2BL2_VIA_C);
#endif
        }
        else { /* user input considered even if LIBXS_PREFETCH_AUTO is disabled */
          switch (atoi(env_prefetch)) {
            case 2:  *internal_prefetch() = LIBXS_PREFETCH_SIGONLY; break;
            case 3:  *internal_prefetch() = LIBXS_PREFETCH_BL2_VIA_C; break;
            case 4:  *internal_prefetch() = LIBXS_PREFETCH_AL2; break;
            case 5:  *internal_prefetch() = LIBXS_PREFETCH_AL2_AHEAD; break;
            case 6:  *internal_prefetch() = LIBXS_PREFETCH_AL2BL2_VIA_C; break;
            case 7:  *internal_prefetch() = LIBXS_PREFETCH_AL2BL2_VIA_C_AHEAD; break;
            case 8:  *internal_prefetch() = LIBXS_PREFETCH_AL2_JPST; break;
            case 9:  *internal_prefetch() = LIBXS_PREFETCH_AL2BL2_VIA_C_JPST; break;
            default: *internal_prefetch() = LIBXS_PREFETCH_NONE;
          }
        }
      }
      libxs_hash_init(target_archid);
      libxs_gemm_diff_init(target_archid);
      init_code = libxs_gemm_init(target_archid, *internal_prefetch());
#if defined(__TRACE)
      {
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
      }
#endif
      if (EXIT_SUCCESS == init_code) {
        assert(0 == *internal_registry_keys() && 0 == *internal_registry()); /* should never happen */
        result = (internal_code_type*)libxs_malloc(LIBXS_REGSIZE * sizeof(internal_code_type));
        *internal_registry_keys() = (internal_regkey_type*)libxs_malloc(LIBXS_REGSIZE * sizeof(internal_regkey_type));
        if (0 != result && 0 != *internal_registry_keys()) {
          const char *const env_verbose = getenv("LIBXS_VERBOSE");
          *internal_statistic_mnk() = (unsigned int)(pow((double)(LIBXS_MAX_MNK), 0.3333333333333333) + 0.5);
          if (0 != env_verbose && 0 != *env_verbose) {
            *internal_verbose() = atoi(env_verbose);
          }
          for (i = 0; i < LIBXS_REGSIZE; ++i) result[i].pmm = 0;
          /* omit registering code if JIT is enabled and if an ISA extension is found
           * which is beyond the static code path used to compile the library
           */
#if defined(LIBXS_BUILD)
# if (0 != LIBXS_JIT) && !defined(__MIC__)
          if (LIBXS_STATIC_TARGET_ARCH <= target_archid && LIBXS_X86_AVX > target_archid)
# endif
          { /* opening a scope for eventually declaring variables */
            /* setup the dispatch table for the statically generated code */
#           include <libxs_dispatch.h>
          }
#endif
          atexit(libxs_finalize);
          LIBXS_ATOMIC_STORE(internal_registry(), result, LIBXS_ATOMIC_SEQ_CST);
        }
        else {
#if !defined(NDEBUG) && defined(__TRACE) /* library code is expected to be mute */
          fprintf(stderr, "LIBXS: failed to allocate code registry!\n");
#endif
          libxs_free(*internal_registry_keys());
          libxs_free(result);
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
  for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXS_LOCK_RELEASE(internal_reglock(i));
#endif
  assert(result);
  return result;
}


LIBXS_API_DEFINITION
#if defined(__GNUC__)
LIBXS_ATTRIBUTE(constructor)
#endif
LIBXS_RETARGETABLE void libxs_init(void)
{
  const void *const registry = LIBXS_ATOMIC_LOAD(internal_registry(), LIBXS_ATOMIC_RELAXED);
  if (0 == registry) {
    internal_init();
  }
}


LIBXS_API_DEFINITION
#if defined(__GNUC__)
LIBXS_ATTRIBUTE(destructor)
LIBXS_ATTRIBUTE(no_instrument_function)
#endif
LIBXS_RETARGETABLE void libxs_finalize(void)
{
  internal_code_type* registry = LIBXS_ATOMIC_LOAD(internal_registry(), LIBXS_ATOMIC_SEQ_CST);

  if (0 != registry) {
    int i;
#if !defined(LIBXS_OPENMP)
    /* acquire locks and thereby shortcut lazy initialization later on */
    for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXS_LOCK_ACQUIRE(internal_reglock(i));
#else
#   pragma omp critical(internal_reglock)
#endif
    {
      registry = *internal_registry();

      if (0 != registry) {
        internal_regkey_type *const registry_keys = *internal_registry_keys();
        const char *const target_arch = internal_get_target_arch(*internal_target_archid());
        unsigned int heapmem = (LIBXS_REGSIZE) * (sizeof(internal_code_type) + sizeof(internal_regkey_type));

        /* serves as an id to invalidate the thread-local cache; never decremented */
        ++*internal_teardown();
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

        LIBXS_ATOMIC_STORE_ZERO(internal_registry(), LIBXS_ATOMIC_SEQ_CST);
        *internal_registry_keys() = 0;

        for (i = 0; i < LIBXS_REGSIZE; ++i) {
          internal_code_type code = registry[i];
          if (0 != code.pmm/*potentially allocated*/) {
            const libxs_gemm_descriptor *const desc = &registry_keys[i].descriptor;
            const unsigned long long kernel_size = LIBXS_MNK_SIZE(desc->m, desc->n, desc->k);
            const int precision = (0 == (LIBXS_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1);
            const unsigned int statistic_sml = *internal_statistic_sml();
            int bucket = 2;
            if (LIBXS_MNK_SIZE(statistic_sml, statistic_sml, statistic_sml) >= kernel_size) {
              bucket = 0;
            }
            else {
              const unsigned int statistic_med = *internal_statistic_med();
              if (LIBXS_MNK_SIZE(statistic_med, statistic_med, statistic_med) >= kernel_size) {
                bucket = 1;
              }
            }
            if (0 == (LIBXS_CODE_STATIC & code.imm)/*check non-JIT flag (dynamically allocated/generated*/) {
              void* buffer = 0;
              size_t size = 0;
              /* make address valid by clearing the collision flag */
              code.imm &= ~LIBXS_HASH_COLLISION;
              if (EXIT_SUCCESS == libxs_alloc_info(code.pmm, &size, 0/*flags*/, &buffer)) {
                libxs_deallocate(code.pmm);
                ++internal_statistic(precision)[bucket].njit;
                heapmem += (unsigned int)(size + (((char*)code.pmm) - (char*)buffer));
              }
            }
            else {
              ++internal_statistic(precision)[bucket].nsta;
            }
          }
        }
        if (0 != *internal_verbose()) { /* print statistic on termination */
          LIBXS_FLOCK(stderr);
          LIBXS_FLOCK(stdout);
          fflush(stdout); /* synchronize with standard output */
          {
            const unsigned int linebreak = 0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0) ? 1 : 0;
            if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak) {
              fprintf(stderr, "LIBXS_TARGET=%s ", target_arch);
            }
            fprintf(stderr, "HEAP: %.f MB\n", 1.0 * heapmem / (1 << 20));
          }
          LIBXS_FUNLOCK(stdout);
          LIBXS_FUNLOCK(stderr);
        }
        libxs_free(registry_keys);
        libxs_free(registry);
      }
    }
#if !defined(LIBXS_OPENMP) /* release locks */
  for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXS_LOCK_RELEASE(internal_reglock(i));
#endif
  }
}


LIBXS_API_DEFINITION int libxs_get_target_archid(void)
{
  LIBXS_INIT
#if !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  return *internal_target_archid();
#else /* no JIT support */
  return LIBXS_MIN(*internal_target_archid(), LIBXS_X86_SSE4_2);
#endif
}


LIBXS_API_DEFINITION void libxs_set_target_archid(int id)
{
  switch (id) {
    case LIBXS_X86_AVX512_CORE:
    case LIBXS_X86_AVX512_MIC:
    case LIBXS_X86_AVX2:
    case LIBXS_X86_AVX:
    case LIBXS_X86_SSE4_2:
    case LIBXS_X86_SSE4_1:
    case LIBXS_X86_SSE3:
    case LIBXS_TARGET_ARCH_GENERIC: {
      *internal_target_archid() = id;
    } break;
    default: if (LIBXS_X86_GENERIC <= id) {
      *internal_target_archid() = LIBXS_X86_GENERIC;
    }
    else {
      *internal_target_archid() = LIBXS_TARGET_ARCH_UNKNOWN;
    }
  }

#if !defined(NDEBUG) /* library code is expected to be mute */
  {
    const int target_archid = *internal_target_archid();
    const int cpuid = libxs_cpuid_x86();
    if (cpuid < target_archid) {
      const char *const target_arch = internal_get_target_arch(target_archid);
      fprintf(stderr, "LIBXS: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, internal_get_target_arch(cpuid));
    }
  }
#endif
}


LIBXS_API_DEFINITION const char* libxs_get_target_arch(void)
{
  LIBXS_INIT
  return internal_get_target_arch(*internal_target_archid());
}


/* function serves as a helper for implementing the Fortran interface */
LIBXS_API const char* get_target_arch(int* length);
LIBXS_API_DEFINITION const char* get_target_arch(int* length)
{
  const char *const arch = libxs_get_target_arch();
  /* valid here since function is not in the public interface */
  assert(0 != arch && 0 != length);
  *length = (int)strlen(arch);
  return arch;
}


LIBXS_API_DEFINITION void libxs_set_target_arch(const char* arch)
{
  int target_archid = LIBXS_TARGET_ARCH_UNKNOWN;

  if (0 != arch && 0 != *arch) {
    const int jit = atoi(arch);
    if (0 == strcmp("0", arch)) {
      target_archid = LIBXS_TARGET_ARCH_GENERIC;
    }
    else if (1 < jit) {
      target_archid = LIBXS_X86_GENERIC + jit;
    }
    else if (0 == strcmp("skx", arch) || 0 == strcmp("avx3", arch) || 0 == strcmp("avx512", arch)) {
      target_archid = LIBXS_X86_AVX512_CORE;
    }
    else if (0 == strcmp("knl", arch) || 0 == strcmp("mic2", arch)) {
      target_archid = LIBXS_X86_AVX512_MIC;
    }
    else if (0 == strcmp("hsw", arch) || 0 == strcmp("avx2", arch)) {
      target_archid = LIBXS_X86_AVX2;
    }
    else if (0 == strcmp("snb", arch) || 0 == strcmp("avx", arch)) {
      target_archid = LIBXS_X86_AVX;
    }
    else if (0 == strcmp("wsm", arch) || 0 == strcmp("nhm", arch) || 0 == strcmp("sse4", arch) || 0 == strcmp("sse4_2", arch) || 0 == strcmp("sse4.2", arch)) {
      target_archid = LIBXS_X86_SSE4_2;
    }
    else if (0 == strcmp("sse4_1", arch) || 0 == strcmp("sse4.1", arch)) {
      target_archid = LIBXS_X86_SSE4_1;
    }
    else if (0 == strcmp("sse3", arch) || 0 == strcmp("sse", arch)) {
      target_archid = LIBXS_X86_SSE3;
    }
    else if (0 == strcmp("x86", arch) || 0 == strcmp("sse2", arch)) {
      target_archid = LIBXS_X86_GENERIC;
    }
    else if (0 == strcmp("generic", arch) || 0 == strcmp("none", arch)) {
      target_archid = LIBXS_TARGET_ARCH_GENERIC;
    }
  }

  if (LIBXS_TARGET_ARCH_UNKNOWN == target_archid || LIBXS_X86_AVX512_CORE < target_archid) {
    target_archid = libxs_cpuid_x86();
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    const int cpuid = libxs_cpuid_x86();
    if (cpuid < target_archid) {
      const char *const target_arch = internal_get_target_arch(target_archid);
      fprintf(stderr, "LIBXS: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, internal_get_target_arch(cpuid));
    }
  }
#endif
  *internal_target_archid() = target_archid;
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_build(const libxs_gemm_descriptor* descriptor,
  const char* jit_kind, const internal_desc_extra_type* desc_extra, internal_code_type* code)
{
#if !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  const char *const target_arch = internal_get_target_arch(*internal_target_archid());
  libxs_generated_code generated_code;
  assert(0 != descriptor && 0 != code);
  assert(0 != *internal_target_archid());
  assert(0 == code->pmm);

  /* allocate temporary buffer which is large enough to cover the generated code */
  generated_code.generated_code = malloc(131072);
  generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
  generated_code.code_size = 0;
  generated_code.code_type = 2;
  generated_code.last_error = 0;

  /* generate kernel */
  if (0 == desc_extra) {
    libxs_generator_gemm_kernel(&generated_code, descriptor, target_arch);
  }
  else if (0 != desc_extra->row_ptr && 0 != desc_extra->column_idx &&
    0 != desc_extra->values)
  { /* currently only one additional kernel kind */
    assert(0 == (LIBXS_GEMM_FLAG_F32PREC & (descriptor->flags)));
    libxs_generator_spgemm_csr_soa_kernel(&generated_code, descriptor, target_arch,
      desc_extra->row_ptr, desc_extra->column_idx, (const double*)desc_extra->values);
  }

  /* handle an eventual error in the else-branch */
  if (0 == generated_code.last_error) {
    /* attempt to create executable buffer, and check for success */
    if (0 == libxs_allocate(&code->pmm, generated_code.code_size, 0/*auto*/,
      /* must be a superset of what mprotect populates (see below) */
      LIBXS_ALLOC_FLAG_RWX,
      0/*extra*/, 0/*extra_size*/))
    {
#if (!defined(NDEBUG) && defined(_DEBUG)) || defined(LIBXS_VTUNE)
      char jit_code_name[256];
      internal_get_code_name(target_arch, jit_kind, descriptor, sizeof(jit_code_name), jit_code_name);
#else
      const char *const jit_code_name = 0;
      LIBXS_UNUSED(jit_kind);
#endif
      /* copy temporary buffer into the prepared executable buffer */
      memcpy(code->pmm, generated_code.generated_code, generated_code.code_size);
      free(generated_code.generated_code); /* free temporary/initial code buffer */
      /* revoke unnecessary memory protection flags; continue on error */
      libxs_alloc_attribute(code->pmm, LIBXS_ALLOC_FLAG_RW, jit_code_name);
    }
    else {
      free(generated_code.generated_code);
    }
  }
  else {
#if !defined(NDEBUG) /* library code is expected to be mute */
    static LIBXS_TLS int error_jit = 0;
    if (0 == error_jit) {
      fprintf(stderr, "%s (error #%u)\n", libxs_strerror(generated_code.last_error),
        generated_code.last_error);
      error_jit = 1;
    }
#endif
    free(generated_code.generated_code);
  }
#else /* unsupported platform */
# if !defined(__MIC__)
  LIBXS_MESSAGE("================================================================================")
  LIBXS_MESSAGE("LIBXS: The JIT BACKEND is currently not supported under Microsoft Windows!")
  LIBXS_MESSAGE("================================================================================")
# endif
  LIBXS_UNUSED(descriptor); LIBXS_UNUSED(jit_kind); LIBXS_UNUSED(desc_extra);  LIBXS_UNUSED(code);
  /* libxs_get_target_arch also serves as a runtime check whether JIT is available or not */
  assert(LIBXS_X86_AVX > *internal_target_archid());
#endif
}


LIBXS_API_DEFINITION libxs_xmmfunction libxs_xmmdispatch(const libxs_gemm_descriptor* descriptor)
{
  const libxs_xmmfunction null_mmfunction = { 0 };
  if (0 != descriptor && INTERNAL_DISPATCH_BYPASS_CHECK(descriptor->flags, descriptor->alpha, descriptor->beta)) {
    libxs_gemm_descriptor backend_descriptor;

    if (0 > descriptor->prefetch) {
      backend_descriptor = *descriptor;
      backend_descriptor.prefetch = *internal_prefetch();
      descriptor = &backend_descriptor;
    }
    {
      INTERNAL_FIND_CODE_DECLARE(code);
      INTERNAL_FIND_CODE(descriptor, code);
    }
  }
  else { /* TODO: not supported (bypass) */
    return null_mmfunction;
  }
}


LIBXS_API_DEFINITION libxs_smmfunction libxs_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch)
{
  INTERNAL_SMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


LIBXS_API_DEFINITION libxs_dmmfunction libxs_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
  INTERNAL_DMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


LIBXS_API_DEFINITION libxs_dmmfunction libxs_create_dcsr_soa(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  internal_code_type code = { {0} };
  internal_desc_extra_type desc_extra;
  memset(&desc_extra, 0, sizeof(desc_extra));
  desc_extra.row_ptr = row_ptr;
  desc_extra.column_idx = column_idx;
  desc_extra.values = values;
  internal_build(descriptor, "csr", &desc_extra, &code);
  return code.xmm.dmm;
}


LIBXS_API_DEFINITION void libxs_destroy(const void* jit_code)
{
  libxs_deallocate(jit_code);
}


#if defined(LIBXS_GEMM_EXTWRAP)
#if defined(__STATIC)

LIBXS_API_DEFINITION void LIBXS_FSYMBOL(__real_sgemm)(
  const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc)
{
  sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXS_API_DEFINITION void LIBXS_FSYMBOL(__real_dgemm)(
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
