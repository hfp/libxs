/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_SYNC_H
#define LIBXS_SYNC_H

#include "libxs_intrinsics_x86.h"

#if !defined(LIBXS_TLS)
# if (0 != LIBXS_SYNC) && !defined(LIBXS_NO_TLS)
#   if defined(__CYGWIN__) && defined(__clang__)
#     define LIBXS_NO_TLS
#     define LIBXS_TLS
#   else
#     if (defined(_WIN32) && !defined(__GNUC__) && !defined(__clang__)) || (defined(__PGI) && !defined(__PGLLVM__))
#       define LIBXS_TLS LIBXS_ATTRIBUTE(thread)
#     elif defined(__GNUC__) || defined(__clang__) || defined(__PGLLVM__) || defined(_CRAYC)
#       define LIBXS_TLS __thread
#     elif defined(__cplusplus)
#       define LIBXS_TLS thread_local
#     else
#       error Missing TLS support!
#     endif
#   endif
# else
#   if !defined(LIBXS_NO_TLS)
#     define LIBXS_NO_TLS
#   endif
#   define LIBXS_TLS
# endif
#endif

#if !defined(LIBXS_GCC_BASELINE) && !defined(LIBXS_SYNC_LEGACY) && ((defined(_WIN32) && defined(__clang__)) || \
    (defined(__GNUC__) && LIBXS_VERSION2(4, 7) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__)))
# define LIBXS_GCC_BASELINE
#endif

#if defined(__MIC__)
# define LIBXS_SYNC_PAUSE _mm_delay_32(8/*delay*/)
#elif !defined(LIBXS_INTRINSICS_NONE)
# if defined(LIBXS_GCC_BASELINE) && !defined(__INTEL_COMPILER)
#   define LIBXS_SYNC_PAUSE __builtin_ia32_pause()
# else
#   define LIBXS_SYNC_PAUSE _mm_pause()
# endif
#elif (LIBXS_X86_GENERIC <= LIBXS_STATIC_TARGET_ARCH) && defined(__GNUC__)
# define LIBXS_SYNC_PAUSE __asm__ __volatile__("pause" ::: "memory")
#else
# define LIBXS_SYNC_PAUSE
#endif

/* permit thread-unsafe */
#if !defined(LIBXS_SYNC_NONE) && ( \
  (defined(__PGI) && (!defined(LIBXS_LIBATOMIC) || !defined(__STATIC))) || \
  (defined(_CRAYC) && !defined(__GNUC__)))
# define LIBXS_SYNC_NONE
#endif

#if !defined(LIBXS_ATOMIC_TRYLOCK_CMPSWP) && 0
# define LIBXS_ATOMIC_TRYLOCK_CMPSWP
#endif
#if !defined(LIBXS_ATOMIC_ZERO_STORE) && defined(_CRAYC)
# define LIBXS_ATOMIC_ZERO_STORE
#endif
#if !defined(LIBXS_ATOMIC_LOCKTYPE)
# if defined(_WIN32) || 1/*alignment*/
#   define LIBXS_ATOMIC_LOCKTYPE int
# else
#   define LIBXS_ATOMIC_LOCKTYPE char
# endif
#endif

typedef enum libxs_atomic_kind {
#if defined(__ATOMIC_SEQ_CST)
  LIBXS_ATOMIC_SEQ_CST = __ATOMIC_SEQ_CST,
#else
  LIBXS_ATOMIC_SEQ_CST = 0,
#endif
#if defined(__ATOMIC_RELAXED)
  LIBXS_ATOMIC_RELAXED = __ATOMIC_RELAXED
#else
  LIBXS_ATOMIC_RELAXED = LIBXS_ATOMIC_SEQ_CST
#endif
} libxs_atomic_kind;

#define LIBXS_NONATOMIC_LOCKTYPE LIBXS_ATOMIC_LOCKTYPE
#define LIBXS_NONATOMIC_LOAD(SRC_PTR, KIND) (*(SRC_PTR))
#define LIBXS_NONATOMIC_STORE(DST_PTR, VALUE, KIND) { LIBXS_UNUSED(KIND); *(DST_PTR) = (VALUE); }
#define LIBXS_NONATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXS_NONATOMIC_STORE(DST_PTR, 0, KIND)
#define LIBXS_NONATOMIC_FETCH_OR(DST_PTR, VALUE/*side-effect*/, KIND) (/* 1st step: swap(dst, val) */ \
  ((*DST_PTR) = (*DST_PTR) ^ (VALUE)), (VALUE = (VALUE) ^ (*DST_PTR)), ((*DST_PTR) = (*DST_PTR) ^ (VALUE)), \
  (*(DST_PTR) |= VALUE), (VALUE) /* 2nd step: or, and 3rd/last step: original dst-value */)
#define LIBXS_NONATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) += VALUE)
#define LIBXS_NONATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) (*(DST_PTR) -= VALUE)
#define LIBXS_NONATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) (LIBXS_NONATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND), (*(DST_PTR) - (VALUE)))
#define LIBXS_NONATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) (LIBXS_NONATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND), (*(DST_PTR) + (VALUE)))
#define LIBXS_NONATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) ((NEWVAL) == (*(DST_PTR) == (OLDVAL) ? (*(DST_PTR) = (NEWVAL)) : (OLDVAL)))
#define LIBXS_NONATOMIC_TRYLOCK(DST_PTR, KIND) LIBXS_NONATOMIC_CMPSWP(DST_PTR, 0, 1, KIND)
#define LIBXS_NONATOMIC_ACQUIRE(DST_PTR, NPAUSE, KIND) { LIBXS_UNUSED(NPAUSE); \
          LIBXS_ASSERT_MSG(0 == *(DST_PTR), "LIBXS_NONATOMIC_ACQUIRE"); LIBXS_NONATOMIC_STORE(DST_PTR, 1, KIND); \
          LIBXS_ASSERT_MSG(0 != *(DST_PTR), "LIBXS_NONATOMIC_ACQUIRE"); }
#define LIBXS_NONATOMIC_RELEASE(DST_PTR, KIND) { LIBXS_UNUSED(DST_PTR); LIBXS_UNUSED(KIND); \
          LIBXS_ASSERT_MSG(0 != *(DST_PTR), "LIBXS_NONATOMIC_RELEASE"); LIBXS_NONATOMIC_STORE(DST_PTR, 0, KIND); \
          LIBXS_ASSERT_MSG(0 == *(DST_PTR), "LIBXS_NONATOMIC_RELEASE"); }
#define LIBXS_NONATOMIC_SYNC(KIND) LIBXS_UNUSED(KIND)

#if (0 == LIBXS_SYNC) || defined(LIBXS_SYNC_NONE)
# define LIBXS_ATOMIC(FN, BITS) FN
# define LIBXS_ATOMIC_LOAD LIBXS_NONATOMIC_LOAD
# define LIBXS_ATOMIC_STORE LIBXS_NONATOMIC_STORE
# define LIBXS_ATOMIC_STORE_ZERO LIBXS_NONATOMIC_STORE_ZERO
# define LIBXS_ATOMIC_FETCH_OR LIBXS_NONATOMIC_FETCH_OR
# define LIBXS_ATOMIC_ADD_FETCH LIBXS_NONATOMIC_ADD_FETCH
# define LIBXS_ATOMIC_SUB_FETCH LIBXS_NONATOMIC_SUB_FETCH
# define LIBXS_ATOMIC_FETCH_ADD LIBXS_NONATOMIC_FETCH_ADD
# define LIBXS_ATOMIC_FETCH_SUB LIBXS_NONATOMIC_FETCH_SUB
# define LIBXS_ATOMIC_CMPSWP LIBXS_NONATOMIC_CMPSWP
# define LIBXS_ATOMIC_TRYLOCK LIBXS_NONATOMIC_TRYLOCK
# define LIBXS_ATOMIC_ACQUIRE LIBXS_NONATOMIC_ACQUIRE
# define LIBXS_ATOMIC_RELEASE LIBXS_NONATOMIC_RELEASE
# define LIBXS_ATOMIC_SYNC LIBXS_NONATOMIC_SYNC
# if !defined(LIBXS_SYNC_NPAUSE)
#   define LIBXS_SYNC_NPAUSE 0
# endif
#elif (defined(LIBXS_GCC_BASELINE) || defined(LIBXS_LIBATOMIC) /* GNU's libatomic required */ || \
      (defined(__GNUC__) && LIBXS_VERSION2(4, 1) <= LIBXS_VERSION2(__GNUC__, __GNUC_MINOR__)))
# if defined(LIBXS_LIBATOMIC)
#   define LIBXS_ATOMIC(FN, BITS) LIBXS_CONCATENATE(LIBXS_ATOMIC, BITS)(FN)
#   define LIBXS_ATOMIC8(FN) LIBXS_CONCATENATE(FN, 8)
#   define LIBXS_ATOMIC16(FN) LIBXS_CONCATENATE(FN, 16)
#   define LIBXS_ATOMIC32(FN) FN/*default*/
#   define LIBXS_ATOMIC64(FN) LIBXS_CONCATENATE(FN, 64)
#   if defined(__PGI)
#     define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) LIBXS_NONATOMIC_LOAD(SRC_PTR, KIND)
#     define LIBXS_ATOMIC_LOAD8(SRC_PTR, KIND) LIBXS_NONATOMIC_LOAD(SRC_PTR, KIND)
#     define LIBXS_ATOMIC_LOAD16(SRC_PTR, KIND) LIBXS_NONATOMIC_LOAD(SRC_PTR, KIND)
#     define LIBXS_ATOMIC_LOAD64(SRC_PTR, KIND) LIBXS_NONATOMIC_LOAD(SRC_PTR, KIND)
#     define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) LIBXS_NONATOMIC_STORE(DST_PTR, VALUE, KIND)
#     define LIBXS_ATOMIC_STORE8(DST_PTR, VALUE, KIND) LIBXS_NONATOMIC_STORE(DST_PTR, VALUE, KIND)
#     define LIBXS_ATOMIC_STORE16(DST_PTR, VALUE, KIND) LIBXS_NONATOMIC_STORE(DST_PTR, VALUE, KIND)
#     define LIBXS_ATOMIC_STORE64(DST_PTR, VALUE, KIND) LIBXS_NONATOMIC_STORE(DST_PTR, VALUE, KIND)
#   else
#     define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) __atomic_load_4(SRC_PTR, KIND)
#     define LIBXS_ATOMIC_LOAD8(SRC_PTR, KIND) __atomic_load_1(SRC_PTR, KIND)
#     define LIBXS_ATOMIC_LOAD16(SRC_PTR, KIND) __atomic_load_2(SRC_PTR, KIND)
#     define LIBXS_ATOMIC_LOAD64(SRC_PTR, KIND) __atomic_load_8(SRC_PTR, KIND)
#     define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) __atomic_store_4(DST_PTR, (unsigned int)(VALUE), KIND)
#     define LIBXS_ATOMIC_STORE8(DST_PTR, VALUE, KIND) __atomic_store_1(DST_PTR, (unsigned char)(VALUE), KIND)
#     define LIBXS_ATOMIC_STORE16(DST_PTR, VALUE, KIND) __atomic_store_2(DST_PTR, (unsigned short)(VALUE), KIND)
#     define LIBXS_ATOMIC_STORE64(DST_PTR, VALUE, KIND) __atomic_store_8(DST_PTR, (unsigned long long)(VALUE), KIND)
#   endif
#   define LIBXS_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) __atomic_fetch_or_4(DST_PTR, (unsigned int)(VALUE), KIND)
#   define LIBXS_ATOMIC_FETCH_OR8(DST_PTR, VALUE, KIND) __atomic_fetch_or_1(DST_PTR, (unsigned char)(VALUE), KIND)
#   define LIBXS_ATOMIC_FETCH_OR16(DST_PTR, VALUE, KIND) __atomic_fetch_or_2(DST_PTR, (unsigned short)(VALUE), KIND)
#   define LIBXS_ATOMIC_FETCH_OR64(DST_PTR, VALUE, KIND) __atomic_fetch_or_8(DST_PTR, (unsigned long long)(VALUE), KIND)
#   define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) ((int)__atomic_add_fetch_4(DST_PTR, (int)(VALUE), KIND))
#   define LIBXS_ATOMIC_ADD_FETCH8(DST_PTR, VALUE, KIND) ((signed char)__atomic_add_fetch_1(DST_PTR, (signed char)(VALUE), KIND))
#   define LIBXS_ATOMIC_ADD_FETCH16(DST_PTR, VALUE, KIND) ((short)__atomic_add_fetch_2(DST_PTR, (short)(VALUE), KIND))
#   define LIBXS_ATOMIC_ADD_FETCH64(DST_PTR, VALUE, KIND) ((long long)__atomic_add_fetch_8(DST_PTR, (long long)(VALUE), KIND))
#   define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) ((int)__atomic_sub_fetch_4(DST_PTR, (int)(VALUE), KIND))
#   define LIBXS_ATOMIC_SUB_FETCH8(DST_PTR, VALUE, KIND) ((signed char)__atomic_sub_fetch_1(DST_PTR, (signed char)(VALUE), KIND))
#   define LIBXS_ATOMIC_SUB_FETCH16(DST_PTR, VALUE, KIND) ((short)__atomic_sub_fetch_2(DST_PTR, (short)(VALUE), KIND))
#   define LIBXS_ATOMIC_SUB_FETCH64(DST_PTR, VALUE, KIND) ((long long)__atomic_sub_fetch_8(DST_PTR, (long long)(VALUE), KIND))
#   define LIBXS_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) ((int)__atomic_fetch_add_4(DST_PTR, (int)(VALUE), KIND))
#   define LIBXS_ATOMIC_FETCH_ADD8(DST_PTR, VALUE, KIND) ((signed char)__atomic_fetch_add_1(DST_PTR, (signed char)(VALUE), KIND))
#   define LIBXS_ATOMIC_FETCH_ADD16(DST_PTR, VALUE, KIND) ((short)__atomic_fetch_add_2(DST_PTR, (short)(VALUE), KIND))
#   define LIBXS_ATOMIC_FETCH_ADD64(DST_PTR, VALUE, KIND) ((long long)__atomic_fetch_add_8(DST_PTR, (long long)(VALUE), KIND))
#   define LIBXS_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) ((int)__atomic_fetch_sub_4(DST_PTR, (int)(VALUE), KIND))
#   define LIBXS_ATOMIC_FETCH_SUB8(DST_PTR, VALUE, KIND) ((signed char)__atomic_fetch_sub_1(DST_PTR, (signed char)(VALUE), KIND))
#   define LIBXS_ATOMIC_FETCH_SUB16(DST_PTR, VALUE, KIND) ((short)__atomic_fetch_sub_2(DST_PTR, (short)(VALUE), KIND))
#   define LIBXS_ATOMIC_FETCH_SUB64(DST_PTR, VALUE, KIND) ((long long)__atomic_fetch_sub_8(DST_PTR, (long long)(VALUE), KIND))
#   define LIBXS_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) \
            __atomic_compare_exchange_4(DST_PTR, &(OLDVAL), (NEWVAL), 0/*false*/, KIND, LIBXS_ATOMIC_RELAXED)
#   define LIBXS_ATOMIC_CMPSWP8(DST_PTR, OLDVAL, NEWVAL, KIND) \
            __atomic_compare_exchange_1(DST_PTR, &(OLDVAL), (NEWVAL), 0/*false*/, KIND, LIBXS_ATOMIC_RELAXED)
#   define LIBXS_ATOMIC_CMPSWP16(DST_PTR, OLDVAL, NEWVAL, KIND) \
            __atomic_compare_exchange_2(DST_PTR, &(OLDVAL), (NEWVAL), 0/*false*/, KIND, LIBXS_ATOMIC_RELAXED)
#   define LIBXS_ATOMIC_CMPSWP64(DST_PTR, OLDVAL, NEWVAL, KIND) \
            __atomic_compare_exchange_8(DST_PTR, &(OLDVAL), (NEWVAL), 0/*false*/, KIND, LIBXS_ATOMIC_RELAXED)
#   if defined(LIBXS_ATOMIC_TRYLOCK_CMPSWP)
#     define LIBXS_ATOMIC_TRYLOCK(DST_PTR, KIND) (!__atomic_test_and_set(DST_PTR, KIND))
#   endif
#   if defined(__PGI)
#     define LIBXS_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXS_ASSERT_MSG(0 != *(DST_PTR), "LIBXS_ATOMIC_RELEASE"); \
              LIBXS_ATOMIC_STORE_ZERO8(DST_PTR, KIND); } /* matches bit-width of LIBXS_ATOMIC_LOCKTYPE */
#   else
#     define LIBXS_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXS_ASSERT_MSG(0 != *(DST_PTR), "LIBXS_ATOMIC_RELEASE"); \
              __atomic_clear(DST_PTR, KIND); }
#   endif
#   define LIBXS_ATOMIC_SYNC(KIND) __sync_synchronize()
#   if !defined(LIBXS_ATOMIC_ZERO_STORE)
#     define LIBXS_ATOMIC_ZERO_STORE
#   endif
# elif defined(LIBXS_GCC_BASELINE)
#   define LIBXS_ATOMIC(FN, BITS) FN
#   define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) __atomic_load_n(SRC_PTR, KIND)
#   define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) __atomic_store_n(DST_PTR, VALUE, KIND)
#   if !defined(LIBXS_ATOMIC_ZERO_STORE)
#     define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) do {} while (__atomic_and_fetch(DST_PTR, 0, KIND))
#   endif
#   define LIBXS_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) __atomic_fetch_or(DST_PTR, VALUE, KIND)
#   define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) __atomic_add_fetch(DST_PTR, VALUE, KIND)
#   define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) __atomic_sub_fetch(DST_PTR, VALUE, KIND)
#   define LIBXS_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) __atomic_fetch_add(DST_PTR, VALUE, KIND)
#   define LIBXS_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) __atomic_fetch_sub(DST_PTR, VALUE, KIND)
#   define LIBXS_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) __sync_bool_compare_and_swap(DST_PTR, OLDVAL, NEWVAL)
#   if defined(LIBXS_ATOMIC_TRYLOCK_CMPSWP)
#     define LIBXS_ATOMIC_TRYLOCK(DST_PTR, KIND) (!__atomic_test_and_set(DST_PTR, KIND))
#   endif
#   define LIBXS_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXS_ASSERT_MSG(0 != *(DST_PTR), "LIBXS_ATOMIC_RELEASE"); \
            __atomic_clear(DST_PTR, KIND); }
#   if 0 /* __atomic_thread_fence: incorrect behavior in libxs_barrier (even with LIBXS_ATOMIC_SEQ_CST) */
#     define LIBXS_ATOMIC_SYNC(KIND) __atomic_thread_fence(KIND)
#   else
#     define LIBXS_ATOMIC_SYNC(KIND) __sync_synchronize()
#   endif
# else /* GCC legacy atomics */
#   define LIBXS_ATOMIC(FN, BITS) FN
#   define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) __sync_or_and_fetch(SRC_PTR, 0)
#   if (LIBXS_X86_GENERIC <= LIBXS_STATIC_TARGET_ARCH)
#     define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) { \
              __asm__ __volatile__("" ::: "memory"); *(DST_PTR) = (VALUE); \
              __asm__ __volatile__("" ::: "memory"); }
#   else
#     define LIBXS_ATOMIC_SYNC_NOFENCE(KIND)
#     define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) *(DST_PTR) = (VALUE)
#   endif
#   if !defined(LIBXS_ATOMIC_ZERO_STORE)
#     define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) do {} while (__sync_and_and_fetch(DST_PTR, 0))
#   endif
#   define LIBXS_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) __sync_fetch_and_or(DST_PTR, VALUE)
#   define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) __sync_add_and_fetch(DST_PTR, VALUE)
#   define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) __sync_sub_and_fetch(DST_PTR, VALUE)
#   define LIBXS_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) __sync_fetch_and_add(DST_PTR, VALUE)
#   define LIBXS_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) __sync_fetch_and_sub(DST_PTR, VALUE)
#   define LIBXS_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) __sync_bool_compare_and_swap(DST_PTR, OLDVAL, NEWVAL)
#   if defined(LIBXS_ATOMIC_TRYLOCK_CMPSWP)
#     define LIBXS_ATOMIC_TRYLOCK(DST_PTR, KIND) (0 == __sync_lock_test_and_set(DST_PTR, 1))
#   endif
#   define LIBXS_ATOMIC_RELEASE(DST_PTR, KIND) { LIBXS_ASSERT_MSG(0 != *(DST_PTR), "LIBXS_ATOMIC_RELEASE"); \
            __sync_lock_release(DST_PTR); }
#   define LIBXS_ATOMIC_SYNC(KIND) __sync_synchronize()
# endif
# if defined(LIBXS_ATOMIC_ZERO_STORE)
#   define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXS_ATOMIC_STORE(DST_PTR, 0, KIND)
#   define LIBXS_ATOMIC_STORE_ZERO8(DST_PTR, KIND) LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, 8)(DST_PTR, 0, KIND)
#   define LIBXS_ATOMIC_STORE_ZERO16(DST_PTR, KIND) LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, 16)(DST_PTR, 0, KIND)
#   define LIBXS_ATOMIC_STORE_ZERO64(DST_PTR, KIND) LIBXS_ATOMIC(LIBXS_ATOMIC_STORE, 64)(DST_PTR, 0, KIND)
# endif
# if !defined(LIBXS_ATOMIC_TRYLOCK_CMPSWP)
#   define LIBXS_ATOMIC_TRYLOCK(DST_PTR, KIND) /* matches bit-width of LIBXS_ATOMIC_LOCKTYPE */ \
            (0 == LIBXS_ATOMIC(LIBXS_ATOMIC_FETCH_OR, 8)(DST_PTR, 1, KIND))
# endif
# define LIBXS_ATOMIC_ACQUIRE(DST_PTR, NPAUSE, KIND) \
          LIBXS_ASSERT(0 == LIBXS_MOD2((uintptr_t)(DST_PTR), 4)); \
          while (!LIBXS_ATOMIC_TRYLOCK(DST_PTR, KIND)) LIBXS_SYNC_CYCLE(DST_PTR, 0/*free*/, NPAUSE); \
          LIBXS_ASSERT_MSG(0 != *(DST_PTR), "LIBXS_ATOMIC_ACQUIRE")
# if !defined(LIBXS_SYNC_NPAUSE)
#   define LIBXS_SYNC_NPAUSE 4096
# endif
#elif defined(_WIN32)
# define LIBXS_ATOMIC(FN, BITS) LIBXS_CONCATENATE(LIBXS_ATOMIC, BITS)(FN)
# define LIBXS_ATOMIC8(FN) LIBXS_CONCATENATE(FN, 8)
# define LIBXS_ATOMIC16(FN) LIBXS_CONCATENATE(FN, 16)
# define LIBXS_ATOMIC32(FN) FN/*default*/
# define LIBXS_ATOMIC64(FN) LIBXS_CONCATENATE(FN, 64)
# define LIBXS_ATOMIC_LOAD(SRC_PTR, KIND) InterlockedOr((volatile LONG*)(SRC_PTR), 0)
# define LIBXS_ATOMIC_LOAD8(SRC_PTR, KIND) _InterlockedOr8((volatile char*)(SRC_PTR), 0)
# define LIBXS_ATOMIC_LOAD64(SRC_PTR, KIND) InterlockedOr64((volatile LONGLONG*)(SRC_PTR), 0)
# define LIBXS_ATOMIC_STORE(DST_PTR, VALUE, KIND) InterlockedExchange((volatile LONG*)(DST_PTR), (LONG)(VALUE))
# define LIBXS_ATOMIC_STORE8(DST_PTR, VALUE, KIND) InterlockedExchange8((volatile char*)(DST_PTR), (LONGLONG)(VALUE))
# define LIBXS_ATOMIC_STORE64(DST_PTR, VALUE, KIND) InterlockedExchange64((volatile LONGLONG*)(DST_PTR), (LONGLONG)(VALUE))
# if defined(LIBXS_ATOMIC_ZERO_STORE)
#   define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) LIBXS_ATOMIC_STORE(DST_PTR, 0, KIND)
#   define LIBXS_ATOMIC_STORE_ZERO8(DST_PTR, KIND) LIBXS_ATOMIC_STORE8(DST_PTR, 0, KIND)
#   define LIBXS_ATOMIC_STORE_ZERO64(DST_PTR, KIND) LIBXS_ATOMIC_STORE64(DST_PTR, 0, KIND)
# else
#   define LIBXS_ATOMIC_STORE_ZERO(DST_PTR, KIND) InterlockedAnd((volatile LONG*)(DST_PTR), 0)
#   define LIBXS_ATOMIC_STORE_ZERO8(DST_PTR, KIND) InterlockedAnd8((volatile char*)(DST_PTR), 0)
#   define LIBXS_ATOMIC_STORE_ZERO64(DST_PTR, KIND) InterlockedAnd64((volatile LONGLONG*)(DST_PTR), 0)
# endif
# define LIBXS_ATOMIC_FETCH_OR(DST_PTR, VALUE, KIND) InterlockedOr((volatile LONG*)(DST_PTR), VALUE)
# define LIBXS_ATOMIC_FETCH_OR8(DST_PTR, VALUE, KIND) _InterlockedOr8((volatile char*)(DST_PTR), VALUE)
# define LIBXS_ATOMIC_ADD_FETCH(DST_PTR, VALUE, KIND) (LIBXS_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) + (VALUE))
# define LIBXS_ATOMIC_ADD_FETCH16(DST_PTR, VALUE, KIND) (LIBXS_ATOMIC_FETCH_ADD16(DST_PTR, VALUE, KIND) + (VALUE))
# define LIBXS_ATOMIC_ADD_FETCH64(DST_PTR, VALUE, KIND) (LIBXS_ATOMIC_FETCH_ADD64(DST_PTR, VALUE, KIND) + (VALUE))
# define LIBXS_ATOMIC_SUB_FETCH(DST_PTR, VALUE, KIND) ((size_t)LIBXS_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) - ((size_t)VALUE))
# define LIBXS_ATOMIC_SUB_FETCH16(DST_PTR, VALUE, KIND) (LIBXS_ATOMIC_FETCH_SUB16(DST_PTR, VALUE, KIND) - (VALUE))
# define LIBXS_ATOMIC_SUB_FETCH64(DST_PTR, VALUE, KIND) (LIBXS_ATOMIC_FETCH_SUB64(DST_PTR, VALUE, KIND) - (VALUE))
# define LIBXS_ATOMIC_FETCH_ADD(DST_PTR, VALUE, KIND) InterlockedExchangeAdd((volatile LONG*)(DST_PTR), VALUE)
# define LIBXS_ATOMIC_FETCH_ADD16(DST_PTR, VALUE, KIND) _InterlockedExchangeAdd16((volatile SHORT*)(DST_PTR), VALUE)
# define LIBXS_ATOMIC_FETCH_ADD64(DST_PTR, VALUE, KIND) InterlockedExchangeAdd64((volatile LONGLONG*)(DST_PTR), VALUE)
# define LIBXS_ATOMIC_FETCH_SUB(DST_PTR, VALUE, KIND) LIBXS_ATOMIC_FETCH_ADD(DST_PTR, -1 * (VALUE), KIND)
# define LIBXS_ATOMIC_FETCH_SUB16(DST_PTR, VALUE, KIND) LIBXS_ATOMIC_FETCH_ADD16(DST_PTR, -1 * (VALUE), KIND)
# define LIBXS_ATOMIC_FETCH_SUB64(DST_PTR, VALUE, KIND) LIBXS_ATOMIC_FETCH_ADD64(DST_PTR, -1 * (VALUE), KIND)
# define LIBXS_ATOMIC_CMPSWP(DST_PTR, OLDVAL, NEWVAL, KIND) (((LONG)(OLDVAL)) == InterlockedCompareExchange((volatile LONG*)(DST_PTR), NEWVAL, OLDVAL))
# define LIBXS_ATOMIC_CMPSWP8(DST_PTR, OLDVAL, NEWVAL, KIND) ((OLDVAL) == _InterlockedCompareExchange8((volatile char*)(DST_PTR), NEWVAL, OLDVAL))
# if defined(LIBXS_ATOMIC_TRYLOCK_CMPSWP)
#   define LIBXS_ATOMIC_TRYLOCK(DST_PTR, KIND) LIBXS_ATOMIC(LIBXS_ATOMIC_CMPSWP, 8)(DST_PTR, 0, 1, KIND)
# else
#   define LIBXS_ATOMIC_TRYLOCK(DST_PTR, KIND) (0 == LIBXS_ATOMIC(LIBXS_ATOMIC_FETCH_OR, 8)(DST_PTR, 1, KIND))
# endif
# define LIBXS_ATOMIC_ACQUIRE(DST_PTR, NPAUSE, KIND) \
          LIBXS_ASSERT(0 == LIBXS_MOD2((uintptr_t)(DST_PTR), 4)); \
          while (!LIBXS_ATOMIC_TRYLOCK(DST_PTR, KIND)) LIBXS_SYNC_CYCLE(DST_PTR, 0/*free*/, NPAUSE); \
          LIBXS_ASSERT_MSG(0 != *(DST_PTR), "LIBXS_ATOMIC_ACQUIRE")
# define LIBXS_ATOMIC_RELEASE(DST_PTR, KIND) { \
          LIBXS_ASSERT_MSG(0 != *(DST_PTR), "LIBXS_ATOMIC_RELEASE"); \
          LIBXS_ATOMIC(LIBXS_ATOMIC_STORE_ZERO, 8)(DST_PTR, KIND); }
# define LIBXS_ATOMIC_SYNC(KIND) _ReadWriteBarrier()
# if !defined(LIBXS_SYNC_NPAUSE)
#   define LIBXS_SYNC_NPAUSE 4096
# endif
#else /* consider to permit LIBXS_SYNC_NONE */
# error LIBXS is missing atomic compiler builtins!
#endif

#if !defined(LIBXS_SYNC_CYCLE)
# if (0 < LIBXS_SYNC_NPAUSE)
#   define LIBXS_SYNC_CYCLE_ELSE(DST_PTR, EXP_STATE, NPAUSE, ELSE) do { int libxs_sync_cycle_npause_ = 1; \
      do { int libxs_sync_cycle_counter_ = 0; \
        for (; libxs_sync_cycle_counter_ < libxs_sync_cycle_npause_; ++libxs_sync_cycle_counter_) LIBXS_SYNC_PAUSE; \
        if (libxs_sync_cycle_npause_ < (NPAUSE)) { \
          libxs_sync_cycle_npause_ *= 2; \
        } \
        else { \
          libxs_sync_cycle_npause_ = (NPAUSE); \
          LIBXS_SYNC_YIELD; \
          ELSE \
        } \
      } while(((EXP_STATE) & 1) != (*(DST_PTR) & 1)); \
    } while(0)
# else
#   define LIBXS_SYNC_CYCLE_ELSE(DST_PTR, EXP_STATE, NPAUSE, ELSE) LIBXS_SYNC_PAUSE
# endif
# define LIBXS_SYNC_CYCLE(DST_PTR, EXP_STATE, NPAUSE) \
    LIBXS_SYNC_CYCLE_ELSE(DST_PTR, EXP_STATE, NPAUSE, /*else*/;)
#endif

#if (0 != LIBXS_SYNC)
# define LIBXS_LOCK_DEFAULT LIBXS_LOCK_SPINLOCK
# if !defined(LIBXS_LOCK_SYSTEM_SPINLOCK) && !(defined(_OPENMP) && defined(LIBXS_SYNC_OMP)) && \
    (!defined(__linux__) || defined(__USE_XOPEN2K)) && 0/*disabled*/
#   define LIBXS_LOCK_SYSTEM_SPINLOCK
# endif
# if !defined(LIBXS_LOCK_SYSTEM_MUTEX) && !(defined(_OPENMP) && defined(LIBXS_SYNC_OMP))
#   define LIBXS_LOCK_SYSTEM_MUTEX
# endif
# if !defined(LIBXS_LOCK_SYSTEM_RWLOCK) && !(defined(_OPENMP) && defined(LIBXS_SYNC_OMP)) && \
    (!defined(__linux__) || defined(__USE_XOPEN2K) || defined(__USE_UNIX98))
#   define LIBXS_LOCK_SYSTEM_RWLOCK
# endif
  /* Lock type, initialization, destruction, (try-)lock, unlock, etc */
# define LIBXS_LOCK_ACQUIRED(KIND) LIBXS_CONCATENATE(LIBXS_LOCK_ACQUIRED_, KIND)
# define LIBXS_LOCK_TYPE_ISPOD(KIND) LIBXS_CONCATENATE(LIBXS_LOCK_TYPE_ISPOD_, KIND)
# define LIBXS_LOCK_TYPE_ISRW(KIND) LIBXS_CONCATENATE(LIBXS_LOCK_TYPE_ISRW_, KIND)
# define LIBXS_LOCK_TYPE(KIND) LIBXS_CONCATENATE(LIBXS_LOCK_TYPE_, KIND)
# define LIBXS_LOCK_INIT(KIND, LOCK, ATTR) LIBXS_CONCATENATE(LIBXS_LOCK_INIT_, KIND)(LOCK, ATTR)
# define LIBXS_LOCK_DESTROY(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_DESTROY_, KIND)(LOCK)
# define LIBXS_LOCK_TRYLOCK(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_TRYLOCK_, KIND)(LOCK)
# define LIBXS_LOCK_ACQUIRE(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_ACQUIRE_, KIND)(LOCK)
# define LIBXS_LOCK_RELEASE(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_RELEASE_, KIND)(LOCK)
# define LIBXS_LOCK_TRYREAD(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_TRYREAD_, KIND)(LOCK)
# define LIBXS_LOCK_ACQREAD(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_ACQREAD_, KIND)(LOCK)
# define LIBXS_LOCK_RELREAD(KIND, LOCK) LIBXS_CONCATENATE(LIBXS_LOCK_RELREAD_, KIND)(LOCK)
  /* Attribute type, initialization, destruction */
# define LIBXS_LOCK_ATTR_TYPE(KIND) LIBXS_CONCATENATE(LIBXS_LOCK_ATTR_TYPE_, KIND)
# define LIBXS_LOCK_ATTR_INIT(KIND, ATTR) LIBXS_CONCATENATE(LIBXS_LOCK_ATTR_INIT_, KIND)(ATTR)
# define LIBXS_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXS_CONCATENATE(LIBXS_LOCK_ATTR_DESTROY_, KIND)(ATTR)
  /* Cygwin's Pthread implementation appears to be broken; use Win32 */
# if !defined(LIBXS_WIN32_THREADS) && (defined(_WIN32) || defined(__CYGWIN__))
#   define LIBXS_WIN32_THREADS _WIN32_WINNT
#   if defined(__CYGWIN__) || defined(__MINGW32__) /* hack: make SRW-locks available */
#     if defined(_WIN32_WINNT)
#       undef _WIN32_WINNT
#       if !defined(NTDDI_VERSION)
#         define NTDDI_VERSION 0x0600
#       endif
#       define _WIN32_WINNT ((LIBXS_WIN32_THREADS) | 0x0600)
#     else
#       define _WIN32_WINNT 0x0600
#     endif
#   endif
# endif
# if defined(LIBXS_WIN32_THREADS)
#   define LIBXS_TLS_TYPE DWORD
#   define LIBXS_TLS_CREATE(KEYPTR) *(KEYPTR) = TlsAlloc()
#   define LIBXS_TLS_DESTROY(KEY) TlsFree(KEY)
#   define LIBXS_TLS_SETVALUE(KEY, PTR) TlsSetValue(KEY, PTR)
#   define LIBXS_TLS_GETVALUE(KEY) TlsGetValue(KEY)
#   define LIBXS_LOCK_SPINLOCK spin
#   if ((LIBXS_WIN32_THREADS) & 0x0600)
#     define LIBXS_LOCK_MUTEX rwlock
#     define LIBXS_LOCK_RWLOCK rwlock
#   else /* mutex exposes high latency */
#     define LIBXS_LOCK_MUTEX mutex
#     define LIBXS_LOCK_RWLOCK mutex
#   endif
#   if defined(LIBXS_LOCK_SYSTEM_SPINLOCK)
#     define LIBXS_LOCK_ACQUIRED_spin TRUE
#     define LIBXS_LOCK_TYPE_ISPOD_spin 0
#     define LIBXS_LOCK_TYPE_spin CRITICAL_SECTION
#     define LIBXS_LOCK_INIT_spin(LOCK, ATTR) { LIBXS_UNUSED(ATTR); InitializeCriticalSection(LOCK); }
#     define LIBXS_LOCK_DESTROY_spin(LOCK) DeleteCriticalSection((LIBXS_LOCK_TYPE_spin*)(LOCK))
#     define LIBXS_LOCK_TRYLOCK_spin(LOCK) TryEnterCriticalSection(LOCK)
#     define LIBXS_LOCK_ACQUIRE_spin(LOCK) EnterCriticalSection(LOCK)
#     define LIBXS_LOCK_RELEASE_spin(LOCK) LeaveCriticalSection(LOCK)
#     define LIBXS_LOCK_TRYREAD_spin(LOCK) LIBXS_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXS_LOCK_ACQREAD_spin(LOCK) LIBXS_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXS_LOCK_RELREAD_spin(LOCK) LIBXS_LOCK_RELEASE_spin(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_spin int
#     define LIBXS_LOCK_ATTR_INIT_spin(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_ATTR_DESTROY_spin(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   if defined(LIBXS_LOCK_SYSTEM_MUTEX)
#     define LIBXS_LOCK_ACQUIRED_mutex WAIT_OBJECT_0
#     define LIBXS_LOCK_TYPE_ISPOD_mutex 0
#     define LIBXS_LOCK_TYPE_ISRW_mutex 0
#     define LIBXS_LOCK_TYPE_mutex HANDLE
#     define LIBXS_LOCK_INIT_mutex(LOCK, ATTR) (*(LOCK) = CreateMutex(*(ATTR), FALSE, NULL))
#     define LIBXS_LOCK_DESTROY_mutex(LOCK) CloseHandle(*(LOCK))
#     define LIBXS_LOCK_TRYLOCK_mutex(LOCK) WaitForSingleObject(*(LOCK), 0)
#     define LIBXS_LOCK_ACQUIRE_mutex(LOCK) WaitForSingleObject(*(LOCK), INFINITE)
#     define LIBXS_LOCK_RELEASE_mutex(LOCK) ReleaseMutex(*(LOCK))
#     define LIBXS_LOCK_TRYREAD_mutex(LOCK) LIBXS_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXS_LOCK_ACQREAD_mutex(LOCK) LIBXS_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXS_LOCK_RELREAD_mutex(LOCK) LIBXS_LOCK_RELEASE_mutex(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_mutex LPSECURITY_ATTRIBUTES
#     define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) (*(ATTR) = NULL)
#     define LIBXS_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   if defined(LIBXS_LOCK_SYSTEM_RWLOCK)
#     define LIBXS_LOCK_ACQUIRED_rwlock TRUE
#     define LIBXS_LOCK_TYPE_ISPOD_rwlock 1
#     define LIBXS_LOCK_TYPE_ISRW_rwlock 1
#     define LIBXS_LOCK_TYPE_rwlock SRWLOCK
#     define LIBXS_LOCK_INIT_rwlock(LOCK, ATTR) { LIBXS_UNUSED(ATTR); InitializeSRWLock(LOCK); }
#     define LIBXS_LOCK_DESTROY_rwlock(LOCK) LIBXS_UNUSED(LOCK)
#     define LIBXS_LOCK_TRYLOCK_rwlock(LOCK) TryAcquireSRWLockExclusive(LOCK)
#     define LIBXS_LOCK_ACQUIRE_rwlock(LOCK) AcquireSRWLockExclusive(LOCK)
#     define LIBXS_LOCK_RELEASE_rwlock(LOCK) ReleaseSRWLockExclusive(LOCK)
#     define LIBXS_LOCK_TRYREAD_rwlock(LOCK) TryAcquireSRWLockShared(LOCK)
#     define LIBXS_LOCK_ACQREAD_rwlock(LOCK) AcquireSRWLockShared(LOCK)
#     define LIBXS_LOCK_RELREAD_rwlock(LOCK) ReleaseSRWLockShared(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_rwlock int
#     define LIBXS_LOCK_ATTR_INIT_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   define LIBXS_SYNC_YIELD YieldProcessor()
# else
#   define LIBXS_TLS_TYPE pthread_key_t
#   define LIBXS_TLS_CREATE(KEYPTR) pthread_key_create(KEYPTR, NULL)
#   define LIBXS_TLS_DESTROY(KEY) pthread_key_delete(KEY)
#   define LIBXS_TLS_SETVALUE(KEY, PTR) pthread_setspecific(KEY, PTR)
#   define LIBXS_TLS_GETVALUE(KEY) pthread_getspecific(KEY)
#   if defined(__APPLE__) && defined(__MACH__)
#     define LIBXS_SYNC_YIELD pthread_yield_np()
#   elif defined(__GLIBC__) && defined(__GLIBC_MINOR__) \
      && LIBXS_VERSION2(2, 34) <= LIBXS_VERSION2(__GLIBC__, __GLIBC_MINOR__)
      LIBXS_EXTERN int sched_yield(void); /* sched.h */
#     define LIBXS_SYNC_YIELD sched_yield()
#   else
#     if defined(__USE_GNU) || !defined(__BSD_VISIBLE)
      LIBXS_EXTERN int pthread_yield(void) LIBXS_THROW;
#     else
      LIBXS_EXTERN void pthread_yield(void);
#     endif
#     define LIBXS_SYNC_YIELD pthread_yield()
#   endif
#   if defined(LIBXS_LOCK_SYSTEM_SPINLOCK) && defined(__APPLE__) && defined(__MACH__)
#     define LIBXS_LOCK_SPINLOCK mutex
#   else
#     define LIBXS_LOCK_SPINLOCK spin
#   endif
#   define LIBXS_LOCK_MUTEX mutex
#   define LIBXS_LOCK_RWLOCK rwlock
#   if defined(LIBXS_LOCK_SYSTEM_SPINLOCK)
#     define LIBXS_LOCK_ACQUIRED_spin 0
#     define LIBXS_LOCK_TYPE_ISPOD_spin 0
#     define LIBXS_LOCK_TYPE_ISRW_spin 0
#     define LIBXS_LOCK_TYPE_spin pthread_spinlock_t
#     define LIBXS_LOCK_INIT_spin(LOCK, ATTR) LIBXS_EXPECT(0 == pthread_spin_init(LOCK, *(ATTR)))
#     define LIBXS_LOCK_DESTROY_spin(LOCK) LIBXS_EXPECT(0 == pthread_spin_destroy(LOCK))
#     define LIBXS_LOCK_TRYLOCK_spin(LOCK) pthread_spin_trylock(LOCK)
#     define LIBXS_LOCK_ACQUIRE_spin(LOCK) LIBXS_EXPECT(0 == pthread_spin_lock(LOCK))
#     define LIBXS_LOCK_RELEASE_spin(LOCK) LIBXS_EXPECT(0 == pthread_spin_unlock(LOCK))
#     define LIBXS_LOCK_TRYREAD_spin(LOCK) LIBXS_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXS_LOCK_ACQREAD_spin(LOCK) LIBXS_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXS_LOCK_RELREAD_spin(LOCK) LIBXS_LOCK_RELEASE_spin(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_spin int
#     define LIBXS_LOCK_ATTR_INIT_spin(ATTR) (*(ATTR) = 0)
#     define LIBXS_LOCK_ATTR_DESTROY_spin(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   if defined(LIBXS_LOCK_SYSTEM_MUTEX)
#     define LIBXS_LOCK_ACQUIRED_mutex 0
#     define LIBXS_LOCK_TYPE_ISPOD_mutex 0
#     define LIBXS_LOCK_TYPE_ISRW_mutex 0
#     define LIBXS_LOCK_TYPE_mutex pthread_mutex_t
#     define LIBXS_LOCK_INIT_mutex(LOCK, ATTR) LIBXS_EXPECT(0 == pthread_mutex_init(LOCK, ATTR))
#     define LIBXS_LOCK_DESTROY_mutex(LOCK) LIBXS_EXPECT_DEBUG(0 == pthread_mutex_destroy(LOCK))
#     define LIBXS_LOCK_TRYLOCK_mutex(LOCK) pthread_mutex_trylock(LOCK) /*!LIBXS_EXPECT*/
#     define LIBXS_LOCK_ACQUIRE_mutex(LOCK) LIBXS_EXPECT(0 == pthread_mutex_lock(LOCK))
#     define LIBXS_LOCK_RELEASE_mutex(LOCK) LIBXS_EXPECT(0 == pthread_mutex_unlock(LOCK))
#     define LIBXS_LOCK_TRYREAD_mutex(LOCK) LIBXS_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXS_LOCK_ACQREAD_mutex(LOCK) LIBXS_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXS_LOCK_RELREAD_mutex(LOCK) LIBXS_LOCK_RELEASE_mutex(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_mutex pthread_mutexattr_t
#if !defined(__linux__) || defined(__USE_UNIX98) || defined(__USE_XOPEN2K8)
#     if defined(_DEBUG)
#       define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) (LIBXS_EXPECT(0 == pthread_mutexattr_init(ATTR)), \
                LIBXS_EXPECT(0 == pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_ERRORCHECK)))
#     else
#       define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) (pthread_mutexattr_init(ATTR), \
                pthread_mutexattr_settype(ATTR, PTHREAD_MUTEX_NORMAL))
#     endif
#else
#     define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) pthread_mutexattr_init(ATTR)
#endif
#     define LIBXS_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXS_EXPECT(0 == pthread_mutexattr_destroy(ATTR))
#   endif
#   if defined(LIBXS_LOCK_SYSTEM_RWLOCK)
#     define LIBXS_LOCK_ACQUIRED_rwlock 0
#     define LIBXS_LOCK_TYPE_ISPOD_rwlock 0
#     define LIBXS_LOCK_TYPE_ISRW_rwlock 1
#     define LIBXS_LOCK_TYPE_rwlock pthread_rwlock_t
#     define LIBXS_LOCK_INIT_rwlock(LOCK, ATTR) LIBXS_EXPECT(0 == pthread_rwlock_init(LOCK, ATTR))
#     define LIBXS_LOCK_DESTROY_rwlock(LOCK) LIBXS_EXPECT(0 == pthread_rwlock_destroy(LOCK))
#     define LIBXS_LOCK_TRYLOCK_rwlock(LOCK) pthread_rwlock_trywrlock(LOCK)
#     define LIBXS_LOCK_ACQUIRE_rwlock(LOCK) LIBXS_EXPECT(0 == pthread_rwlock_wrlock(LOCK))
#     define LIBXS_LOCK_RELEASE_rwlock(LOCK) LIBXS_EXPECT(0 == pthread_rwlock_unlock(LOCK))
#     define LIBXS_LOCK_TRYREAD_rwlock(LOCK) pthread_rwlock_tryrdlock(LOCK)
#     define LIBXS_LOCK_ACQREAD_rwlock(LOCK) LIBXS_EXPECT(0 == pthread_rwlock_rdlock(LOCK))
#     define LIBXS_LOCK_RELREAD_rwlock(LOCK) LIBXS_LOCK_RELEASE_rwlock(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_rwlock pthread_rwlockattr_t
#     define LIBXS_LOCK_ATTR_INIT_rwlock(ATTR) LIBXS_EXPECT(0 == pthread_rwlockattr_init(ATTR))
#     define LIBXS_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXS_EXPECT(0 == pthread_rwlockattr_destroy(ATTR))
#   endif
# endif
/* OpenMP based locks need to stay disabled unless both
 * libxs and libxsext are built with OpenMP support.
 */
# if defined(_OPENMP) && defined(LIBXS_SYNC_OMP)
#   if !defined(LIBXS_LOCK_SYSTEM_SPINLOCK)
#     define LIBXS_LOCK_ACQUIRED_spin 1
#     define LIBXS_LOCK_TYPE_ISPOD_spin 0
#     define LIBXS_LOCK_TYPE_ISRW_spin 0
#     define LIBXS_LOCK_TYPE_spin omp_lock_t
#     define LIBXS_LOCK_DESTROY_spin(LOCK) omp_destroy_lock(LOCK)
#     define LIBXS_LOCK_TRYLOCK_spin(LOCK) omp_test_lock(LOCK)
#     define LIBXS_LOCK_ACQUIRE_spin(LOCK) omp_set_lock(LOCK)
#     define LIBXS_LOCK_RELEASE_spin(LOCK) omp_unset_lock(LOCK)
#     define LIBXS_LOCK_TRYREAD_spin(LOCK) LIBXS_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXS_LOCK_ACQREAD_spin(LOCK) LIBXS_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXS_LOCK_RELREAD_spin(LOCK) LIBXS_LOCK_RELEASE_spin(LOCK)
#     if (201811 <= _OPENMP/*v5.0*/)
#       define LIBXS_LOCK_INIT_spin(LOCK, ATTR) omp_init_lock_with_hint(LOCK, *(ATTR))
#       define LIBXS_LOCK_ATTR_TYPE_spin omp_lock_hint_t
#       define LIBXS_LOCK_ATTR_INIT_spin(ATTR) (*(ATTR) = omp_lock_hint_none)
#     else
#       define LIBXS_LOCK_INIT_spin(LOCK, ATTR) { LIBXS_UNUSED(ATTR); omp_init_lock(LOCK); }
#       define LIBXS_LOCK_ATTR_TYPE_spin const void*
#       define LIBXS_LOCK_ATTR_INIT_spin(ATTR) LIBXS_UNUSED(ATTR)
#     endif
#     define LIBXS_LOCK_ATTR_DESTROY_spin(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   if !defined(LIBXS_LOCK_SYSTEM_MUTEX)
#     define LIBXS_LOCK_ACQUIRED_mutex 1
#     define LIBXS_LOCK_TYPE_ISPOD_mutex 0
#     define LIBXS_LOCK_TYPE_ISRW_mutex 0
#     define LIBXS_LOCK_TYPE_mutex omp_lock_t
#     define LIBXS_LOCK_DESTROY_mutex(LOCK) omp_destroy_lock(LOCK)
#     define LIBXS_LOCK_TRYLOCK_mutex(LOCK) omp_test_lock(LOCK)
#     define LIBXS_LOCK_ACQUIRE_mutex(LOCK) omp_set_lock(LOCK)
#     define LIBXS_LOCK_RELEASE_mutex(LOCK) omp_unset_lock(LOCK)
#     define LIBXS_LOCK_TRYREAD_mutex(LOCK) LIBXS_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXS_LOCK_ACQREAD_mutex(LOCK) LIBXS_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXS_LOCK_RELREAD_mutex(LOCK) LIBXS_LOCK_RELEASE_mutex(LOCK)
#     if (201811 <= _OPENMP/*v5.0*/)
#       define LIBXS_LOCK_INIT_mutex(LOCK, ATTR) omp_init_lock_with_hint(LOCK, *(ATTR))
#       define LIBXS_LOCK_ATTR_TYPE_mutex omp_lock_hint_t
#       define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) (*(ATTR) = omp_lock_hint_none)
#     else
#       define LIBXS_LOCK_INIT_mutex(LOCK, ATTR) { LIBXS_UNUSED(ATTR); omp_init_lock(LOCK); }
#       define LIBXS_LOCK_ATTR_TYPE_mutex const void*
#       define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) LIBXS_UNUSED(ATTR)
#     endif
#     define LIBXS_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   if !defined(LIBXS_LOCK_SYSTEM_RWLOCK)
#     define LIBXS_LOCK_ACQUIRED_rwlock 1
#     define LIBXS_LOCK_TYPE_ISPOD_rwlock 0
#     define LIBXS_LOCK_TYPE_ISRW_rwlock 0
#     define LIBXS_LOCK_TYPE_rwlock omp_lock_t
#     define LIBXS_LOCK_DESTROY_rwlock(LOCK) omp_destroy_lock(LOCK)
#     define LIBXS_LOCK_TRYLOCK_rwlock(LOCK) omp_test_lock(LOCK)
#     define LIBXS_LOCK_ACQUIRE_rwlock(LOCK) omp_set_lock(LOCK)
#     define LIBXS_LOCK_RELEASE_rwlock(LOCK) omp_unset_lock(LOCK)
#     define LIBXS_LOCK_TRYREAD_rwlock(LOCK) LIBXS_LOCK_TRYLOCK_rwlock(LOCK)
#     define LIBXS_LOCK_ACQREAD_rwlock(LOCK) LIBXS_LOCK_ACQUIRE_rwlock(LOCK)
#     define LIBXS_LOCK_RELREAD_rwlock(LOCK) LIBXS_LOCK_RELEASE_rwlock(LOCK)
#     if (201811 <= _OPENMP/*v5.0*/)
#       define LIBXS_LOCK_INIT_rwlock(LOCK, ATTR) omp_init_lock_with_hint(LOCK, *(ATTR))
#       define LIBXS_LOCK_ATTR_TYPE_rwlock omp_lock_hint_t
#       define LIBXS_LOCK_ATTR_INIT_rwlock(ATTR) (*(ATTR) = omp_lock_hint_none)
#     else
#       define LIBXS_LOCK_INIT_rwlock(LOCK, ATTR) { LIBXS_UNUSED(ATTR); omp_init_lock(LOCK); }
#       define LIBXS_LOCK_ATTR_TYPE_rwlock const void*
#       define LIBXS_LOCK_ATTR_INIT_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#     endif
#     define LIBXS_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#   endif
# elif !defined(LIBXS_SYNC_NONE) /* based on atomic primitives */
#   if !defined(LIBXS_LOCK_SYSTEM_SPINLOCK)
#     define LIBXS_LOCK_ACQUIRED_spin 0
#     define LIBXS_LOCK_TYPE_ISPOD_spin 1
#     define LIBXS_LOCK_TYPE_ISRW_spin 0
#     define LIBXS_LOCK_TYPE_spin volatile LIBXS_ATOMIC_LOCKTYPE
#     define LIBXS_LOCK_INIT_spin(LOCK, ATTR) { LIBXS_UNUSED(ATTR); (*(LOCK) = 0); }
#     define LIBXS_LOCK_DESTROY_spin(LOCK) LIBXS_UNUSED(LOCK)
#     define LIBXS_LOCK_TRYLOCK_spin(LOCK) (LIBXS_LOCK_ACQUIRED_spin + !LIBXS_ATOMIC_TRYLOCK(LOCK, LIBXS_ATOMIC_RELAXED))
#     define LIBXS_LOCK_ACQUIRE_spin(LOCK) LIBXS_ATOMIC_ACQUIRE(LOCK, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_RELAXED)
#     define LIBXS_LOCK_RELEASE_spin(LOCK) LIBXS_ATOMIC_RELEASE(LOCK, LIBXS_ATOMIC_RELAXED)
#     define LIBXS_LOCK_TRYREAD_spin(LOCK) LIBXS_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXS_LOCK_ACQREAD_spin(LOCK) LIBXS_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXS_LOCK_RELREAD_spin(LOCK) LIBXS_LOCK_RELEASE_spin(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_spin int
#     define LIBXS_LOCK_ATTR_INIT_spin(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_ATTR_DESTROY_spin(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   if !defined(LIBXS_LOCK_SYSTEM_MUTEX)
#     define LIBXS_LOCK_ACQUIRED_mutex 0
#     define LIBXS_LOCK_TYPE_ISPOD_mutex 1
#     define LIBXS_LOCK_TYPE_ISRW_mutex 0
#     define LIBXS_LOCK_TYPE_mutex volatile LIBXS_ATOMIC_LOCKTYPE
#     define LIBXS_LOCK_INIT_mutex(LOCK, ATTR) { LIBXS_UNUSED(ATTR); (*(LOCK) = 0); }
#     define LIBXS_LOCK_DESTROY_mutex(LOCK) LIBXS_UNUSED(LOCK)
#     define LIBXS_LOCK_TRYLOCK_mutex(LOCK) (LIBXS_LOCK_ACQUIRED_mutex + !LIBXS_ATOMIC_TRYLOCK(LOCK, LIBXS_ATOMIC_RELAXED))
#     define LIBXS_LOCK_ACQUIRE_mutex(LOCK) LIBXS_ATOMIC_ACQUIRE(LOCK, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_RELAXED)
#     define LIBXS_LOCK_RELEASE_mutex(LOCK) LIBXS_ATOMIC_RELEASE(LOCK, LIBXS_ATOMIC_RELAXED)
#     define LIBXS_LOCK_TRYREAD_mutex(LOCK) LIBXS_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXS_LOCK_ACQREAD_mutex(LOCK) LIBXS_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXS_LOCK_RELREAD_mutex(LOCK) LIBXS_LOCK_RELEASE_mutex(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_mutex int
#     define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   if !defined(LIBXS_LOCK_SYSTEM_RWLOCK)
#     define LIBXS_LOCK_ACQUIRED_rwlock 0
#     define LIBXS_LOCK_TYPE_ISPOD_rwlock 1
#     define LIBXS_LOCK_TYPE_ISRW_rwlock 0
#     define LIBXS_LOCK_TYPE_rwlock volatile LIBXS_ATOMIC_LOCKTYPE
#     define LIBXS_LOCK_INIT_rwlock(LOCK, ATTR) { LIBXS_UNUSED(ATTR); (*(LOCK) = 0); }
#     define LIBXS_LOCK_DESTROY_rwlock(LOCK) LIBXS_UNUSED(LOCK)
#     define LIBXS_LOCK_TRYLOCK_rwlock(LOCK) (LIBXS_LOCK_ACQUIRED_rwlock + !LIBXS_ATOMIC_TRYLOCK(LOCK, LIBXS_ATOMIC_RELAXED))
#     define LIBXS_LOCK_ACQUIRE_rwlock(LOCK) LIBXS_ATOMIC_ACQUIRE(LOCK, LIBXS_SYNC_NPAUSE, LIBXS_ATOMIC_RELAXED)
#     define LIBXS_LOCK_RELEASE_rwlock(LOCK) LIBXS_ATOMIC_RELEASE(LOCK, LIBXS_ATOMIC_RELAXED)
#     define LIBXS_LOCK_TRYREAD_rwlock(LOCK) LIBXS_LOCK_TRYLOCK_rwlock(LOCK)
#     define LIBXS_LOCK_ACQREAD_rwlock(LOCK) LIBXS_LOCK_ACQUIRE_rwlock(LOCK)
#     define LIBXS_LOCK_RELREAD_rwlock(LOCK) LIBXS_LOCK_RELEASE_rwlock(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_rwlock int
#     define LIBXS_LOCK_ATTR_INIT_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#   endif
# else /* experimental */
#   if !defined(LIBXS_LOCK_SYSTEM_SPINLOCK)
#     define LIBXS_LOCK_ACQUIRED_spin 0
#     define LIBXS_LOCK_TYPE_ISPOD_spin 0
#     define LIBXS_LOCK_TYPE_ISRW_spin 0
#     define LIBXS_LOCK_TYPE_spin libxs_spinlock*
#     define LIBXS_LOCK_INIT_spin(LOCK, ATTR) { LIBXS_UNUSED(ATTR); (*(LOCK) = libxs_spinlock_create()); }
#     define LIBXS_LOCK_DESTROY_spin(LOCK) libxs_spinlock_destroy(*(LOCK))
#     define LIBXS_LOCK_TRYLOCK_spin(LOCK) libxs_spinlock_trylock(*(LOCK))
#     define LIBXS_LOCK_ACQUIRE_spin(LOCK) libxs_spinlock_acquire(*(LOCK))
#     define LIBXS_LOCK_RELEASE_spin(LOCK) libxs_spinlock_release(*(LOCK))
#     define LIBXS_LOCK_TRYREAD_spin(LOCK) LIBXS_LOCK_TRYLOCK_spin(LOCK)
#     define LIBXS_LOCK_ACQREAD_spin(LOCK) LIBXS_LOCK_ACQUIRE_spin(LOCK)
#     define LIBXS_LOCK_RELREAD_spin(LOCK) LIBXS_LOCK_RELEASE_spin(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_spin int
#     define LIBXS_LOCK_ATTR_INIT_spin(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_ATTR_DESTROY_spin(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   if !defined(LIBXS_LOCK_SYSTEM_MUTEX)
#     define LIBXS_LOCK_ACQUIRED_mutex 0
#     define LIBXS_LOCK_TYPE_ISPOD_mutex 0
#     define LIBXS_LOCK_TYPE_ISRW_mutex 0
#     define LIBXS_LOCK_TYPE_mutex libxs_mutex*
#     define LIBXS_LOCK_INIT_mutex(LOCK, ATTR) { LIBXS_UNUSED(ATTR); (*(LOCK) = libxs_mutex_create()); }
#     define LIBXS_LOCK_DESTROY_mutex(LOCK) libxs_mutex_destroy(*(LOCK))
#     define LIBXS_LOCK_TRYLOCK_mutex(LOCK) libxs_mutex_trylock(*(LOCK))
#     define LIBXS_LOCK_ACQUIRE_mutex(LOCK) libxs_mutex_acquire(*(LOCK))
#     define LIBXS_LOCK_RELEASE_mutex(LOCK) libxs_mutex_release(*(LOCK))
#     define LIBXS_LOCK_TRYREAD_mutex(LOCK) LIBXS_LOCK_TRYLOCK_mutex(LOCK)
#     define LIBXS_LOCK_ACQREAD_mutex(LOCK) LIBXS_LOCK_ACQUIRE_mutex(LOCK)
#     define LIBXS_LOCK_RELREAD_mutex(LOCK) LIBXS_LOCK_RELEASE_mutex(LOCK)
#     define LIBXS_LOCK_ATTR_TYPE_mutex int
#     define LIBXS_LOCK_ATTR_INIT_mutex(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_ATTR_DESTROY_mutex(ATTR) LIBXS_UNUSED(ATTR)
#   endif
#   if !defined(LIBXS_LOCK_SYSTEM_RWLOCK)
#     define LIBXS_LOCK_ACQUIRED_rwlock 0
#     define LIBXS_LOCK_TYPE_ISPOD_rwlock 0
#     define LIBXS_LOCK_TYPE_ISRW_rwlock 1
#     define LIBXS_LOCK_TYPE_rwlock libxs_rwlock*
#     define LIBXS_LOCK_INIT_rwlock(LOCK, ATTR) { LIBXS_UNUSED(ATTR); (*(LOCK) = libxs_rwlock_create()); }
#     define LIBXS_LOCK_DESTROY_rwlock(LOCK) libxs_rwlock_destroy(*(LOCK))
#     define LIBXS_LOCK_TRYLOCK_rwlock(LOCK) libxs_rwlock_trylock(*(LOCK))
#     define LIBXS_LOCK_ACQUIRE_rwlock(LOCK) libxs_rwlock_acquire(*(LOCK))
#     define LIBXS_LOCK_RELEASE_rwlock(LOCK) libxs_rwlock_release(*(LOCK))
#     define LIBXS_LOCK_TRYREAD_rwlock(LOCK) libxs_rwlock_tryread(*(LOCK))
#     define LIBXS_LOCK_ACQREAD_rwlock(LOCK) libxs_rwlock_acqread(*(LOCK))
#     define LIBXS_LOCK_RELREAD_rwlock(LOCK) libxs_rwlock_relread(*(LOCK))
#     define LIBXS_LOCK_ATTR_TYPE_rwlock int
#     define LIBXS_LOCK_ATTR_INIT_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#     define LIBXS_LOCK_ATTR_DESTROY_rwlock(ATTR) LIBXS_UNUSED(ATTR)
#   endif
# endif
#else /* no synchronization */
# define LIBXS_SYNC_YIELD LIBXS_SYNC_PAUSE
# define LIBXS_LOCK_SPINLOCK spinlock_dummy
# define LIBXS_LOCK_MUTEX mutex_dummy
# define LIBXS_LOCK_RWLOCK rwlock_dummy
# define LIBXS_LOCK_ACQUIRED(KIND) 0
# define LIBXS_LOCK_TYPE_ISPOD(KIND) 1
# define LIBXS_LOCK_TYPE_ISRW(KIND) 0
# define LIBXS_LOCK_ATTR_TYPE(KIND) int
# define LIBXS_LOCK_ATTR_INIT(KIND, ATTR) LIBXS_UNUSED(ATTR)
# define LIBXS_LOCK_ATTR_DESTROY(KIND, ATTR) LIBXS_UNUSED(ATTR)
# define LIBXS_LOCK_TYPE(KIND) int
# define LIBXS_LOCK_INIT(KIND, LOCK, ATTR) { LIBXS_UNUSED(LOCK); LIBXS_UNUSED(ATTR); }
# define LIBXS_LOCK_DESTROY(KIND, LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_TRYLOCK(KIND, LOCK) LIBXS_LOCK_ACQUIRED(KIND)
# define LIBXS_LOCK_ACQUIRE(KIND, LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_RELEASE(KIND, LOCK) LIBXS_UNUSED(LOCK)
# define LIBXS_LOCK_TRYREAD(KIND, LOCK) LIBXS_LOCK_TRYLOCK(KIND, LOCK)
# define LIBXS_LOCK_ACQREAD(KIND, LOCK) LIBXS_LOCK_ACQUIRE(KIND, LOCK)
# define LIBXS_LOCK_RELREAD(KIND, LOCK) LIBXS_LOCK_RELEASE(KIND, LOCK)
#endif

#if (0 == LIBXS_SYNC)
# define LIBXS_FLOCK(FILE)
# define LIBXS_FUNLOCK(FILE)
#elif defined(_WIN32)
# define LIBXS_FLOCK(FILE) _lock_file(FILE)
# define LIBXS_FUNLOCK(FILE) _unlock_file(FILE)
#else
# if !defined(__CYGWIN__)
#   define LIBXS_FLOCK(FILE) flockfile(FILE)
#   define LIBXS_FUNLOCK(FILE) funlockfile(FILE)
    LIBXS_EXTERN void flockfile(FILE*) LIBXS_THROW;
    LIBXS_EXTERN void funlockfile(FILE*) LIBXS_THROW;
# else /* Only available with __CYGWIN__ *and* C++0x. */
#   define LIBXS_FLOCK(FILE)
#   define LIBXS_FUNLOCK(FILE)
# endif
#endif

/** Synchronize console output */
#define LIBXS_STDIO_ACQUIRE() LIBXS_FLOCK(stdout); LIBXS_FLOCK(stderr)
#define LIBXS_STDIO_RELEASE() LIBXS_FUNLOCK(stderr); LIBXS_FUNLOCK(stdout)


/** Opaque type which represents a barrier. */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_barrier libxs_barrier;

/** Create barrier from one of the threads. */
LIBXS_API libxs_barrier* libxs_barrier_create(int ncores, int nthreads_per_core);
/** Initialize the barrier from each thread of the team. */
LIBXS_API void libxs_barrier_init(libxs_barrier* barrier, int tid);
/** Wait for the entire team to arrive. */
LIBXS_API void libxs_barrier_wait(libxs_barrier* barrier, int tid);
/** Destroy the resources associated with this barrier. */
LIBXS_API void libxs_barrier_destroy(const libxs_barrier* barrier);
/** DEPRECATED: use libxs_barrier_destroy instead. */
#define libxs_barrier_release libxs_barrier_destroy

/** Spin-lock, which eventually differs from LIBXS_LOCK_TYPE(LIBXS_LOCK_SPINLOCK). */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_spinlock libxs_spinlock;
LIBXS_API libxs_spinlock* libxs_spinlock_create(void);
LIBXS_API void libxs_spinlock_destroy(const libxs_spinlock* spinlock);
LIBXS_API int libxs_spinlock_trylock(libxs_spinlock* spinlock);
LIBXS_API void libxs_spinlock_acquire(libxs_spinlock* spinlock);
LIBXS_API void libxs_spinlock_release(libxs_spinlock* spinlock);

/** Mutual-exclusive lock (Mutex), which eventually differs from LIBXS_LOCK_TYPE(LIBXS_LOCK_MUTEX). */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_mutex libxs_mutex;
LIBXS_API libxs_mutex* libxs_mutex_create(void);
LIBXS_API void libxs_mutex_destroy(const libxs_mutex* mutex);
LIBXS_API int libxs_mutex_trylock(libxs_mutex* mutex);
LIBXS_API void libxs_mutex_acquire(libxs_mutex* mutex);
LIBXS_API void libxs_mutex_release(libxs_mutex* mutex);

/** Reader-Writer lock (RW-lock), which eventually differs from LIBXS_LOCK_TYPE(LIBXS_LOCK_RWLOCK). */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_rwlock libxs_rwlock;
LIBXS_API libxs_rwlock* libxs_rwlock_create(void);
LIBXS_API void libxs_rwlock_destroy(const libxs_rwlock* rwlock);
LIBXS_API int libxs_rwlock_trylock(libxs_rwlock* rwlock);
LIBXS_API void libxs_rwlock_acquire(libxs_rwlock* rwlock);
LIBXS_API void libxs_rwlock_release(libxs_rwlock* rwlock);
LIBXS_API int libxs_rwlock_tryread(libxs_rwlock* rwlock);
LIBXS_API void libxs_rwlock_acqread(libxs_rwlock* rwlock);
LIBXS_API void libxs_rwlock_relread(libxs_rwlock* rwlock);

/** Utility function to receive the process ID of the calling process. */
LIBXS_API unsigned int libxs_get_pid(void);
/**
 * Utility function to receive a Thread-ID (TID) for the calling thread.
 * The TID is not related to a specific threading runtime. TID=0 may not
 * represent the main thread. TIDs are zero-based and consecutive numbers.
 */
LIBXS_API unsigned int libxs_get_tid(void);

#endif /*LIBXS_SYNC_H*/
