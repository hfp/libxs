/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MAIN_H
#define LIBXS_MAIN_H

#include <libxs.h>
#include <libxs_timer.h>
#include <libxs_sync.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXS_CAPACITY_REGISTRY) /* must be POT */
# define LIBXS_CAPACITY_REGISTRY 131072
#endif
#if !defined(LIBXS_CAPACITY_CACHE) /* must be POT */
# define LIBXS_CAPACITY_CACHE 16
#endif

#if !defined(LIBXS_PAGE_MINSIZE)
# if defined(LIBXS_PLATFORM_X86)
#   define LIBXS_PAGE_MINSIZE 4096 /* 4 KB */
# elif defined(__APPLE__)
#   define LIBXS_PAGE_MINSIZE 16384 /* 16 KB */
# else
#   define LIBXS_PAGE_MINSIZE 4096 /* 4 KB */
# endif
#endif

#if !defined(LIBXS_BATCH_CHECK) && !defined(NDEBUG)
# define LIBXS_BATCH_CHECK
#endif

#if !defined(LIBXS_NTHREADS_MAX)
# if (0 != LIBXS_SYNC)
#   define LIBXS_NTHREADS_MAX 1024
# else
#   define LIBXS_NTHREADS_MAX 1
# endif
#endif
/* relies on LIBXS_NTHREADS_MAX */
#if !defined(LIBXS_NTHREADS_USE) && 0
# define LIBXS_NTHREADS_USE
#endif

#if !defined(LIBXS_INTERCEPT_DYNAMIC) && defined(LIBXS_BUILD) && \
    (defined(__GNUC__) || defined(_CRAYC)) && !defined(_WIN32) && !defined(__CYGWIN__) && \
   !(defined(__APPLE__) && defined(__MACH__) && LIBXS_VERSION2(6, 1) >= \
      LIBXS_VERSION2(__clang_major__, __clang_minor__))
# define LIBXS_INTERCEPT_DYNAMIC
#endif

#if defined(LIBXS_INTERCEPT_DYNAMIC)
# include <dlfcn.h>
# if !defined(RTLD_NEXT)
#   define LIBXS_RTLD_NEXT ((void*)-1l)
# else
#   define LIBXS_RTLD_NEXT RTLD_NEXT
# endif
#endif

#if defined(LIBXS_PLATFORM_AARCH64)
# if defined(_MSC_VER)
#   define LIBXS_ARM_ENC16(OP0, OP1, CRN, CRM, OP2) ( \
      (((OP0) & 1) << 14) | \
      (((OP1) & 7) << 11) | \
      (((CRN) & 15) << 7) | \
      (((CRM) & 15) << 3) | \
      (((OP2) & 7) << 0))
#   define ID_AA64ISAR1_EL1 LIBXS_ARM_ENC16(0b11, 0b000, 0b0000, 0b0110, 0b001)
#   define ID_AA64PFR0_EL1  LIBXS_ARM_ENC16(0b11, 0b000, 0b0000, 0b0100, 0b000)
#   define MIDR_EL1         LIBXS_ARM_ENC16(0b11, 0b000, 0b0000, 0b0000, 0b000)
#   define LIBXS_ARM_MRS(RESULT, ID) RESULT = _ReadStatusReg(ID)
# else
#   define LIBXS_ARM_MRS(RESULT, ID) __asm__ __volatile__( \
      "mrs %0," LIBXS_STRINGIFY(ID) : "=r"(RESULT))
# endif
#endif

#if defined(__powerpc64__)
# define LIBXS_TIMER_RDTSC(CYCLE) do { \
    CYCLE = __ppc_get_timebase(); \
  } while(0)
#elif ((defined(LIBXS_PLATFORM_X86) && (64 <= (LIBXS_BITS))) && \
      (defined(__GNUC__) || defined(LIBXS_INTEL_COMPILER) || defined(__PGI)))
# define LIBXS_TIMER_RDTSC(CYCLE) do { \
    libxs_timer_tick_t libxs_timer_rdtsc_hi_; \
    __asm__ __volatile__ ("rdtsc" : "=a"(CYCLE), "=d"(libxs_timer_rdtsc_hi_)); \
    CYCLE |= libxs_timer_rdtsc_hi_ << 32; \
  } while(0)
#elif (defined(_rdtsc) || defined(_WIN32)) && defined(LIBXS_PLATFORM_X86)
# define LIBXS_TIMER_RDTSC(CYCLE) (CYCLE = __rdtsc())
#elif defined(LIBXS_PLATFORM_AARCH64) && 1
# if defined(ARM64_CNTVCT) /* Windows */
#   define LIBXS_TIMER_RDTSC(CYCLE) LIBXS_ARM_MRS(CYCLE, ARM64_CNTVCT)
# else
#   define LIBXS_TIMER_RDTSC(CYCLE) LIBXS_ARM_MRS(CYCLE, CNTVCT_EL0)
# endif
#endif

#if !defined(LIBXS_VERBOSITY_HIGH)
# define LIBXS_VERBOSITY_HIGH 3 /* secondary warning or info-verbosity */
#endif
#if !defined(LIBXS_VERBOSITY_WARN)
# define LIBXS_VERBOSITY_WARN ((LIBXS_VERBOSITY_HIGH) - LIBXS_MIN(1, LIBXS_VERBOSITY_HIGH))
#endif

#if !defined(LIBXS_LOCK)
# define LIBXS_LOCK LIBXS_LOCK_DEFAULT
#endif

/** Check if M, N, K, or LDx fits into the descriptor. */
#if (0 != LIBXS_ILP64)
# define LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K) (0xFFFFFFFF >= (M) && 0xFFFFFFFF >= (N) && 0xFFFFFFFF >= (K))
#else /* always fits */
# define LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K) 1
#endif

#if defined(LIBXS_ASSERT) /* assert available */
# define LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K) LIBXS_ASSERT(LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K))
#else
# define LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K)
#endif

#define LIBXS_DESCRIPTOR_CLEAR_AUX(DST, SIZE, FLAGS) LIBXS_MEMSET127(DST, 0, SIZE)
#define LIBXS_DESCRIPTOR_CLEAR(BLOB) \
  LIBXS_ASSERT((LIBXS_DESCRIPTOR_MAXSIZE) == sizeof(*(BLOB))); \
  LIBXS_DESCRIPTOR_CLEAR_AUX(BLOB, LIBXS_DESCRIPTOR_MAXSIZE, 0)

/** Low-level/internal GEMM descriptor initialization. */
#define LIBXS_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE0, DATA_TYPE1, DATA_TYPE2, FLAGS, M, N, K, LDA, LDB, LDC, PREFETCH) \
  LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K); LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(LDA, LDB, LDC); \
  LIBXS_DESCRIPTOR_CLEAR_AUX(&(DESCRIPTOR), sizeof(DESCRIPTOR), FLAGS); \
  (DESCRIPTOR).datatype[0] = (unsigned char)(DATA_TYPE0); (DESCRIPTOR).datatype[1] = (unsigned char)(DATA_TYPE1); \
  (DESCRIPTOR).datatype[2] = (unsigned char)(DATA_TYPE2); (DESCRIPTOR).prefetch = (unsigned char)(PREFETCH); \
  (DESCRIPTOR).flags = (unsigned int)(FLAGS); \
  (DESCRIPTOR).m   = (unsigned int)(M);   (DESCRIPTOR).n   = (unsigned int)(N);   (DESCRIPTOR).k   = (unsigned int)(K); \
  (DESCRIPTOR).lda = (unsigned int)(LDA); (DESCRIPTOR).ldb = (unsigned int)(LDB); (DESCRIPTOR).ldc = (unsigned int)(LDC)

/** Declare and construct a GEMM descriptor. */
#define LIBXS_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, DATA_TYPE0, DATA_TYPE1, DATA_TYPE2, FLAGS, M, N, K, LDA, LDB, LDC, PREFETCH) \
  libxs_gemm_descriptor DESCRIPTOR; LIBXS_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE0, DATA_TYPE1, DATA_TYPE2 \
    FLAGS, M, N, K, LDA, LDB, LDC, PREFETCH)


/** Integral type (libxs_kernel_kind, libxs_build_kind). */
#if defined(LIBXS_UNPACKED)
# define LIBXS_DESCRIPTOR_BIG(KIND) ((libxs_descriptor_kind)((KIND) | 0x8000000000000000))
# define LIBXS_DESCRIPTOR_ISBIG(KIND) ((int)(((libxs_descriptor_kind)(KIND)) >> 63))
# define LIBXS_DESCRIPTOR_KIND(KIND) ((int)(((libxs_descriptor_kind)(KIND)) & 0x7FFFFFFFFFFFFFFF))
typedef uint64_t libxs_descriptor_kind;
#else
# define LIBXS_DESCRIPTOR_BIG(KIND) ((libxs_descriptor_kind)((KIND) | 0x80))
# define LIBXS_DESCRIPTOR_ISBIG(KIND) ((unsigned char)((KIND) >> 7))
# define LIBXS_DESCRIPTOR_KIND(KIND) ((unsigned char)((KIND) & 0x7F))
typedef unsigned char libxs_descriptor_kind;
#endif

/**
 * Print the command line arguments of the current process, and get the number of written
 * characters including the prefix, the postfix, but not the terminating NULL character.
 * If zero is returned, nothing was printed (no prefix, no postfix).
 * If buffer_size is zero, buffer is assumed to be a FILE-pointer.
 */
LIBXS_API_INTERN int libxs_print_cmdline(void* buffer, size_t buffer_size, const char* prefix, const char* postfix);

/**
 * Dump data, (optionally) check attempt to dump different data into an existing file (unique),
 * or (optionally) permit overwriting an existing file.
 */
LIBXS_API_INTERN int libxs_dump(const char* title, const char* name, const void* data, size_t size, int unique, int overwrite);

/** Calculates duration in seconds from given RTC ticks. */
LIBXS_API double libxs_timer_duration_rtc(libxs_timer_tick_t tick0, libxs_timer_tick_t tick1);
/** Returns the current tick of platform-specific real-time clock. */
LIBXS_API libxs_timer_tick_t libxs_timer_tick_rtc(void);
/** Returns the current tick of a (monotonic) platform-specific counter. */
LIBXS_API libxs_timer_tick_t libxs_timer_tick_tsc(void);

/** Global lock; create an own lock for an independent domain. */
LIBXS_APIVAR_PUBLIC(LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_lock_global);
/** Initialization counter that can be used to check whether the library is initialized (!=0) or not (==0). */
LIBXS_APIVAR_PUBLIC(unsigned int libxs_ninit);
/** Used for system/user specific locking (I/O). */
LIBXS_APIVAR_PUBLIC(int libxs_stdio_handle);
/** Verbosity level (0: quiet, 1: errors, 2: warnings, 3: info, neg.: all). */
LIBXS_APIVAR_PUBLIC(int libxs_verbosity);
/** Determines whether a threaded implementation is synchronized or not. */
LIBXS_APIVAR_PUBLIC(int libxs_nosync);
/** Security-enhanced environment. */
LIBXS_APIVAR_PUBLIC(int libxs_se);

/** Number of seconds per RDTSC-cycle (zero or negative if RDTSC invalid). */
LIBXS_APIVAR_PRIVATE(double libxs_timer_scale);
/** Counts the maximum number of thread that have been active. */
LIBXS_APIVAR_PRIVATE(unsigned int libxs_thread_count);

#endif /*LIBXS_MAIN_H*/
