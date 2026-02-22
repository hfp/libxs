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

#include <libxs_timer.h>
#include <libxs_cpuid.h>
#include <libxs_sync.h>

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


LIBXS_API_INTERN void libxs_memory_init(int target_arch);
LIBXS_API_INTERN void libxs_memory_finalize(void);

/** Architecture-specific CPUID: returns ISA level and optionally fills info. */
LIBXS_API_INTERN int libxs_cpuid_x86(libxs_cpuid_info_t* info);
LIBXS_API_INTERN int libxs_cpuid_arm(libxs_cpuid_info_t* info);
LIBXS_API_INTERN int libxs_cpuid_rv64(libxs_cpuid_info_t* info);
/** Reads the CPU model name from OS-specific interfaces. */
LIBXS_API_INTERN void libxs_cpuid_model(char model[], size_t* model_size);

/** Calculates duration in seconds from given RTC ticks. */
LIBXS_API_INTERN double libxs_timer_duration_rtc(libxs_timer_tick_t tick0, libxs_timer_tick_t tick1);
/** Returns the current tick of platform-specific real-time clock. */
LIBXS_API_INTERN libxs_timer_tick_t libxs_timer_tick_rtc(void);
/** Returns the current tick of a (monotonic) platform-specific counter. */
LIBXS_API_INTERN libxs_timer_tick_t libxs_timer_tick_tsc(void);

/**
 * Dump data, (optionally) check attempt to dump different data into an existing file (unique),
 * or (optionally) permit overwriting an existing file.
 */
LIBXS_API_INTERN int libxs_dump(const char* title, const char* name, const void* data, size_t size, int unique, int overwrite);

/**
 * Print the command line arguments of the current process, and get the number of written
 * characters including the prefix, the postfix, but not the terminating NULL character.
 * If zero is returned, nothing was printed (no prefix, no postfix).
 * If buffer_size is zero, buffer is assumed to be a FILE-pointer.
 */
LIBXS_API_INTERN int libxs_print_cmdline(void* buffer, size_t buffer_size, const char* prefix, const char* postfix);

/** Global lock; create an own lock for an independent domain. */
LIBXS_APIVAR_PRIVATE(LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_lock_global);
/** Initialization counter that can be used to check whether the library is initialized (!=0) or not (==0). */
LIBXS_APIVAR_PRIVATE(unsigned int libxs_ninit);
/** Used for system/user specific locking (I/O). */
LIBXS_APIVAR_PRIVATE(int libxs_stdio_handle);
/** Verbosity level (0: quiet, 1: errors, 2: warnings, 3: info, neg.: all). */
LIBXS_APIVAR_PRIVATE(int libxs_verbosity);
/** Security-enhanced environment. */
LIBXS_APIVAR_PRIVATE(int libxs_se);

/** Number of seconds per RDTSC-cycle (zero or negative if RDTSC invalid). */
LIBXS_APIVAR_PRIVATE(double libxs_timer_scale);
/** Counts the maximum number of thread that have been active. */
LIBXS_APIVAR_PRIVATE(unsigned int libxs_thread_count);

#endif /*LIBXS_MAIN_H*/
