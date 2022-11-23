/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_cpuid.h>
#include <libxs_generator.h>
#include <libxs_mem.h>
#include <libxs_sync.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <signal.h>
#include <setjmp.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_CPUID_ARM_BASELINE)
# if defined(__APPLE__) && defined(__arm64__) && 1
#   define LIBXS_CPUID_ARM_BASELINE LIBXS_AARCH64_APPL_M1
# elif 0
#   define LIBXS_CPUID_ARM_BASELINE LIBXS_AARCH64_V82
# endif
#endif

#if defined(LIBXS_PLATFORM_AARCH64) && !defined(LIBXS_CPUID_ARM_BASELINE)
# if defined(_MSC_VER)
#   define LIBXS_CPUID_ARM_ENC16(OP0, OP1, CRN, CRM, OP2) ( \
      (((OP0) & 1) << 14) | \
      (((OP1) & 7) << 11) | \
      (((CRN) & 15) << 7) | \
      (((CRM) & 15) << 3) | \
      (((OP2) & 7) << 0))
#   define ID_AA64ISAR1_EL1 LIBXS_CPUID_ARM_ENC16(0b11, 0b000, 0b0000, 0b0110, 0b001)
#   define ID_AA64PFR0_EL1  LIBXS_CPUID_ARM_ENC16(0b11, 0b000, 0b0000, 0b0100, 0b000)
#   define LIBXS_CPUID_ARM_MRS(RESULT, ID) RESULT = _ReadStatusReg(ID)
# else
#   define LIBXS_CPUID_ARM_MRS(RESULT, ID) __asm__ __volatile__( \
      "mrs %0," LIBXS_STRINGIFY(ID) : "=r"(RESULT))
#   define LIBXS_CPUID_ARM_CNTB(RESULT) __asm__ __volatile__( \
      "cntb %0" : "=r"(RESULT))
# endif
LIBXS_APIVAR_DEFINE(jmp_buf internal_cpuid_arm_jmp_buf);
LIBXS_API_INTERN void internal_cpuid_arm_sigill(int /*signum*/);
LIBXS_API_INTERN void internal_cpuid_arm_sigill(int signum) {
  void (* const handler)(int) = signal(signum, internal_cpuid_arm_sigill);
  LIBXS_ASSERT(SIGILL == signum);
  if (SIG_ERR != handler) longjmp(internal_cpuid_arm_jmp_buf, 1);
}
#endif


#if defined(LIBXS_PLATFORM_AARCH64)
# if defined(__has_builtin) && __has_builtin(__builtin_sve_svcntb)
#   define libxs_svcntb __builtin_sve_svcntb
# elif defined(LIBXS_CPUID_ARM_CNTB)
LIBXS_API_INTERN int libxs_svcntb(void);
LIBXS_API_INTERN LIBXS_ATTRIBUTE(target("arch=armv8-a+sve"))
uint64_t libxs_svcntb(void) {
  uint64_t result = 0;
  LIBXS_CPUID_ARM_CNTB(result);
  return result;
}
# endif
#endif


LIBXS_API int libxs_cpuid_arm(libxs_cpuid_info* info)
{
  static int result = LIBXS_TARGET_ARCH_UNKNOWN;
#if defined(LIBXS_PLATFORM_AARCH64)
  if (NULL != info) LIBXS_MEMZERO127(info);
  if (LIBXS_TARGET_ARCH_UNKNOWN == result) { /* avoid redetecting features */
# if defined(LIBXS_CPUID_ARM_BASELINE)
    result = LIBXS_CPUID_ARM_BASELINE;
# else
    void (*const handler)(int) = signal(SIGILL, internal_cpuid_arm_sigill);
    result = LIBXS_AARCH64_V81;
    if (SIG_ERR != handler) {
      uint64_t capability; /* 64-bit value */
      if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
        LIBXS_CPUID_ARM_MRS(capability, ID_AA64ISAR1_EL1);
        if (0 != (0xF & capability)) { /* DPB */
          result = LIBXS_AARCH64_V82;
#   if defined(LIBXS_CPUID_ARM_CNTB)
          if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
            LIBXS_CPUID_ARM_MRS(capability, ID_AA64PFR0_EL1);
            if (0 != (0xF & (capability >> 32))) { /* SVE */
              switch (libxs_svcntb()) {
                case 16: result = LIBXS_AARCH64_SVE128; break;
                case 32: result = LIBXS_AARCH64_SVE256; break;
                case 64: /* SVE 512-bit */
                  result = (1 == (0xF & (capability >> 16))
                    ? LIBXS_AARCH64_A64FX /* FP16 */
                    : LIBXS_AARCH64_SVE512);
                  break;
#     if defined(NDEBUG)
                default: ;
#     else
                default: if (0 != libxs_verbosity) { /* library code is expected to be mute */
                  fprintf(stderr, "LIBXS WARNING: libxs_cpuid_arm discovered an unexpected SVE vector length!\n");
                }
#     endif
              }
            }
          }
#   endif
        }
      }
      /* restore original state */
      signal(SIGILL, handler);
    }
# endif
  }
#else
# if !defined(NDEBUG)
  static int error_once = 0;
  if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: libxs_cpuid_arm called on non-ARM platform!\n");
  }
# endif
  if (NULL != info) LIBXS_MEMZERO127(info);
#endif
  return result;
}
