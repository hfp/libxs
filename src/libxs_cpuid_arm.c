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
#include <libxs_sync.h>
#include "libxs_main.h"

#include <signal.h>
#include <setjmp.h>

#if !defined(LIBXS_CPUID_ARM_BASELINE) && 0
# define LIBXS_CPUID_ARM_BASELINE LIBXS_AARCH64_NEOV1
#endif
#if !defined(LIBXS_CPUID_ARM_CNTB_FALLBACK) && 1
# define LIBXS_CPUID_ARM_CNTB_FALLBACK
#endif
#if !defined(LIBXS_CPUID_ARM_MODEL_FALLBACK)
# if 0
#   define LIBXS_CPUID_ARM_MODEL_FALLBACK
# elif defined(__APPLE__) && defined(__arm64__)
#   define LIBXS_CPUID_ARM_MODEL_FALLBACK
# endif
#endif

#if defined(LIBXS_PLATFORM_AARCH64)
# if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 4611)
# endif
LIBXS_APIVAR_DEFINE(jmp_buf internal_cpuid_arm_jmp_buf);
LIBXS_API_INTERN void internal_cpuid_arm_sigill(int /*signum*/);
LIBXS_API_INTERN void internal_cpuid_arm_sigill(int signum) {
  void (*const handler)(int) = signal(signum, internal_cpuid_arm_sigill);
  LIBXS_ASSERT(SIGILL == signum);
  if (SIG_ERR != handler) longjmp(internal_cpuid_arm_jmp_buf, 1);
}
LIBXS_API_INTERN int libxs_cpuid_arm_svcntb(void);
LIBXS_API_INTERN int libxs_cpuid_arm_svcntb(void) {
  int result = 0;
  if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
# if (defined(__has_builtin) && __has_builtin(__builtin_sve_svcntb)) && 0
    const uint64_t vlen_bytes = __builtin_sve_svcntb();
    if (0 < vlen_bytes && 256 >= vlen_bytes) result = (int)vlen_bytes;
# elif !defined(_MSC_VER) /* TODO: improve condition */
    register uint64_t vlen_bytes __asm__("x0") = 0;
    __asm__ __volatile__(".byte 0xe0, 0xe3, 0x20, 0x04" /*cntb %0*/ : "=r"(vlen_bytes));
    if (0 < vlen_bytes && 256 >= vlen_bytes) result = (int)vlen_bytes;
# endif
  }
  return result;
}
/* Call late (not upfront) since MIDR_EL1 failure cannot always be trapped. */
LIBXS_API_INTERN char libxs_cpuid_arm_vendor(void);
LIBXS_API_INTERN char libxs_cpuid_arm_vendor(void) {
  uint64_t result = 0;
  if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
    LIBXS_ARM_MRS(result, MIDR_EL1);
  }
  return (char)(0xFF & (result >> 24));
}
#endif


LIBXS_API unsigned int libxs_cpuid_arm_mmla_gemm_pack_b_to_vnnit_on_stack(void)
{
#if defined(LIBXS_PLATFORM_X86)
  return 0;
#else
  const char *const l_env_b_vnnit_in_stack = getenv("LIBXS_AARCH64_MMLA_GEMM_B_INPUT_PACKING_ON_STACK");
  unsigned int l_b_vnnit_in_stack = 0;
  if ( 0 == l_env_b_vnnit_in_stack ) {
  } else {
    l_b_vnnit_in_stack = atoi(l_env_b_vnnit_in_stack);
  }
  return l_b_vnnit_in_stack;
#endif
}


LIBXS_API int libxs_cpuid_arm_use_bfdot(void)
{
#if defined(LIBXS_PLATFORM_X86)
  return 0;
#else
  const char *const l_env_aarch64_bfdot = getenv("LIBXS_AARCH64_USE_BFDOT");
  int result = 0;
  if ( 0 == l_env_aarch64_bfdot ) {
    result = 0;
  } else {
    if ( atoi(l_env_aarch64_bfdot) != 0 ) {
      result = 1;
    }
  }
  return result;
#endif
}


LIBXS_API int libxs_cpuid_arm(libxs_cpuid_info* info)
{
  static int result = LIBXS_TARGET_ARCH_UNKNOWN;
#if defined(LIBXS_PLATFORM_AARCH64)
  libxs_cpuid_info cpuid_info;
  size_t model_size = 0;
# if !defined(LIBXS_CPUID_ARM_MODEL_FALLBACK)
  LIBXS_UNUSED(model_size);
  if (NULL != info)
# endif
  {
    size_t cpuinfo_model_size = sizeof(cpuid_info.model);
    libxs_cpuid_model(cpuid_info.model, &cpuinfo_model_size);
    LIBXS_ASSERT(0 != cpuinfo_model_size || '\0' == *cpuid_info.model);
    model_size = cpuinfo_model_size;
    cpuid_info.constant_tsc = 1;
  }
  if (LIBXS_TARGET_ARCH_UNKNOWN == result) { /* avoid re-detecting features */
# if defined(LIBXS_CPUID_ARM_BASELINE)
    result = LIBXS_CPUID_ARM_BASELINE;
# else
    void (*const handler)(int) = signal(SIGILL, internal_cpuid_arm_sigill);
#   if defined(__APPLE__) && defined(__arm64__)
    result = LIBXS_AARCH64_APPL_M1;
#   else
    result = LIBXS_AARCH64_V81;
#   endif
    if (SIG_ERR != handler) {
      uint64_t id_aa64isar1_el1 = 0;
      if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
        LIBXS_ARM_MRS(id_aa64isar1_el1, ID_AA64ISAR1_EL1);
      }
      if (LIBXS_AARCH64_V82 <= result
        || /* DPB */ 0 != (0xF & id_aa64isar1_el1))
      {
        volatile uint64_t id_aa64pfr0_el1 = 0;
        volatile int no_access = 0; /* try libxs_cpuid_arm_svcntb */
        if (LIBXS_AARCH64_V82 > result) result = LIBXS_AARCH64_V82;
        if (0 == setjmp(internal_cpuid_arm_jmp_buf)) {
          LIBXS_ARM_MRS(id_aa64pfr0_el1, ID_AA64PFR0_EL1);
        }
        else no_access = 1;
        if (0 != (0xF & (id_aa64pfr0_el1 >> 32)) || 0 != no_access) { /* SVE */
          const int vlen_bytes = libxs_cpuid_arm_svcntb();
          switch (vlen_bytes) {
            case 16: { /* SVE 128-bit */
              if (LIBXS_AARCH64_SVE128 > result) result = LIBXS_AARCH64_SVE128;
            } break;
            case 32: { /* SVE 256-bit */
              const int sve256 = (1 == (0xF & (id_aa64isar1_el1 >> 44))
                ? LIBXS_AARCH64_NEOV1 /* BF16 */
                : LIBXS_AARCH64_SVE256);
              if (sve256 > result) result = sve256;
            } break;
#   if defined(LIBXS_CPUID_ARM_CNTB_FALLBACK)
            case 0: /* fallback (hack) */
#   endif
            case 64: { /* SVE 512-bit */
              const char vendor = libxs_cpuid_arm_vendor();
              if ('F' == vendor) { /* Fujitsu */
                if (LIBXS_AARCH64_A64FX > result) {
#   if defined(LIBXS_CPUID_ARM_CNTB_FALLBACK)
                  if (0 != libxs_verbosity && 0 == vlen_bytes) { /* library code is expected to be mute */
                    fprintf(stderr, "LIBXS WARNING: assuming SVE 512-bit vector length!\n");
                  }
#   endif
                  result = LIBXS_AARCH64_A64FX;
                }
              }
              else
#   if defined(LIBXS_CPUID_ARM_CNTB_FALLBACK)
              if (64 == vlen_bytes)
#   endif
              {
                LIBXS_ASSERT(0 == no_access);
                if (LIBXS_AARCH64_SVE512 > result) result = LIBXS_AARCH64_SVE512;
              }
            } break;
            default: if (0 != libxs_verbosity) { /* library code is expected to be mute */
              if (0 != no_access || 0 == vlen_bytes) {
                fprintf(stderr, "LIBXS WARNING: cannot determine SVE vector length!\n");
              }
              else {
                fprintf(stderr, "LIBXS WARNING: unexpected SVE %i-bit vector length!\n",
                  vlen_bytes * 8);
              }
            }
          }
        }
      }
#   if defined(LIBXS_CPUID_ARM_MODEL_FALLBACK)
      else if (0 != model_size) { /* determine CPU based on vendor-string (everything else failed) */
        if (LIBXS_AARCH64_APPL_M1 > result && 0 == strncmp("Apple M1", cpuid_info.model, model_size)) {
          result = LIBXS_AARCH64_APPL_M1;
        }
      }
#   endif
      /* restore original state */
      signal(SIGILL, handler);
    }
# endif
  }
  if (NULL != info) memcpy(info, &cpuid_info, sizeof(cpuid_info));
#else
# if !defined(NDEBUG)
  static int error_once = 0;
  if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: libxs_cpuid_arm called on non-ARM platform!\n");
  }
# endif
  if (NULL != info) memset(info, 0, sizeof(*info));
#endif
  return result;
}

#if defined(LIBXS_PLATFORM_AARCH64) && defined(_MSC_VER)
# pragma warning(pop)
#endif
