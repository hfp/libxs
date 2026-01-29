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
#include <ctype.h>
#if !defined(_WIN32)
# if !defined(__linux__)
#   include <sys/sysctl.h>
#   include <sys/types.h>
# endif
# include <sys/mman.h>
#endif

#define LIBXS_CPUID_CHECK(VALUE, CHECK) ((CHECK) == ((CHECK) & (VALUE)))

#if !defined(LIBXS_CPUID_SYSCTL_BYNAME) && 1
# define LIBXS_CPUID_SYSCTL_BYNAME "machdep.cpu.brand_string"
#endif
#if !defined(LIBXS_CPUID_PROC_CPUINFO) && 1
# define LIBXS_CPUID_PROC_CPUINFO "model name"
#endif

#if defined(LIBXS_PLATFORM_X86)
/* XGETBV: receive results (EAX, EDX) for eXtended Control Register (XCR). */
/* CPUID, receive results (EAX, EBX, ECX, EDX) for requested FUNCTION/SUBFN. */
#if defined(_MSC_VER) /*defined(_WIN32) && !defined(__GNUC__)*/
#   define LIBXS_XGETBV(XCR, EAX, EDX) do { \
      unsigned long long libxs_xgetbv_ = _xgetbv(XCR); \
      (EAX) = (int)libxs_xgetbv_; \
      (EDX) = (int)(libxs_xgetbv_ >> 32); \
    } while(0)
#   define LIBXS_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) do { \
      int libxs_cpuid_x86_[/*4*/] = { 0, 0, 0, 0 }; \
      __cpuidex(libxs_cpuid_x86_, FUNCTION, SUBFN); \
      (EAX) = (unsigned int)libxs_cpuid_x86_[0]; \
      (EBX) = (unsigned int)libxs_cpuid_x86_[1]; \
      (ECX) = (unsigned int)libxs_cpuid_x86_[2]; \
      (EDX) = (unsigned int)libxs_cpuid_x86_[3]; \
    } while(0)
# elif defined(__GNUC__) || !defined(_CRAYC)
#   if (64 > (LIBXS_BITS))
      LIBXS_EXTERN int __get_cpuid( /* prototype */
        unsigned int, unsigned int*, unsigned int*, unsigned int*, unsigned int*);
#     define LIBXS_XGETBV(XCR, EAX, EDX) (EAX) = (EDX) = 0xFFFFFFFF
#     define LIBXS_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) \
        (EAX) = (EBX) = (EDX) = 0; ECX = (SUBFN); \
        __get_cpuid(FUNCTION, &(EAX), &(EBX), &(ECX), &(EDX))
#   else /* 64-bit */
#     define LIBXS_XGETBV(XCR, EAX, EDX) __asm__ __volatile__( \
        ".byte 0x0f, 0x01, 0xd0" /*xgetbv*/ : "=a"(EAX), "=d"(EDX) : "c"(XCR) \
      )
#     define LIBXS_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) \
        (ECX) = (EDX) = 0; \
        __asm__ __volatile__ (".byte 0x0f, 0xa2" /*cpuid*/ \
        : "=a"(EAX), "=b"(EBX), "=c"(ECX), "=d"(EDX) \
        : "a"(FUNCTION), "b"(0), "c"(SUBFN), "d"(0) \
      ); if (0 == (EDX)) LIBXS_UNUSED(EDX)
#   endif
# else /* legacy Cray Compiler */
#   define LIBXS_XGETBV(XCR, EAX, EDX) (EAX) = (EDX) = 0
#   define LIBXS_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) (EAX) = (EBX) = (ECX) = (EDX) = 0
# endif
#endif


LIBXS_API_INTERN void libxs_cpuid_model(char model[], size_t* model_size)
{
  if (NULL != model_size && 0 != *model_size) {
    if (NULL != model) {
      size_t size = *model_size;
      *model_size = 0;
      *model = '\0';
      { /* OS specific discovery */
#if defined(_WIN32) /* TODO */
        LIBXS_UNUSED(size);
#else
        FILE *const cpuinfo = fopen("/proc/cpuinfo", "r");
        if (NULL != cpuinfo) {
          while (NULL != fgets(model, (int)size, cpuinfo)) {
            if (0 != strncmp(LIBXS_CPUID_PROC_CPUINFO, model, sizeof(LIBXS_CPUID_PROC_CPUINFO) - 1)) *model = '\0';
            else {
              char* s = strchr(model, ':');
              if (NULL != s) {
                ++s; /* skip separator */
                while (isspace(*s)) ++s;
                *model_size = strlen(s);
                memmove(model, s, *model_size);
                s = strchr(model, '\n');
                if (NULL != s) *s = '\0';
                break;
              }
            }
          }
          fclose(cpuinfo);
        }
# if !defined(__linux__)
        if (0 == *model_size && 0 == sysctlbyname(LIBXS_CPUID_SYSCTL_BYNAME, model, &size, NULL, 0)
          && 0 != size)
        {
          *model_size = size;
        }
# endif
#endif
      }
    }
    else *model_size = 0; /* error */
  }
}


LIBXS_API int libxs_cpuid_x86(libxs_cpuid_info* info)
{
  static int result = LIBXS_TARGET_ARCH_UNKNOWN;
#if defined(LIBXS_PLATFORM_X86)
  unsigned int eax, ebx, ecx, edx;
  LIBXS_CPUID_X86(0, 0/*ecx*/, eax, ebx, ecx, edx);
  if (1 <= eax) { /* CPUID max. leaf */
    /* avoid re-detecting features but re-detect on request (info given) */
    if (LIBXS_TARGET_ARCH_UNKNOWN == result || NULL != info) {
      int feature_cpu = LIBXS_X86_GENERIC, feature_os = LIBXS_X86_GENERIC, has_context = 0;
      unsigned int maxleaf = eax;
# if defined(__linux__)
      if (0 == libxs_se && LIBXS_TARGET_ARCH_UNKNOWN == result) {
        FILE *const selinux = fopen("/sys/fs/selinux/enforce", "rb");
        if (NULL != selinux) {
          if (1 == fread(&libxs_se, 1/*sizeof(char)*/, 1/*count*/, selinux)) {
            libxs_se = ('0' != libxs_se ? 1 : 0);
          }
          else { /* conservative assumption in case of read-error */
            libxs_se = 1;
          }
          fclose(selinux);
        }
      }
# elif defined(MAP_JIT)
      libxs_se = 1;
# endif
      LIBXS_CPUID_X86(1, 0/*ecx*/, eax, ebx, ecx, edx);
      if (LIBXS_CPUID_CHECK(ecx, 0x00000001)) { /* SSE3(0x00000001) */
        if (LIBXS_CPUID_CHECK(ecx, 0x00100000)) { /* SSE42(0x00100000) */
          if (LIBXS_CPUID_CHECK(ecx, 0x10000000)) { /* AVX(0x10000000) */
            if (LIBXS_CPUID_CHECK(ecx, 0x00001000)) { /* FMA(0x00001000) */
              unsigned int ecx2;
              LIBXS_CPUID_X86(7, 0/*ecx*/, eax, ebx, ecx2, edx);
              if ( /* AVX512F(0x00010000), AVX512CD(0x10000000) */
#if 0           /* AVX512DQ(0x00020000), AVX512BW(0x40000000), AVX512VL(0x80000000) */
                LIBXS_CPUID_CHECK(ebx, 0xC0020000) &&
#endif
                LIBXS_CPUID_CHECK(ebx, 0x10010000)) /* Common */
              {
                feature_cpu = LIBXS_X86_AVX512; /* AVX512-Core/SKX/baseline */
              }
              else feature_cpu = LIBXS_X86_AVX2;
            }
            else feature_cpu = LIBXS_X86_AVX;
          }
          else feature_cpu = LIBXS_X86_SSE42;
        }
        else feature_cpu = LIBXS_X86_SSE3;
      }
# if !defined(LIBXS_INTRINSICS_DEBUG)
      LIBXS_ASSERT_MSG(LIBXS_STATIC_TARGET_ARCH <= LIBXS_MAX(LIBXS_X86_GENERIC, feature_cpu), "missed detecting ISA extensions");
      /* coverity[dead_error_line] */
      if (LIBXS_STATIC_TARGET_ARCH > feature_cpu) feature_cpu = LIBXS_STATIC_TARGET_ARCH;
# endif
      /* XSAVE/XGETBV(0x04000000), OSXSAVE(0x08000000) */
      if (LIBXS_CPUID_CHECK(ecx, 0x0C000000)) { /* OS SSE support */
        feature_os = LIBXS_MIN(LIBXS_X86_SSE42, feature_cpu);
        if (LIBXS_X86_AVX <= feature_cpu) {
          LIBXS_XGETBV(0, eax, edx);
          if (LIBXS_CPUID_CHECK(eax, 0x00000006)) { /* OS XSAVE 256-bit */
            feature_os = LIBXS_MIN(LIBXS_X86_AVX2, feature_cpu);
            if (LIBXS_CPUID_CHECK(eax, 0x000000E0)) { /* OS XSAVE 512-bit */
              feature_os = LIBXS_MIN(LIBXS_X86_AVX512, feature_cpu);
            }
          }
        }
      }
      else if (LIBXS_X86_GENERIC <= feature_cpu) {
        /* assume FXSAVE-enabled/manual state-saving OS,
           as it was introduced 1999 even for 32bit */
        feature_os = LIBXS_X86_SSE42;
      }
      else feature_os = LIBXS_TARGET_ARCH_GENERIC;
      has_context = (LIBXS_STATIC_TARGET_ARCH >= feature_cpu || feature_os >= feature_cpu) ? 1 : 0;
      if (LIBXS_TARGET_ARCH_UNKNOWN == result && 0 != libxs_verbosity) { /* library code is expected to be mute */
# if !defined(LIBXS_TARGET_ARCH)
        const int target_vlen32 = libxs_cpuid_vlen32(feature_cpu);
        const char *const compiler_support = (libxs_cpuid_vlen32(LIBXS_MAX_STATIC_TARGET_ARCH) < target_vlen32
          ? "" : (((2 <= libxs_verbosity || 0 > libxs_verbosity) && LIBXS_MAX_STATIC_TARGET_ARCH < feature_cpu)
            ? "highly " : NULL));
        if (NULL != compiler_support) {
          const char *const name = libxs_cpuid_name(LIBXS_MAX_STATIC_TARGET_ARCH);
          fprintf(stderr, "LIBXS WARNING: %soptimized non-JIT code paths are limited to \"%s\"!\n", compiler_support, name);
        }
# endif
# if defined(__OPTIMIZE__) && !defined(NDEBUG)
#   if defined(_DEBUG)
        fprintf(stderr, "LIBXS WARNING: library is optimized without -DNDEBUG and contains extra debug code!\n");
#   elif !defined(LIBXS_BUILD) /* warning limited to header-only */
        fprintf(stderr, "LIBXS WARNING: library is optimized without -DNDEBUG and contains debug code!\n");
#   endif
# endif
# if !defined(__APPLE__) || !defined(__MACH__) /* permitted features */
        if (0 == has_context) {
          fprintf(stderr, "LIBXS WARNING: detected CPU features are not permitted by the OS!\n");
          if (0 == libxs_se) {
            fprintf(stderr, "LIBXS WARNING: downgraded code generation to supported features!\n");
          }
        }
# endif
      }
      /* macOS is faulting AVX-512 (on-demand larger state) */
      result = feature_cpu;
# if !defined(__APPLE__) || !defined(__MACH__)
#   if 0 /* opportunistic */
      if (0 == libxs_se)
#   endif
      { /* only permitted features */
        result = LIBXS_MIN(feature_cpu, feature_os);
      }
# endif
      if (NULL != info) {
        size_t model_size = sizeof(info->model);
        libxs_cpuid_model(info->model, &model_size);
        LIBXS_ASSERT(0 != model_size || '\0' == *info->model);
        LIBXS_CPUID_X86(0x80000007, 0/*ecx*/, eax, ebx, ecx, edx);
        info->constant_tsc = LIBXS_CPUID_CHECK(edx, 0x00000100);
        info->has_context = has_context;
      }
    }
  }
  else {
    if (NULL != info) memset(info, 0, sizeof(*info));
    result = LIBXS_X86_GENERIC;
  }
#else
# if !defined(NDEBUG)
  static int error_once = 0;
  if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: libxs_cpuid_x86 called on non-x86 platform!\n");
  }
# endif
  if (NULL != info) memset(info, 0, sizeof(*info));
#endif
  return result;
}


LIBXS_API int libxs_cpuid(libxs_cpuid_info* info)
{
#if defined(LIBXS_PLATFORM_X86)
  return libxs_cpuid_x86(info);
#elif defined(LIBXS_PLATFORM_AARCH64)
  return libxs_cpuid_arm(info);
#elif defined(LIBXS_PLATFORM_RV64)
  return libxs_cpuid_rv64(info);
#else
  memset(info, 0, sizeof(info));
  return LIBXS_TARGET_ARCH_UNKNOWN;
#endif
}


/**
 * This implementation also accounts for non-x86 platforms,
 * which not only allows to resolve any given ID but to
 * fallback gracefully ("unknown").
 */
LIBXS_API const char* libxs_cpuid_name(int id)
{
  const char* target_arch = NULL;
  switch (id) {
    case LIBXS_X86_AVX512: {
      target_arch = "avx512";
    } break;
    case LIBXS_X86_AVX2: {
      target_arch = "hsw";
    } break;
    case LIBXS_X86_AVX: {
      target_arch = "snb";
    } break;
    case LIBXS_X86_SSE42: {
      target_arch = "wsm";
    } break;
    case LIBXS_X86_SSE3: {
      target_arch = "sse3";
    } break;
    case LIBXS_AARCH64_V81:
    case LIBXS_AARCH64_V82: {
      target_arch = "aarch64";
    } break;
    case LIBXS_AARCH64_APPL_M1: {
      target_arch = "appl_m1";
    } break;
    case LIBXS_AARCH64_APPL_M4: {
      target_arch = "appl_m4";
    } break;
    case LIBXS_AARCH64_SVE128: {
      target_arch = "sve128";
    } break;
    case LIBXS_AARCH64_SVE256: {
      target_arch = "sve256";
    } break;
    case LIBXS_AARCH64_SVE512: {
      target_arch = "sve512";
    } break;
    case LIBXS_RV64_MVL128: {
      target_arch = "rv64_mvl128";
    } break;
    case LIBXS_RV64_MVL256: {
      target_arch = "rv64_mvl256";
    } break;
    case LIBXS_RV64_MVL256_LMUL: {
      target_arch = "rv64_mvl256_lmul";
    } break;
    case LIBXS_RV64_MVL128_LMUL: {
      target_arch = "rv64_mvl128_lmul";
    } break;
    case LIBXS_TARGET_ARCH_GENERIC: {
      target_arch = "generic";
    } break;
    default: if (LIBXS_X86_GENERIC <= id
              && LIBXS_X86_ALLFEAT >= id)
    {
      target_arch = "x86_64";
    }
    else {
      target_arch = "unknown";
    }
  }
  LIBXS_ASSERT(NULL != target_arch);
  return target_arch;
}


LIBXS_API int libxs_cpuid_id(const char* arch)
{
  int target_archid = LIBXS_TARGET_ARCH_UNKNOWN;

  if (strcmp(arch, "skx") == 0 || strcmp(arch, "skl") == 0
    || strcmp(arch, "avx3") == 0 || strcmp(arch, "avx512") == 0)
  {
    target_archid = LIBXS_X86_AVX512;
  }
  else if (strcmp(arch, "hsw") == 0 || strcmp(arch, "avx2") == 0) {
    target_archid = LIBXS_X86_AVX2;
  }
  else if (strcmp(arch, "snb") == 0 || strcmp(arch, "avx") == 0) {
    target_archid = LIBXS_X86_AVX;
  }
  else if (strcmp(arch, "wsm") == 0 || strcmp(arch, "nhm") == 0
       || strcmp(arch, "sse4_2") == 0 || strcmp(arch, "sse4.2") == 0
       || strcmp(arch, "sse42") == 0  || strcmp(arch, "sse4") == 0)
  {
    target_archid = LIBXS_X86_SSE42;
  }
  else if (strcmp(arch, "sse3") == 0) {
    target_archid = LIBXS_X86_SSE3;
  }
  else if (strcmp(arch, "x86") == 0|| strcmp(arch, "x86_64") == 0
          || strcmp(arch, "x64") == 0 || strcmp(arch, "sse2") == 0
          || strcmp(arch, "sse") == 0)
  {
    target_archid = LIBXS_X86_GENERIC;
  }
  else if  (strcmp(arch, "arm") == 0 || strcmp(arch, "arm64") == 0
        || strcmp(arch, "arm_v81") == 0
        || strcmp(arch, "aarch64") == 0)
  {
    target_archid = LIBXS_AARCH64_V81;
  }
  else if (strcmp(arch, "arm_v82") == 0) {
    target_archid = LIBXS_AARCH64_V82;
  }
  else if (strcmp(arch, "appl_m1") == 0) {
    target_archid = LIBXS_AARCH64_APPL_M1;
  }
  else if (strcmp(arch, "sve128") == 0) {
    target_archid = LIBXS_AARCH64_SVE128;
  }
  else if (strcmp(arch, "sve256") == 0) {
    target_archid = LIBXS_AARCH64_SVE256;
  }
  else if (strcmp(arch, "sve512") == 0) {
    target_archid = LIBXS_AARCH64_SVE512;
  }
  else if (strcmp(arch, "rv64_mvl128") == 0) {
    target_archid = LIBXS_RV64_MVL128;
  }
  else if (strcmp(arch, "rv64_mvl256") == 0) {
    target_archid = LIBXS_RV64_MVL256;
  }
  else if (strcmp(arch, "rv64_mvl256_lmul") == 0) {
    target_archid = LIBXS_RV64_MVL256_LMUL;
  }
  else if (strcmp(arch, "rv64_mvl128_lmul") == 0) {
    target_archid = LIBXS_RV64_MVL128_LMUL;
  } else {
    target_archid = LIBXS_TARGET_ARCH_UNKNOWN;
  }

  return target_archid;
}


/**
 * This implementation also accounts for non-x86 platforms,
 * which not only allows to resolve any given ID but to
 * fallback gracefully (scalar).
 */
LIBXS_API int libxs_cpuid_vlen32(int id)
{
  int result;
  if (LIBXS_RV64_MVL128 == id)
  {
    result = 4;
  }
  else if (LIBXS_RV64_MVL256 == id)
  {
    result = 8;
  }
  else if (LIBXS_AARCH64_V81 == id
        || LIBXS_AARCH64_V82 == id
        || LIBXS_AARCH64_APPL_M1 == id
        || LIBXS_AARCH64_SVE128  == id)
  {
    result = 4;
  }
  else if (LIBXS_AARCH64_SVE256 == id) {
    result = 8;
  }
  else if (LIBXS_AARCH64_SVE512 == id
        || LIBXS_AARCH64_APPL_M4 == id)
  {
    result = 16;
  }
  else if (LIBXS_X86_AVX512 <= id) {
    result = 16;
  }
  else if (LIBXS_X86_AVX <= id) {
    result = 8;
  }
  else if (LIBXS_X86_GENERIC <= id) {
    result = 4;
  }
  else { /* scalar */
    result = 1;
  }
  return result;
}
