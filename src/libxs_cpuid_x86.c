/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_generator.h>
#include <libxs_mem.h>
#include <libxs_sync.h>
#if !defined(_WIN32)
# include <sys/mman.h>
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
      LIBXS_EXTERN LIBXS_RETARGETABLE int __get_cpuid( /* prototype */
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
      ); LIBXS_UNUSED(EDX)
#   endif
# else /* legacy Cray Compiler */
#   define LIBXS_XGETBV(XCR, EAX, EDX) (EAX) = (EDX) = 0
#   define LIBXS_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) (EAX) = (EBX) = (ECX) = (EDX) = 0
# endif
#endif

#define LIBXS_CPUID_CHECK(VALUE, CHECK) ((CHECK) == ((CHECK) & (VALUE)))

LIBXS_API_INTERN int libxs_cpuid_x86_amx_enable(void);
#if defined(__linux__)
# include <sys/syscall.h>
# include <unistd.h>
# if !defined(LIBXS_BUILD) || (1 >= (LIBXS_BUILD))
LIBXS_EXTERN long syscall(long number, ...) LIBXS_THROW;
# endif
LIBXS_API_INTERN int libxs_cpuid_x86_amx_enable(void)
{
  unsigned long bitmask = 0;
  long status = syscall(SYS_arch_prctl, 0x1022, &bitmask);
  if (0 != status) return -1;
  if (bitmask & (1<<18)) return 0;

  status = syscall(SYS_arch_prctl, 0x1023, 18);
  if (0 != status) return -1; /* setup failed */
  status = syscall(SYS_arch_prctl, 0x1022, &bitmask);

  /* setup failed */
  if (0 != status || !(bitmask & (1<18))) return -1;

  /* setup successfull */
  return 0;
}
#else
LIBXS_API_INTERN int libxs_cpuid_x86_amx_enable(void)
{
  return -1;
}
#endif

LIBXS_API int libxs_cpuid_x86(libxs_cpuid_info* info)
{
  static int result = LIBXS_TARGET_ARCH_UNKNOWN;
#if defined(LIBXS_PLATFORM_X86)
  unsigned int eax, ebx, ecx, edx;
  LIBXS_CPUID_X86(0, 0/*ecx*/, eax, ebx, ecx, edx);
  if (1 <= eax) { /* CPUID max. leaf */
    /* avoid redetecting features but redetect on request (info given) */
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
              /* AVX512F(0x00010000), AVX512CD(0x10000000) */
              if (LIBXS_CPUID_CHECK(ebx, 0x10010000)) { /* Common */
                /* AVX512DQ(0x00020000), AVX512BW(0x40000000), AVX512VL(0x80000000) */
                if (LIBXS_CPUID_CHECK(ebx, 0xC0020000)) { /* AVX512-Core */
                  if (LIBXS_CPUID_CHECK(ecx2, 0x00000800)) { /* VNNI */
                    unsigned int edx2; /* we need to save edx for AMX check */
# if 0 /* no check required yet */
                    unsigned int ecx3;
                    LIBXS_CPUID_X86(7, 1/*ecx*/, eax, ebx, ecx3, edx);
# else
                    LIBXS_CPUID_X86(7, 1/*ecx*/, eax, ebx, ecx2, edx2);
# endif
                    if (LIBXS_CPUID_CHECK(eax, 0x00000020)) { /* BF16 */
                      feature_cpu = LIBXS_X86_AVX512_CPX;
                      if (LIBXS_CPUID_CHECK(edx, 0x03400000)) { /* AMX-TILE, AMX-INT8, AMX-BF16 */
                        feature_cpu = LIBXS_X86_AVX512_SPR;
                      }
                    }
                    else feature_cpu = LIBXS_X86_AVX512_CLX; /* CLX */
                  }
                  else feature_cpu = LIBXS_X86_AVX512_CORE; /* SKX */
                }
                /* AVX512PF(0x04000000), AVX512ER(0x08000000) */
                else if (LIBXS_CPUID_CHECK(ebx, 0x0C000000)) { /* AVX512-MIC */
                  if (LIBXS_CPUID_CHECK(edx, 0x0000000C)) { /* KNM */
                    feature_cpu = LIBXS_X86_AVX512_KNM;
                  }
                  else feature_cpu = LIBXS_X86_AVX512_MIC; /* KNL */
                }
                else feature_cpu = LIBXS_X86_AVX512; /* AVX512-Common */
              }
              else feature_cpu = LIBXS_X86_AVX2;
            }
            else feature_cpu = LIBXS_X86_AVX;
          }
          else feature_cpu = LIBXS_X86_SSE42;
        }
        else feature_cpu = LIBXS_X86_SSE3;
      }
      /* enable AMX state in the OS on SPR and later */
      if (feature_cpu >= LIBXS_X86_AVX512_SPR) {
        if (0 != libxs_cpuid_x86_amx_enable()) {
          static int error_once = 0;
          if (0 != libxs_verbosity /* library code is expected to be mute */
            && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXS WARNING: AMX state allocation in the OS failed!\n");
          }
          feature_cpu = LIBXS_X86_AVX512_CLX;
        }
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
              feature_os = LIBXS_MIN(LIBXS_X86_AVX512_CPX, feature_cpu);
              if (LIBXS_X86_AVX512_SPR <= feature_cpu && 7 <= maxleaf
                && LIBXS_CPUID_CHECK(eax, 0x00060000)) /* OS XSAVE 512-bit */
              {
                feature_os = feature_cpu; /* unlimited AMX */
              }
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
          const char *const name = libxs_cpuid_name( /* exclude MIC when running on Core processors */
            (((LIBXS_X86_AVX512_MIC == LIBXS_MAX_STATIC_TARGET_ARCH) ||
              (LIBXS_X86_AVX512_KNM == LIBXS_MAX_STATIC_TARGET_ARCH)) && (LIBXS_X86_AVX512_CORE <= feature_cpu))
              ? LIBXS_X86_AVX2 : LIBXS_MAX_STATIC_TARGET_ARCH);
          fprintf(stderr, "LIBXS WARNING: %soptimized non-JIT code paths are limited to \"%s\"!\n", compiler_support, name);
        }
# endif
# if !defined(NDEBUG) && defined(__OPTIMIZE__)
        fprintf(stderr, "LIBXS WARNING: library is optimized without -DNDEBUG and contains debug code!\n");
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
        LIBXS_CPUID_X86(0x80000007, 0/*ecx*/, eax, ebx, ecx, edx);
        info->constant_tsc = LIBXS_CPUID_CHECK(edx, 0x00000100);
        info->has_context = has_context;
      }
    }
  }
  else {
    if (NULL != info) LIBXS_MEMZERO127(info);
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
  if (NULL != info) LIBXS_MEMZERO127(info);
#endif
  return result;
}


LIBXS_API int libxs_cpuid(void)
{
#if defined(LIBXS_PLATFORM_X86)
  return libxs_cpuid_x86(NULL/*info*/);
#else
  return libxs_cpuid_arm(NULL/*info*/);
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
    case LIBXS_X86_AVX512_SPR: {
      target_arch = "spr";
    } break;
    case LIBXS_X86_AVX512_CPX: {
      target_arch = "cpx";
    } break;
    case LIBXS_X86_AVX512_CLX: {
      target_arch = "clx";
    } break;
    case LIBXS_X86_AVX512_CORE: {
      target_arch = "skx";
    } break;
    case LIBXS_X86_AVX512_KNM: {
      target_arch = "knm";
    } break;
    case LIBXS_X86_AVX512_MIC: {
      target_arch = "knl";
    } break;
    case LIBXS_X86_AVX512: {
      /* TODO: rework BE to use target ID instead of set of strings (target_arch = "avx3") */
      target_arch = "hsw";
    } break;
    case LIBXS_X86_AVX512_VL256: {
      target_arch = "avx512_vl256";
    } break;
    case LIBXS_X86_AVX512_VL256_CLX: {
      target_arch = "avx512_vl256_clx";
    } break;
    case LIBXS_X86_AVX512_VL256_CPX: {
      target_arch = "avx512_vl256_cpx";
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
    case LIBXS_AARCH64_A64FX: {
      target_arch = "a64fx";
    } break;
    case LIBXS_AARCH64_APPL_M1: {
      target_arch = "appl_m1";
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


/**
 * This implementation also accounts for non-x86 platforms,
 * which not only allows to resolve any given ID but to
 * fallback gracefully (scalar).
 */
LIBXS_API int libxs_cpuid_vlen32(int id)
{
  int result;
#if defined(LIBXS_PLATFORM_X86)
  if (LIBXS_X86_AVX512 <= id) {
    result = 16;
  }
  else if (LIBXS_X86_AVX <= id) {
    result = 8;
  }
  else if (LIBXS_X86_SSE42 <= id) {
    result = 4;
  }
  else
#elif defined(LIBXS_PLATFORM_AARCH64)
  if (LIBXS_AARCH64_V81 == id ||
      LIBXS_AARCH64_V82 == id ||
      LIBXS_AARCH64_APPL_M1 == id) {
    result = 4;
  }
  else if (LIBXS_AARCH64_A64FX == id) {
    result = 16;
  }
  else
#else
  LIBXS_UNUSED(id);
#endif
  { /* scalar */
    result = 1;
  }
  return result;
}
