/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
#include <libxs_intrinsics_x86.h>
#include <libxs_generator.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdio.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if defined(LIBXS_PLATFORM_SUPPORTED)
/* XGETBV: receive results (EAX, EDX) for eXtended Control Register (XCR). */
/* CPUID, receive results (EAX, EBX, ECX, EDX) for requested FUNCTION/SUBFN. */
  #if defined(_MSC_VER) /*defined(_WIN32) && !defined(__GNUC__)*/
#   define LIBXS_XGETBV(XCR, EAX, EDX) { \
      unsigned long long libxs_xgetbv_ = _xgetbv(XCR); \
      EAX = (int)libxs_xgetbv_; \
      EDX = (int)(libxs_xgetbv_ >> 32); \
    }
#   define LIBXS_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) { \
      int libxs_cpuid_x86_[/*4*/] = { 0, 0, 0, 0 }; \
      __cpuidex(libxs_cpuid_x86_, FUNCTION, SUBFN); \
      EAX = (unsigned int)libxs_cpuid_x86_[0]; \
      EBX = (unsigned int)libxs_cpuid_x86_[1]; \
      ECX = (unsigned int)libxs_cpuid_x86_[2]; \
      EDX = (unsigned int)libxs_cpuid_x86_[3]; \
    }
# elif defined(__GNUC__) || !defined(_CRAYC)
#   if (64 > (LIBXS_BITS))
      LIBXS_EXTERN LIBXS_RETARGETABLE int __get_cpuid(unsigned int, unsigned int*, unsigned int*, unsigned int*, unsigned int*);
#     define LIBXS_XGETBV(XCR, EAX, EDX) EAX = (EDX) = 0xFFFFFFFF
#     define LIBXS_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) \
        EAX = (EBX) = (EDX) = 0; ECX = (SUBFN); \
        __get_cpuid(FUNCTION, &(EAX), &(EBX), &(ECX), &(EDX))
#   else /* 64-bit */
#     define LIBXS_XGETBV(XCR, EAX, EDX) __asm__ __volatile__( \
        ".byte 0x0f, 0x01, 0xd0" /*xgetbv*/ : "=a"(EAX), "=d"(EDX) : "c"(XCR) \
      )
#     define LIBXS_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) \
        __asm__ __volatile__ (".byte 0x0f, 0xa2" /*cpuid*/ \
        : "=a"(EAX), "=b"(EBX), "=c"(ECX), "=d"(EDX) \
        : "a"(FUNCTION), "b"(0), "c"(SUBFN), "d"(0) \
      )
#   endif
# else /* legacy Cray Compiler */
#   define LIBXS_XGETBV(XCR, EAX, EDX) EAX = (EDX) = 0
#   define LIBXS_CPUID_X86(FUNCTION, SUBFN, EAX, EBX, ECX, EDX) EAX = (EBX) = (ECX) = (EDX) = 0
# endif
#endif

#define LIBXS_CPUID_CHECK(VALUE, CHECK) ((CHECK) == ((CHECK) & (VALUE)))


LIBXS_API int libxs_cpuid_x86(void)
{
#if defined(LIBXS_INTRINSICS_DEBUG)
  int target_arch = LIBXS_X86_GENERIC;
#else
  int target_arch = LIBXS_STATIC_TARGET_ARCH;
#endif
#if defined(LIBXS_PLATFORM_SUPPORTED)
  unsigned int eax, ebx, ecx, edx;
  LIBXS_CPUID_X86(0, 0/*ecx*/, eax, ebx, ecx, edx);
  if (1 <= eax) { /* CPUID max. leaf */
    const unsigned int maxleaf = eax;
    static int error_once = 0;
    LIBXS_CPUID_X86(1, 0/*ecx*/, eax, ebx, ecx, edx);
    /* Check for CRC32 (this is not a proper test for SSE 4.2 as a whole!) */
    if (LIBXS_CPUID_CHECK(ecx, 0x00100000)) {
      target_arch = LIBXS_X86_SSE4;
    }
    /* XSAVE/XGETBV(0x04000000), OSXSAVE(0x08000000) */
    if (LIBXS_CPUID_CHECK(ecx, 0x0C000000)) {
      LIBXS_XGETBV(0, eax, edx);
      if (LIBXS_CPUID_CHECK(eax, 0x00000006)) { /* OS XSAVE 256-bit */
        if (7 <= maxleaf && LIBXS_CPUID_CHECK(eax, 0x000000E0)) { /* OS XSAVE 512-bit */
          LIBXS_CPUID_X86(7, 0/*ecx*/, eax, ebx, ecx, edx);
          /* AVX512F(0x00010000), AVX512CD(0x10000000) */
          if (LIBXS_CPUID_CHECK(ebx, 0x10010000)) { /* Common */
            /* AVX512DQ(0x00020000), AVX512BW(0x40000000), AVX512VL(0x80000000) */
            if (LIBXS_CPUID_CHECK(ebx, 0xC0020000)) { /* AVX512-Core */
              if (LIBXS_CPUID_CHECK(ecx, 0x00000800)) { /* VNNI */
                LIBXS_CPUID_X86(7, 1/*ecx*/, eax, ebx, ecx, edx);
                if (LIBXS_CPUID_CHECK(eax, 0x00000020)) { /* BF16 */
                  target_arch = LIBXS_X86_AVX512_CPX;
                }
                else { /* CLX */
                  target_arch = LIBXS_X86_AVX512_CLX;
                }
              }
              else { /* SKX */
                target_arch = LIBXS_X86_AVX512_CORE;
              }
            }
            /* AVX512PF(0x04000000), AVX512ER(0x08000000) */
            else if (LIBXS_CPUID_CHECK(ebx, 0x0C000000)) { /* AVX512-MIC */
              if (LIBXS_CPUID_CHECK(edx, 0x0000000C)) { /* KNM */
                target_arch = LIBXS_X86_AVX512_KNM;
              }
              else { /* KNL */
                target_arch = LIBXS_X86_AVX512_MIC;
              }
            }
            else { /* AVX512-Common */
              target_arch = LIBXS_X86_AVX512;
            }
          }
        }
        else if (LIBXS_CPUID_CHECK(ecx, 0x10000000)) { /* AVX(0x10000000) */
          if (LIBXS_CPUID_CHECK(ecx, 0x00001000)) { /* FMA(0x00001000) */
            target_arch = LIBXS_X86_AVX2;
          }
          else {
            target_arch = LIBXS_X86_AVX;
          }
        }
      }
    }
    else if (LIBXS_STATIC_TARGET_ARCH < target_arch &&
      0 != libxs_verbosity && 1 == ++error_once) /* library code is expected to be mute */
    {
      fprintf(stderr, "LIBXS WARNING: detected CPU features are not permitted by the OS!\n");
    }
  }
#endif
#if defined(LIBXS_INTRINSICS_DEBUG)
  return target_arch;
#else /* check if procedure obviously failed to detect the highest available instruction set extension */
  LIBXS_ASSERT(LIBXS_STATIC_TARGET_ARCH <= target_arch);
  return LIBXS_MAX(target_arch, LIBXS_STATIC_TARGET_ARCH);
#endif
}


LIBXS_API int libxs_cpuid(void)
{
  return libxs_cpuid_x86();
}


LIBXS_API const char* libxs_cpuid_name(int id)
{
  const char* target_arch = NULL;
  switch (id) {
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
    case LIBXS_X86_AVX2: {
      target_arch = "hsw";
    } break;
    case LIBXS_X86_AVX: {
      target_arch = "snb";
    } break;
    case LIBXS_X86_SSE4: {
      /* TODO: rework BE to use target ID instead of set of strings (target_arch = "sse4") */
      target_arch = "wsm";
    } break;
    case LIBXS_X86_SSE3: {
      /* WSM includes SSE4, but BE relies on SSE3 only,
       * hence we enter "wsm" path starting with SSE3.
       */
      target_arch = "wsm";
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

  LIBXS_ASSERT(NULL != target_arch);
  return target_arch;
}

