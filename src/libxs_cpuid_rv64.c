/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_cpuid.h>
#include "libxs_main.h"

#include <signal.h>
#include <setjmp.h>

#if defined(LIBXS_PLATFORM_RV64)
LIBXS_APIVAR_DEFINE(jmp_buf internal_cpuid_rv64_jmp_buf);
LIBXS_API_INTERN void internal_cpuid_rv64_sigill(int /*signum*/);
LIBXS_API_INTERN void internal_cpuid_rv64_sigill(int signum) {
  void (*const handler)(int) = signal(signum, internal_cpuid_rv64_sigill);
  LIBXS_ASSERT(SIGILL == signum);
  if (SIG_ERR != handler) longjmp(internal_cpuid_rv64_jmp_buf, 1);
}
#endif


LIBXS_API_INTERN int libxs_cpuid_rv64(libxs_cpuid_info_t* info)
{
  int mvl = 0;
  libxs_cpuid_info_t cpuid_info;
  size_t cpuinfo_model_size = sizeof(cpuid_info.model);
#if defined(LIBXS_PLATFORM_RV64)
  { void (*const handler)(int) = signal(SIGILL, internal_cpuid_rv64_sigill);
    if (SIG_ERR != handler) {
      if (0 == setjmp(internal_cpuid_rv64_jmp_buf)) {
        int rvl = 65536;
        __asm__(".option arch, +zve64x\n\t"
                "vsetvli %0, %1, e8, m1, ta, ma\n"
                : "=r"(mvl) : "r"(rvl));
      }
      signal(SIGILL, handler);
    }
  }
#endif

  libxs_cpuid_model(cpuid_info.model, &cpuinfo_model_size);
  LIBXS_ASSERT(0 != cpuinfo_model_size || '\0' == *cpuid_info.model);
  cpuid_info.constant_tsc = 1;

  /* Get MVL in bits */
  switch (mvl * 8) {
    case 128: mvl = LIBXS_RV64_MVL128; break;
    case 256: mvl = LIBXS_RV64_MVL256; break;
    default: {
      if (0 != libxs_verbosity) {
        if (0 == mvl) {
          fprintf(stderr, "LIBXS WARNING: cannot determine RVV vector length!\n");
        }
        else {
          fprintf(stderr, "LIBXS WARNING: unexpected RVV %i-bit vector length!\n",
            mvl * 8);
        }
      }
      mvl = LIBXS_RV64_MVL128;
    }
  }

  if (NULL != info) memcpy(info, &cpuid_info, sizeof(cpuid_info));

  return mvl;
}
