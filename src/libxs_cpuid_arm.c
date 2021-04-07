/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_cpuid.h>
#include <libxs_generator.h>
#include <libxs_mem.h>

#if defined(LIBXS_PLATFORM_AARCH64)
# include "libxs_main.h"
# include <asm/hwcap.h>
# if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD))) /* GLIBC */
#   include <sys/auxv.h>
# endif
#else
# include <libxs_sync.h>
#endif


LIBXS_API int libxs_cpuid_arm(libxs_cpuid_info* info)
{
  static int result = LIBXS_TARGET_ARCH_UNKNOWN;
#if defined(LIBXS_PLATFORM_X86)
# if !defined(NDEBUG)
  static int error_once = 0;
  if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS WARNING: libxs_cpuid_arm called on x86 platform!\n");
  }
# endif
  if (NULL != info) LIBXS_MEMZERO127(info);
#else
  if (NULL != info) LIBXS_MEMZERO127(info);
# if defined(LIBXS_PLATFORM_AARCH64)
  result = LIBXS_AARCH64_V81;
  {
#   if defined(LIBXS_INTERCEPT_DYNAMIC)
    typedef unsigned long (*getcap_fn)(unsigned long);
    static getcap_fn getcap = NULL;
    if (NULL == getcap) {
#     if defined(RTLD_DEFAULT)
      void *const handle = RTLD_DEFAULT;
#     else
      void *const handle = dlopen(NULL, RTLD_LOCAL);
#     endif
      dlerror();
      getcap = (getcap_fn)dlsym(handle, "getauxval");
      if (NULL != dlerror()) getcap = NULL;
#     if !defined(RTLD_DEFAULT)
      if (NULL != handle) result = dlclose(handle);
#     endif
    }
    if (NULL != getcap) {
      const unsigned long capabilities = getcap(AT_HWCAP);
#     if defined(HWCAP_DCPOP)
      if (HWCAP_DCPOP & capabilities) {
#       if defined(HWCAP_SVE)
        if (HWCAP_SVE & capabilities) {
          result = LIBXS_AARCH64_A64FX;
        }
        else
#       endif
        {
          result = LIBXS_AARCH64_V82;
        }
      } /* HWCAP_DCPOP */
#     endif
    }
    else {
    }
#   endif
  }
# endif
#endif
  return result;
}
