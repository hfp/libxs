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


LIBXS_API int libxs_cpuid_rv64(libxs_cpuid_info_t* info)
{
  int mvl;
  libxs_cpuid_info_t cpuid_info;
  size_t cpuinfo_model_size = sizeof(cpuid_info.model);
#ifdef LIBXS_PLATFORM_RV64
  int rvl = 65536;
  __asm__(".option arch, +zve64x\n\t""vsetvli %0, %1, e8, m1, ta, ma\n": "=r"(mvl): "r" (rvl));
#else
  mvl = 0;
#endif

  libxs_cpuid_model(cpuid_info.model, &cpuinfo_model_size);
  LIBXS_ASSERT(0 != cpuinfo_model_size || '\0' == *cpuid_info.model);
  cpuid_info.constant_tsc = 1;

  /* Get MVL in bits */
  switch (mvl * 8){
    case 128: mvl = LIBXS_RV64_MVL128; break;
    case 256: mvl = LIBXS_RV64_MVL256; break;
    default: mvl = LIBXS_RV64_MVL128;
  }

  if (NULL != info) memcpy(info, &cpuid_info, sizeof(cpuid_info));

  return mvl;
}
