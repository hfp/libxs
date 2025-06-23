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

#define VLMAX (65536)

LIBXS_API int libxs_cpuid_rv64(libxs_cpuid_info* info)
{
  int mvl;
  libxs_cpuid_info cpuid_info;
  size_t cpuinfo_model_size = sizeof(cpuid_info.model);
#ifdef LIBXS_PLATFORM_RV64
  int rvl = VLMAX;
  __asm__(".option arch, +zve64x\n\t""vsetvli %0, %1, e8, m1, ta, ma\n": "=r"(mvl): "r" (rvl));
#else
  mvl = 0;
#endif

  libxs_cpuid_model(cpuid_info.model, &cpuinfo_model_size);
  LIBXS_ASSERT(0 != cpuinfo_model_size || '\0' == *cpuid_info.model);
  cpuid_info.constant_tsc = 1;

  /* Get MVL in bits */
  switch (mvl * 8){
    case 128:
      mvl = LIBXS_RV64_MVL128;
      break;

    case 256:
      mvl = LIBXS_RV64_MVL256;
      break;

    default:
      mvl = LIBXS_RV64_MVL128;
      break;
  }

  if (NULL != info) memcpy(info, &cpuid_info, sizeof(cpuid_info));

  return mvl;
}

LIBXS_API unsigned int libxs_cpuid_rv64_gemm_prefetch_reuse_a(void){
  const char *const env_gemm_prefetch_reuse_a = getenv("LIBXS_RV64_GEMM_PREFETCH_REUSE_A");
  unsigned int result = (env_gemm_prefetch_reuse_a == 0) ? 0 : atoi(env_gemm_prefetch_reuse_a);
  return result;
}

LIBXS_API unsigned int libxs_cpuid_rv64_gemm_prefetch_reuse_b(void){
  const char *const env_gemm_prefetch_reuse_b = getenv("LIBXS_RV64_GEMM_PREFETCH_REUSE_B");
  unsigned int result = (env_gemm_prefetch_reuse_b == 0) ? 0 : atoi(env_gemm_prefetch_reuse_b);
  return result;
}

LIBXS_API unsigned int libxs_cpuid_rv64_gemm_prefetch_reuse_c(void){
  const char *const env_gemm_prefetch_reuse_c = getenv("LIBXS_RV64_GEMM_PREFETCH_REUSE_C");
  unsigned int result = (env_gemm_prefetch_reuse_c == 0) ? 0 : atoi(env_gemm_prefetch_reuse_c);
  return result;
}

LIBXS_API unsigned int libxs_cpuid_rv64_gemm_prefetch_a(void){
  const char *const env_gemm_prefetch_a = getenv("LIBXS_RV64_GEMM_PREFETCH_A");
  unsigned int result = (env_gemm_prefetch_a == 0) ? 0 : atoi(env_gemm_prefetch_a);
  return result;
}

LIBXS_API unsigned int libxs_cpuid_rv64_gemm_prefetch_b(void){
  const char *const env_gemm_prefetch_b = getenv("LIBXS_RV64_GEMM_PREFETCH_B");
  unsigned int result = (env_gemm_prefetch_b == 0) ? 0 : atoi(env_gemm_prefetch_b);
  return result;
}

LIBXS_API unsigned int libxs_cpuid_rv64_gemm_m_prefetch_stride(void){
  const char *const env_gemm_m_prefetch_stride = getenv("LIBXS_RV64_GEMM_M_PREFETCH_STRIDE");
  unsigned int result = (env_gemm_m_prefetch_stride == 0) ? 0 : atoi(env_gemm_m_prefetch_stride);
  return result;
}

#undef VLMAX
