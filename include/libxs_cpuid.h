/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_CPUID_H
#define LIBXS_CPUID_H

#include "libxs_typedefs.h"

/** Enumerates the target architectures and instruction set extensions. */
#define LIBXS_TARGET_ARCH_UNKNOWN   0
#define LIBXS_TARGET_ARCH_GENERIC   1
#define LIBXS_X86_GENERIC           1002
#define LIBXS_X86_SSE3              1003
#define LIBXS_X86_SSE42             1004
#define LIBXS_X86_AVX               1005
#define LIBXS_X86_AVX2              1006
#define LIBXS_X86_AVX512            1100
#define LIBXS_X86_ALLFEAT           1999
#define LIBXS_AARCH64               2001 /* Baseline v8.1 */
#define LIBXS_AARCH64_SVE128        2201 /* SVE 128 */
#define LIBXS_AARCH64_SVE256        2301 /* SVE 256 */
#define LIBXS_AARCH64_SVE512        2401 /* SVE 512 */
#define LIBXS_AARCH64_ALLFEAT       2999
#define LIBXS_RV64_MVL128           3001 /* RISCV 128-bit RVV */
#define LIBXS_RV64_MVL256           3002 /* RISCV 256-bit RVV */
#define LIBXS_RV64_MVL128_LMUL      3003 /* RISCV 128-bit RVV with non-unit LMUL */
#define LIBXS_RV64_MVL256_LMUL      3004 /* RISCV 256-bit RVV witb non-unit LMUL */
#define LIBXS_RV64_ALLFEAT          3999

 /** Zero-initialized structure; assumes conservative properties. */
LIBXS_EXTERN_C typedef struct libxs_cpuid_info {
  char model[1024]; /** CPU-name (OS-specific implementation). */
  int constant_tsc; /** Timer stamp counter is monotonic. */
#if defined(LIBXS_PLATFORM_X86)
  int has_context;  /** Context switches are permitted. */
#endif
} libxs_cpuid_info;

/** Returns the target architecture and instruction set extensions. */
LIBXS_API int libxs_cpuid_x86(libxs_cpuid_info* LIBXS_ARGDEF(info, NULL));
LIBXS_API int libxs_cpuid_arm(libxs_cpuid_info* LIBXS_ARGDEF(info, NULL));
LIBXS_API int libxs_cpuid_rv64(libxs_cpuid_info* LIBXS_ARGDEF(info, NULL));

/** Similar to libxs_cpuid_x86, but conceptually not arch-specific. */
LIBXS_API int libxs_cpuid(libxs_cpuid_info* LIBXS_ARGDEF(info, NULL));

/** Names the CPU architecture given by CPUID. */
LIBXS_API const char* libxs_cpuid_name(int id);

/** Translate the CPU name to LIBXS's internal ID. */
LIBXS_API int libxs_cpuid_id(const char* name);

/** SIMD vector length (VLEN) in Bytes; zero if scalar. */
LIBXS_API int libxs_cpuid_vlen(int id);

#endif /*LIBXS_CPUID_H*/
