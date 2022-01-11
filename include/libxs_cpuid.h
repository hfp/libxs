/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_CPUID_H
#define LIBXS_CPUID_H

#include "libxs_macros.h"

/**
 * Enumerates the available target architectures and instruction
 * set extensions as returned by libxs_get_target_archid().
 * LIBXS_X86_ALLFEAT: pseudo-value enabling all features
 * used anywhere in LIBXS (never set as an architecture,
 * used as an upper bound in comparisons to distinct x86).
 */
#define LIBXS_TARGET_ARCH_UNKNOWN   0
#define LIBXS_TARGET_ARCH_GENERIC   1
#define LIBXS_X86_GENERIC           1002
#define LIBXS_X86_SSE3              1003
#define LIBXS_X86_SSE42             1004
#define LIBXS_X86_AVX               1005
#define LIBXS_X86_AVX2              1006
#define LIBXS_X86_AVX512_VL256      1007
#define LIBXS_X86_AVX512_VL256_CLX  1008
#define LIBXS_X86_AVX512_VL256_CPX  1009
#define LIBXS_X86_AVX512            1010
#define LIBXS_X86_AVX512_MIC        1011 /* KNL */
#define LIBXS_X86_AVX512_KNM        1012
#define LIBXS_X86_AVX512_CORE       1020 /* SKX */
#define LIBXS_X86_AVX512_CLX        1021
#define LIBXS_X86_AVX512_CPX        1022
#define LIBXS_X86_AVX512_SPR        1023
#define LIBXS_X86_ALLFEAT           1999
#define LIBXS_AARCH64_V81           2001 /* Baseline */
#define LIBXS_AARCH64_V82           2002 /* A64FX minus SVE */
#define LIBXS_AARCH64_A64FX         2100 /* SVE */
#define LIBXS_AARCH64_APPL_M1       2200 /* Apple M1 */
#define LIBXS_AARCH64_ALLFEAT       2999

#if defined(LIBXS_PLATFORM_X86)
LIBXS_API_INTERN int lixsmm_cpuid_x86_amx_enable();
/** Zero-initialized structure; assumes conservative properties. */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_cpuid_info {
  int constant_tsc; /** Timer stamp counter is monotonic. */
  int has_context;  /** Context switches are permitted. */
} libxs_cpuid_info;
#else
typedef int libxs_cpuid_info;
#endif

/** Returns the target architecture and instruction set extensions. */
#if defined(__cplusplus) /* note: stay compatible with TF */
LIBXS_API int libxs_cpuid_x86(libxs_cpuid_info* info = NULL);
LIBXS_API int libxs_cpuid_arm(libxs_cpuid_info* info = NULL);
#else
LIBXS_API int libxs_cpuid_x86(libxs_cpuid_info* info);
LIBXS_API int libxs_cpuid_arm(libxs_cpuid_info* info);
#endif

/**
 * Similar to libxs_cpuid_x86, but conceptually not x86-specific.
 * The actual code path (as used by LIBXS) is determined by
 * libxs_[get|set]_target_archid/libxs_[get|set]_target_arch.
 */
LIBXS_API int libxs_cpuid(void);

/**
 * Names the CPU architecture given by CPUID.
 * Do not use libxs_cpuid() to match the current CPU!
 * Use libxs_get_target_archid() instead.
 */
LIBXS_API const char* libxs_cpuid_name(int id);

/**
 * SIMD vector length (VLEN) in 32-bit elements.
 * Do not use libxs_cpuid() to match the current CPU!
 * Use libxs_get_target_archid() instead.
 */
LIBXS_API int libxs_cpuid_vlen32(int id);

#endif /*LIBXS_CPUID_H*/
