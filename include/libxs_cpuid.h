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
 */
#define LIBXS_TARGET_ARCH_UNKNOWN 0
#define LIBXS_TARGET_ARCH_GENERIC 1
#define LIBXS_X86_GENERIC      1002
#define LIBXS_X86_SSE3         1003
#define LIBXS_X86_SSE4         1004
#define LIBXS_X86_AVX          1005
#define LIBXS_X86_AVX2         1006
#define LIBXS_X86_AVX512       1007
#define LIBXS_X86_AVX512_MIC   1010 /* KNL */
#define LIBXS_X86_AVX512_KNM   1011
#define LIBXS_X86_AVX512_CORE  1020 /* SKX */
#define LIBXS_X86_AVX512_CLX   1021
#define LIBXS_X86_AVX512_CPX   1022
#define LIBXS_X86_AVX512_SPR   1023
#define LIBXS_X86_ALLFEAT      1999 /* all features supported which are used anywhere in LIBXS, this value should never be used to set arch, only for compares */
/** A zero-initialized structure assumes conservative properties. */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_cpuid_x86_info {
  int constant_tsc; /** Timer stamp counter is monotonic. */
  int has_context;  /** Context switches are permitted. */
} libxs_cpuid_x86_info;

/** Returns the target architecture and instruction set extensions. */
#if defined(__cplusplus) /* note: stay compatible with TF */
LIBXS_API int libxs_cpuid_x86(libxs_cpuid_x86_info* info = NULL);
#else
LIBXS_API int libxs_cpuid_x86(libxs_cpuid_x86_info* info);
#endif

/**
 * Similar to libxs_cpuid_x86, but conceptually not x86-specific.
 * The actual code path (as used by LIBXS) is determined by
 * libxs_[get|set]_target_archid/libxs_[get|set]_target_arch.
 */
LIBXS_API int libxs_cpuid(void);

/** Names the CPU architecture given by CPUID. */
LIBXS_API const char* libxs_cpuid_name(int id);

/** SIMD vector length (VLEN) in 32-bit elements. */
LIBXS_API int libxs_cpuid_vlen32(int id);

#endif /*LIBXS_CPUID_H*/
