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

#include "libxs.h"

/** Enumerates the target architectures and instruction set extensions.
 * x86 levels use thermometer ordering: higher numeric value implies all
 * features of lower levels.  AVX10/256 sits below AVX512 because it lacks
 * 512-bit vectors -- existing LIBXS_X86_AVX512 <= arch guards remain safe.
 * AVX10_512 (1200) implies AVX512_INT8 (1110); CPUs with AVX10.1/512 but
 * without AVX-VNNI-INT8 (e.g. Granite Rapids P-core) report as AVX512. */
#define LIBXS_TARGET_ARCH_UNKNOWN   0
#define LIBXS_TARGET_ARCH_GENERIC   1
#define LIBXS_X86_GENERIC           1002
#define LIBXS_X86_SSE3              1003
#define LIBXS_X86_SSE42             1004
#define LIBXS_X86_AVX               1005
#define LIBXS_X86_AVX2              1006
#define LIBXS_X86_AVX10_256         1030 /* AVX10.1/256: all features, 256-bit max (Sierra Forest) */
#define LIBXS_X86_AVX512            1100 /* AVX-512: f+cd+dq+bw+vl+vnni (Cascade Lake) */
#define LIBXS_X86_AVX512_AMX        1105 /* + AMX-TILE+INT8+BF16: TDPBUSD (Sapphire/Granite Rapids) */
#define LIBXS_X86_AVX512_INT8       1110 /* + AVX-VNNI-INT8: VPDPBUUD/BSSD */
#define LIBXS_X86_AVX10_512         1200 /* AVX10.1/512 + AMX + INT8: full feature set */
#define LIBXS_X86_ALLFEAT           1999
#define LIBXS_AARCH64               2001 /* Baseline v8.1 */
#define LIBXS_AARCH64_SVE128        2201 /* SVE 128 */
#define LIBXS_AARCH64_SVE256        2301 /* SVE 256 */
#define LIBXS_AARCH64_SVE512        2401 /* SVE 512 */
#define LIBXS_AARCH64_ALLFEAT       2999
#define LIBXS_RV64_MVL128           3001 /* RISCV 128-bit RVV */
#define LIBXS_RV64_MVL256           3002 /* RISCV 256-bit RVV */
#define LIBXS_RV64_MVL128_LMUL      3003 /* RISCV 128-bit RVV with non-unit LMUL */
#define LIBXS_RV64_MVL256_LMUL      3004 /* RISCV 256-bit RVV with non-unit LMUL */
#define LIBXS_RV64_ALLFEAT          3999

/**
 * CPU identification result. Zero-initialization yields
 * conservative (least-capable) properties.
 */
LIBXS_EXTERN_C typedef struct libxs_cpuid_t {
  /** CPU model name obtained from the OS (e.g., /proc/cpuinfo). */
  char model[256];
  /** Non-zero if the timestamp counter is invariant/monotonic (see libxs_timer_info). */
  int constant_tsc;
} libxs_cpuid_t;

/**
 * Returns the detected ISA level for the current platform and optionally
 * fills info with CPU properties (model name, TSC capability).
 * Dispatches to the appropriate architecture-specific implementation.
 */
LIBXS_API int libxs_cpuid(libxs_cpuid_t* LIBXS_ARGDEF(info, NULL));

/** Returns a human-readable name for the given ISA level id. */
LIBXS_API const char* libxs_cpuid_name(int id);

/** Translates a CPU architecture name (e.g., "avx2") to the corresponding LIBXS id constant. */
LIBXS_API int libxs_cpuid_id(const char* name);

/** Returns the SIMD vector length (VLEN) in bytes for the given ISA level; zero if scalar. */
LIBXS_API int libxs_cpuid_vlen(int id);

/**
 * Request AMX tile state (XTILE_DATA) from the OS.
 * Must be called before any AMX tile instruction. On Linux, this
 * requests ARCH_REQ_XCOMP_PERM for component 18 via arch_prctl.
 * Returns EXIT_SUCCESS (or if already enabled), EXIT_FAILURE otherwise.
 * Not called automatically by libxs_cpuid to avoid the per-thread
 * XSAVE overhead (~8KB context-switch cost) for non-AMX workloads.
 */
LIBXS_API int libxs_cpuid_amx_enable(void);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_UTILS_H) && !defined(LIBXS_SYNC_H) \
 && !defined(LIBXS_MALLOC_H) && !defined(LIBXS_HIST_H) \
 && !defined(LIBXS_REG_H) && !defined(LIBXS_GEMM_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_CPUID_H*/
