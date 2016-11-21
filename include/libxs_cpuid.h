/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#ifndef LIBXS_CPUID_H
#define LIBXS_CPUID_H

#include "libxs_macros.h"

/**
 * Enumerates the available target architectures and instruction
 * set extensions as returned by libxs_get_target_archid().
 */
#define LIBXS_TARGET_ARCH_UNKNOWN 0
#define LIBXS_TARGET_ARCH_GENERIC 1
#define LIBXS_X86_GENERIC      1000
#define LIBXS_X86_IMCI         1001
#define LIBXS_X86_SSE3         1002
#define LIBXS_X86_SSE4_1       1003
#define LIBXS_X86_SSE4_2       1004
#define LIBXS_X86_AVX          1005
#define LIBXS_X86_AVX2         1006
#define LIBXS_X86_AVX512       1007
#define LIBXS_X86_AVX512_MIC   1008
#define LIBXS_X86_AVX512_CORE  1009


/** Returns the target architecture and instruction set extension (code path). */
LIBXS_API int libxs_cpuid_x86(void);

/** Returns the target architecture and instruction set extension (code path). */
LIBXS_INLINE LIBXS_RETARGETABLE int libxs_cpuid(void) { return libxs_cpuid_x86(); }

#endif /*LIBXS_CPUID_H*/
