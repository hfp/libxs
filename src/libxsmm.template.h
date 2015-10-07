/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
#ifndef LIBXS_H
#define LIBXS_H

/** Parameters the library was built for. */
#define LIBXS_ALIGNMENT $ALIGNMENT
#define LIBXS_ALIGNED_STORES $ALIGNED_STORES
#define LIBXS_ALIGNED_LOADS $ALIGNED_LOADS
#define LIBXS_ALIGNED_MAX $ALIGNED_MAX
#define LIBXS_ROW_MAJOR $ROW_MAJOR
#define LIBXS_COL_MAJOR $COL_MAJOR
#define LIBXS_PREFETCH $PREFETCH
#define LIBXS_PREFETCH_A $PREFETCH_A
#define LIBXS_PREFETCH_B $PREFETCH_B
#define LIBXS_PREFETCH_C $PREFETCH_C
#define LIBXS_MAX_MNK $MAX_MNK
#define LIBXS_MAX_M $MAX_M
#define LIBXS_MAX_N $MAX_N
#define LIBXS_MAX_K $MAX_K
#define LIBXS_AVG_M $AVG_M
#define LIBXS_AVG_N $AVG_N
#define LIBXS_AVG_K $AVG_K
#define LIBXS_BETA $BETA
#define LIBXS_OFFLOAD_ENABLED $OFFLOAD

#include "libxs_prefetch.h"
#include "libxs_fallback.h"

/** Type of a function generated for a specific M, N, and K. */
typedef LIBXS_RETARGETABLE void (*libxs_smm_function)(const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c
                                    LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pa)
                                    LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pb)
                                    LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pc));
typedef LIBXS_RETARGETABLE void (*libxs_dmm_function)(const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c
                                    LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pa)
                                    LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pb)
                                    LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pc));

/** Query the pointer of a generated function; zero if it does not exist. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_smm_function libxs_smm_dispatch(int m, int n, int k);
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dmm_function libxs_dmm_dispatch(int m, int n, int k);

/** JIT a function and return the pointer to the executable memory. */
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_smm_function libxs_smm_jit_build(int m, int n, int k);
LIBXS_EXTERN_C LIBXS_RETARGETABLE libxs_dmm_function libxs_dmm_jit_build(int m, int n, int k);

/** Dispatched matrix-matrix multiplication; single-precision. */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_smm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pa) LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pb) LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  LIBXS_MM(float, m, n, k, a, b, c, LIBXS_PREFETCH_ARG_pa, LIBXS_PREFETCH_ARG_pb, LIBXS_PREFETCH_ARG_pc);
}

/** Dispatched matrix-matrix multiplication; double-precision. */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dmm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pa) LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pb) LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  LIBXS_MM(double, m, n, k, a, b, c, LIBXS_PREFETCH_ARG_pa, LIBXS_PREFETCH_ARG_pb, LIBXS_PREFETCH_ARG_pc);
}

/** Non-dispatched matrix-matrix multiplication using inline code; single-precision. */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_simm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pa) LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pb) LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  LIBXS_IMM(float, int, m, n, k, a, b, c, LIBXS_PREFETCH_ARG_pa, LIBXS_PREFETCH_ARG_pb, LIBXS_PREFETCH_ARG_pc);
}

/** Non-dispatched matrix-matrix multiplication using inline code; double-precision. */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dimm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pa) LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pb) LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  LIBXS_IMM(double, int, m, n, k, a, b, c, LIBXS_PREFETCH_ARG_pa, LIBXS_PREFETCH_ARG_pb, LIBXS_PREFETCH_ARG_pc);
}

/** Non-dispatched matrix-matrix multiplication using BLAS; single-precision. */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_sblasmm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c) {
  LIBXS_BLASMM(float, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS; double-precision. */
LIBXS_INLINE LIBXS_RETARGETABLE void libxs_dblasmm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c) {
  LIBXS_BLASMM(double, m, n, k, a, b, c);
}
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Dispatched matrix-matrix multiplication. */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pa) LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pb) LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  libxs_smm(m, n, k, a, b, c LIBXS_PREFETCH_ARGA(pa) LIBXS_PREFETCH_ARGB(pb) LIBXS_PREFETCH_ARGC(pc));
}

/** Dispatched matrix-matrix multiplication. */
LIBXS_RETARGETABLE inline void libxs_mm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pa) LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pb) LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  libxs_dmm(m, n, k, a, b, c LIBXS_PREFETCH_ARGA(pa) LIBXS_PREFETCH_ARGB(pb) LIBXS_PREFETCH_ARGC(pc));
}

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXS_RETARGETABLE inline void libxs_imm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pa) LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pb) LIBXS_PREFETCH_DECL(const float *LIBXS_RESTRICT, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  libxs_simm(m, n, k, a, b, c LIBXS_PREFETCH_ARGA(pa) LIBXS_PREFETCH_ARGB(pb) LIBXS_PREFETCH_ARGC(pc));
}

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXS_RETARGETABLE inline void libxs_imm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c
  LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pa) LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pb) LIBXS_PREFETCH_DECL(const double *LIBXS_RESTRICT, pc))
{
  LIBXS_USE(pa); LIBXS_USE(pb); LIBXS_USE(pc);
  libxs_dimm(m, n, k, a, b, c LIBXS_PREFETCH_ARGA(pa) LIBXS_PREFETCH_ARGB(pb) LIBXS_PREFETCH_ARGC(pc));
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXS_RETARGETABLE inline void libxs_blasmm(int m, int n, int k, const float *LIBXS_RESTRICT a, const float *LIBXS_RESTRICT b, float *LIBXS_RESTRICT c)
{
  libxs_sblasmm(m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXS_RETARGETABLE inline void libxs_blasmm(int m, int n, int k, const double *LIBXS_RESTRICT a, const double *LIBXS_RESTRICT b, double *LIBXS_RESTRICT c)
{
  libxs_dblasmm(m, n, k, a, b, c);
}

/** Call libxs_smm_dispatch, or libxs_dmm_dispatch depending on T. */
template<typename T> class LIBXS_RETARGETABLE libxs_mm_dispatch { typedef void function_type; };

template<> class LIBXS_RETARGETABLE libxs_mm_dispatch<float> {
  typedef libxs_smm_function function_type;
  mutable/*retargetable*/ function_type m_function;
public:
  libxs_mm_dispatch(): m_function(0) {}
  libxs_mm_dispatch(int m, int n, int k): m_function(libxs_smm_dispatch(m, n, k)) {}
  operator function_type() const { return m_function; }
};

template<> class LIBXS_RETARGETABLE libxs_mm_dispatch<double> {
  typedef libxs_dmm_function function_type;
  mutable/*retargetable*/ function_type m_function;
public:
  libxs_mm_dispatch(): m_function(0) {}
  libxs_mm_dispatch(int m, int n, int k): m_function(libxs_dmm_dispatch(m, n, k)) {}
  operator function_type() const { return m_function; }
};

#endif /*__cplusplus*/
#endif /*LIBXS_H*/
