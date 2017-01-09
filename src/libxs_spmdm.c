/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
#include <libxs_spmdm.h>
#include <libxs_intrinsics_x86.h>
#include <libxs.h>
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_SPMDM_MALLOC_INTRINSIC) && !defined(LIBXS_INTRINSICS_NONE)
# define LIBXS_SPMDM_MALLOC_INTRINSIC
#endif
#if defined(LIBXS_SPMDM_MALLOC_INTRINSIC)
# define LIBXS_SPMDM_MALLOC(SIZE, ALIGNMENT) _mm_malloc(SIZE, ALIGNMENT)
# define LIBXS_SPMDM_FREE(BUFFER) _mm_free((void*)(BUFFER))
#else
# define LIBXS_SPMDM_MALLOC(SIZE, ALIGNMENT) libxs_aligned_malloc(SIZE, -(ALIGNMENT))
# define LIBXS_SPMDM_FREE(BUFFER) libxs_free(BUFFER)
#endif

/* Enable/disable specific code paths */
#if !defined(LIBXS_SPMDM_AVX512_CORE)
# define LIBXS_SPMDM_AVX512_CORE
#endif
#if !defined(LIBXS_SPMDM_AVX2)
# define LIBXS_SPMDM_AVX2
#endif


#if !defined(LIBXS_INTRINSICS_NONE) && (LIBXS_X86_AVX <= LIBXS_MAX_STATIC_TARGET_ARCH)
LIBXS_EXTERN_C LIBXS_RETARGETABLE __m256i internal_spmdm_shufmasks_32[256];
LIBXS_EXTERN_C LIBXS_RETARGETABLE __m256i internal_spmdm_shufmasks_16[256];
#endif


/* function pointer for the CPUID-dispatched implementation */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void (*internal_spmdm_createSparseSlice_fp32_thread)(const libxs_spmdm_handle*, char,
  const float*, libxs_CSR_sparseslice*, int, int, int);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void (*internal_spmdm_createSparseSlice_bfloat16_thread)(const libxs_spmdm_handle*, char,
  const uint16_t*, libxs_CSR_sparseslice*, int, int, int);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void (*internal_spmdm_compute_fp32_thread)(const libxs_spmdm_handle*, char, char,
  const float*, libxs_CSR_sparseslice*, const float*, char, const float*, float*, int, int, int);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void (*internal_spmdm_compute_bfloat16_thread)(const libxs_spmdm_handle*, char, char,
  const uint16_t*, libxs_CSR_sparseslice*, const uint16_t*, char, const uint16_t*, float*, int, int, int);


LIBXS_INLINE LIBXS_RETARGETABLE LIBXS_INTRINSICS(LIBXS_X86_AVX)
void internal_spmdm_init_shufmask_avx()
{
#if !defined(LIBXS_INTRINSICS_NONE) && !defined(LIBXS_INTRINSICS_LEGACY) \
  && (LIBXS_X86_AVX <= LIBXS_MAX_STATIC_TARGET_ARCH)
  unsigned int i, j, c, last_bit;
  LIBXS_ALIGNED(int temp_shufmasks[8], 64);
  LIBXS_ALIGNED(uint16_t temp_shufmasks2[16], 64);
  int cnt;
  for (i = 0; i < 256; i++) {
    cnt = 0;
    j = i;
    for (c = 0; c < 8; c++) temp_shufmasks[c] = 0;
    for (c = 0; c < 16; c++) temp_shufmasks2[c] = 0;
    while ( j) {
      last_bit = LIBXS_INTRINSICS_BITSCANFWD(j);
      temp_shufmasks[cnt] = last_bit;
      temp_shufmasks2[cnt] = (uint16_t)last_bit;
      j &= (~(1<<last_bit));
      cnt++;
    }
    internal_spmdm_shufmasks_32[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks);
    internal_spmdm_shufmasks_16[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks2);
  }
#endif
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_spmdm_allocate_csr_a(libxs_spmdm_handle* handle, libxs_CSR_sparseslice** libxs_output_csr)
{
  int kb, mb;
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;

  size_t sz_block = ((handle->bm + 1)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(float) + sizeof(libxs_CSR_sparseslice));
  size_t sz_all_blocks = sz_block * handle->mb * handle->kb;

  char * memory_block = (char *)LIBXS_SPMDM_MALLOC( sz_all_blocks, 2097152);
  char * memory_head  = memory_block;

  libxs_CSR_sparseslice* libxs_output_csr_a = (libxs_CSR_sparseslice*)(memory_head);
  memory_head += handle->mb * handle->kb * sizeof(libxs_CSR_sparseslice);

  for (kb = 0; kb < k_blocks; kb++) {
    for (mb = 0; mb < m_blocks; mb++) {
      int i = kb*m_blocks + mb;
      libxs_output_csr_a[i].rowidx = (uint16_t *)(memory_head);
      memory_head += (handle->bm + 1)*sizeof(uint16_t);
      libxs_output_csr_a[i].colidx = (uint16_t *)(memory_head);
      memory_head += (handle->bm)*(handle->bk)*sizeof(uint16_t);
      libxs_output_csr_a[i].values = (float*)(memory_head);
      memory_head += (handle->bm)*(handle->bk)*sizeof(float);
    }
  }
  assert(memory_head == (memory_block + sz_all_blocks));
  *libxs_output_csr = libxs_output_csr_a;
  handle->base_ptr_scratch_A = memory_block;
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_spmdm_allocate_scratch(libxs_spmdm_handle* handle, int max_threads)
{
  size_t sz_memory_for_scratch_per_thread = ((handle->bm)*(handle->bn)*sizeof(float) + (handle->bk)*(handle->bn)*sizeof(float))*max_threads, sz_total_memory;
  sz_memory_for_scratch_per_thread = (sz_memory_for_scratch_per_thread + 4095)/4096 * 4096;
  sz_total_memory = sz_memory_for_scratch_per_thread * max_threads;

  handle->base_ptr_scratch_B_scratch_C = (char *)LIBXS_SPMDM_MALLOC(sz_total_memory, 2097152);
  handle->memory_for_scratch_per_thread = (int)sz_memory_for_scratch_per_thread;
}


LIBXS_INLINE LIBXS_RETARGETABLE void internal_spmdm_deallocate_csr_a(libxs_spmdm_handle* handle)
{
  LIBXS_SPMDM_FREE(handle->base_ptr_scratch_A);
  handle->base_ptr_scratch_A= NULL;
  LIBXS_SPMDM_FREE(handle->base_ptr_scratch_B_scratch_C);
  handle->base_ptr_scratch_B_scratch_C = NULL;
}


LIBXS_API_DEFINITION void libxs_spmdm_destroy(libxs_spmdm_handle* handle)
{
  internal_spmdm_deallocate_csr_a(handle);
}


LIBXS_API_DEFINITION int libxs_spmdm_get_num_createSparseSlice_blocks(const libxs_spmdm_handle* handle)
{
  return handle->mb * handle->kb;
}


LIBXS_API_DEFINITION int libxs_spmdm_get_num_compute_blocks(const libxs_spmdm_handle* handle)
{
  return handle->mb * handle->nb;
}


LIBXS_INLINE LIBXS_RETARGETABLE
void internal_spmdm_createSparseSlice_fp32_thread_sw(
  const libxs_spmdm_handle* handle,
  char transA,
  const float* A,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
# include "libxs_spmdm_begin.h"
# include "template/libxs_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
}


LIBXS_INLINE LIBXS_RETARGETABLE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_spmdm_createSparseSlice_fp32_thread_avx2(
  const libxs_spmdm_handle* handle,
  char transA,
  const float* A,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_SPMDM_AVX2) && (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
# include "libxs_spmdm_begin_avx2.h"
# include "template/libxs_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
# if !defined(NDEBUG) && defined(LIBXS_SPMDM_AVX2) /* library code is expected to be mute */
  { static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: unable to enter AVX2 code path!\n");
    }
  }
# endif
  internal_spmdm_createSparseSlice_fp32_thread_sw(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXS_INLINE LIBXS_RETARGETABLE LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
void internal_spmdm_createSparseSlice_fp32_thread_avx512_core(
  const libxs_spmdm_handle* handle,
  char transA,
  const float* A,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_SPMDM_AVX512_CORE) && (LIBXS_X86_AVX512_CORE <= LIBXS_MAX_STATIC_TARGET_ARCH)
# include "libxs_spmdm_begin_avx512.h"
# include "template/libxs_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
# if !defined(NDEBUG) && defined(LIBXS_SPMDM_AVX512_CORE) /* library code is expected to be mute */
  { static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: unable to enter AVX-512/Core code path!\n");
    }
  }
# endif
  internal_spmdm_createSparseSlice_fp32_thread_avx2(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXS_API_DEFINITION
void libxs_spmdm_createSparseSlice_fp32_thread(
  const libxs_spmdm_handle* handle,
  char transA,
  const float* A,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if (LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_fp32_thread_avx512_core(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_fp32_thread_avx2(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#else /* pointer based function call */
  assert(0 != internal_spmdm_createSparseSlice_fp32_thread);
  internal_spmdm_createSparseSlice_fp32_thread(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXS_INLINE LIBXS_RETARGETABLE
void internal_spmdm_createSparseSlice_bfloat16_thread_sw(
  const libxs_spmdm_handle* handle,
  char transA,
  const uint16_t* A,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
# include "libxs_spmdm_begin.h"
# include "template/libxs_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
}


LIBXS_INLINE LIBXS_RETARGETABLE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_spmdm_createSparseSlice_bfloat16_thread_avx2(
  const libxs_spmdm_handle* handle,
  char transA,
  const uint16_t* A,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_SPMDM_AVX2) && (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
# include "libxs_spmdm_begin_avx2.h"
# include "template/libxs_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
# if !defined(NDEBUG) && defined(LIBXS_SPMDM_AVX2) /* library code is expected to be mute */
  { static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: unable to enter AVX2 code path!\n");
    }
  }
# endif
  internal_spmdm_createSparseSlice_bfloat16_thread_sw(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXS_INLINE LIBXS_RETARGETABLE LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
void internal_spmdm_createSparseSlice_bfloat16_thread_avx512_core(
  const libxs_spmdm_handle* handle,
  char transA,
  const uint16_t* A,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_SPMDM_AVX512_CORE) && (LIBXS_X86_AVX512_CORE <= LIBXS_MAX_STATIC_TARGET_ARCH)
# include "libxs_spmdm_begin_avx512.h"
# include "template/libxs_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
# if !defined(NDEBUG) && defined(LIBXS_SPMDM_AVX512_CORE) /* library code is expected to be mute */
  { static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: unable to enter AVX-512/Core code path!\n");
    }
  }
# endif
  internal_spmdm_createSparseSlice_bfloat16_thread_avx2(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXS_API_DEFINITION
void libxs_spmdm_createSparseSlice_bfloat16_thread(
  const libxs_spmdm_handle* handle,
  char transA,
  const uint16_t* A,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if (LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_bfloat16_thread_avx512_core(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_bfloat16_thread_avx2(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#else /* pointer based function call */
  assert(0 != internal_spmdm_createSparseSlice_fp32_thread);
  internal_spmdm_createSparseSlice_bfloat16_thread(handle, transA, A, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXS_INLINE LIBXS_RETARGETABLE
void internal_spmdm_compute_fp32_thread_sw(
  const libxs_spmdm_handle* handle,
  char transA,
  char transB,
  const float* alpha,
  libxs_CSR_sparseslice* A_sparse,
  const float* B,
  char transC,
  const float* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
# include "libxs_spmdm_begin.h"
# include "template/libxs_spmdm_compute_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
}


LIBXS_INLINE LIBXS_RETARGETABLE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_spmdm_compute_fp32_thread_avx2(
  const libxs_spmdm_handle* handle,
  char transA,
  char transB,
  const float* alpha,
  libxs_CSR_sparseslice* A_sparse,
  const float* B,
  char transC,
  const float* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_SPMDM_AVX2) && (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
# include "libxs_spmdm_begin_avx2.h"
# include "template/libxs_spmdm_compute_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
# if !defined(NDEBUG) && defined(LIBXS_SPMDM_AVX2) /* library code is expected to be mute */
  { static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: unable to enter AVX2 code path!\n");
    }
  }
# endif
  internal_spmdm_compute_fp32_thread_sw(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXS_INLINE LIBXS_RETARGETABLE LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
void internal_spmdm_compute_fp32_thread_avx512_core(
  const libxs_spmdm_handle* handle,
  char transA,
  char transB,
  const float* alpha,
  libxs_CSR_sparseslice* A_sparse,
  const float* B,
  char transC,
  const float* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_SPMDM_AVX512_CORE) && (LIBXS_X86_AVX512_CORE <= LIBXS_MAX_STATIC_TARGET_ARCH)
# include "libxs_spmdm_begin_avx512.h"
# include "template/libxs_spmdm_compute_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
# if !defined(NDEBUG) && defined(LIBXS_SPMDM_AVX512_CORE) /* library code is expected to be mute */
  { static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: unable to enter AVX-512/Core code path!\n");
    }
  }
# endif
  internal_spmdm_compute_fp32_thread_avx2(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXS_API_DEFINITION
void libxs_spmdm_compute_fp32_thread(
  const libxs_spmdm_handle* handle,
  char transA,
  char transB,
  const float* alpha,
  libxs_CSR_sparseslice* A_sparse,
  const float* B,
  char transC,
  const float* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
#if (LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH)
  internal_spmdm_compute_fp32_thread_avx512_core(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  internal_spmdm_compute_fp32_thread_avx2(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#else /* pointer based function call */
  assert(0 != internal_spmdm_compute_fp32_thread);
  internal_spmdm_compute_fp32_thread(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXS_INLINE LIBXS_RETARGETABLE
void internal_spmdm_compute_bfloat16_thread_sw(
  const libxs_spmdm_handle* handle,
  char transA,
  char transB,
  const uint16_t* alpha,
  libxs_CSR_sparseslice* A_sparse,
  const uint16_t* B,
  char transC,
  const uint16_t* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
# include "libxs_spmdm_begin.h"
# include "template/libxs_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
}


LIBXS_INLINE LIBXS_RETARGETABLE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_spmdm_compute_bfloat16_thread_avx2(
  const libxs_spmdm_handle* handle,
  char transA,
  char transB,
  const uint16_t* alpha,
  libxs_CSR_sparseslice* A_sparse,
  const uint16_t* B,
  char transC,
  const uint16_t* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_SPMDM_AVX2) && (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
# include "libxs_spmdm_begin_avx2.h"
# include "template/libxs_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
# if !defined(NDEBUG) && defined(LIBXS_SPMDM_AVX2) /* library code is expected to be mute */
  { static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: unable to enter AVX2 code path!\n");
    }
  }
# endif
  internal_spmdm_compute_bfloat16_thread_sw(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXS_INLINE LIBXS_RETARGETABLE LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
void internal_spmdm_compute_bfloat16_thread_avx512_core(
  const libxs_spmdm_handle* handle,
  char transA,
  char transB,
  const uint16_t* alpha,
  libxs_CSR_sparseslice* A_sparse,
  const uint16_t* B,
  char transC,
  const uint16_t* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
#if !defined(LIBXS_INTRINSICS_NONE) && defined(LIBXS_SPMDM_AVX512_CORE) && (LIBXS_X86_AVX512_CORE <= LIBXS_MAX_STATIC_TARGET_ARCH)
# include "libxs_spmdm_begin_avx512.h"
# include "template/libxs_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
# if !defined(NDEBUG) && defined(LIBXS_SPMDM_AVX512_CORE) /* library code is expected to be mute */
  { static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: unable to enter AVX-512/Core code path!\n");
    }
  }
# endif
  internal_spmdm_compute_bfloat16_thread_avx2(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXS_API_DEFINITION
void libxs_spmdm_compute_bfloat16_thread(
  const libxs_spmdm_handle* handle,
  char transA,
  char transB,
  const uint16_t* alpha,
  libxs_CSR_sparseslice* A_sparse,
  const uint16_t* B,
  char transC,
  const uint16_t* beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
#if (LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH)
  internal_spmdm_compute_bfloat16_thread_avx512_core(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  internal_spmdm_compute_bfloat16_thread_avx2(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#else /* pointer based function call */
  assert(0 != internal_spmdm_compute_bfloat16_thread);
  internal_spmdm_compute_bfloat16_thread(handle, transA, transB, alpha, A_sparse, B, transC, beta, C, block_id, tid, nthreads);
#endif
}


LIBXS_API_DEFINITION void libxs_spmdm_init(int M, int N, int K, int max_threads,
  libxs_spmdm_handle* handle, libxs_CSR_sparseslice** libxs_output_csr)
{
  /* initialize internal library structures */
  LIBXS_INIT

  handle->m  = M;
  handle->n  = N;
  handle->k  = K;

  handle->bm = (M >= 4096 || M <= 1024) ? 512 : 256;
  if (LIBXS_X86_AVX512_CORE <= libxs_target_archid) {
    internal_spmdm_createSparseSlice_fp32_thread = internal_spmdm_createSparseSlice_fp32_thread_avx512_core;
    internal_spmdm_createSparseSlice_bfloat16_thread = internal_spmdm_createSparseSlice_bfloat16_thread_avx512_core;
    internal_spmdm_compute_fp32_thread = internal_spmdm_compute_fp32_thread_avx512_core;
    internal_spmdm_compute_bfloat16_thread = internal_spmdm_compute_bfloat16_thread_avx512_core;
    handle->bn = 96;
  }
  else if (LIBXS_X86_AVX2 <= libxs_target_archid) {
    internal_spmdm_createSparseSlice_fp32_thread = internal_spmdm_createSparseSlice_fp32_thread_avx2;
    internal_spmdm_createSparseSlice_bfloat16_thread = internal_spmdm_createSparseSlice_bfloat16_thread_avx2;
    internal_spmdm_compute_fp32_thread = internal_spmdm_compute_fp32_thread_avx2;
    internal_spmdm_compute_bfloat16_thread = internal_spmdm_compute_bfloat16_thread_avx2;
    handle->bn = 48;
  }
  else {
    internal_spmdm_createSparseSlice_fp32_thread = internal_spmdm_createSparseSlice_fp32_thread_sw;
    internal_spmdm_createSparseSlice_bfloat16_thread = internal_spmdm_createSparseSlice_bfloat16_thread_sw;
    internal_spmdm_compute_fp32_thread = internal_spmdm_compute_fp32_thread_sw;
    internal_spmdm_compute_bfloat16_thread = internal_spmdm_compute_bfloat16_thread_sw;
    handle->bn = 6;
  }
  handle->bk = 128;
  handle->mb = (handle->m + handle->bm - 1) / handle->bm;
  handle->nb = (handle->n + handle->bn - 1) / handle->bn;
  handle->kb = (handle->k + handle->bk - 1) / handle->bk;

  /* This is temporary space needed; allocate for each different size of A */
  internal_spmdm_allocate_csr_a(handle, libxs_output_csr);
  internal_spmdm_allocate_scratch(handle, max_threads);

  /* Initialize shuffle masks for the computation */
  if (LIBXS_X86_AVX <= libxs_target_archid) {
    internal_spmdm_init_shufmask_avx();
  }

  /* post-conditions */
  assert(0 != internal_spmdm_createSparseSlice_fp32_thread);
  assert(0 != internal_spmdm_createSparseSlice_bfloat16_thread);
  assert(0 != internal_spmdm_compute_fp32_thread);
  assert(0 != internal_spmdm_compute_bfloat16_thread);
}

