/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_spmdm.h>
#include "libxs_main.h"

/* Enable/disable specific code paths */
#if defined(LIBXS_INTRINSICS_AVX) && !defined(LIBXS_SPMDM_AVX)
# define LIBXS_SPMDM_AVX
#endif
#if defined(LIBXS_INTRINSICS_AVX2) && !defined(LIBXS_SPMDM_AVX2) && \
  !(defined(__PGI) && defined(__cplusplus))
# define LIBXS_SPMDM_AVX2
#endif
#if defined(LIBXS_INTRINSICS_AVX512_CORE) && !defined(LIBXS_SPMDM_AVX512_CORE) && \
  !(defined(__PGI) && defined(__cplusplus))
# define LIBXS_SPMDM_AVX512_CORE
#endif


/* function pointer for the CPUID-dispatched implementation (separate typedef for legacy Cray C++ needed) */
typedef void (*internal_spmdm_createSparseSlice_fp32_thread_fn)(const libxs_spmdm_handle*, char, const float*, libxs_CSR_sparseslice*, int, int, int);
LIBXS_APIVAR_DEFINE(internal_spmdm_createSparseSlice_fp32_thread_fn internal_spmdm_createSparseSlice_fp32_thread);
typedef void (*internal_spmdm_createSparseSlice_bfloat16_thread_fn)(const libxs_spmdm_handle*, char, const libxs_bfloat16*, libxs_CSR_sparseslice*, int, int, int);
LIBXS_APIVAR_DEFINE(internal_spmdm_createSparseSlice_bfloat16_thread_fn internal_spmdm_createSparseSlice_bfloat16_thread);
typedef void (*internal_spmdm_compute_fp32_thread_fn)(const libxs_spmdm_handle*, char, char, const float*, libxs_CSR_sparseslice*, const float*, char, const float*, float*, int, int, int);
LIBXS_APIVAR_DEFINE(internal_spmdm_compute_fp32_thread_fn internal_spmdm_compute_fp32_thread);
typedef void (*internal_spmdm_compute_bfloat16_thread_fn)(const libxs_spmdm_handle*, char, char, const libxs_bfloat16*, libxs_CSR_sparseslice*, const libxs_bfloat16*, char, const libxs_bfloat16*, float*, int, int, int);
LIBXS_APIVAR_DEFINE(internal_spmdm_compute_bfloat16_thread_fn internal_spmdm_compute_bfloat16_thread);

#if defined(LIBXS_SPMDM_AVX)
LIBXS_APIVAR_DEFINE(__m256i* internal_spmdm_shufmasks_32);
LIBXS_APIVAR_DEFINE(__m256i* internal_spmdm_shufmasks_16);
#endif


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX)
LIBXS_ATTRIBUTE_UNUSED void internal_spmdm_init_shufmask_avx(void)
{
#if defined(LIBXS_SPMDM_AVX)
  static __m256i spmdm_shufmasks_32[256], spmdm_shufmasks_16[256];
  LIBXS_ALIGNED(int temp_shufmasks[8], 64);
  LIBXS_ALIGNED(uint16_t temp_shufmasks2[16], 64);
  unsigned int i, j, c, last_bit;
  int cnt;
  for (i = 0; i < 256; i++) {
    cnt = 0;
    j = i;
    for (c = 0; c < 8; c++) temp_shufmasks[c] = 0;
    for (c = 0; c < 16; c++) temp_shufmasks2[c] = 0;
    while (j) {
      last_bit = LIBXS_INTRINSICS_BITSCANFWD32(j);
      temp_shufmasks[cnt] = last_bit;
      temp_shufmasks2[cnt] = (uint16_t)last_bit;
      j &= (~(1<<last_bit));
      cnt++;
    }
    spmdm_shufmasks_32[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks);
    spmdm_shufmasks_16[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks2);
  }
  internal_spmdm_shufmasks_32 = spmdm_shufmasks_32;
  internal_spmdm_shufmasks_16 = spmdm_shufmasks_16;
#endif
}


LIBXS_API_INLINE void internal_spmdm_allocate_csr_a(libxs_spmdm_handle* handle, libxs_CSR_sparseslice** libxs_output_csr)
{
  int kb, mb;
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;

  const size_t sz_block = (((size_t)handle->bm + 1) * sizeof(uint16_t)
    + (size_t)handle->bm * handle->bk * sizeof(uint16_t)
    + (size_t)handle->bm * handle->bk * sizeof(float)
    + sizeof(libxs_CSR_sparseslice));
  size_t sz_all_blocks = sz_block * handle->mb * handle->kb;
  char* memory_block = 0;
  void *const pv = &memory_block;

  /* use low-level scratch memory allocation since life-time of this buffer is unknown */
  if (EXIT_SUCCESS == libxs_xmalloc((void**)pv, sz_all_blocks, 2097152,
    LIBXS_MALLOC_FLAG_SCRATCH | LIBXS_MALLOC_FLAG_PRIVATE, 0/*extra*/, 0/*extra_size*/))
  {
    char* memory_head  = memory_block;
    libxs_CSR_sparseslice* libxs_output_csr_a = (libxs_CSR_sparseslice*)(memory_head);
    memory_head += (size_t)handle->mb * handle->kb * sizeof(libxs_CSR_sparseslice);
    LIBXS_ASSERT(0 != libxs_output_csr_a/*sanity check*/);

    for (kb = 0; kb < k_blocks; kb++) {
      for (mb = 0; mb < m_blocks; mb++) {
        int i = kb*m_blocks + mb;
        libxs_output_csr_a[i].rowidx = (uint16_t*)(memory_head);
        memory_head += ((size_t)handle->bm + 1) * sizeof(uint16_t);
        libxs_output_csr_a[i].colidx = (uint16_t*)(memory_head);
        memory_head += (size_t)handle->bm * handle->bk * sizeof(uint16_t);
        libxs_output_csr_a[i].values = (float*)(memory_head);
        memory_head += (size_t)handle->bm * handle->bk * sizeof(float);
      }
    }
    LIBXS_ASSERT(memory_head == (memory_block + sz_all_blocks));
    *libxs_output_csr = libxs_output_csr_a;
  }
  else if (0 != libxs_verbosity) { /* library code is expected to be mute */
    fprintf(stderr, "LIBXS ERROR: SPMDM CSR scratch memory allocation failed!\n");
  }

  handle->base_ptr_scratch_A = memory_block;
}


LIBXS_API_INLINE void internal_spmdm_allocate_scratch(libxs_spmdm_handle* handle, int max_threads)
{
  void *const pv = &handle->base_ptr_scratch_B_scratch_C;
  size_t sz_total_memory, sz_memory_for_scratch_per_thread =
    (size_t)handle->bm * handle->bn * sizeof(float) +
    (size_t)handle->bk * handle->bn * sizeof(float);
  sz_memory_for_scratch_per_thread = LIBXS_UP2(sz_memory_for_scratch_per_thread, 4096);
  sz_total_memory = sz_memory_for_scratch_per_thread * max_threads;
  handle->base_ptr_scratch_B_scratch_C = 0;

  /* use low-level scratch memory allocation since life-time of this buffer is unknown */
  if (EXIT_SUCCESS == libxs_xmalloc((void**)pv, sz_total_memory, 2097152,
    LIBXS_MALLOC_FLAG_SCRATCH | LIBXS_MALLOC_FLAG_PRIVATE, 0/*extra*/, 0/*extra_size*/))
  {
    handle->memory_for_scratch_per_thread = (int)sz_memory_for_scratch_per_thread;
  }
  else {
    if (0 != libxs_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXS ERROR: SPMDM scratch memory allocation failed!\n");
    }
    handle->memory_for_scratch_per_thread = 0;
  }
}


LIBXS_API_INLINE void internal_spmdm_deallocate_csr_a(libxs_spmdm_handle* handle)
{
  libxs_xfree(handle->base_ptr_scratch_A, 0/*no check*/);
  handle->base_ptr_scratch_A = NULL;
  libxs_xfree(handle->base_ptr_scratch_B_scratch_C, 0/*no check*/);
  handle->base_ptr_scratch_B_scratch_C = NULL;
}


LIBXS_API void libxs_spmdm_destroy(libxs_spmdm_handle* handle)
{
  internal_spmdm_deallocate_csr_a(handle);
}


LIBXS_API int libxs_spmdm_get_num_createSparseSlice_blocks(const libxs_spmdm_handle* handle)
{
  return handle->mb * handle->kb;
}


LIBXS_API int libxs_spmdm_get_num_compute_blocks(const libxs_spmdm_handle* handle)
{
  return handle->mb * handle->nb;
}


LIBXS_API_INLINE
void internal_spmdm_createSparseSlice_fp32_thread_sw(
  const libxs_spmdm_handle* handle,
  char transa,
  const float* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
# include "libxs_spmdm_begin.h"
# include "template/libxs_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
LIBXS_ATTRIBUTE_UNUSED void internal_spmdm_createSparseSlice_fp32_thread_avx2(
  const libxs_spmdm_handle* handle,
  char transa,
  const float* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXS_SPMDM_AVX2)
# include "libxs_spmdm_begin_avx2.h"
# include "template/libxs_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
  internal_spmdm_createSparseSlice_fp32_thread_sw(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


#if defined(LIBXS_SPMDM_AVX512_CORE)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
LIBXS_ATTRIBUTE_UNUSED void internal_spmdm_createSparseSlice_fp32_thread_avx512_core(
  const libxs_spmdm_handle* handle,
  char transa,
  const float* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXS_SPMDM_AVX512_CORE)
# include "libxs_spmdm_begin_avx512.h"
# include "template/libxs_spmdm_createSparseSlice_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
  internal_spmdm_createSparseSlice_fp32_thread_avx2(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}
#endif


LIBXS_API
void libxs_spmdm_createSparseSlice_fp32_thread(
  const libxs_spmdm_handle* handle,
  char transa,
  const float* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
  /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH) && defined(LIBXS_SPMDM_AVX512_CORE)
  internal_spmdm_createSparseSlice_fp32_thread_avx512_core(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH) && /* no need for an indirect call */ \
      (LIBXS_X86_AVX512_CORE > LIBXS_MAX_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_fp32_thread_avx2(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#else /* pointer based function call */
  LIBXS_ASSERT(0 != internal_spmdm_createSparseSlice_fp32_thread);
  internal_spmdm_createSparseSlice_fp32_thread(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXS_API_INLINE
void internal_spmdm_createSparseSlice_bfloat16_thread_sw(
  const libxs_spmdm_handle* handle,
  char transa,
  const libxs_bfloat16* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
# include "libxs_spmdm_begin.h"
# include "template/libxs_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
LIBXS_ATTRIBUTE_UNUSED void internal_spmdm_createSparseSlice_bfloat16_thread_avx2(
  const libxs_spmdm_handle* handle,
  char transa,
  const libxs_bfloat16* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXS_SPMDM_AVX2)
# include "libxs_spmdm_begin_avx2.h"
# include "template/libxs_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
  internal_spmdm_createSparseSlice_bfloat16_thread_sw(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


#if defined(LIBXS_SPMDM_AVX512_CORE)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
LIBXS_ATTRIBUTE_UNUSED void internal_spmdm_createSparseSlice_bfloat16_thread_avx512_core(
  const libxs_spmdm_handle* handle,
  char transa,
  const libxs_bfloat16* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXS_SPMDM_AVX512_CORE)
# include "libxs_spmdm_begin_avx512.h"
# include "template/libxs_spmdm_createSparseSlice_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
  internal_spmdm_createSparseSlice_bfloat16_thread_avx2(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}
#endif


LIBXS_API
void libxs_spmdm_createSparseSlice_bfloat16_thread(
  const libxs_spmdm_handle* handle,
  char transa,
  const libxs_bfloat16* a,
  libxs_CSR_sparseslice* libxs_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
  /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH) && defined(LIBXS_SPMDM_AVX512_CORE)
  internal_spmdm_createSparseSlice_bfloat16_thread_avx512_core(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH) && /* no need for an indirect call */ \
      (LIBXS_X86_AVX512_CORE > LIBXS_MAX_STATIC_TARGET_ARCH)
  internal_spmdm_createSparseSlice_bfloat16_thread_avx2(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#else /* pointer based function call */
  LIBXS_ASSERT(0 != internal_spmdm_createSparseSlice_fp32_thread);
  internal_spmdm_createSparseSlice_bfloat16_thread(handle, transa, a, libxs_output_csr_a, block_id, tid, nthreads);
#endif
}


LIBXS_API_INLINE
void internal_spmdm_compute_fp32_thread_sw(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const float* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const float* b,
  char transc,
  const float* beta,
  float* c,
  int block_id,
  int tid, int nthreads)
{
# include "libxs_spmdm_begin.h"
# include "template/libxs_spmdm_compute_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
LIBXS_ATTRIBUTE_UNUSED void internal_spmdm_compute_fp32_thread_avx2(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const float* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const float* b,
  char transc,
  const float* beta,
  float* c,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXS_SPMDM_AVX2)
# include "libxs_spmdm_begin_avx2.h"
# include "template/libxs_spmdm_compute_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
  internal_spmdm_compute_fp32_thread_sw(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#endif
}


#if defined(LIBXS_SPMDM_AVX512_CORE)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
LIBXS_ATTRIBUTE_UNUSED void internal_spmdm_compute_fp32_thread_avx512_core(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const float* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const float* b,
  char transc,
  const float* beta,
  float* c,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXS_SPMDM_AVX512_CORE)
# include "libxs_spmdm_begin_avx512.h"
# include "template/libxs_spmdm_compute_fp32_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
  internal_spmdm_compute_fp32_thread_avx2(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#endif
}
#endif


LIBXS_API
void libxs_spmdm_compute_fp32_thread(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const float* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const float* b,
  char transc,
  const float* beta,
  float* c,
  int block_id,
  int tid, int nthreads)
{
  /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH) && defined(LIBXS_SPMDM_AVX512_CORE)
  internal_spmdm_compute_fp32_thread_avx512_core(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH) && /* no need for an indirect call */ \
      (LIBXS_X86_AVX512_CORE > LIBXS_MAX_STATIC_TARGET_ARCH)
  internal_spmdm_compute_fp32_thread_avx2(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#else /* pointer based function call */
  LIBXS_ASSERT(0 != internal_spmdm_compute_fp32_thread);
  internal_spmdm_compute_fp32_thread(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#endif
}


LIBXS_API_INLINE
void internal_spmdm_compute_bfloat16_thread_sw(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const libxs_bfloat16* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const libxs_bfloat16* b,
  char transc,
  const libxs_bfloat16* beta,
  float* c,
  int block_id,
  int tid, int nthreads)
{
# include "libxs_spmdm_begin.h"
# include "template/libxs_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
LIBXS_ATTRIBUTE_UNUSED void internal_spmdm_compute_bfloat16_thread_avx2(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const libxs_bfloat16* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const libxs_bfloat16* b,
  char transc,
  const libxs_bfloat16* beta,
  float* c,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXS_SPMDM_AVX2)
# include "libxs_spmdm_begin_avx2.h"
# include "template/libxs_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
  internal_spmdm_compute_bfloat16_thread_sw(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#endif
}


#if defined(LIBXS_SPMDM_AVX512_CORE)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
LIBXS_ATTRIBUTE_UNUSED void internal_spmdm_compute_bfloat16_thread_avx512_core(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const libxs_bfloat16* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const libxs_bfloat16* b,
  char transc,
  const libxs_bfloat16* beta,
  float* c,
  int block_id,
  int tid, int nthreads)
{
#if defined(LIBXS_SPMDM_AVX512_CORE)
# include "libxs_spmdm_begin_avx512.h"
# include "template/libxs_spmdm_compute_bfloat16_thread.tpl.c"
# include "libxs_spmdm_end.h"
#else
  internal_spmdm_compute_bfloat16_thread_avx2(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#endif
}
#endif


LIBXS_API
void libxs_spmdm_compute_bfloat16_thread(
  const libxs_spmdm_handle* handle,
  char transa,
  char transb,
  const libxs_bfloat16* alpha,
  libxs_CSR_sparseslice* a_sparse,
  const libxs_bfloat16* b,
  char transc,
  const libxs_bfloat16* beta,
  float* c,
  int block_id,
  int tid, int nthreads)
{
  /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH) && defined(LIBXS_SPMDM_AVX512_CORE)
  internal_spmdm_compute_bfloat16_thread_avx512_core(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH) && /* no need for an indirect call */ \
      (LIBXS_X86_AVX512_CORE > LIBXS_MAX_STATIC_TARGET_ARCH)
  internal_spmdm_compute_bfloat16_thread_avx2(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#else /* pointer based function call */
  LIBXS_ASSERT(0 != internal_spmdm_compute_bfloat16_thread);
  internal_spmdm_compute_bfloat16_thread(handle, transa, transb, alpha, a_sparse, b, transc, beta, c, block_id, tid, nthreads);
#endif
}


LIBXS_API void libxs_spmdm_init(int M, int N, int K, int max_threads,
  libxs_spmdm_handle* handle, libxs_CSR_sparseslice** libxs_output_csr)
{
  double load_imbalance_tolerate = 1.1;
  int max_work_per_block;
  double avg_work_per_block;
  int max_blocks_per_thread;
  double avg_blocks_per_thread;
  double load_imbalance_1, load_imbalance_2, load_imbalance;

  libxs_init(); /* !LIBXS_INIT */
  { unsigned int dummy =
    LIBXS_ATOMIC_ADD_FETCH(&libxs_statistic_num_spmdm, 1,
      LIBXS_ATOMIC_RELAXED); /* count number of invocations */
    LIBXS_UNUSED(dummy);
  }

  handle->m  = M;
  handle->n  = N;
  handle->k  = K;
  handle->bm = (M >= 4096 || M <= 1024) ? 512 : 256;

#if defined(LIBXS_SPMDM_AVX512_CORE)
  if (LIBXS_X86_AVX512_CORE <= libxs_target_archid || LIBXS_X86_AVX512_CORE <= LIBXS_STATIC_TARGET_ARCH) {
    internal_spmdm_createSparseSlice_fp32_thread = internal_spmdm_createSparseSlice_fp32_thread_avx512_core;
    internal_spmdm_createSparseSlice_bfloat16_thread = internal_spmdm_createSparseSlice_bfloat16_thread_avx512_core;
    internal_spmdm_compute_fp32_thread = internal_spmdm_compute_fp32_thread_avx512_core;
    internal_spmdm_compute_bfloat16_thread = internal_spmdm_compute_bfloat16_thread_avx512_core;
    handle->bn = 96;
  }
  else
#endif
#if defined(LIBXS_SPMDM_AVX2)
  if (LIBXS_X86_AVX2 <= libxs_target_archid || LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH) {
    internal_spmdm_createSparseSlice_fp32_thread = internal_spmdm_createSparseSlice_fp32_thread_avx2;
    internal_spmdm_createSparseSlice_bfloat16_thread = internal_spmdm_createSparseSlice_bfloat16_thread_avx2;
    internal_spmdm_compute_fp32_thread = internal_spmdm_compute_fp32_thread_avx2;
    internal_spmdm_compute_bfloat16_thread = internal_spmdm_compute_bfloat16_thread_avx2;
    handle->bn = 48;
  }
  else
#endif
  {
    internal_spmdm_createSparseSlice_fp32_thread = internal_spmdm_createSparseSlice_fp32_thread_sw;
    internal_spmdm_createSparseSlice_bfloat16_thread = internal_spmdm_createSparseSlice_bfloat16_thread_sw;
    internal_spmdm_compute_fp32_thread = internal_spmdm_compute_fp32_thread_sw;
    internal_spmdm_compute_bfloat16_thread = internal_spmdm_compute_bfloat16_thread_sw;
    handle->bn = 6;
  }
  handle->bk = 128;
  handle->mb = LIBXS_UPDIV(handle->m, handle->bm);
  handle->nb = LIBXS_UPDIV(handle->n, handle->bn);
  handle->kb = LIBXS_UPDIV(handle->k, handle->bk);

  max_work_per_block    = handle->bm * handle->bn;
  avg_work_per_block    = (double)((size_t)handle->m * handle->n) / ((size_t)handle->mb * handle->nb);
  load_imbalance_1      = max_work_per_block / avg_work_per_block;
  max_blocks_per_thread = LIBXS_UPDIV(handle->mb * handle->nb, max_threads);
  avg_blocks_per_thread = (double)handle->mb * handle->nb / max_threads;
  load_imbalance_2      = max_blocks_per_thread / avg_blocks_per_thread;
  load_imbalance        = load_imbalance_1 * load_imbalance_2;

  while (32 < handle->bm && load_imbalance > load_imbalance_tolerate) {
    handle->bm--;
    handle->mb = LIBXS_UPDIV(handle->m, handle->bm);

    max_blocks_per_thread = LIBXS_UPDIV(handle->mb * handle->nb, max_threads);
    avg_blocks_per_thread = (double)handle->mb * handle->nb / max_threads;
    load_imbalance_2      = max_blocks_per_thread / avg_blocks_per_thread;
    max_work_per_block    = handle->bm * handle->bn;
    avg_work_per_block    = (double)((size_t)handle->m * handle->n) / ((size_t)handle->mb * handle->nb);
    load_imbalance_1      = max_work_per_block / avg_work_per_block;
    load_imbalance        = load_imbalance_1 * load_imbalance_2;
  }

  /* This is temporary space needed; allocate for each different size of a */
  internal_spmdm_allocate_csr_a(handle, libxs_output_csr);
  internal_spmdm_allocate_scratch(handle, max_threads);

  /* Initialize shuffle masks for the computation */
#if defined(LIBXS_SPMDM_AVX)
  if (LIBXS_X86_AVX <= libxs_target_archid || LIBXS_X86_AVX <= LIBXS_STATIC_TARGET_ARCH) {
    internal_spmdm_init_shufmask_avx();
    LIBXS_ASSERT(0 != internal_spmdm_shufmasks_32);
    LIBXS_ASSERT(0 != internal_spmdm_shufmasks_16);
  }
#endif
  /* post-conditions */
  LIBXS_ASSERT(0 != internal_spmdm_createSparseSlice_fp32_thread);
  LIBXS_ASSERT(0 != internal_spmdm_createSparseSlice_bfloat16_thread);
  LIBXS_ASSERT(0 != internal_spmdm_compute_fp32_thread);
  LIBXS_ASSERT(0 != internal_spmdm_compute_bfloat16_thread);
}

