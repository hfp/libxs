/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_mem.h>
#include "libxs_main.h"
#include <libxs_malloc.h>
#include <libxs_math.h>
#include "libxs_hash.h"
#include "libxs_diff.h"

#include <ctype.h>

#if !defined(LIBXS_MEM_STDLIB) && 0
# define LIBXS_MEM_STDLIB
#endif
#if !defined(LIBXS_MEM_SW) && 0
# define LIBXS_MEM_SW
#endif

#define LIBXS_MEM_SHUFFLE_COPRIME(N) libxs_coprime2(N)
#define LIBXS_MEM_SHUFFLE(INOUT, ELEMSIZE, COUNT, SHUFFLE, NREPEAT) do { \
  unsigned char *const LIBXS_RESTRICT data = (unsigned char*)(INOUT); \
  const size_t c = (COUNT) - 1, c2 = ((COUNT) + 1) / 2; \
  size_t i; \
  for (i = (0 != (NREPEAT) ? 0 : (COUNT)); i < (COUNT); ++i) { \
    size_t j = i, k = 0; \
    for (; k < (NREPEAT) || j < i; ++k) j = ((SHUFFLE) * j) % (COUNT); \
    if (i < j) LIBXS_MEMSWP( \
      data + (ELEMSIZE) * (c - j), \
      data + (ELEMSIZE) * (c - i), \
      ELEMSIZE); \
    if (c2 <= i) LIBXS_MEMSWP( \
      data + (ELEMSIZE) * (c - i), \
      data + (ELEMSIZE) * i, \
      ELEMSIZE); \
  } \
} while(0)

/* xcopy kernel: consecutive loads and stores (matcopy) */
#define LIBXS_MCOPY_KERNEL(TYPE, TS, OUT, IN, LDI, LDO, I, J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*)(IN)) \
    + (size_t)(TS) * ((size_t)(I) * (LDI) + (J))); \
  TYPE *const DST = (TYPE*)(((char*)(OUT)) \
    + (size_t)(TS) * ((size_t)(I) * (LDO) + (J)))
/* xcopy kernel: zero stores (matzero) */
#define LIBXS_MZERO_KERNEL(TYPE, TS, OUT, IN, LDI, LDO, I, J, SRC, DST) \
  static const double libxs_mzero_zero_[32]; \
  const TYPE *const SRC = (const TYPE*)libxs_mzero_zero_; \
  TYPE *const DST = (TYPE*)(((char*)(OUT)) \
    + (size_t)(TS) * ((size_t)(I) * (LDO) + (J)))
/* xcopy kernel: strided loads, consecutive stores (transpose) */
#define LIBXS_TCOPY_KERNEL(TYPE, TS, OUT, IN, LDI, LDO, I, J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*)(IN)) \
    + (size_t)(TS) * ((size_t)(J) * (LDI) + (I))); \
  TYPE *const DST = (TYPE*)(((char*)(OUT)) \
    + (size_t)(TS) * ((size_t)(I) * (LDO) + (J)))

/* typed double loop: outer [M0,M1), inner [N0,N1) */
#define LIBXS_XCOPY_LOOP(TYPE, TS, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1) do { \
  unsigned int libxs_xcopy_loop_i_, libxs_xcopy_loop_j_; \
  for (libxs_xcopy_loop_i_ = (M0); libxs_xcopy_loop_i_ < (unsigned int)(M1); \
    ++libxs_xcopy_loop_i_) \
  { \
    LIBXS_PRAGMA_NONTEMPORAL(OUT) \
    for (libxs_xcopy_loop_j_ = (N0); libxs_xcopy_loop_j_ < (unsigned int)(N1); \
      ++libxs_xcopy_loop_j_) \
    { \
      XKERNEL(TYPE, TS, OUT, IN, LDI, LDO, libxs_xcopy_loop_i_, libxs_xcopy_loop_j_, \
        libxs_xcopy_loop_src_, libxs_xcopy_loop_dst_); \
      *libxs_xcopy_loop_dst_ = *libxs_xcopy_loop_src_; \
    } \
  } \
} while(0)

/* typesize-specialized tile: switches on TS, falls back to byte loop */
#define LIBXS_XCOPY_TILE(XKERNEL, TS, OUT, IN, LDI, LDO, M0, M1, N0, N1) do { \
  switch(TS) { \
    case 1: { \
      LIBXS_XCOPY_LOOP(char, 1, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 2: { \
      LIBXS_XCOPY_LOOP(short, 2, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 4: { \
      LIBXS_XCOPY_LOOP(float, 4, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    case 8: { \
      LIBXS_XCOPY_LOOP(double, 8, XKERNEL, OUT, IN, LDI, LDO, M0, M1, N0, N1); \
    } break; \
    default: { \
      unsigned int libxs_xcopy_tile_i_, libxs_xcopy_tile_j_, libxs_xcopy_tile_k_; \
      for (libxs_xcopy_tile_i_ = (M0); libxs_xcopy_tile_i_ < (unsigned int)(M1); \
        ++libxs_xcopy_tile_i_) \
      { \
        for (libxs_xcopy_tile_j_ = (N0); libxs_xcopy_tile_j_ < (unsigned int)(N1); \
          ++libxs_xcopy_tile_j_) \
        { \
          XKERNEL(char, TS, OUT, IN, LDI, LDO, libxs_xcopy_tile_i_, libxs_xcopy_tile_j_, \
            libxs_xcopy_tile_src_, libxs_xcopy_tile_dst_); \
          LIBXS_PRAGMA_UNROLL \
          for (libxs_xcopy_tile_k_ = 0; libxs_xcopy_tile_k_ < (unsigned int)(TS); \
            ++libxs_xcopy_tile_k_) \
          { \
            libxs_xcopy_tile_dst_[libxs_xcopy_tile_k_] = \
              libxs_xcopy_tile_src_[libxs_xcopy_tile_k_]; \
          } \
        } \
      } \
    } break; \
  } \
} while(0)

/* matcopy tile: outer=N, inner=M for consecutive stores */
#define LIBXS_MCOPY_TILE(TS, OUT, IN, LDI, LDO, M0, M1, N0, N1) \
  LIBXS_XCOPY_TILE(LIBXS_MCOPY_KERNEL, TS, OUT, IN, LDI, LDO, N0, N1, M0, M1)
/* matzero tile: outer=N, inner=M for consecutive stores */
#define LIBXS_MZERO_TILE(TS, OUT, LDO, M0, M1, N0, N1) \
  LIBXS_XCOPY_TILE(LIBXS_MZERO_KERNEL, TS, OUT, NULL, 0, LDO, N0, N1, M0, M1)
/* transpose tile: outer=M, inner=N for consecutive stores */
#define LIBXS_TCOPY_TILE(TS, OUT, IN, LDI, LDO, M0, M1, N0, N1) \
  LIBXS_XCOPY_TILE(LIBXS_TCOPY_KERNEL, TS, OUT, IN, LDI, LDO, M0, M1, N0, N1)

/* in-place transpose of square region (typed swap) */
#define LIBXS_ITRANS_LOOP(TYPE, INOUT, LD, M) do { \
  unsigned int libxs_itrans_loop_i_, libxs_itrans_loop_j_; \
  for (libxs_itrans_loop_i_ = 0; libxs_itrans_loop_i_ < (unsigned int)(M); \
    ++libxs_itrans_loop_i_) \
  { \
    for (libxs_itrans_loop_j_ = 0; libxs_itrans_loop_j_ < libxs_itrans_loop_i_; \
      ++libxs_itrans_loop_j_) \
    { \
      TYPE *const libxs_itrans_loop_a_ = ((TYPE*)(INOUT)) \
        + (size_t)(LD) * libxs_itrans_loop_i_ + libxs_itrans_loop_j_; \
      TYPE *const libxs_itrans_loop_b_ = ((TYPE*)(INOUT)) \
        + (size_t)(LD) * libxs_itrans_loop_j_ + libxs_itrans_loop_i_; \
      LIBXS_ISWAP(*libxs_itrans_loop_a_, *libxs_itrans_loop_b_); \
    } \
  } \
} while(0)

#define LIBXS_ITRANS_TILE(TS, INOUT, LD, M) do { \
  switch(TS) { \
    case 1: { LIBXS_ITRANS_LOOP(char, INOUT, LD, M); } break; \
    case 2: { LIBXS_ITRANS_LOOP(short, INOUT, LD, M); } break; \
    case 4: { LIBXS_ITRANS_LOOP(int, INOUT, LD, M); } break; \
    case 8: { LIBXS_ITRANS_LOOP(int64_t, INOUT, LD, M); } break; \
    default: { \
      unsigned int libxs_itrans_tile_i_, libxs_itrans_tile_j_, libxs_itrans_tile_k_; \
      for (libxs_itrans_tile_i_ = 0; libxs_itrans_tile_i_ < (unsigned int)(M); \
        ++libxs_itrans_tile_i_) \
      { \
        for (libxs_itrans_tile_j_ = 0; libxs_itrans_tile_j_ < libxs_itrans_tile_i_; \
          ++libxs_itrans_tile_j_) \
        { \
          char *const libxs_itrans_tile_a_ = ((char*)(INOUT)) \
            + (size_t)(TS) * ((size_t)(LD) * libxs_itrans_tile_i_ + libxs_itrans_tile_j_); \
          char *const libxs_itrans_tile_b_ = ((char*)(INOUT)) \
            + (size_t)(TS) * ((size_t)(LD) * libxs_itrans_tile_j_ + libxs_itrans_tile_i_); \
          LIBXS_PRAGMA_UNROLL \
          for (libxs_itrans_tile_k_ = 0; libxs_itrans_tile_k_ < (unsigned int)(TS); \
            ++libxs_itrans_tile_k_) \
          { \
            LIBXS_ISWAP(libxs_itrans_tile_a_[libxs_itrans_tile_k_], \
              libxs_itrans_tile_b_[libxs_itrans_tile_k_]); \
          } \
        } \
      } \
    } break; \
  } \
} while(0)

/* typed swap of triangle range [BEGIN,END) for in-place square transpose */
#define LIBXS_ITRANS_RANGE_LOOP(TYPE, INOUT, LD, BEGIN, END, ROW, COL) do { \
  unsigned int libxs_itrans_range_idx_; \
  for (libxs_itrans_range_idx_ = (BEGIN); libxs_itrans_range_idx_ < (END); \
    ++libxs_itrans_range_idx_) \
  { \
    TYPE *const libxs_itrans_range_a_ = ((TYPE*)(INOUT)) \
      + (size_t)(LD) * (ROW) + (COL); \
    TYPE *const libxs_itrans_range_b_ = ((TYPE*)(INOUT)) \
      + (size_t)(LD) * (COL) + (ROW); \
    LIBXS_ISWAP(*libxs_itrans_range_a_, *libxs_itrans_range_b_); \
    if (++(COL) >= (ROW)) { ++(ROW); (COL) = 0; } \
  } \
} while(0)

#define LIBXS_ITRANS_RANGE(TS, INOUT, LD, BEGIN, END, ROW, COL) do { \
  switch(TS) { \
    case 1: { LIBXS_ITRANS_RANGE_LOOP(char, INOUT, LD, BEGIN, END, ROW, COL); } break; \
    case 2: { LIBXS_ITRANS_RANGE_LOOP(short, INOUT, LD, BEGIN, END, ROW, COL); } break; \
    case 4: { LIBXS_ITRANS_RANGE_LOOP(int, INOUT, LD, BEGIN, END, ROW, COL); } break; \
    case 8: { LIBXS_ITRANS_RANGE_LOOP(int64_t, INOUT, LD, BEGIN, END, ROW, COL); } break; \
    default: { \
      unsigned int libxs_itrans_range_idx_, libxs_itrans_range_k_; \
      for (libxs_itrans_range_idx_ = (BEGIN); libxs_itrans_range_idx_ < (END); \
        ++libxs_itrans_range_idx_) \
      { \
        char *const libxs_itrans_range_a_ = ((char*)(INOUT)) \
          + (size_t)(TS) * ((size_t)(LD) * (ROW) + (COL)); \
        char *const libxs_itrans_range_b_ = ((char*)(INOUT)) \
          + (size_t)(TS) * ((size_t)(LD) * (COL) + (ROW)); \
        LIBXS_PRAGMA_UNROLL \
        for (libxs_itrans_range_k_ = 0; libxs_itrans_range_k_ < (unsigned int)(TS); \
          ++libxs_itrans_range_k_) \
        { \
          LIBXS_ISWAP(libxs_itrans_range_a_[libxs_itrans_range_k_], \
            libxs_itrans_range_b_[libxs_itrans_range_k_]); \
        } \
        if (++(COL) >= (ROW)) { ++(ROW); (COL) = 0; } \
      } \
    } break; \
  } \
} while(0)

/* 2D task partitioning over M and N */
#define LIBXS_XCOPY_TASKS(UM, UN, TID, NTASKS, M0, M1, N0, N1) do { \
  const int libxs_xcopy_tasks_nm_ = (int)(UM); \
  if ((NTASKS) <= libxs_xcopy_tasks_nm_) { \
    const unsigned int libxs_xcopy_tasks_mt_ = LIBXS_UPDIV(UM, (unsigned int)(NTASKS)); \
    (M0) = LIBXS_MIN((unsigned int)(TID) * libxs_xcopy_tasks_mt_, (UM)); \
    (M1) = LIBXS_MIN((M0) + libxs_xcopy_tasks_mt_, (UM)); \
    (N0) = 0; (N1) = (UN); \
  } \
  else { \
    const int libxs_xcopy_tasks_nn_ = (NTASKS) / libxs_xcopy_tasks_nm_; \
    const int libxs_xcopy_tasks_mt_ = (TID) / libxs_xcopy_tasks_nn_; \
    const int libxs_xcopy_tasks_nt_ = (TID) - libxs_xcopy_tasks_mt_ * libxs_xcopy_tasks_nn_; \
    const unsigned int libxs_xcopy_tasks_ns_ = \
      LIBXS_UPDIV(UN, (unsigned int)libxs_xcopy_tasks_nn_); \
    (M0) = LIBXS_MIN((unsigned int)libxs_xcopy_tasks_mt_, (UM)); \
    (M1) = LIBXS_MIN((M0) + 1, (UM)); \
    (N0) = LIBXS_MIN((unsigned int)libxs_xcopy_tasks_nt_ * libxs_xcopy_tasks_ns_, (UN)); \
    (N1) = LIBXS_MIN((N0) + libxs_xcopy_tasks_ns_, (UN)); \
  } \
} while(0)


#if !defined(LIBXS_MEM_SW)
LIBXS_APIVAR_DEFINE(unsigned char (*internal_libxs_diff_function)(const void*, const void*, unsigned char));
LIBXS_APIVAR_DEFINE(int (*internal_libxs_memcmp_function)(const void*, const void*, size_t));
LIBXS_APIVAR_DEFINE(void (*internal_libxs_mcopy_tile_function)(void*, const void*, unsigned int,
  unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int));
LIBXS_APIVAR_DEFINE(void (*internal_libxs_tcopy_tile_function)(void*, const void*, unsigned int,
  unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int));
#endif


LIBXS_API size_t libxs_offset(size_t ndims, const size_t offset[], const size_t shape[], size_t* size)
{
  size_t result = 0, size1 = 0;
  if (0 != ndims && NULL != shape) {
    size_t i;
    result = (NULL != offset ? offset[0] : 0);
    size1 = shape[0];
    for (i = 1; i < ndims; ++i) {
      result += ((NULL != offset && 0 != offset[i]) ? (offset[i] - 1) : 0) * size1;
      size1 *= shape[i];
    }
  }
  if (NULL != size) *size = size1;
  return result;
}


LIBXS_API int libxs_aligned(const void* ptr, const size_t* inc, int* alignment)
{
  const int minalign = libxs_cpuid_vlen(libxs_cpuid(NULL));
  const uintptr_t address = (uintptr_t)ptr;
  int ptr_is_aligned;
  LIBXS_ASSERT(LIBXS_ISPOT(minalign));
  if (NULL == alignment) {
    ptr_is_aligned = !LIBXS_MOD2(address, (uintptr_t)minalign);
  }
  else {
    const unsigned int nbits = LIBXS_INTRINSICS_BITSCANFWD64(address);
    *alignment = (32 > nbits ? (1 << nbits) : INT_MAX);
    ptr_is_aligned = (minalign <= *alignment);
  }
  return ptr_is_aligned && (NULL == inc || !LIBXS_MOD2(*inc, (size_t)minalign));
}


LIBXS_API_INLINE
unsigned char internal_libxs_diff_sw(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_MEM_STDLIB) && defined(LIBXS_MEM_SW)
  return (unsigned char)memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xF0); i += 16) {
    LIBXS_DIFF_16_DECL(aa);
    LIBXS_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_16(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
unsigned char internal_libxs_diff_sse(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_X86) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xF0); i += 16) {
    LIBXS_DIFF_SSE_DECL(aa);
    LIBXS_DIFF_SSE_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_SSE(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_libxs_diff_sw(a, b, size);
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
unsigned char internal_libxs_diff_avx2(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_AVX2) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xE0); i += 32) {
    LIBXS_DIFF_AVX2_DECL(aa);
    LIBXS_DIFF_AVX2_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX2(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_libxs_diff_sw(a, b, size);
#endif
}


#if defined(LIBXS_DIFF_AVX512_ENABLED)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
unsigned char internal_libxs_diff_avx512(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_AVX512) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xC0); i += 64) {
    LIBXS_DIFF_AVX512_DECL(aa);
    LIBXS_DIFF_AVX512_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX512(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_libxs_diff_sw(a, b, size);
#endif
}
#endif


LIBXS_API_INLINE
int internal_libxs_memcmp_sw(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_MEM_STDLIB)
  return memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_16_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & ~(size_t)0xF); i += 16) {
    LIBXS_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_16(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
int internal_libxs_memcmp_sse(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_X86) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_SSE_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & ~(size_t)0xF); i += 16) {
    LIBXS_DIFF_SSE_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_SSE(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_libxs_memcmp_sw(a, b, size);
#endif
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
int internal_libxs_memcmp_avx2(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_AVX2) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_AVX2_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & ~(size_t)0x1F); i += 32) {
    LIBXS_DIFF_AVX2_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX2(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_libxs_memcmp_sw(a, b, size);
#endif
}


#if defined(LIBXS_DIFF_AVX512_ENABLED)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int internal_libxs_memcmp_avx512(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_AVX512) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXS_DIFF_AVX512_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & ~(size_t)0x3F); i += 64) {
    LIBXS_DIFF_AVX512_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX512(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_libxs_memcmp_sw(a, b, size);
#endif
}
#endif


LIBXS_API_INLINE void internal_libxs_mcopy_tile_sw(
  void* out, const void* in, unsigned int typesize,
  unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1)
{
  if (NULL != in) {
    LIBXS_MCOPY_TILE(typesize, out, in, ldi, ldo, m0, m1, n0, n1);
  }
  else {
    LIBXS_MZERO_TILE(typesize, out, ldo, m0, m1, n0, n1);
  }
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_mcopy_tile_avx2(
  void* out, const void* in, unsigned int typesize,
  unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1)
{
#if defined(LIBXS_INTRINSICS_AVX2)
  if (NULL != in) {
    LIBXS_MCOPY_TILE(typesize, out, in, ldi, ldo, m0, m1, n0, n1);
  }
  else {
    LIBXS_MZERO_TILE(typesize, out, ldo, m0, m1, n0, n1);
  }
#else
  internal_libxs_mcopy_tile_sw(out, in, typesize, ldi, ldo, m0, m1, n0, n1);
#endif
}


LIBXS_API_INLINE void internal_libxs_tcopy_tile_sw(
  void* out, const void* in, unsigned int typesize,
  unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1)
{
  LIBXS_TCOPY_TILE(typesize, out, in, ldi, ldo, m0, m1, n0, n1);
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_tcopy_tile_avx2(
  void* out, const void* in, unsigned int typesize,
  unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1)
{
#if defined(LIBXS_INTRINSICS_AVX2)
  LIBXS_TCOPY_TILE(typesize, out, in, ldi, ldo, m0, m1, n0, n1);
#else
  internal_libxs_tcopy_tile_sw(out, in, typesize, ldi, ldo, m0, m1, n0, n1);
#endif
}


LIBXS_API_INTERN void internal_libxs_memory_init(int target_arch)
{
  internal_libxs_hash_init(target_arch);
#if !defined(LIBXS_MEM_SW)
  if (LIBXS_X86_AVX512 <= target_arch) {
# if defined(LIBXS_DIFF_AVX512_ENABLED)
    internal_libxs_diff_function = internal_libxs_diff_avx512;
# else
    internal_libxs_diff_function = internal_libxs_diff_avx2;
# endif
# if defined(LIBXS_DIFF_AVX512_ENABLED)
    internal_libxs_memcmp_function = internal_libxs_memcmp_avx512;
# else
    internal_libxs_memcmp_function = internal_libxs_memcmp_avx2;
# endif
  }
  else if (LIBXS_X86_AVX2 <= target_arch) {
    internal_libxs_diff_function = internal_libxs_diff_avx2;
    internal_libxs_memcmp_function = internal_libxs_memcmp_avx2;
  }
  else if (LIBXS_X86_GENERIC <= target_arch) {
    internal_libxs_diff_function = internal_libxs_diff_sse;
    internal_libxs_memcmp_function = internal_libxs_memcmp_sse;
  }
  else {
    internal_libxs_diff_function = internal_libxs_diff_sw;
    internal_libxs_memcmp_function = internal_libxs_memcmp_sw;
  }
  LIBXS_ASSERT(NULL != internal_libxs_diff_function);
  LIBXS_ASSERT(NULL != internal_libxs_memcmp_function);
# if (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  /* mcopy/tcopy: direct call, no pointer dispatch needed */
# elif (LIBXS_X86_AVX2 > LIBXS_MAX_STATIC_TARGET_ARCH)
  internal_libxs_mcopy_tile_function = internal_libxs_mcopy_tile_sw;
  internal_libxs_tcopy_tile_function = internal_libxs_tcopy_tile_sw;
# else
  if (LIBXS_X86_AVX2 <= target_arch) {
    internal_libxs_mcopy_tile_function = internal_libxs_mcopy_tile_avx2;
    internal_libxs_tcopy_tile_function = internal_libxs_tcopy_tile_avx2;
  }
  else {
    internal_libxs_mcopy_tile_function = internal_libxs_mcopy_tile_sw;
    internal_libxs_tcopy_tile_function = internal_libxs_tcopy_tile_sw;
  }
# endif
  LIBXS_ASSERT(NULL != internal_libxs_mcopy_tile_function
    || LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH);
  LIBXS_ASSERT(NULL != internal_libxs_tcopy_tile_function
    || LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH);
#endif
}


LIBXS_API_INTERN void internal_libxs_memory_finalize(void)
{
#if !defined(NDEBUG) && !defined(LIBXS_MEM_SW) && 0
  internal_libxs_diff_function = NULL;
  internal_libxs_memcmp_function = NULL;
#endif
}


LIBXS_API_INTERN unsigned char internal_libxs_diff_4(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_libxs_diff_sw(a, b, 4);
#else
  LIBXS_DIFF_4_DECL(a4);
  LIBXS_DIFF_4_LOAD(a4, a);
  return LIBXS_DIFF_4(a4, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char internal_libxs_diff_8(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_libxs_diff_sw(a, b, 8);
#else
  LIBXS_DIFF_8_DECL(a8);
  LIBXS_DIFF_8_LOAD(a8, a);
  return LIBXS_DIFF_8(a8, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char internal_libxs_diff_16(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_libxs_diff_sw(a, b, 16);
#else
  LIBXS_DIFF_16_DECL(a16);
  LIBXS_DIFF_16_LOAD(a16, a);
  return LIBXS_DIFF_16(a16, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char internal_libxs_diff_32(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_libxs_diff_sw(a, b, 32);
#else
  LIBXS_DIFF_32_DECL(a32);
  LIBXS_DIFF_32_LOAD(a32, a);
  return LIBXS_DIFF_32(a32, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char internal_libxs_diff_48(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_libxs_diff_sw(a, b, 48);
#else
  LIBXS_DIFF_48_DECL(a48);
  LIBXS_DIFF_48_LOAD(a48, a);
  return LIBXS_DIFF_48(a48, b, 0/*dummy*/);
#endif
}


LIBXS_API_INTERN unsigned char internal_libxs_diff_64(const void* a, const void* b, ...)
{
#if defined(LIBXS_MEM_SW)
  return internal_libxs_diff_sw(a, b, 64);
#else
  LIBXS_DIFF_64_DECL(a64);
  LIBXS_DIFF_64_LOAD(a64, a);
  return LIBXS_DIFF_64(a64, b, 0/*dummy*/);
#endif
}


LIBXS_API unsigned char libxs_diff(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_MEM_SW) && !defined(LIBXS_MEM_STDLIB)
  return internal_libxs_diff_sw(a, b, size);
#else
# if defined(LIBXS_MEM_STDLIB)
  return 0 != memcmp(a, b, size);
# elif (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH) && defined(LIBXS_DIFF_AVX512_ENABLED)
  return internal_libxs_diff_avx512(a, b, size);
# elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  return internal_libxs_diff_avx2(a, b, size);
# elif (LIBXS_X86_SSE3 <= LIBXS_STATIC_TARGET_ARCH)
# if (LIBXS_X86_AVX2 > LIBXS_MAX_STATIC_TARGET_ARCH)
  return internal_libxs_diff_sse(a, b, size);
# else /* pointer based function call */
# if defined(LIBXS_INIT_COMPLETED)
  LIBXS_ASSERT(NULL != internal_libxs_diff_function);
  return (unsigned char)(64 <= size
    ? internal_libxs_diff_function(a, b, size)
    : internal_libxs_diff_sse(a, b, size));
# else
  return (unsigned char)((NULL != internal_libxs_diff_function && 64 <= size)
    ? internal_libxs_diff_function(a, b, size)
    : internal_libxs_diff_sse(a, b, size));
# endif
# endif
# else
  return internal_libxs_diff_sw(a, b, size);
# endif
#endif
}


LIBXS_API unsigned int libxs_diff_n(const void* a, const void* bn, unsigned char elemsize,
  unsigned char stride, unsigned int hint, unsigned int count)
{
  unsigned int result;
  LIBXS_ASSERT(elemsize <= stride);
#if defined(LIBXS_MEM_STDLIB) && !defined(LIBXS_MEM_SW)
  LIBXS_DIFF_N(unsigned int, result, memcmp, a, bn, elemsize, stride, hint, count);
#else
# if !defined(LIBXS_MEM_SW)
  switch (elemsize) {
    case 64: {
      LIBXS_DIFF_64_DECL(a64);
      LIBXS_DIFF_64_LOAD(a64, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_64, a64, bn, 64, stride, hint, count);
    } break;
    case 48: {
      LIBXS_DIFF_48_DECL(a48);
      LIBXS_DIFF_48_LOAD(a48, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_48, a48, bn, 48, stride, hint, count);
    } break;
    case 32: {
      LIBXS_DIFF_32_DECL(a32);
      LIBXS_DIFF_32_LOAD(a32, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_32, a32, bn, 32, stride, hint, count);
    } break;
    case 16: {
      LIBXS_DIFF_16_DECL(a16);
      LIBXS_DIFF_16_LOAD(a16, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_16, a16, bn, 16, stride, hint, count);
    } break;
    case 8: {
      LIBXS_DIFF_8_DECL(a8);
      LIBXS_DIFF_8_LOAD(a8, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_8, a8, bn, 8, stride, hint, count);
    } break;
    case 4: {
      LIBXS_DIFF_4_DECL(a4);
      LIBXS_DIFF_4_LOAD(a4, a);
      LIBXS_DIFF_N(unsigned int, result, LIBXS_DIFF_4, a4, bn, 4, stride, hint, count);
    } break;
    default:
# endif
    {
      LIBXS_DIFF_N(unsigned int, result, libxs_diff, a, bn, elemsize, stride, hint, count);
    }
# if !defined(LIBXS_MEM_SW)
  }
# endif
#endif
  return result;
}


LIBXS_API int libxs_memcmp(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_MEM_SW) && !defined(LIBXS_MEM_STDLIB)
  return internal_libxs_memcmp_sw(a, b, size);
#else
# if defined(LIBXS_MEM_STDLIB)
  return memcmp(a, b, size);
# elif (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH) && defined(LIBXS_DIFF_AVX512_ENABLED)
  return internal_libxs_memcmp_avx512(a, b, size);
# elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  return internal_libxs_memcmp_avx2(a, b, size);
# elif (LIBXS_X86_SSE3 <= LIBXS_STATIC_TARGET_ARCH)
# if (LIBXS_X86_AVX2 > LIBXS_MAX_STATIC_TARGET_ARCH)
  return internal_libxs_memcmp_sse(a, b, size);
# else /* pointer based function call */
# if defined(LIBXS_INIT_COMPLETED)
  LIBXS_ASSERT(NULL != internal_libxs_memcmp_function);
  return (64 <= size
    ? internal_libxs_memcmp_function(a, b, size)
    : internal_libxs_memcmp_sse(a, b, size));
# else
  return ((NULL != internal_libxs_memcmp_function && 64 <= size)
    ? internal_libxs_memcmp_function(a, b, size)
    : internal_libxs_memcmp_sse(a, b, size));
# endif
# endif
# else
  return internal_libxs_memcmp_sw(a, b, size);
# endif
#endif
}


LIBXS_API unsigned int libxs_hash(const void* data, unsigned int size, unsigned int seed)
{
  /*LIBXS_INIT*/
  return libxs_crc32(seed, data, size);
}


LIBXS_API unsigned int libxs_hash8(unsigned int data)
{
  const unsigned int hash = libxs_hash16(data);
  uint8_t tmp_data = (uint8_t)hash;
  unsigned int tmp_seed = (unsigned int)(hash >> 8);
  return libxs_crc32_u8(tmp_seed, &tmp_data) & 0xFF;
}


LIBXS_API unsigned int libxs_hash16(unsigned int data)
{
  uint16_t tmp_data = (uint16_t)data;
  unsigned int tmp_seed = (unsigned int)(data >> 16);
  return libxs_crc32_u16(tmp_seed, &tmp_data) & 0xFFFF;
}


LIBXS_API unsigned int libxs_hash32(unsigned long long data)
{
  uint32_t tmp_data = (uint32_t)data;
  unsigned int tmp_seed = (unsigned int)(data >> 32);
  return libxs_crc32_u32(tmp_seed, &tmp_data) & 0xFFFFFFFF;
}


LIBXS_API unsigned long long libxs_hash_string(const char string[])
{
  unsigned long long result = 0;
  const size_t length = (NULL != string ? strlen(string) : 0);
  if (sizeof(result) < length) {
    const size_t length2 = LIBXS_MAX(length / 2, sizeof(result));
    unsigned int hash32, seed32 = 0; /* seed=0: match else-optimization */
    /*LIBXS_INIT*/
    seed32 = libxs_crc32(seed32, string, length2);
    hash32 = libxs_crc32(seed32, string + length2, length - length2);
    result = hash32; result = (result << 32) | seed32;
  }
  else { /* length <= sizeof(result) */
    char *const s = (char*)&result; signed char i;
    for (i = 0; i < (signed char)length; ++i) s[i] = string[i];
    for (; i < (signed char)sizeof(result); ++i) s[i] = 0;
  }
  return result;
}


LIBXS_API const char* libxs_stristrn(const char a[], const char b[], size_t maxlen)
{
  const char* result = NULL;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b && 0 != maxlen) {
    do {
      if (tolower(*a) != tolower(*b)) ++a;
      else {
        const char* const start = a;
        const char* c = b;
        size_t i = 0;
        result = a;
        while ('\0' != c[++i] && i != maxlen && '\0' != *++a) {
          if (tolower(*a) != tolower(c[i])) {
            result = NULL;
            break;
          }
        }
        if ('\0' == c[i] || i == maxlen) break; /* full match */
        result = NULL;
        a = start + 1; /* backtrack past match start */
      }
    } while ('\0' != *a);
  }
  return result;
}


LIBXS_API const char* libxs_stristr(const char a[], const char b[])
{
  return libxs_stristrn(a, b, (size_t)-1);
}


LIBXS_API int libxs_strimatch(const char a[], const char b[], const char delims[], int* count)
{
  int result = 0, na = 0, nb = 0;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b) {
    const char* const sep = ((NULL == delims || '\0' == *delims) ? " \t;,:-" : delims);
    const char *c, *tmp;
    char s[2] = {'\0'};
    size_t m, n;
    for (;;) {
      while (*s = *b, NULL != strpbrk(s, sep)) ++b; /* left-trim */
      if ('\0' != *b && '[' != *b) ++nb; /* count words */
      else break;
      tmp = b;
      while ('\0' != *tmp && (*s = *tmp, NULL == strpbrk(s, sep))) ++tmp;
      m = tmp - b;
      c = libxs_stristrn(a, b, LIBXS_MIN(1, m));
      if (NULL != c) {
        const char* d = c;
        while ('\0' != *d && (*s = *d, NULL == strpbrk(s, sep))) ++d;
        n = d - c;
        if (1 >= n || NULL != libxs_stristrn(c, b, LIBXS_MIN(m, n))) ++result;
      }
      b = tmp;
    }
    for (;;) { /* count number of words */
      while (*s = *a, NULL != strpbrk(s, sep)) ++a; /* left-trim */
      if ('\0' != *a && '[' != *a) ++na; /* count words */
      else break;
      while ('\0' != *a && (*s = *a, NULL == strpbrk(s, sep))) ++a;
    }
    if (na < result) result = na;
  }
  else result = -1;
  if (NULL != count) *count = LIBXS_MAX(na, nb);
  return result;
}


LIBXS_API size_t libxs_format_value(char buffer[],
  int buffer_size, size_t nbytes, const char scale[], const char* unit, int base)
{
  const int len = (NULL != scale ? ((int)strlen(scale)) : 0);
  const int m = LIBXS_INTRINSICS_BITSCANBWD64(nbytes) / LIBXS_MAX(base, 1), n = LIBXS_MIN(m, len);
  int i;
  buffer[0] = 0; /* clear */
  LIBXS_ASSERT(NULL != unit && 0 < base);
  for (i = 0; i < n; ++i) nbytes >>= base;
  LIBXS_SNPRINTF(buffer, buffer_size, "%lu %c%s",
    (unsigned long)nbytes, 0 < n ? scale[n-1] : *unit, 0 < n ? unit : "");
  return nbytes;
}


LIBXS_API int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat)
{
  int result;
  if (0 == count || 0 == elemsize) {
    result = EXIT_SUCCESS;
  }
  else if (NULL != inout) {
    const size_t s = (NULL == shuffle ? LIBXS_MEM_SHUFFLE_COPRIME(count) : *shuffle);
    const size_t n = (NULL == nrepeat ? 1 : *nrepeat);
    switch (elemsize) {
      case 8:   LIBXS_MEM_SHUFFLE(inout, 8, count, s, n); break;
      case 4:   LIBXS_MEM_SHUFFLE(inout, 4, count, s, n); break;
      case 2:   LIBXS_MEM_SHUFFLE(inout, 2, count, s, n); break;
      case 1:   LIBXS_MEM_SHUFFLE(inout, 1, count, s, n); break;
      default:  LIBXS_MEM_SHUFFLE(inout, elemsize, count, s, n);
    }
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXS_API int libxs_shuffle2(void* dst, const void* src, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat)
{
  const unsigned char *const LIBXS_RESTRICT inp = (const unsigned char*)src;
  unsigned char *const LIBXS_RESTRICT out = (unsigned char*)dst;
  const size_t size = elemsize * count;
  int result;
  if ((NULL != inp && NULL != out && ((out + size) <= inp || (inp + size) <= out)) || 0 == size) {
    const size_t s = (NULL == shuffle ? LIBXS_MEM_SHUFFLE_COPRIME(count) : *shuffle);
    size_t i = 0, j = 1;
    if (NULL == nrepeat || 1 == *nrepeat) {
      if (elemsize < 128) {
        switch (elemsize) {
          case 8: for (; i < size; i += 8, j += s) {
            if (count < j) j -= count;
            LIBXS_MEMCPY(out + i, inp + size - 8 * j, 8);
          } break;
          case 4: for (; i < size; i += 4, j += s) {
            if (count < j) j -= count;
            LIBXS_MEMCPY(out + i, inp + size - 4 * j, 4);
          } break;
          case 2: for (; i < size; i += 2, j += s) {
            if (count < j) j -= count;
            LIBXS_MEMCPY(out + i, inp + size - 2 * j, 2);
          } break;
          case 1: for (; i < size; ++i, j += s) {
            if (count < j) j -= count;
            out[i] = inp[size-j];
          } break;
          default: for (; i < size; i += elemsize, j += s) {
            if (count < j) j -= count;
            LIBXS_MEMCPY(out + i, inp + size - elemsize * j, elemsize);
          }
        }
      }
      else { /* generic path */
        for (; i < size; i += elemsize, j += s) {
          if (count < j) j -= count;
          memcpy(out + i, inp + size - elemsize * j, elemsize);
        }
      }
    }
    else if (0 != *nrepeat) { /* generic path */
      const size_t c = count - 1;
      for (; i < count; ++i) {
        size_t k = 0;
        LIBXS_ASSERT(NULL != inp && NULL != out);
        for (j = i; k < *nrepeat; ++k) j = c - ((s * j) % count);
        memcpy(out + elemsize * i, inp + elemsize * j, elemsize);
      }
    }
    else { /* ordinary copy */
      memcpy(out, inp, size);
    }
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXS_API size_t libxs_unshuffle(size_t count, const size_t* shuffle)
{
  size_t result = 0;
  if (0 < count) {
    const size_t n = (NULL == shuffle ? LIBXS_MEM_SHUFFLE_COPRIME(count) : *shuffle);
    size_t c = count - 1, j = c, d = 0;
    for (; result < count; ++result, j = c - d) {
      d = (j * n) % count;
      if (0 == d) break;
    }
  }
  LIBXS_ASSERT(result <= count);
  return result;
}


LIBXS_API_INLINE void internal_libxs_itrans_scratch(
  void* inout, void* scratch, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo)
{
#if defined(LIBXS_MEM_SW)
  LIBXS_MCOPY_TILE(typesize, scratch, inout, ldi, m, 0, m, 0, n);
  LIBXS_TCOPY_TILE(typesize, inout, scratch, m, ldo, 0, m, 0, n);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
  internal_libxs_mcopy_tile_avx2(scratch, inout, typesize, ldi, m, 0, m, 0, n);
  internal_libxs_tcopy_tile_avx2(inout, scratch, typesize, m, ldo, 0, m, 0, n);
#elif (LIBXS_X86_AVX2 > LIBXS_MAX_STATIC_TARGET_ARCH)
  internal_libxs_mcopy_tile_sw(scratch, inout, typesize, ldi, m, 0, m, 0, n);
  internal_libxs_tcopy_tile_sw(inout, scratch, typesize, m, ldo, 0, m, 0, n);
#else /* pointer based function call */
  internal_libxs_mcopy_tile_function(scratch, inout, typesize, ldi, m, 0, m, 0, n);
  internal_libxs_tcopy_tile_function(inout, scratch, typesize, m, ldo, 0, m, 0, n);
#endif
}


LIBXS_API void libxs_matcopy_task(void* out, const void* in, unsigned int typesize,
  int m, int n, int ldi, int ldo,
  int tid, int ntasks)
{
  if (0 < typesize && typesize < 256 && m <= ldi && m <= ldo
    && ((NULL != out && 0 < m && 0 < n) || (0 == m && 0 == n))
    && 0 <= tid && tid < ntasks)
  {
    if (0 < m && 0 < n) {
      unsigned int m0, m1, n0, n1;
      LIBXS_XCOPY_TASKS((unsigned int)m, (unsigned int)n, tid, ntasks, m0, m1, n0, n1);
#if defined(LIBXS_MEM_SW)
      if (NULL != in) {
        LIBXS_MCOPY_TILE(typesize, out, in,
          (unsigned int)ldi, (unsigned int)ldo, m0, m1, n0, n1);
      }
      else {
        LIBXS_MZERO_TILE(typesize, out, (unsigned int)ldo, m0, m1, n0, n1);
      }
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
      internal_libxs_mcopy_tile_avx2(out, in, typesize,
        (unsigned int)ldi, (unsigned int)ldo, m0, m1, n0, n1);
#elif (LIBXS_X86_AVX2 > LIBXS_MAX_STATIC_TARGET_ARCH)
      internal_libxs_mcopy_tile_sw(out, in, typesize,
        (unsigned int)ldi, (unsigned int)ldo, m0, m1, n0, n1);
#else /* pointer based function call */
      internal_libxs_mcopy_tile_function(out, in, typesize,
        (unsigned int)ldi, (unsigned int)ldo, m0, m1, n0, n1);
#endif
    }
  }
}


LIBXS_API void libxs_matcopy(void* out, const void* in, unsigned int typesize,
  int m, int n, int ldi, int ldo)
{
  libxs_matcopy_task(out, in, typesize, m, n, ldi, ldo, 0, 1);
}


LIBXS_API void libxs_otrans_task(void* out, const void* in, unsigned int typesize,
  int m, int n, int ldi, int ldo,
  int tid, int ntasks)
{
  if (0 < typesize && typesize < 256 && m <= ldi && n <= ldo
    && ((NULL != out && NULL != in && 0 < m && 0 < n) || (0 == m && 0 == n))
    && 0 <= tid && tid < ntasks)
  {
    if (0 < m && 0 < n) {
      unsigned int m0, m1, n0, n1;
      LIBXS_ASSERT(out != in);
      LIBXS_XCOPY_TASKS((unsigned int)m, (unsigned int)n, tid, ntasks, m0, m1, n0, n1);
#if defined(LIBXS_MEM_SW)
      LIBXS_TCOPY_TILE(typesize, out, in,
        (unsigned int)ldi, (unsigned int)ldo, m0, m1, n0, n1);
#elif (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
      internal_libxs_tcopy_tile_avx2(out, in, typesize,
        (unsigned int)ldi, (unsigned int)ldo, m0, m1, n0, n1);
#elif (LIBXS_X86_AVX2 > LIBXS_MAX_STATIC_TARGET_ARCH)
      internal_libxs_tcopy_tile_sw(out, in, typesize,
        (unsigned int)ldi, (unsigned int)ldo, m0, m1, n0, n1);
#else /* pointer based function call */
      internal_libxs_tcopy_tile_function(out, in, typesize,
        (unsigned int)ldi, (unsigned int)ldo, m0, m1, n0, n1);
#endif
    }
  }
}


LIBXS_API void libxs_otrans(void* out, const void* in, unsigned int typesize,
  int m, int n, int ldi, int ldo)
{
  libxs_otrans_task(out, in, typesize, m, n, ldi, ldo, 0, 1);
}


LIBXS_API void libxs_itrans_task(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo, void* scratch,
  int tid, int ntasks)
{
  if (NULL != inout && 0 < typesize && m <= ldi && n <= ldo
    && 0 <= tid && tid < ntasks)
  {
    if (m == n && ldi == ldo && 1 < m) {
      const unsigned int um = (unsigned int)m;
      const unsigned int ntriangles = um * (um - 1) / 2;
      const unsigned int tasksize = LIBXS_UPDIV(ntriangles, (unsigned int)ntasks);
      const unsigned int begin = LIBXS_MIN((unsigned int)tid * tasksize, ntriangles);
      const unsigned int end = LIBXS_MIN(begin + tasksize, ntriangles);
      unsigned int row, col;
      /* map linear index to triangular (i,j) pair where j < i */
      row = (unsigned int)((1 + libxs_isqrt_u64(1 + 8 * (unsigned long long)begin)) / 2);
      if (row * (row - 1) / 2 > begin && 0 < row) --row;
      col = begin - row * (row - 1) / 2;
      LIBXS_ITRANS_RANGE(typesize, inout, (unsigned int)ldi, begin, end, row, col);
    }
    else if (0 == tid) {
      if (NULL != scratch) {
        internal_libxs_itrans_scratch(inout, scratch, typesize,
          (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo);
      }
      else {
        const size_t scratchsize = (size_t)m * n * typesize;
        void* scratch_alloc = libxs_malloc(NULL/*pool*/, scratchsize, LIBXS_MALLOC_AUTO);
        if (NULL != scratch_alloc) {
          internal_libxs_itrans_scratch(inout, scratch_alloc, typesize,
            (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo);
          libxs_free(scratch_alloc);
        }
      }
    }
  }
}


LIBXS_API void libxs_itrans(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo, void* scratch)
{
  libxs_itrans_task(inout, typesize, m, n, ldi, ldo, scratch, 0, 1);
}


LIBXS_API void libxs_itrans_batch(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo,
  int index_base, int index_stride,
  const int stride[], int batchsize,
  int tid, int ntasks)
{
  if (NULL != inout && 0 < typesize && m <= ldi && n <= ldo
    && 0 <= tid && tid < ntasks)
  {
    const int size = (batchsize < 0 ? -batchsize : batchsize);
    const int tasksize = LIBXS_UPDIV(size, ntasks);
    const int begin = tid * tasksize;
    const int end = LIBXS_MIN(begin + tasksize, size);
    char *const mat0 = (char*)inout;
    void* scratch = NULL;
    int need_scratch = (m != n || ldi != ldo);
    if (need_scratch) {
      scratch = libxs_malloc(NULL/*pool*/, (size_t)m * n * typesize, LIBXS_MALLOC_AUTO);
    }
    if (NULL != stride) {
      if (0 != index_stride) {
        int i;
        if (NULL == scratch) {
          for (i = begin; i < end; ++i) {
            const int idx = i * index_stride;
            char *const mat = mat0 + (size_t)(stride[idx] - index_base) * typesize;
            LIBXS_ITRANS_TILE(typesize, mat, (unsigned int)ldi, (unsigned int)m);
          }
        }
        else {
          for (i = begin; i < end; ++i) {
            const int idx = i * index_stride;
            char *const mat = mat0 + (size_t)(stride[idx] - index_base) * typesize;
            internal_libxs_itrans_scratch(mat, scratch, typesize,
              (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo);
          }
        }
      }
      else {
        const size_t d = (size_t)(*stride - index_base * (int)sizeof(void*));
        size_t i;
        if (NULL == scratch) {
          for (i = (size_t)begin; i < (size_t)end; ++i) {
            void *const mat = *(void**)(mat0 + d * i);
            if (NULL != mat) {
              LIBXS_ITRANS_TILE(typesize, mat, (unsigned int)ldi, (unsigned int)m);
            }
          }
        }
        else {
          for (i = (size_t)begin; i < (size_t)end; ++i) {
            void *const mat = *(void**)(mat0 + d * i);
            if (NULL != mat) {
              internal_libxs_itrans_scratch(mat, scratch, typesize,
                (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo);
            }
          }
        }
      }
    }
    else {
      int i;
      if (NULL == scratch) {
        for (i = begin; i < end; ++i) {
          LIBXS_ITRANS_TILE(typesize,
            mat0 + (size_t)i * m * n * typesize,
            (unsigned int)ldi, (unsigned int)m);
        }
      }
      else {
        for (i = begin; i < end; ++i) {
          internal_libxs_itrans_scratch(
            mat0 + (size_t)i * m * n * typesize,
            scratch, typesize,
            (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo);
        }
      }
    }
    libxs_free(scratch);
  }
}
