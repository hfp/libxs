/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_perm.h>
#include <libxs_mem.h>
#include <libxs_math.h>
#include <libxs_malloc.h>
#include "libxs_main.h"

#define LIBXS_MEM_SHUFFLE_COPRIME(N) libxs_coprime2(N)
#define LIBXS_MEM_SHUFFLE_MALLOC(SIZE, POOL) \
  internal_libxs_perm_shuffle_malloc(SIZE, &(POOL))
#define LIBXS_MEM_SHUFFLE_FREE(PTR, POOL) \
  internal_libxs_perm_shuffle_free(PTR, POOL)
#define LIBXS_MEM_SHUFFLE(INOUT, ELEMSIZE, COUNT, SHUFFLE, NREPEAT) do { \
  unsigned char *const LIBXS_RESTRICT shfl_data = (unsigned char*)(INOUT); \
  const size_t shfl_count = (COUNT), shfl_stride = (SHUFFLE); \
  const size_t shfl_last = shfl_count - 1, shfl_nrep = (NREPEAT); \
  const size_t shfl_nbitmask = (shfl_count + 7) / 8; \
  int shfl_pool_v = 0, shfl_pool_t = 0; \
  unsigned char *const shfl_visited = (unsigned char*) \
    LIBXS_MEM_SHUFFLE_MALLOC(shfl_nbitmask, shfl_pool_v); \
  if (NULL != shfl_visited && 0 != shfl_nrep) { \
    void *const shfl_tmp = LIBXS_MEM_SHUFFLE_MALLOC(ELEMSIZE, shfl_pool_t); \
    memset(shfl_visited, 0, shfl_nbitmask); \
    if (NULL != shfl_tmp) { \
      size_t shfl_i; \
      for (shfl_i = 0; shfl_i < shfl_count; ++shfl_i) { \
        size_t shfl_src, shfl_dst, shfl_k; \
        if (0 != (shfl_visited[shfl_i / 8] & (1u << (shfl_i % 8)))) continue; \
        shfl_src = shfl_i; \
        for (shfl_k = 0; shfl_k < shfl_nrep; ++shfl_k) { \
          shfl_src = shfl_last - ((shfl_stride * shfl_src) % shfl_count); \
        } \
        if (shfl_src == shfl_i) { \
          shfl_visited[shfl_i / 8] |= (unsigned char)(1u << (shfl_i % 8)); \
          continue; \
        } \
        LIBXS_MEMCPY(shfl_tmp, shfl_data + (ELEMSIZE) * shfl_i, ELEMSIZE); \
        shfl_dst = shfl_i; \
        do { \
          shfl_src = shfl_dst; \
          for (shfl_k = 0; shfl_k < shfl_nrep; ++shfl_k) { \
            shfl_src = shfl_last - ((shfl_stride * shfl_src) % shfl_count); \
          } \
          shfl_visited[shfl_dst / 8] |= (unsigned char)(1u << (shfl_dst % 8)); \
          if (shfl_src != shfl_i) { \
            LIBXS_MEMCPY(shfl_data + (ELEMSIZE) * shfl_dst, \
              shfl_data + (ELEMSIZE) * shfl_src, ELEMSIZE); \
          } \
          else { \
            LIBXS_MEMCPY(shfl_data + (ELEMSIZE) * shfl_dst, shfl_tmp, ELEMSIZE); \
          } \
          shfl_dst = shfl_src; \
        } while (shfl_src != shfl_i); \
      } \
      LIBXS_MEM_SHUFFLE_FREE(shfl_tmp, shfl_pool_t); \
    } \
    LIBXS_MEM_SHUFFLE_FREE(shfl_visited, shfl_pool_v); \
  } \
} while(0)


LIBXS_API_INLINE void* internal_libxs_perm_shuffle_malloc(size_t size, int* pool) {
  void* p = libxs_malloc(internal_libxs_default_pool, size, LIBXS_MALLOC_AUTO);
  if (NULL != p) { *pool = 1; return p; }
  *pool = 0; return malloc(size);
}

LIBXS_API_INLINE void internal_libxs_perm_shuffle_free(void* ptr, int pool) {
  if (0 != pool) libxs_free(ptr); else free(ptr);
}


LIBXS_API_INLINE void* internal_libxs_sort_malloc(size_t size, int* pool) {
  void* p = libxs_malloc(internal_libxs_default_pool, size, LIBXS_MALLOC_AUTO);
  if (NULL != p) { *pool = 1; return p; }
  *pool = 0; return malloc(size);
}


LIBXS_API_INLINE void internal_libxs_sort_free(void* ptr, int pool) {
  if (0 != pool) libxs_free(ptr); else free(ptr);
}


#if defined(LIBXS_INTRINSICS_AVX2)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_shuffle2_u32_avx2(
  unsigned int* LIBXS_RESTRICT dst, const unsigned int* LIBXS_RESTRICT src,
  size_t count, size_t stride)
{
  const size_t count8 = count & ~(size_t)7;
  size_t i = 0, j = 1;
  for (; i < count8; i += 8) {
    LIBXS_ALIGNED(int idx[8], LIBXS_ALIGNMENT);
    __m256i vidx, vdata;
    int lane;
    for (lane = 0; lane < 8; ++lane) {
      if (count < j) j -= count;
      idx[lane] = (int)(count - j);
      j += stride;
    }
    vidx = _mm256_load_si256((__m256i*)idx);
    vdata = _mm256_i32gather_epi32((const int*)src, vidx, 4);
    _mm256_storeu_si256((__m256i*)(dst + i), vdata);
  }
  for (; i < count; ++i) {
    if (count < j) j -= count;
    dst[i] = src[count - j];
    j += stride;
  }
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_shuffle2_u64_avx2(
  unsigned long long* LIBXS_RESTRICT dst, const unsigned long long* LIBXS_RESTRICT src,
  size_t count, size_t stride)
{
  const size_t count4 = count & ~(size_t)3;
  size_t i = 0, j = 1;
  for (; i < count4; i += 4) {
    LIBXS_ALIGNED(long long idx[4], LIBXS_ALIGNMENT);
    __m256i vidx, vdata;
    int lane;
    for (lane = 0; lane < 4; ++lane) {
      if (count < j) j -= count;
      idx[lane] = (long long)(count - j);
      j += stride;
    }
    vidx = _mm256_load_si256((__m256i*)idx);
    vdata = _mm256_i64gather_epi64((const long long*)src, vidx, 8);
    _mm256_storeu_si256((__m256i*)(dst + i), vdata);
  }
  for (; i < count; ++i) {
    if (count < j) j -= count;
    dst[i] = src[count - j];
    j += stride;
  }
}
#endif /* LIBXS_INTRINSICS_AVX2 */


#if defined(LIBXS_INTRINSICS_AVX512)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_shuffle2_u32_avx512(
  unsigned int* LIBXS_RESTRICT dst, const unsigned int* LIBXS_RESTRICT src,
  size_t count, size_t stride)
{
  const size_t count16 = count & ~(size_t)15;
  size_t i = 0, j = 1;
  for (; i < count16; i += 16) {
    LIBXS_ALIGNED(int idx[16], LIBXS_ALIGNMENT);
    __m512i vidx, vdata;
    int lane;
    for (lane = 0; lane < 16; ++lane) {
      if (count < j) j -= count;
      idx[lane] = (int)(count - j);
      j += stride;
    }
    vidx = _mm512_load_si512((__m512i*)idx);
    vdata = _mm512_i32gather_epi32(vidx, src, 4);
    _mm512_storeu_si512((__m512i*)(dst + i), vdata);
  }
  for (; i < count; ++i) {
    if (count < j) j -= count;
    dst[i] = src[count - j];
    j += stride;
  }
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_shuffle2_u64_avx512(
  unsigned long long* LIBXS_RESTRICT dst, const unsigned long long* LIBXS_RESTRICT src,
  size_t count, size_t stride)
{
  const size_t count8 = count & ~(size_t)7;
  size_t i = 0, j = 1;
  for (; i < count8; i += 8) {
    LIBXS_ALIGNED(long long idx[8], LIBXS_ALIGNMENT);
    __m512i vidx, vdata;
    int lane;
    for (lane = 0; lane < 8; ++lane) {
      if (count < j) j -= count;
      idx[lane] = (long long)(count - j);
      j += stride;
    }
    vidx = _mm512_load_si512((__m512i*)idx);
    vdata = _mm512_i64gather_epi64(vidx, src, 8);
    _mm512_storeu_si512((__m512i*)(dst + i), vdata);
  }
  for (; i < count; ++i) {
    if (count < j) j -= count;
    dst[i] = src[count - j];
    j += stride;
  }
}
#endif /* LIBXS_INTRINSICS_AVX512 */


LIBXS_API int libxs_sort_smooth(libxs_sort_t method, int m, int n,
  const void* mat, int ld, libxs_data_t datatype, int* perm)
{
  int result = EXIT_SUCCESS;
  if (NULL == perm || NULL == mat || m < 0 || n < 0 || ld < m) {
    return EXIT_FAILURE;
  }
  if (0 == m || LIBXS_SORT_NONE == method) {
    return EXIT_SUCCESS;
  }
  {
    int pool = 0;
    const size_t scores_size = (size_t)m * sizeof(double);
    const size_t visited_size = (LIBXS_SORT_GREEDY == method) ? (size_t)m : 0;
    double* scores = NULL;
    char* visited = NULL;

    if (LIBXS_SORT_IDENTITY != method) {
      scores = (double*)internal_libxs_sort_malloc(
        scores_size + visited_size, &pool);
      if (NULL == scores) return EXIT_FAILURE;
      if (0 != visited_size) visited = (char*)(scores + m);
    }

    switch ((int)datatype) {
      case LIBXS_DATATYPE_F64: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) (VALUE)
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE double
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_F32: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE float
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_I32: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE int
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_U32: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE unsigned int
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_I16: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE short
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_U16: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE unsigned short
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_I8: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE signed char
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_U8: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE unsigned char
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      default: {
        static int error_once = 0;
        if (0 != libxs_verbosity
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1,
                    LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr,
            "LIBXS ERROR: unsupported data-type for sort!\n");
        }
        result = EXIT_FAILURE;
      }
    }

    internal_libxs_sort_free(scores, pool);
  }
  return result;
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
#if defined(LIBXS_INTRINSICS_AVX512) || defined(LIBXS_INTRINSICS_AVX2)
      if (4 == elemsize && count <= 0x7FFFFFFFU) {
# if defined(LIBXS_INTRINSICS_AVX512)
#   if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
        internal_libxs_shuffle2_u32_avx512((unsigned int*)out, (const unsigned int*)inp, count, s);
        i = size;
#   elif (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
        if (LIBXS_X86_AVX512 <= libxs_cpuid(NULL)) {
          internal_libxs_shuffle2_u32_avx512((unsigned int*)out, (const unsigned int*)inp, count, s);
          i = size;
        }
#   endif
# endif
# if defined(LIBXS_INTRINSICS_AVX2)
        if (i < size) {
#   if (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
          internal_libxs_shuffle2_u32_avx2((unsigned int*)out, (const unsigned int*)inp, count, s);
          i = size;
#   elif (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
          if (LIBXS_X86_AVX2 <= libxs_cpuid(NULL)) {
            internal_libxs_shuffle2_u32_avx2((unsigned int*)out, (const unsigned int*)inp, count, s);
            i = size;
          }
#   endif
        }
# endif
      }
      else if (8 == elemsize) {
# if defined(LIBXS_INTRINSICS_AVX512)
#   if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
        internal_libxs_shuffle2_u64_avx512((unsigned long long*)out, (const unsigned long long*)inp, count, s);
        i = size;
#   elif (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
        if (LIBXS_X86_AVX512 <= libxs_cpuid(NULL)) {
          internal_libxs_shuffle2_u64_avx512((unsigned long long*)out, (const unsigned long long*)inp, count, s);
          i = size;
        }
#   endif
# endif
# if defined(LIBXS_INTRINSICS_AVX2)
        if (i < size) {
#   if (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
          internal_libxs_shuffle2_u64_avx2((unsigned long long*)out, (const unsigned long long*)inp, count, s);
          i = size;
#   elif (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
          if (LIBXS_X86_AVX2 <= libxs_cpuid(NULL)) {
            internal_libxs_shuffle2_u64_avx2((unsigned long long*)out, (const unsigned long long*)inp, count, s);
            i = size;
          }
#   endif
        }
# endif
      }
#endif
      if (i < size && elemsize < 128) {
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
      if (i < size) { /* generic path */
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
    size_t c = count - 1, j = c, d;
    do {
      d = (j * n) % count;
      ++result;
      j = c - d;
    } while (0 != d && result < count);
  }
  LIBXS_ASSERT(result <= count);
  return result;
}
