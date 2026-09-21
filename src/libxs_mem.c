/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_mem.h>
#include <libxs/libxs_malloc.h>
#include <libxs/libxs_math.h>

#include "libxs_main.h"
#include "libxs_crc32.h"
#include "libxs_diff.h"

#if defined(_WIN32)
# include <windows.h>
#else
# include <unistd.h>
# if defined(__APPLE__) && defined(__MACH__)
#   include <sys/sysctl.h>
# endif
#endif

#if !defined(LIBXS_MEM_STDLIB) && 0
# define LIBXS_MEM_STDLIB
#endif
#if !defined(LIBXS_MEM_SW) && 0
# define LIBXS_MEM_SW
#endif
#if !defined(LIBXS_TCOPY_BLOCK)
# define LIBXS_TCOPY_BLOCK 32
#endif
#if !defined(LIBXS_ITRANS_BLOCK)
# define LIBXS_ITRANS_BLOCK 8
#endif

/* xcopy kernel: consecutive loads and stores (matcopy) */
#define LIBXS_MCOPY_KERNEL(TYPE, TS, OUT, IN, LDI, LDO, I, J, SRC, DST) \
  const TYPE *const SRC = (const TYPE*)(((const char*)(IN)) \
    + (size_t)(TS) * ((size_t)(I) * (LDI) + (J))); \
  TYPE *const DST = (TYPE*)(((char*)(OUT)) \
    + (size_t)(TS) * ((size_t)(I) * (LDO) + (J)))
/**
 * xcopy kernel: zero stores (matzero). Three constraints meet in the zero
 * source. The read member is a character array, which keeps the load out of
 * type-based aliasing for every TYPE: reading a typed array through a wider or
 * narrower TYPE is the violation that made libxs_crc32_u64 return a stale digest
 * at -O2. It holds 256 Bytes because libxs_matcopy_task admits typesize < 256
 * and the byte-loop fallback reads that many. And it is a union with a double so
 * that the alignment TYPE needs comes from the type system, rather than from
 * LIBXS_ALIGNED, which expands to nothing outside MSVC and GNU.
 */
#define LIBXS_MZERO_KERNEL(TYPE, TS, OUT, IN, LDI, LDO, I, J, SRC, DST) \
  static const union { double align; unsigned char byte[256]; } \
    libxs_mzero_zero_ = { 0 }; \
  const TYPE *const SRC = (const TYPE*)libxs_mzero_zero_.byte; \
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
/* transpose tile: outer=M, inner=N for consecutive stores, in blocks keeping the strided loads resident */
#define LIBXS_TCOPY_TILE(TS, OUT, IN, LDI, LDO, M0, M1, N0, N1) do { \
  unsigned int libxs_tcopy_tile_i0_, libxs_tcopy_tile_j0_; \
  for (libxs_tcopy_tile_i0_ = (M0); libxs_tcopy_tile_i0_ < (unsigned int)(M1); \
    libxs_tcopy_tile_i0_ += LIBXS_TCOPY_BLOCK) \
  { \
    const unsigned int libxs_tcopy_tile_i1_ = LIBXS_MIN( \
      libxs_tcopy_tile_i0_ + LIBXS_TCOPY_BLOCK, (unsigned int)(M1)); \
    for (libxs_tcopy_tile_j0_ = (N0); libxs_tcopy_tile_j0_ < (unsigned int)(N1); \
      libxs_tcopy_tile_j0_ += LIBXS_TCOPY_BLOCK) \
    { \
      const unsigned int libxs_tcopy_tile_j1_ = LIBXS_MIN( \
        libxs_tcopy_tile_j0_ + LIBXS_TCOPY_BLOCK, (unsigned int)(N1)); \
      LIBXS_XCOPY_TILE(LIBXS_TCOPY_KERNEL, TS, OUT, IN, LDI, LDO, \
        libxs_tcopy_tile_i0_, libxs_tcopy_tile_i1_, \
        libxs_tcopy_tile_j0_, libxs_tcopy_tile_j1_); \
    } \
  } \
} while(0)

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

/* typed swap of whole triangle rows [RA,RB) for in-place square transpose, columns in blocks */
#define LIBXS_ITRANS_ROWS_LOOP(TYPE, INOUT, LD, RA, RB) do { \
  unsigned int libxs_itrans_rows_c0_, libxs_itrans_rows_row_, libxs_itrans_rows_col_; \
  for (libxs_itrans_rows_c0_ = 0; libxs_itrans_rows_c0_ + 1 < (unsigned int)(RB); \
    libxs_itrans_rows_c0_ += LIBXS_ITRANS_BLOCK) \
  { \
    const unsigned int libxs_itrans_rows_c1_ = libxs_itrans_rows_c0_ + LIBXS_ITRANS_BLOCK; \
    for (libxs_itrans_rows_row_ = LIBXS_MAX((unsigned int)(RA), libxs_itrans_rows_c0_ + 1); \
      libxs_itrans_rows_row_ < (unsigned int)(RB); ++libxs_itrans_rows_row_) \
    { \
      const unsigned int libxs_itrans_rows_cend_ = LIBXS_MIN( \
        libxs_itrans_rows_c1_, libxs_itrans_rows_row_); \
      for (libxs_itrans_rows_col_ = libxs_itrans_rows_c0_; \
        libxs_itrans_rows_col_ < libxs_itrans_rows_cend_; ++libxs_itrans_rows_col_) \
      { \
        TYPE *const libxs_itrans_rows_a_ = ((TYPE*)(INOUT)) \
          + (size_t)(LD) * libxs_itrans_rows_row_ + libxs_itrans_rows_col_; \
        TYPE *const libxs_itrans_rows_b_ = ((TYPE*)(INOUT)) \
          + (size_t)(LD) * libxs_itrans_rows_col_ + libxs_itrans_rows_row_; \
        LIBXS_ISWAP(*libxs_itrans_rows_a_, *libxs_itrans_rows_b_); \
      } \
    } \
  } \
} while(0)

#define LIBXS_ITRANS_ROWS(TS, INOUT, LD, RA, RB) do { \
  switch(TS) { \
    case 1: { LIBXS_ITRANS_ROWS_LOOP(char, INOUT, LD, RA, RB); } break; \
    case 2: { LIBXS_ITRANS_ROWS_LOOP(short, INOUT, LD, RA, RB); } break; \
    case 4: { LIBXS_ITRANS_ROWS_LOOP(int, INOUT, LD, RA, RB); } break; \
    case 8: { LIBXS_ITRANS_ROWS_LOOP(int64_t, INOUT, LD, RA, RB); } break; \
    default: { \
      unsigned int libxs_itrans_rows_r_ = (RA), libxs_itrans_rows_c_ = 0; \
      const unsigned int libxs_itrans_rows_begin_ = libxs_itrans_rows_r_ * (libxs_itrans_rows_r_ - 1) / 2; \
      const unsigned int libxs_itrans_rows_end_ = (unsigned int)(RB) * ((unsigned int)(RB) - 1) / 2; \
      LIBXS_ITRANS_RANGE(TS, INOUT, LD, libxs_itrans_rows_begin_, libxs_itrans_rows_end_, \
        libxs_itrans_rows_r_, libxs_itrans_rows_c_); \
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


#if !defined(_WIN32) && defined(__linux__)
/**
 * The cgroup memory limit that applies to this process, or zero where none does.
 * The physical size is what sysconf reports, and in a container that is the
 * machine's rather than the process's: four gigabytes of limit on a host holding
 * a terabyte reads as a terabyte, which is the wrong number precisely where a
 * caller sizing itself against memory needs a right one.
 *
 * The limit is not necessarily on the process's own cgroup. A batch system puts
 * it on the job while the process runs in a leaf below it, so the walk goes from
 * the leaf to the root and takes the smallest limit found: reading the leaf alone
 * finds "max" and concludes there is no limit.
 */
LIBXS_API_INLINE size_t internal_libxs_mem_cgroup(void)
{
  size_t result = 0;
  FILE *const self = fopen("/proc/self/cgroup", "r");
  if (NULL != self) {
    char line[512];
    while (NULL != fgets(line, sizeof(line), self)) {
      char* path = NULL;
      int v2 = 0;
      if ('0' == line[0] && ':' == line[1] && ':' == line[2]) {
        path = line + 3; /* unified hierarchy */
        v2 = 1;
      }
      else { /* one controller per line, and only the memory one carries a limit */
        char *const m = strstr(line, ":memory:");
        if (NULL != m) path = m + 8;
      }
      if (NULL != path) {
        char *const nl = strchr(path, '\n');
        if (NULL != nl) *nl = '\0';
        for (;;) {
          char file[1024];
          FILE* handle;
          if (0 != v2) {
            LIBXS_SNPRINTF(file, sizeof(file),
              "/sys/fs/cgroup%s/memory.max", path);
          }
          else {
            LIBXS_SNPRINTF(file, sizeof(file),
              "/sys/fs/cgroup/memory%s/memory.limit_in_bytes", path);
          }
          handle = fopen(file, "r");
          if (NULL != handle) {
            char buffer[64];
            if (NULL != fgets(buffer, sizeof(buffer), handle)) {
              char* end = NULL;
              const unsigned long long value = strtoull(buffer, &end, 10);
              /* v2 spells no limit "max", which parses as zero; v1 spells it as a
               * number past what the machine holds, which the caller clamps */
              if (end != buffer && 0 < value
                && (0 == result || (size_t)value < result))
              {
                result = (size_t)value;
              }
            }
            fclose(handle);
          }
          if ('\0' == *path) break; /* the root was the last one to try */
          { char *const slash = strrchr(path, '/');
            if (NULL == slash) break;
            *slash = '\0'; /* one level up; the root becomes the empty string */
          }
        }
      }
    }
    fclose(self);
  }
  return result;
}
#endif


LIBXS_API int libxs_mem_info(size_t* mem_free, size_t* mem_total)
{
  int result = EXIT_FAILURE;
  size_t size_free = 0, size_total = 0;
#if defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    size_total = (size_t)status.ullTotalPhys;
    size_free = (size_t)status.ullAvailPhys;
  }
#else
# if defined(_SC_PAGE_SIZE)
  const long page_size = sysconf(_SC_PAGE_SIZE);
# else
  const long page_size = 4096;
# endif
  long pages_free = 0, pages_total = 0;
# if defined(__linux__)
#   if defined(_SC_PHYS_PAGES)
  pages_total = sysconf(_SC_PHYS_PAGES);
#   endif
#   if defined(_SC_AVPHYS_PAGES)
  pages_free = sysconf(_SC_AVPHYS_PAGES);
#   else
  pages_free = pages_total;
#   endif
# elif defined(__APPLE__) && defined(__MACH__)
  { size_t nfree = sizeof(long), ntotal = sizeof(long);
    if (0 != sysctlbyname("hw.memsize", &pages_total, &ntotal, NULL, 0)) {
      pages_total = 0;
    }
    else if (0 < page_size) pages_total /= page_size;
    if (0 != sysctlbyname("vm.page_free_count", &pages_free, &nfree, NULL, 0)) {
      pages_free = pages_total;
    }
  }
# endif
  if (0 < page_size && 0 <= pages_free && 0 <= pages_total) {
    size_total = (size_t)page_size * (size_t)pages_total;
    size_free = (size_t)page_size * (size_t)pages_free;
  }
# if defined(__linux__)
  { const size_t limit = internal_libxs_mem_cgroup();
    if (0 != limit && limit < size_total) {
      size_total = limit;
      if (size_total < size_free) size_free = size_total;
    }
  }
# endif
#endif
  if (0 != size_total) {
    if (NULL != mem_total) *mem_total = size_total;
    if (NULL != mem_free) *mem_free = size_free;
    result = EXIT_SUCCESS;
  }
  return result;
}


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
  const unsigned char result = (unsigned char)memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char result = 0, i;
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xF0); i += 16) {
    LIBXS_DIFF_16_DECL(aa);
    LIBXS_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_16(aa, b8 + i, 0/*dummy*/)) { result = 1; break; }
  }
  if (0 == result) {
    for (; i < size; ++i) if (a8[i] ^ b8[i]) { result = 1; break; }
  }
#endif
  return result;
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
unsigned char internal_libxs_diff_sse(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_X86) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char result = 0, i;
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xF0); i += 16) {
    LIBXS_DIFF_SSE_DECL(aa);
    LIBXS_DIFF_SSE_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_SSE(aa, b8 + i, 0/*dummy*/)) { result = 1; break; }
  }
  if (0 == result) {
    for (; i < size; ++i) if (a8[i] ^ b8[i]) { result = 1; break; }
  }
#else
  const unsigned char result = internal_libxs_diff_sw(a, b, size);
#endif
  return result;
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
unsigned char internal_libxs_diff_avx2(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_AVX2) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char result = 0, i;
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xE0); i += 32) {
    LIBXS_DIFF_AVX2_DECL(aa);
    LIBXS_DIFF_AVX2_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX2(aa, b8 + i, 0/*dummy*/)) { result = 1; break; }
  }
  if (0 == result) {
    for (; i < size; ++i) if (a8[i] ^ b8[i]) { result = 1; break; }
  }
#else
  const unsigned char result = internal_libxs_diff_sw(a, b, size);
#endif
  return result;
}


#if defined(LIBXS_DIFF_AVX512_ENABLED)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
unsigned char internal_libxs_diff_avx512(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXS_INTRINSICS_AVX512) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char result = 0, i;
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xC0); i += 64) {
    LIBXS_DIFF_AVX512_DECL(aa);
    LIBXS_DIFF_AVX512_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX512(aa, b8 + i, 0/*dummy*/)) { result = 1; break; }
  }
  if (0 == result) {
    for (; i < size; ++i) if (a8[i] ^ b8[i]) { result = 1; break; }
  }
#else
  const unsigned char result = internal_libxs_diff_sw(a, b, size);
#endif
  return result;
}
#endif


LIBXS_API_INLINE
int internal_libxs_memcmp_sw(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_MEM_STDLIB)
  const int result = memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  int result = 0;
  size_t i;
  LIBXS_DIFF_16_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & ~(size_t)0xF); i += 16) {
    LIBXS_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_16(aa, b8 + i, 0/*dummy*/)) { result = 1; break; }
  }
  if (0 == result) {
    for (; i < size; ++i) if (a8[i] ^ b8[i]) { result = 1; break; }
  }
#endif
  return result;
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_GENERIC)
int internal_libxs_memcmp_sse(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_X86) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  int result = 0;
  size_t i;
  LIBXS_DIFF_SSE_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & ~(size_t)0xF); i += 16) {
    LIBXS_DIFF_SSE_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_SSE(aa, b8 + i, 0/*dummy*/)) { result = 1; break; }
  }
  if (0 == result) {
    for (; i < size; ++i) if (a8[i] ^ b8[i]) { result = 1; break; }
  }
#else
  const int result = internal_libxs_memcmp_sw(a, b, size);
#endif
  return result;
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
int internal_libxs_memcmp_avx2(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_AVX2) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  int result = 0;
  size_t i;
  LIBXS_DIFF_AVX2_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & ~(size_t)0x1F); i += 32) {
    LIBXS_DIFF_AVX2_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX2(aa, b8 + i, 0/*dummy*/)) { result = 1; break; }
  }
  if (0 == result) {
    for (; i < size; ++i) if (a8[i] ^ b8[i]) { result = 1; break; }
  }
#else
  const int result = internal_libxs_memcmp_sw(a, b, size);
#endif
  return result;
}


#if defined(LIBXS_DIFF_AVX512_ENABLED)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
int internal_libxs_memcmp_avx512(const void* a, const void* b, size_t size)
{
#if defined(LIBXS_INTRINSICS_AVX512) && !defined(LIBXS_MEM_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  int result = 0;
  size_t i;
  LIBXS_DIFF_AVX512_DECL(aa);
  LIBXS_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & ~(size_t)0x3F); i += 64) {
    LIBXS_DIFF_AVX512_LOAD(aa, a8 + i);
    if (LIBXS_DIFF_AVX512(aa, b8 + i, 0/*dummy*/)) { result = 1; break; }
  }
  if (0 == result) {
    for (; i < size; ++i) if (a8[i] ^ b8[i]) { result = 1; break; }
  }
#else
  const int result = internal_libxs_memcmp_sw(a, b, size);
#endif
  return result;
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
  return (unsigned char)((NULL != internal_libxs_diff_function && 64 <= size)
    ? internal_libxs_diff_function(a, b, size)
    : internal_libxs_diff_sse(a, b, size));
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
  return ((NULL != internal_libxs_memcmp_function && 64 <= size)
    ? internal_libxs_memcmp_function(a, b, size)
    : internal_libxs_memcmp_sse(a, b, size));
# endif
# else
  return internal_libxs_memcmp_sw(a, b, size);
# endif
#endif
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


LIBXS_API_INLINE unsigned int internal_libxs_itrans_row(unsigned int index)
{
  unsigned int result = (unsigned int)((1 + libxs_isqrt_u64(1 + 8 * (unsigned long long)index)) / 2);
  if (0 < result && result * (result - 1) / 2 > index) --result;
  return result;
}


LIBXS_API void libxs_itrans_task(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo,
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
      /* map linear index to triangular (i,j) pair where j < i */
      const unsigned int rend = internal_libxs_itrans_row(end);
      unsigned int row = internal_libxs_itrans_row(begin);
      unsigned int col = begin - row * (row - 1) / 2;
      if (row == rend) {
        LIBXS_ITRANS_RANGE(typesize, inout, (unsigned int)ldi, begin, end, row, col);
      }
      else { /* whole rows in column blocks, a partial row at either end in row order */
        const unsigned int head = (row + 1) * row / 2, tail = rend * (rend - 1) / 2;
        if (0 != col) { /* the range macro advances row, hence the bound is fixed first */
          LIBXS_ITRANS_RANGE(typesize, inout, (unsigned int)ldi, begin, head, row, col);
        }
        LIBXS_ITRANS_ROWS(typesize, inout, (unsigned int)ldi, row, rend);
        row = rend; col = 0;
        LIBXS_ITRANS_RANGE(typesize, inout, (unsigned int)ldi, tail, end, row, col);
      }
    }
    else if (0 == tid) {
      void *const scratch = libxs_malloc(NULL/*pool*/,
        (size_t)m * n * typesize, LIBXS_MALLOC_AUTO);
      if (NULL != scratch) {
        internal_libxs_itrans_scratch(inout, scratch, typesize,
          (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo);
        libxs_free(scratch);
      }
    }
  }
}


LIBXS_API void libxs_itrans(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo)
{
  libxs_itrans_task(inout, typesize, m, n, ldi, ldo, 0, 1);
}


LIBXS_API int libxs_mem_ntasks(libxs_mem_op_t op, int m, int n,
  unsigned int typesize, int nthreads)
{
  /* per call, below min_total a team loses to the unsplit operation */
  static const size_t min_total[] = { 1U << 20, 4U << 20, 64U << 10, 512U << 10 };
  static const size_t per_task[] = { 128U << 10, 256U << 10, 32U << 10, 128U << 10 };
  int result = 1;
  if (1 < nthreads && 0 < m && 0 < n && 0 < typesize
    && LIBXS_MEM_OP_MATCOPY <= op && op <= LIBXS_MEM_OP_ITRANS)
  {
    const size_t nbytes = (size_t)m * (size_t)n * typesize;
    if (min_total[op] <= nbytes) {
      const size_t ntasks = nbytes / per_task[op];
      result = (int)LIBXS_MAX(LIBXS_MIN((size_t)nthreads, ntasks), 1);
    }
  }
  return result;
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
