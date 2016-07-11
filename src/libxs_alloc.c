/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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
#include "libxs_alloc.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if !defined(NDEBUG)
# include <string.h>
#endif
#if defined(_WIN32)
# include <windows.h>
#else
# include <sys/mman.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_MAX_ALIGN)
# define LIBXS_MAX_ALIGN (2 * 1024 *1024)
#endif

#if !defined(LIBXS_ALLOC_MMAP)
/*# define LIBXS_ALLOC_MMAP*/
#endif


typedef struct LIBXS_RETARGETABLE internal_alloc_info {
  void* pointer;
  unsigned int size;
} internal_alloc_info;


LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_gcd(unsigned int a, unsigned int b)
{
  while (0 != b) {
    const unsigned int r = a % b;
    a = b;
    b = r;
  }
  return a;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_lcm(unsigned int a, unsigned int b)
{
  return (a * b) / libxs_gcd(a, b);
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE unsigned int libxs_alignment(unsigned int size, unsigned int alignment)
{
  unsigned int result = sizeof(void*);
  if ((LIBXS_MAX_ALIGN) <= size) {
    result = libxs_lcm(0 == alignment ? (LIBXS_ALIGNMENT) : libxs_lcm(alignment, LIBXS_ALIGNMENT), LIBXS_MAX_ALIGN);
  }
  else {
    if ((LIBXS_ALIGNMENT) <= size) {
      result = (0 == alignment ? (LIBXS_ALIGNMENT) : libxs_lcm(alignment, LIBXS_ALIGNMENT));
    }
    else if (0 != alignment) {
      result = libxs_lcm(alignment, result);
    }
  }
  return result;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_alloc_info(const void* memory, unsigned int* size, void** extra)
{
#if !defined(NDEBUG)
  if (0 != size || 0 != extra)
#endif
  {
    if (0 != memory) {
      const internal_alloc_info *const info = (const internal_alloc_info*)(((const char*)memory) - sizeof(internal_alloc_info));
      if (size) *size = info->size;
      if (extra) *extra = info->pointer;
    }
    else {
      if (size) *size = 0;
      if (extra) *extra = 0;
    }
  }
#if !defined(NDEBUG)
  else {
    return EXIT_FAILURE;
  }
#endif
  return EXIT_SUCCESS;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_allocate(void** memory, unsigned int size, unsigned int alignment,
  const void* extra, unsigned int extra_size)
{
  int result = EXIT_SUCCESS;

  if (memory) {
    if (0 < size) {
      const unsigned int auto_alignment = libxs_alignment(size, alignment);
      const unsigned int alloc_size = size + extra_size + sizeof(internal_alloc_info) + auto_alignment - 1;
#if defined(LIBXS_ALLOC_MMAP)
# if defined(_WIN32)
      char* buffer = (char*)VirtualAlloc(0, alloc_size, MEM_RESERVE, PAGE_NOACCESS);
      char *const aligned = LIBXS_ALIGN(buffer + extra_size + sizeof(internal_alloc_info), auto_alignment);
      if (0 != buffer) {
        buffer = (char*)VirtualAlloc(buffer, aligned - buffer, MEM_COMMIT, PAGE_READWRITE);
      }
      if (0 != buffer)
# else
#   if 0 /*TODO*/
      char* buffer = (char*)mmap(0, alloc_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      char *const aligned = LIBXS_ALIGN(buffer + extra_size + sizeof(internal_alloc_info), auto_alignment);
      if (MAP_FAILED != buffer) {
        buffer = (char*)mmap(buffer, aligned - buffer, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      }
#   else
      char *const buffer = (char*)mmap(0, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      char *const aligned = LIBXS_ALIGN(buffer + extra_size + sizeof(internal_alloc_info), auto_alignment);
#   endif
      if (MAP_FAILED != buffer)
# endif
#else
      char *const buffer = malloc(alloc_size);
      char *const aligned = LIBXS_ALIGN(buffer + extra_size + sizeof(internal_alloc_info), auto_alignment);
      if (0 != buffer)
#endif
      {
        internal_alloc_info *const info = (internal_alloc_info*)(aligned - sizeof(internal_alloc_info));
        assert((aligned + size) <= (buffer + alloc_size));
#if !defined(NDEBUG)
        memset(buffer, 0, alloc_size);
#endif
        if (0 < extra_size && 0 != extra) {
          const char *const src = (const char*)extra;
          unsigned int i;
#if (1900 <= _MSC_VER)
#         pragma warning(suppress: 6386)
#endif
          for (i = 0; i < extra_size; ++i) buffer[i] = src[i];
        }
#if !defined(NDEBUG)
        else if (0 == extra && 0 != extra_size) {
          result = EXIT_FAILURE;
        }
#endif
        info->pointer = buffer;
        info->size = size;
        *memory = aligned;
      }
      else {
        result = EXIT_FAILURE;
      }
    }
    else {
      *memory = 0;
    }
  }
#if !defined(NDEBUG)
  else if (0 != size) {
    result = EXIT_FAILURE;
  }
#endif
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE int libxs_deallocate(const void* memory)
{
  int result = EXIT_SUCCESS;
  if (memory) {
    void* buffer = 0;
#if defined(LIBXS_ALLOC_MMAP)
# if defined(_WIN32)
    libxs_alloc_info(memory, 0, &buffer);
    result = FALSE != VirtualFree(buffer, 0, MEM_RELEASE) ? EXIT_SUCCESS : EXIT_FAILURE;
# else
    unsigned int size = 0;
    libxs_alloc_info(memory, &size, &buffer);
    result = 0 == munmap(buffer, size + (((const char*)memory) - ((char*)buffer))) ? EXIT_SUCCESS : EXIT_FAILURE;
# endif
#else
    libxs_alloc_info(memory, 0, &buffer);
    free(buffer);
#endif
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void* libxs_malloc(unsigned int size)
{
  void* result = 0;
  return 0 == libxs_allocate(&result, size, 0, 0/*extra*/, 0/*extra_size*/) ? result : 0;
}


LIBXS_EXTERN_C LIBXS_RETARGETABLE void libxs_free(const void* memory)
{
  libxs_deallocate(memory);
}

