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

#ifndef LIBXS_BGEMM_H
#define LIBXS_BGEMM_H

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <libxs.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#define _USE_LIBXS_PREFETCH
#define BG_type 1 /* 1-float, 2-double */
#if (BG_type==2)
  typedef double real;
  #define _KERNEL libxs_dmmfunction
  #define _KERNEL_JIT libxs_dmmdispatch
#else
  typedef float real;
  #define _KERNEL libxs_smmfunction
  #define _KERNEL_JIT libxs_smmdispatch
#endif

// ORDER: 0-jik, 1-ijk, 2-jki, 3-ikj, 4-kji, 5-kij 
#define jik 0
#define ijk 1
#define jki 2
#define ikj 3
#define kji 4
#define kij 5

#define _alloc(size,alignment)  \
  mmap(NULL, size, PROT_READ | PROT_WRITE, \
      MAP_ANONYMOUS | MAP_SHARED | MAP_HUGETLB| MAP_POPULATE, -1, 0);
#define _free(addr) {munmap(addr, 0);}

//#define OMP_LOCK 
#ifdef OMP_LOCK
#define LOCK_T omp_lock_t
#define LOCK_INIT(x) {omp_init_lock(x);}
#define LOCK_SET(x) {omp_set_lock(x);}
#define LOCK_UNSET(x) {omp_unset_lock(x);}
#else
typedef struct{volatile int var[16];}lock_var;
#define LOCK_T lock_var
#define LOCK_INIT(x) {(x)->var[0] = 0;}
#define LOCK_SET(x) {do { \
        while ((x)->var[0] ==1); \
        _mm_pause(); \
      } while(__sync_lock_test_and_set(&((x)->var[0]), 1) != 0); \
}
#define LOCK_CHECK(x) {while ((x)->var[0] ==0); _mm_pause();}
#define LOCK_UNSET(x) {__sync_lock_release(&((x)->var[0]));}
#endif

#if defined(_OPENMP)
#define BG_BARRIER_INIT(x) { \
  int nthrds; \
_Pragma("omp parallel")\
  { \
    nthrds = omp_get_num_threads(); \
  } \
  int Cores = Cores = nthrds > 68 ? nthrds > 136 ? nthrds/4 : nthrds/2 : nthrds; \
  /*printf ("Cores=%d, Threads=%d\n", Cores, nthrds);*/ \
  /* create a new barrier */ \
  x = libxs_barrier_create(Cores, nthrds/Cores); \
  assert(x!= NULL); \
  /* each thread must initialize with the barrier */ \
_Pragma("omp parallel") \
  { \
    libxs_barrier_init(x, omp_get_thread_num()); \
  } \ 
}
#define BG_BARRIER(x, y) {libxs_barrier_wait((libxs_barrier*)x, y);}
#define BG_BARRIER_DEL(x) {libxs_barrier_release((libxs_barrier*)x);}
#else
#define BG_BARRIER_INIT(x) {}
#define BG_BARRIER(x, y) {}
#define BG_BARRIER_DEL(x) {}
#endif

#ifdef __cplusplus
  extern "C"{
#endif

/******************************************************************************
  Fine grain parallelized version(s) of BGEMM
    - Requires block structure layout for A,B matrices
    - Parallelized across all three dims - M, N, K
    - Uses fine-grain on-demand locks for write to C and fast barrier
    - Allows for calling multiple GEMMs, specified by 'count'
******************************************************************************/
void LIBXS_FSYMBOL(sgemm)(const char*, const char*, const libxs_blasint*, const libxs_blasint*, const libxs_blasint*,
  const real*, const real*, const libxs_blasint*, const real*, const libxs_blasint*,
  const real*, real*, const libxs_blasint*);

typedef struct libxs_blkgemm_handle {
  int m;
  int n;
  int k;
  int bm;
  int bn;
  int bk;
  int mb;
  int nb;
  int kb;
  /* The following have been added by KB */
  int b_m1, b_n1, b_k1, b_k2;
  int _ORDER;
  int C_pre_init;
  LOCK_T* _wlock;
  _KERNEL _l_kernel;
#ifdef _USE_LIBXS_PREFETCH
  _KERNEL _l_kernel_pf;
#endif
  libxs_barrier* bar;
} libxs_blkgemm_handle;

LIBXS_API void libxs_blksgemm_init_a( libxs_blkgemm_handle* handle,
                                          real* libxs_mat_dst,
                                          real* colmaj_mat_src );

LIBXS_API void libxs_blksgemm_init_b( libxs_blkgemm_handle* handle,
                                          real* libxs_mat_dst,
                                          real* colmaj_mat_src );

LIBXS_API void libxs_blksgemm_init_c( libxs_blkgemm_handle* handle,
                                          real* libxs_mat_dst,
                                          real* colmaj_mat_src );

LIBXS_API void libxs_blksgemm_check_c( libxs_blkgemm_handle* handle,
                                           real* libxs_mat_dst,
                                           real* colmaj_mat_src );

LIBXS_API void libxs_blksgemm_exec( const libxs_blkgemm_handle* handle,
                                        const char transA,
                                        const char transB,
                                        const real* alpha,
                                        const real* a,
                                        const real* b,
                                        const real* beta,
                                        real* c );

LIBXS_API void libxs_blkgemm_handle_alloc(libxs_blkgemm_handle* handle, const int M, const int MB, const int N, const int NB);


#ifdef __cplusplus
}
#endif
#endif /* LIBXS_BGEMM_H */
