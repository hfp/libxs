/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_tas.h>
#include <libxs/libxs_malloc.h>
#include <libxs/libxs_sync.h>

/* extent of a panel along M or N, and of a chunk along K */
#if !defined(LIBXS_TAS_PANEL)
# define LIBXS_TAS_PANEL 256
#endif
#if !defined(LIBXS_TAS_CHUNK)
# define LIBXS_TAS_CHUNK 4096
#endif
/* FLOPs per Byte below which a GEMM is planned along its longest dimension (as LIBXS_GEMM_JIT_MAX) */
#if !defined(LIBXS_TAS_MAX)
# define LIBXS_TAS_MAX 7
#endif

#define INTERNAL_TAS_NOTRANS(C) ('N' == (C) || 'n' == (C))
/* the count of finished tasks takes a cache line of its own in front of the partial C */
#define INTERNAL_TAS_HEAD LIBXS_CACHELINE

#define INTERNAL_TAS_REDUCE(TYPE, PLAN, TILES, C) do { \
  const int internal_tas_m_ = (PLAN)->shape.m, internal_tas_n_ = (PLAN)->shape.n; \
  const int internal_tas_ldc_ = (PLAN)->shape.ldc; \
  const size_t internal_tas_tile_ = (size_t)internal_tas_m_ * internal_tas_n_; \
  const TYPE internal_tas_alpha_ = (TYPE)(PLAN)->shape.alpha; \
  const TYPE internal_tas_beta_ = (TYPE)(PLAN)->shape.beta; \
  const TYPE *const internal_tas_p_ = (const TYPE*)(TILES); \
  TYPE *const internal_tas_c_ = (TYPE*)(C); \
  int internal_tas_i_, internal_tas_j_, internal_tas_t_; \
  for (internal_tas_j_ = 0; internal_tas_j_ < internal_tas_n_; ++internal_tas_j_) { \
    for (internal_tas_i_ = 0; internal_tas_i_ < internal_tas_m_; ++internal_tas_i_) { \
      const size_t internal_tas_ij_ = (size_t)internal_tas_j_ * internal_tas_m_ + internal_tas_i_; \
      const size_t internal_tas_cij_ = (size_t)internal_tas_j_ * internal_tas_ldc_ + internal_tas_i_; \
      TYPE internal_tas_sum_ = internal_tas_p_[internal_tas_ij_]; \
      for (internal_tas_t_ = 1; internal_tas_t_ < (PLAN)->nscratch; ++internal_tas_t_) { \
        internal_tas_sum_ += internal_tas_p_[internal_tas_t_ * internal_tas_tile_ + internal_tas_ij_]; \
      } \
      /* BLAS does not read C when beta is zero (C may hold NaN, e.g., fresh memory) */ \
      internal_tas_c_[internal_tas_cij_] = ((TYPE)0 == internal_tas_beta_ \
        ? internal_tas_alpha_ * internal_tas_sum_ \
        : internal_tas_alpha_ * internal_tas_sum_ + internal_tas_beta_ * internal_tas_c_[internal_tas_cij_]); \
    } \
  } \
} while(0)


LIBXS_APIVAR_DEFINE(int internal_libxs_tas_panel);
LIBXS_APIVAR_DEFINE(int internal_libxs_tas_chunk);
LIBXS_APIVAR_DEFINE(double internal_libxs_tas_max);


LIBXS_API_INLINE void internal_libxs_tas_init(void)
{
  static int internal_libxs_tas_init_once = 0;
  if (0 == internal_libxs_tas_init_once) {
    const char *const env_panel = getenv("LIBXS_TAS_PANEL");
    const char *const env_chunk = getenv("LIBXS_TAS_CHUNK");
    const char *const env_max = getenv("LIBXS_TAS_MAX");
    const int panel = (NULL == env_panel ? 0 : atoi(env_panel));
    const int chunk = (NULL == env_chunk ? 0 : atoi(env_chunk));
    internal_libxs_tas_panel = (0 < panel ? panel : LIBXS_TAS_PANEL);
    internal_libxs_tas_chunk = (0 < chunk ? chunk : LIBXS_TAS_CHUNK);
    internal_libxs_tas_max = (NULL == env_max ? LIBXS_TAS_MAX : atof(env_max));
    internal_libxs_tas_init_once = 1;
  }
}


LIBXS_API_INLINE void internal_libxs_tas_shape(libxs_gemm_shape_t* dst,
  const libxs_gemm_shape_t* src, char dim, int extent)
{
  *dst = *src;
  if ('M' == dim) dst->m = extent;
  else if ('N' == dim) dst->n = extent;
  else { /* partial C are tight, and alpha/beta apply when they are summed */
    dst->k = extent; dst->ldc = src->m;
    dst->alpha = 1.0; dst->beta = 0.0;
  }
}


/**
 * Geometry of the plan, kernels aside: dim stays zero for a GEMM that is not
 * bound by memory bandwidth, that yields fewer than two panels, or whose element
 * offsets exceed what the batch addresses (int).
 */
LIBXS_API_INLINE void internal_libxs_tas_plan(libxs_tas_t* plan,
  const libxs_gemm_shape_t* shape, int panel)
{
  const int ta = !INTERNAL_TAS_NOTRANS(shape->transa);
  const int tb = !INTERNAL_TAS_NOTRANS(shape->transb);
  const int m = shape->m, n = shape->n, k = shape->k;
  const double flops = 2.0 * m * n * k;
  const double nbytes = (double)LIBXS_TYPESIZE(shape->datatype)
    * ((double)m * k + (double)k * n + (double)m * n);
  char dim = 0;
  int extent = 0, w = 0;
  size_t stride_a = 0, stride_b = 0, stride_c = 0;
  if (flops < internal_libxs_tas_max * nbytes) {
    if (m >= n && m >= k) { dim = 'M'; extent = m; }
    else if (n >= k) { dim = 'N'; extent = n; }
    else { dim = 'K'; extent = k; }
    w = (0 < panel ? panel : ('K' == dim ? internal_libxs_tas_chunk : internal_libxs_tas_panel));
    w = LIBXS_MIN(w, extent);
  }
  if ('M' == dim) {
    stride_a = (size_t)w * (0 != ta ? shape->lda : 1);
    stride_c = (size_t)w;
  }
  else if ('N' == dim) {
    stride_b = (size_t)w * (0 != tb ? 1 : shape->ldb);
    stride_c = (size_t)w * shape->ldc;
  }
  else if ('K' == dim) {
    stride_a = (size_t)w * (0 != ta ? 1 : shape->lda);
    stride_b = (size_t)w * (0 != tb ? shape->ldb : 1);
    stride_c = (size_t)m * n;
  }
  if (0 != dim) {
    const int npanels = extent / w;
    const size_t nmax = (size_t)INT_MAX / npanels;
    if (2 <= npanels && stride_a <= nmax && stride_b <= nmax && stride_c <= nmax) {
      const int nrest = extent - npanels * w;
      plan->dim = dim;
      plan->npanels = npanels;
      plan->stride_a = (int)stride_a;
      plan->stride_b = (int)stride_b;
      plan->stride_c = (int)stride_c;
      plan->rest_a = stride_a * npanels;
      plan->rest_b = stride_b * npanels;
      plan->rest_c = stride_c * npanels;
      if ('K' == dim) plan->nscratch = npanels + (0 < nrest ? 1 : 0);
      internal_libxs_tas_shape(&plan->panel.shape, shape, dim, w);
      if (0 < nrest) internal_libxs_tas_shape(&plan->rest.shape, shape, dim, nrest);
    }
  }
}


/* dispatches into config, whose shape the plan has set, hinted or not */
LIBXS_API_INLINE int internal_libxs_tas_dispatch(libxs_gemm_config_t* config,
  int hint, const libxs_gemm_backend_t* backend, void* registry)
{
  const libxs_gemm_shape_t shape = config->shape;
  LIBXS_MEMZERO(config);
  if (0 != hint) {
    config->flags = (libxs_gemm_flags_t)(LIBXS_GEMM_FLAG_STREAM | LIBXS_GEMM_FLAG_REUSE);
  }
  return libxs_gemm_dispatch_cpy_rt(config, &shape, NULL, backend, registry);
}


LIBXS_API int libxs_tas_dispatch_rt(libxs_tas_t* plan,
  const libxs_gemm_shape_t* shape, int panel,
  const libxs_gemm_backend_t* backend, void* registry)
{
  int result = 0;
  if (NULL != plan && NULL != shape
    && (LIBXS_DATATYPE_F64 == shape->datatype || LIBXS_DATATYPE_F32 == shape->datatype)
    && 0 < shape->m && 0 < shape->n && 0 < shape->k)
  {
    internal_libxs_tas_init();
    LIBXS_MEMZERO(plan);
    plan->shape = *shape;
    internal_libxs_tas_plan(plan, shape, panel);
    if (0 != plan->dim) {
      result = internal_libxs_tas_dispatch(&plan->panel, 1, backend, registry);
      if (0 != result) { /* panels never share a C */
        plan->panel.flags = (libxs_gemm_flags_t)(plan->panel.flags | LIBXS_GEMM_FLAG_NOLOCK);
      }
      if (0 != result && 0 < plan->rest.shape.m) {
        result = internal_libxs_tas_dispatch(&plan->rest, 1, backend, registry);
      }
      if (0 != result && 0 != plan->nscratch) {
        const size_t size = INTERNAL_TAS_HEAD + (size_t)plan->nscratch
          * shape->m * shape->n * LIBXS_TYPESIZE(shape->datatype);
        plan->scratch = libxs_malloc(libxs_default_pool(), size, LIBXS_CACHELINE);
        if (NULL != plan->scratch) memset(plan->scratch, 0, INTERNAL_TAS_HEAD);
        else result = 0;
      }
    }
    else {
      plan->panel.shape = *shape;
      result = internal_libxs_tas_dispatch(&plan->panel, 0, backend, registry);
    }
    if (0 == result) libxs_tas_release(plan);
  }
  return result;
}


LIBXS_API void libxs_tas_task(const libxs_tas_t* plan,
  const void* a, const void* b, void* c,
  int tid, int ntasks)
{
  LIBXS_ASSERT(NULL != plan);
  if (0 <= tid && tid < ntasks) {
    if (0 == plan->dim) {
      if (0 == tid) libxs_gemm_call(&plan->panel, a, b, c);
    }
    else {
      const size_t esize = LIBXS_TYPESIZE(plan->shape.datatype);
      char *const cp = (0 != plan->nscratch
        ? ((char*)plan->scratch + INTERNAL_TAS_HEAD) : (char*)c);
      libxs_gemm_index_task(a, &plan->stride_a, b, &plan->stride_b, cp, &plan->stride_c,
        0/*index_stride*/, 0/*index_base*/, plan->npanels, &plan->panel, tid, ntasks);
      /* the last task takes the fewest panels (libxs_gemm_index_task), hence the remainder */
      if (0 < plan->rest.shape.m && ntasks - 1 == tid) {
        libxs_gemm_call(&plan->rest,
          (const char*)a + plan->rest_a * esize,
          (const char*)b + plan->rest_b * esize,
          cp + plan->rest_c * esize);
      }
      if (0 != plan->nscratch) {
        int *const nfinished = (int*)plan->scratch;
        /* the last task to finish sees every partial C, and resets the count for the next call */
        if (ntasks == LIBXS_ATOMIC_ADD_FETCH(nfinished, 1, LIBXS_ATOMIC_SEQ_CST)) {
          if (LIBXS_DATATYPE_F64 == plan->shape.datatype) {
            INTERNAL_TAS_REDUCE(double, plan, cp, c);
          }
          else {
            INTERNAL_TAS_REDUCE(float, plan, cp, c);
          }
          LIBXS_ATOMIC_STORE(nfinished, 0, LIBXS_ATOMIC_SEQ_CST);
        }
      }
    }
  }
}


LIBXS_API void libxs_tas(const libxs_tas_t* plan,
  const void* a, const void* b, void* c)
{
  libxs_tas_task(plan, a, b, c, 0, 1);
}


LIBXS_API int libxs_tas_ntasks(const libxs_tas_t* plan, int nthreads)
{
  int result = 1;
  if (NULL != plan && 0 != plan->dim) {
    result = libxs_gemm_ntasks(&plan->panel, plan->npanels, nthreads);
  }
  return result;
}


LIBXS_API void libxs_tas_release(libxs_tas_t* plan)
{
  if (NULL != plan) {
    libxs_free(plan->scratch);
    plan->scratch = NULL;
  }
}
