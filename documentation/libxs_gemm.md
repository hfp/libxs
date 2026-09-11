# GEMM: Matrix Multiplication

Header: `libxs_gemm.h`
Fortran: `USE LIBXS` (libxs/libxs.f)

Batched general matrix-matrix multiplication (GEMM) and symmetric
rank-k/2k updates. Operations are expressed as
C := alpha * op(A) * op(B) + beta * C, where op() is an optional
transpose. Kernels are dispatched via MKL JIT, LIBXSMM, or BLAS;
a built-in default kernel (auto-vectorized) is used as fallback.

## Types

```C
typedef struct libxs_gemm_shape_t {
  libxs_data_t datatype;
  char transa, transb;
  int m, n, k, lda, ldb, ldc;
  double alpha, beta;
} libxs_gemm_shape_t;
```

GEMM shape: problem geometry, transpose flags, and scalar
coefficients. Serves as registry key when caching dispatched
configurations. The key is hashed byte-wise, hence dispatch builds a
zero-initialized copy rather than hashing a caller-supplied struct
(the two chars leave padding before `m`).

```C
typedef struct libxs_gemm_config_t {
  libxs_gemm_dblas_t dgemm_blas;
  libxs_gemm_sblas_t sgemm_blas;
  libxs_gemm_djit_t dgemm_jit;
  libxs_gemm_sjit_t sgemm_jit;
  libxs_gemm_xfn_t xgemm;
  void* jitter;
  libxs_gemm_flags_t flags;
  int warmup;
  libxs_gemm_shape_t shape;
} libxs_gemm_config_t;
```

Configuration holding dispatched GEMM kernels. Kernel priority:
1. JIT kernel (dgemm_jit/sgemm_jit + jitter),
2. XGEMM kernel (xgemm),
3. BLAS kernel (dgemm_blas/sgemm_blas) -- always non-NULL after
   dispatch (falls back to built-in auto-vectorized C code).

warmup counts dispatches of this shape and is maintained by
dispatch (see LIBXS_GEMM_JIT_WARMUP).

```C
typedef enum libxs_gemm_flags_t {
  LIBXS_GEMM_FLAGS_DEFAULT = 0,
  LIBXS_GEMM_FLAG_NOLOCK = 1,
  LIBXS_GEMM_FLAG_OWNJIT = 2
} libxs_gemm_flags_t;
```

Flags controlling batch synchronization. Set LIBXS_GEMM_FLAG_NOLOCK
when no duplicate C pointers exist across the batch.
LIBXS_GEMM_FLAG_OWNJIT is set by dispatch and marks the single config
owning `jitter`; aliasing configs (double-dispatch, and every caller
copy) leave it clear so the handle is released exactly once.

## Dispatch

### C (inline, compile-time backend selection)

```C
libxs_gemm_config_t* libxs_gemm_dispatch(
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta,
  void* registry /* = NULL */);
```

Inline function that selects the backend at compile time:

- MKL JIT (if mkl.h is included before libxs_gemm.h),
- LIBXSMM (if libxsmm.h is included),
- BLAS dgemm/sgemm (if __BLAS, __MKL, or MKL_H is defined),
- built-in default otherwise.

On registry hit, returns a pointer to the cached config (hash
probe only). On miss, dispatches a new kernel and stores it.
If registry is NULL, an internal registry is used.

The detection itself is exposed, so a caller assembling shapes by
hand can reach `libxs_gemm_dispatch_rt` without repeating it:

```C
void libxs_gemm_backend_init(libxs_gemm_backend_t* backend);
```

It zero-initializes backend and fills the pointers available at the
caller's compile time. Unavailable backends are left NULL, i.e.,
dispatch falls through to the built-in default kernel.

### Config ownership (registry pointer or caller copy)

Every dispatch function comes in two flavors. The unsuffixed C form
returns a pointer to the registry-owned config; the `_cpy` form
populates a caller-owned config instead:

```C
int libxs_gemm_dispatch_cpy(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta,
  void* registry /* = NULL */);

int libxs_gemm_config_cpy(
  libxs_gemm_config_t* config, const libxs_gemm_config_t* cached);
```

Both return nonzero if config was populated (config is left
untouched otherwise, hence zero-initialize it before the call).
`libxs_gemm_config_cpy` converts a pointer obtained from any
dispatch function into a caller-owned config.

Pick the flavor by lifetime, not by convenience:

| Flavor | Use when |
| :--- | :--- |
| pointer | the config is cached across many calls and should pick up a kernel once JIT warm-up completes |
| copy | flags are modified, per-thread state is wanted, or dispatch happens per call anyway |

A copy is a snapshot. It is not upgraded in place when warm-up
completes later, so re-dispatch to pick up a kernel. Writing
`flags` through a registry pointer affects every user of that
shape, hence a copy is the safe place for `LIBXS_GEMM_FLAG_NOLOCK`.
`LIBXS_GEMM_FLAG_OWNJIT` is cleared in a copy: the JIT handle stays
owned by the registry, and `libxs_gemm_release` is a no-op for the
copy (release the registry, or the config the registry returned).

In Fortran the plain name covers both flavors, and the first
argument selects: pass a config to fill it (copy), omit it to
receive the registry-owned pointer. The suffixed names remain and
name one flavor each, which is what C has to do throughout:

| Operation | C pointer | C copy | Fortran (either) | Fortran pointer | Fortran copy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GEMM | `libxs_gemm_dispatch` | `libxs_gemm_dispatch_cpy` | `libxs_gemm_dispatch` | `libxs_gemm_dispatch_ptr` | `libxs_gemm_dispatch` |
| SYR2K | `libxs_syr2k_dispatch` | `libxs_syr2k_dispatch_cpy` | `libxs_syr2k_dispatch` | `libxs_syr2k_dispatch` | `libxs_syr2k_dispatch_cpy` |
| SYRK | `libxs_syrk_dispatch` | `libxs_syrk_dispatch_cpy` | `libxs_syrk_dispatch` | `libxs_syrk_dispatch` | `libxs_syrk_dispatch_cpy` |

The result type follows the flavor, so the call tells which one ran:

```fortran
ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, lda, ldc)
rc = libxs_syrk_dispatch(config, LIBXS_DATATYPE_F64, n, k, lda, ldc)
```

All of the above are inlines (or Fortran wrappers) that detect the
backend at the caller's compile time. Backends are never a runtime
argument there, which is what keeps LIBXS decoupled from MKL,
LIBXSMM, and BLAS. To supply backend pointers explicitly, use the
`_rt` forms, which are the only dispatch symbols exported by the
library itself:

| Operation | Explicit backend (library symbol) |
| :--- | :--- |
| GEMM | `libxs_gemm_dispatch_rt` (takes shape structs) |
| SYR2K | `libxs_syr2k_dispatch_rt` |
| SYRK | `libxs_syrk_dispatch_rt` |

### C (runtime, explicit backend selection)

```C
typedef struct libxs_gemm_backend_t {
  libxs_jit_create_dgemm_t jit_create_dgemm;
  libxs_jit_get_dgemm_t   jit_get_dgemm;
  libxs_jit_create_sgemm_t jit_create_sgemm;
  libxs_jit_get_sgemm_t   jit_get_sgemm;
  libxs_xgemm_dispatch_t  xgemm_dispatch;
  libxs_gemm_dblas_t dgemm_blas;
  libxs_gemm_sblas_t sgemm_blas;
} libxs_gemm_backend_t;

libxs_gemm_config_t* libxs_gemm_dispatch_rt(
  const libxs_gemm_shape_t* shape,
  const libxs_gemm_shape_t* kernel_shape,
  const libxs_gemm_backend_t* backend,
  void* registry);
```

Non-inline function that accepts backend and shape structs.
shape: full problem shape (registry key, stored in config).
kernel_shape: actual kernel dimensions (may differ, e.g., tight
ldc for scratch). NULL means same as shape. If kernel_shape
differs from shape, the kernel is looked up under kernel_shape
first (double-dispatch), avoiding redundant code generation.
backend: function pointers for backends. NULL means built-in
default only. Same registry semantics as above.
`LIBXS_GEMM_BACKEND` can restrict the starting point of the
runtime fallback chain: 0 = automatic/default, 1 = MKL JIT,
2 = LIBXSMM, 3 = BLAS/MKL, 4 = built-in fallback. Choices 1-3
still fall through to lower-priority backends if the requested
backend is not supplied or cannot dispatch the shape.

Backend callback signatures (MKL-compatible):

    jit_create_dgemm: int(void** jitter, int layout, int transa,
                          int transb, int m, int n, int k,
                          double alpha, int lda, int ldb,
                          double beta, int ldc)
                      Return: MKL_JIT_SUCCESS (0) or
                              MKL_NO_JIT (1) on success,
                              MKL_JIT_ERROR (2) on failure.

    jit_get_dgemm:    void*(void* jitter)
                      Return kernel function pointer.

    xgemm_dispatch:   libxs_gemm_xfn_t(
                          libxs_xgemm_shape_t shape /* by value */,
                          unsigned int gemm_flags,
                          unsigned int prefetch_flags)
                      Layout-compatible with libxsmm_dispatch_gemm.
                      shape: m,n,k,lda,ldb,ldc then a_in_type,
                             b_in_type, out_type, comp_type.
                      gemm_flags: bit 0 = transa, bit 1 = transb,
                             bit 2 = beta==0.
                      Dispatch is attempted only for alpha==1 with
                      beta in {0,1}; other scalars skip this backend.

### Fortran (LIBXS_JIT, recommended)

```fortran
USE LIBXS_JIT
rc = libxs_gemm_dispatch(config, LIBXS_DATATYPE_F64,
     &  'N', 'N', m, n, k, lda, ldb, ldc, alpha, beta)
```

The LIBXS_JIT adapter module provides the same dispatch API
but fills in BLAS (and MKL JIT when compiled with __MKL)
automatically. Requires BLAS at link time.

### Fortran (LIBXS, explicit backends)

```fortran
USE LIBXS
rc = libxs_gemm_dispatch(config, datatype, transa, transb,
     &  m, n, k, lda, ldb, ldc, alpha, beta,
     &  jit_create_dgemm=..., jit_get_dgemm=...,
     &  dgemm_blas=..., registry=...)
```

All backend arguments are OPTIONAL C_FUNPTR (named arguments).
Returns nonzero on success (dispatch produced a callable config).
The config is populated from the registry-owned copy.

## Single-Kernel Call

```C
void libxs_gemm_call(
  const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c);
```

Call the dispatched GEMM kernel (JIT > XGEMM > BLAS fallback).
The caller must ensure config is non-NULL (dispatch succeeded).

## Release

```C
void libxs_gemm_release(libxs_gemm_config_t* config);
```

Release resources (e.g., MKL jitter handle) held by config, and
clear the JIT kernel such that config falls back to XGEMM/BLAS.
A handle can be shared by several configs (double dispatch), hence
only the owning config (LIBXS_GEMM_FLAG_OWNJIT, set by dispatch)
releases it and a handle is released exactly once. A dispatched
config is owned by the registry, i.e., releasing it affects every
user of that registry: release at teardown, or rely on
libxs_gemm_release_registry.

```C
void libxs_gemm_release_registry(libxs_registry_t* registry);
```

Release all configs in a registry, then destroy the registry.

## Pointer-Array Batch

```C
void libxs_gemm_batch(
  const void* a_array[], const void* b_array[], void* c_array[],
  int batchsize, const libxs_gemm_config_t* config);

void libxs_gemm_batch_task(
  const void* a_array[], const void* b_array[], void* c_array[],
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);
```

Batch of GEMMs from pointer arrays. The _task variant splits
work across ntasks threads (tid = 0..ntasks-1).

## Index/Strided Batch

```C
void libxs_gemm_index(
  const void* a, const int stride_a[],
  const void* b, const int stride_b[],
        void* c, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config);

void libxs_gemm_index_task(
  const void* a, const int stride_a[],
  const void* b, const int stride_b[],
        void* c, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);
```

Batch of GEMMs from element-offset index arrays into contiguous
buffers. index_base: 0 (C) or 1 (Fortran). index_stride: byte
stride between consecutive index entries (sizeof(int) for packed
arrays, 0 for constant-stride mode).

## SYR2K / SYRK

Symmetric rank-2k and rank-k updates built on top of GEMM dispatch.

For small problems (n <= LIBXS_GEMM_BLOCK_M and k <= LIBXS_GEMM_BLOCK_K),
the dispatched kernel handles the full GEMM in one call. For larger
problems, the implementation tiles the output into blocks and
accumulates along K. Full-size tiles use the dispatched JIT kernel;
remainder tiles fall back to BLAS (if available) or the built-in
default. Scratch memory is thread-local.

### Dispatch

```C
libxs_gemm_config_t* libxs_syr2k_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  void* registry /* = NULL */);

libxs_gemm_config_t* libxs_syrk_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  void* registry /* = NULL */);

int libxs_syr2k_dispatch_cpy(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  void* registry /* = NULL */);

int libxs_syrk_dispatch_cpy(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  void* registry /* = NULL */);
```

Dispatch a GEMM config for SYR2K/SYRK. The problem shape (n, k, lda,
ldb, ldc) is stored in the config and used by the call functions,
while the dispatched kernel is the SYRK tile, i.e., the blocking
(LIBXS_GEMM_BM/BN/BK) is applied inside. These are inlines and detect
the backend at the caller's compile time, exactly like
libxs_gemm_dispatch. The `_cpy` flavors populate a caller-owned config
(see [Config ownership](#config-ownership-registry-pointer-or-caller-copy)).

To pass backend pointers explicitly, use the runtime forms:

```C
libxs_gemm_config_t* libxs_syr2k_dispatch_rt(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  const libxs_gemm_backend_t* backend /* = NULL */,
  void* registry /* = NULL */);

libxs_gemm_config_t* libxs_syrk_dispatch_rt(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  const libxs_gemm_backend_t* backend /* = NULL */,
  void* registry /* = NULL */);
```

Fortran mirrors both: LIBXS_JIT auto-detects, module LIBXS accepts
OPTIONAL backend function pointers.

```fortran
ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, lda, ldc,
     &  jit_create_dgemm=C_FUNLOC(mkl_cblas_jit_create_dgemm),
     &  jit_get_dgemm=C_FUNLOC(mkl_jit_get_dgemm_ptr),
     &  dgemm_blas=C_FUNLOC(DGEMM))

rc = libxs_syrk_dispatch_cpy(config, LIBXS_DATATYPE_F64,
     &  n, k, lda, ldc)
```

### Call

```C
void libxs_syr2k(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c);

void libxs_syr2k_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c,
  int tid, int ntasks);
```

C := alpha*(A*B^T + B*A^T) + beta*C. Only the triangle specified
by uplo ('U' or 'L') is written. All dimensions and leading
dimensions come from the dispatched config.

```C
void libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c);

void libxs_syrk_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c,
  int tid, int ntasks);
```

C := alpha*A*A^T + beta*C. Only the triangle specified by uplo
('U' or 'L') is written.

The _task variants partition work across ntasks threads. Each
thread operates on independent output blocks; no locking is
required. Thread-local scratch buffers are used internally.

## Compile-Time Tuning

The block sizes used for tiled SYRK/SYR2K can be overridden at
compile time via preprocessor defines:

    LIBXS_GEMM_BM        Row block size    (default: 24)
    LIBXS_GEMM_BN        Column block size (default: 48)
    LIBXS_GEMM_BK        K-direction block (default: 128)

Problems fitting within these limits use a single specialized
kernel call (MKL JIT or LIBXSMM when available). The same values are
the SYRK tile dispatched as the kernel shape, hence they decide which
kernel the arithmetic-intensity gate (LIBXS_GEMM_JIT_MAX) sees.

    LIBXS_GEMM_NWARMUP   JIT warm-up counters (default: 4096)

Number of slots in the table holding the per-shape reuse counters
(must be a power of two). Shapes share slots by hash; a collision is
detected by an identity tag stored alongside the counter, and the
losing shape simply restarts counting rather than inheriting foreign
progress. Sizing this below the number of distinct shapes in flight
therefore degrades to later JIT, never to a wrong kernel.

## Environment Variables

    LIBXS_GEMM_BM=N         Row block size for tiled SYRK/SYR2K
                            (default: 24).
    LIBXS_GEMM_BN=N         Column block size (default: 48).
    LIBXS_GEMM_BK=N         K-direction block size (default: 128).
    LIBXS_GEMM_BACKEND=N    Select runtime backend chain start:
                            0 auto (default), 1 MKL JIT,
                            2 LIBXSMM, 3 BLAS, 4 built-in fallback.
    LIBXS_GEMM_JIT_MAX=N    Arithmetic-intensity threshold for JIT
                            dispatch. JIT/LIBXSMM kernels are only
                            generated when the kernel shape's AI
                            (flops/bytes) is below N (default: 7,
                            roughly the AI of an 80x80x80 kernel).
                            Set to 0 to disable JIT entirely.
    LIBXS_GEMM_JIT_WARMUP=N Number of calls a shape must
                            accumulate before JIT compilation is
                            attempted. Shapes called fewer than N
                            times use the BLAS fallback; only shapes
                            with proven reuse pay the JIT compile
                            cost. Reuse is counted per dispatch, hence
                            the first shape of an empty registry is
                            compiled immediately (a caller dispatching
                            only once cannot accumulate calls).
                            Counted in a fixed-size table, not in the
                            registry (see LIBXS_GEMM_NWARMUP); once a
                            shape's JIT has been attempted it is never
                            attempted again, whether it succeeded or
                            was refused by LIBXS_GEMM_JIT_MAX.
                            Set to 0 or 1 to compile immediately
                            on first miss (default: 8, clamped to 254).
    LIBXS_GEMM_PRINT=N      Print dispatch info every N-th call
                            to stderr (requires compile-time gate).
                            Set to 0 for a summary at teardown instead
                            (registry, histogram, and warm-up slots).
                            Every line carries the origin of the
                            process, because ranks share one stream:
                            "LIBXS INFO[rid]:" with rid as libxs_rid,
                            hence the local rank where it is known and
                            the process id otherwise.
    LIBXS_SYRK_PRINT=N      Print DSYRK/DSYR2K BLAS fallback calls
                            to stderr (requires compile-time gate).
    LIBXS_SYRK_BLAS=0|1     Resolve the BLAS DSYRK/DSYR2K entry points
                            and prefer them for any shape wider than
                            one tile. The tiled path (and with it the
                            kernel of the dispatched config) is then
                            unreachable, so the SYRK dispatch neither
                            compiles a kernel nor takes a registry
                            entry for such a shape. Set to 0 to tile
                            and to make the kernel matter (default: 1).

## Example (C)

```C
#include <libxs/libxs_gemm.h>

libxs_registry_t* reg = libxs_registry_create();

/* dispatch once per unique shape */
const libxs_gemm_config_t* cfg = libxs_syr2k_dispatch(
  LIBXS_DATATYPE_F64, n, k, lda, ldb, ldc, reg);

/* call many times (registry hit = hash probe only) */
for (batch = 0; batch < nbatches; ++batch) {
  libxs_syr2k(cfg, 'U', 0.5, 0.0, a[batch], b[batch], c[batch]);
}

libxs_gemm_release_registry(reg);
```

## Example (Fortran)

```fortran
USE :: LIBXS

TYPE(libxs_gemm_config_t) :: config
INTEGER(C_INT) :: rc

rc = libxs_gemm_dispatch(config, LIBXS_DATATYPE_F64,
     &  'N', 'N', m, n, k, lda, ldb, ldc, alpha, beta,
     &  dgemm_blas=C_FUNLOC(DGEMM))

IF (0 /= rc) THEN
  CALL libxs_gemm_call(config, C_LOC(a), C_LOC(b), C_LOC(c))
END IF
! rc /= 0 guarantees libxs_gemm_call will succeed
! (JIT, XGEMM, or BLAS/built-in fallback)
```
