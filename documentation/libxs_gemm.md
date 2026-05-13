# GEMM: Matrix Multiplication

Header: `libxs_gemm.h`
Fortran: `USE LIBXS` (include/libxs.f)

Batched general matrix-matrix multiplication (GEMM) and symmetric
rank-k/2k updates. Operations are expressed as
C := alpha * op(A) * op(B) + beta * C, where op() is an optional
transpose. Kernels are dispatched via MKL JIT, LIBXSMM, or BLAS;
a built-in default kernel (auto-vectorized) is used as fallback.

## Types

```C
typedef struct libxs_gemm_shape_t {
  libxs_data_t datatype;
  int transa, transb;
  int m, n, k, lda, ldb, ldc;
  double alpha, beta;
} libxs_gemm_shape_t;
```

GEMM shape: problem geometry, transpose flags, and scalar
coefficients. Serves as registry key when caching dispatched
configurations.

```C
typedef struct libxs_gemm_config_t {
  libxs_gemm_dblas_t dgemm_blas;
  libxs_gemm_sblas_t sgemm_blas;
  libxs_gemm_djit_t dgemm_jit;
  libxs_gemm_sjit_t sgemm_jit;
  libxs_gemm_xfn_t xgemm;
  void* jitter;
  libxs_gemm_flags_t flags;
  libxs_gemm_shape_t shape;
} libxs_gemm_config_t;
```

Configuration holding dispatched GEMM kernels. Kernel priority:
1. JIT kernel (dgemm_jit/sgemm_jit + jitter),
2. XGEMM kernel (xgemm),
3. BLAS kernel (dgemm_blas/sgemm_blas),
4. built-in default.

```C
typedef enum libxs_gemm_flags_t {
  LIBXS_GEMM_FLAGS_DEFAULT = 0,
  LIBXS_GEMM_FLAG_NOLOCK = 1
} libxs_gemm_flags_t;
```

Flags controlling batch synchronization. Set LIBXS_GEMM_FLAG_NOLOCK
when no duplicate C pointers exist across the batch.

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

    xgemm_dispatch:   libxs_gemm_xfn_t(int datatype, int flags,
                          int m, int n, int k,
                          int lda, int ldb, int ldc)
                      flags: bit 0 = transa, bit 1 = transb,
                             bit 2 = beta==0.

### Fortran

```fortran
rc = libxs_gemm_dispatch(config, datatype, transa, transb,
     &  m, n, k, lda, ldb, ldc, alpha, beta,
     &  jit_create_dgemm=..., jit_get_dgemm=...,
     &  dgemm_blas=..., registry=...)
```

All backend arguments are OPTIONAL C_FUNPTR (named arguments).
Returns nonzero if a usable kernel was obtained. The config is
populated from the registry-owned copy.

Typical usage with MKL JIT:

```fortran
rc = libxs_gemm_dispatch(config, LIBXS_DATATYPE_F64,
     &  'N', 'N', m, n, k, lda, ldb, ldc, alpha, beta,
     &  jit_create_dgemm=C_FUNLOC(mkl_jit_create_dgemm),
     &  jit_get_dgemm=C_FUNLOC(mkl_jit_get_dgemm_ptr))
```

Typical usage with BLAS only:

```fortran
rc = libxs_gemm_dispatch(config, LIBXS_DATATYPE_F64,
     &  'N', 'N', m, n, k, lda, ldb, ldc, alpha, beta,
     &  dgemm_blas=C_FUNLOC(DGEMM))
```

## Single-Kernel Call

```C
int libxs_gemm_ready(const libxs_gemm_config_t* config);
```

Returns nonzero if config holds a usable kernel.

```C
int libxs_gemm_call(
  const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c);
```

Call the dispatched GEMM kernel. Returns EXIT_SUCCESS on success.

## Release

```C
void libxs_gemm_release(libxs_gemm_config_t* config);
```

Release resources (e.g., MKL jitter handle) held by config.

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
  const libxs_gemm_backend_t* backend /* = NULL */,
  void* registry /* = NULL */);

libxs_gemm_config_t* libxs_syrk_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  const libxs_gemm_backend_t* backend /* = NULL */,
  void* registry /* = NULL */);
```

Dispatch a GEMM config for SYR2K/SYRK. The shape (n, k, lda, ldb,
ldc) is stored in the config and used by the call functions.
backend supplies JIT and BLAS function pointers (NULL = built-in
default). Returns NULL on failure.

Fortran variants accept OPTIONAL backend function pointers:

```fortran
ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, lda, ldc,
     &  jit_create_dgemm=C_FUNLOC(mkl_cblas_jit_create_dgemm),
     &  jit_get_dgemm=C_FUNLOC(mkl_jit_get_dgemm_ptr),
     &  dgemm_blas=C_FUNLOC(DGEMM))
```

### Call

```C
int libxs_syr2k(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c);

int libxs_syr2k_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c,
  int tid, int ntasks);
```

C := alpha*(A*B^T + B*A^T) + beta*C. Only the triangle specified
by uplo ('U' or 'L') is written. All dimensions and leading
dimensions come from the dispatched config.

```C
int libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c);

int libxs_syrk_task(
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

    LIBXS_GEMM_BLOCK_M   Row block size    (default: 32)
    LIBXS_GEMM_BLOCK_N   Column block size (default: BLOCK_M)
    LIBXS_GEMM_BLOCK_K   K-direction block (default: 128)

Problems fitting within these limits use a single specialized
kernel call (MKL JIT or LIBXSMM when available).

## Environment Variables

    LIBXS_GEMM_PRINT=N   Print dispatch info every N-th call
                          to stderr (compile-time gate).

## Example (C)

```C
#include <libxs_gemm.h>

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
  rc = libxs_gemm_call(config, C_LOC(a), C_LOC(b), C_LOC(c))
END IF
```
