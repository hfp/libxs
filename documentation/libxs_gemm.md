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
- built-in default otherwise.

On registry hit, returns a pointer to the cached config (hash
probe only). On miss, dispatches a new kernel and stores it.
If registry is NULL, an internal registry is used.

### C (runtime, explicit backend selection)

```C
libxs_gemm_config_t* libxs_gemm_dispatch_rt(
  int datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta,
  libxs_jit_create_dgemm_t jit_create_dgemm,
  libxs_jit_get_dgemm_t   jit_get_dgemm,
  libxs_jit_create_sgemm_t jit_create_sgemm,
  libxs_jit_get_sgemm_t   jit_get_sgemm,
  libxs_xgemm_dispatch_t  xgemm_dispatch,
  libxs_gemm_dblas_t dgemm_blas,
  libxs_gemm_sblas_t sgemm_blas,
  void* registry);
```

Non-inline function that accepts backend function pointers
explicitly. All pointers are nullable (NULL = skip that backend).
This is the single entry point used by both the C inline wrapper
and the Fortran module. Same registry semantics as above.

Backend callback signatures (MKL-compatible):

    jit_create_dgemm: int(void** jitter, int layout, int transa,
                          int transb, int m, int n, int k,
                          double alpha, int lda, int ldb,
                          double beta, int ldc)
                      Return 0 on success.

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
Scratch memory is managed internally (thread-local, grown as needed).

### Dispatch

```C
libxs_gemm_config_t* libxs_syr2k_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  void* registry /* = NULL */);

libxs_gemm_config_t* libxs_syrk_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  void* registry /* = NULL */);
```

Dispatch a GEMM config for SYR2K/SYRK. Internally dispatches
GEMM('N','T', n, n, k, ...) with alpha=1, beta=0. The returned
config stores ldc for the output matrix. Returns NULL on failure.

Fortran variants accept the same OPTIONAL backend function
pointers as libxs_gemm_dispatch:

```fortran
ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, lda, ldc,
     &  dgemm_blas=C_FUNLOC(DGEMM))
```

### Call

```C
int libxs_syr2k(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c);
```

C := alpha*(A*B^T + B*A^T) + beta*C. Only the triangle specified
by uplo ('U' or 'L') is written.

```C
int libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c);
```

C := alpha*A*A^T + beta*C. Only the triangle specified by uplo
('U' or 'L') is written.

## Environment Variables

LIBXS_SYRK=N  Print registry info every N-th syr2k/syrk dispatch
               call to stderr (diagnostic, compile-time optional
               via LIBXS_SYRK_TRACE).

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
