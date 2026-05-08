# GEMM: Matrix Multiplication

Header: `libxs_gemm.h`

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

## Functions

### Dispatch

```C
libxs_gemm_config_t* libxs_gemm_dispatch(
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta,
  void* registry /* = NULL */);
```

Dispatch a GEMM kernel. Returns pointer to a ready-to-use config
(registry-owned), or NULL on failure. On registry hit the lookup
is a hash-table probe (zero JIT cost). On miss the kernel is
JIT-compiled and stored. If registry is NULL, an internal registry
is used (lazy-initialized, destroyed by libxs_finalize).

### Single-Kernel Call

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

### Release

```C
void libxs_gemm_release(libxs_gemm_config_t* config);
```

Release resources (e.g., MKL jitter handle) held by config.

```C
void libxs_gemm_release_registry(libxs_registry_t* registry);
```

Release all configs in a registry, then destroy the registry.

### Pointer-Array Batch

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

### Index/Strided Batch

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

## Example

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
