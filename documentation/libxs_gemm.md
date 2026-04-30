# GEMM: Matrix Multiplication

Header: `libxs_gemm.h`

Batched general matrix-matrix multiplication (GEMM). Operations are expressed as C := alpha \* op(A) \* op(B) + beta \* C, where op() is an optional transpose. The caller can supply MKL JIT, LIBXSMM, or BLAS kernels; otherwise a built-in default kernel (auto-vectorized) is used.

## Types

```C
typedef void (*libxs_gemm_dblas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc);
```

Standard Fortran BLAS dgemm signature (e.g., `dgemm_`).

```C
typedef void (*libxs_gemm_sblas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc);
```

Standard Fortran BLAS sgemm signature (e.g., `sgemm_`).

```C
typedef void (*libxs_gemm_djit_t)(void* jitter,
  const double* a, const double* b, double* c);
typedef void (*libxs_gemm_sjit_t)(void* jitter,
  const float* a, const float* b, float* c);
```

MKL JIT kernel signatures. Shape, alpha, beta, and transpose info are baked into the jitter handle.

```C
typedef struct libxs_gemm_param_t {
  void* op[4]; const void* a[6]; const void* b[6]; void* c[6];
} libxs_gemm_param_t;
```

XGEMM parameter struct, layout-compatible with `libxsmm_gemm_param`. Only `a[0]` (a.primary), `b[0]` (b.primary), and `c[0]` (c.primary) are used for plain GEMM.

```C
typedef void (*libxs_gemm_xfn_t)(const void*);
```

Opaque XGEMM kernel signature: `void(const libxs_gemm_param_t*)`.

```C
typedef enum libxs_gemm_flags_t {
  LIBXS_GEMM_FLAGS_DEFAULT = 0,
  LIBXS_GEMM_FLAG_NOLOCK = 1
} libxs_gemm_flags_t;
```

Flags controlling batch synchronization. By default, `_task` variants synchronize C-matrix updates. Set `LIBXS_GEMM_FLAG_NOLOCK` when no duplicate C pointers exist.

```C
typedef struct libxs_gemm_shape_t {
  libxs_data_t datatype;
  int transa, transb;
  int m, n, k, lda, ldb, ldc;
  double alpha, beta;
} libxs_gemm_shape_t;
```

GEMM shape: problem geometry, transpose flags, and scalar coefficients. Alpha and beta are stored as double regardless of datatype (float values are promoted without loss). Also serves as registry key when caching dispatched configurations (all-int fields avoid padding).

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

Configuration supplying GEMM kernels. Pass NULL to batch functions to use the built-in default kernel (auto-vectorized, no BLAS dependency). Kernel selection priority:

1. JIT kernel (`dgemm_jit`/`sgemm_jit` + `jitter`) if non-NULL,
2. XGEMM kernel (`xgemm`) if non-NULL,
3. BLAS kernel (`dgemm_blas`/`sgemm_blas`) if non-NULL,
4. built-in default kernel.

Only the function pointers matching the datatype need to be set. The `shape` member is populated by `libxs_gemm_dispatch`.

## Functions

### Dispatch and Single-Kernel Call

```C
int libxs_gemm_dispatch(
  libxs_gemm_config_t* config,
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta,
  void* registry /* = NULL */);
```

Dispatch a GEMM kernel and populate `config` accordingly. With MKL JIT (highest priority): sets `config->dgemm_jit` or `sgemm_jit` + `jitter`. With LIBXSMM (fallback): sets `config->xgemm` via `libxsmm_dispatch_gemm`. Without either: config is unchanged (falls through to BLAS/default). The caller should memset config to zero before the first call. Returns `EXIT_SUCCESS` on success, `EXIT_FAILURE` when no JIT backend is available. If `registry` is non-NULL, dispatched configs are cached by shape: a hit copies the cached config, a miss dispatches and stores.

```C
int libxs_gemm_ready(const libxs_gemm_config_t* config);
```

Check whether a dispatched config holds a usable kernel. Returns nonzero if `libxs_gemm_call` would succeed, zero otherwise.

```C
int libxs_gemm_call(
  const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c);
```

Call the GEMM kernel previously dispatched into config. Follows the documented priority (JIT > XGEMM > BLAS fallback). Returns `EXIT_SUCCESS` if a kernel was called, `EXIT_FAILURE` otherwise.

```C
void libxs_gemm_release(libxs_gemm_config_t* config);
```

Release resources acquired by `libxs_gemm_dispatch` (e.g., MKL jitter). Safe to call even if dispatch was not used or returned zero.

```C
void libxs_gemm_release_registry(libxs_registry_t* registry);
```

Release all GEMM configs cached in a registry (e.g., MKL jitter), then destroy the registry itself. Safe to call with NULL.

### Pointer-Array Batch

```C
void libxs_gemm_batch(
  const void* a_array[], const void* b_array[], void* c_array[],
  int batchsize, const libxs_gemm_config_t* config);
```

Process a batch of GEMMs given arrays of pointers to matrices. Shape, alpha, beta, datatype, and transpose info come from `config->shape`.

```C
void libxs_gemm_batch_task(
  const void* a_array[], const void* b_array[], void* c_array[],
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);
```

Per-thread form of `libxs_gemm_batch`.

### Index/Strided Batch

```C
void libxs_gemm_index(
  const void* a, const int stride_a[],
  const void* b, const int stride_b[],
        void* c, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config);
```

Process a batch of GEMMs given index arrays into contiguous buffers. Each `stride_a[i]`, `stride_b[i]`, `stride_c[i]` is an element-offset from the respective base pointer. `index_base` selects the indexing convention: 0 for zero-based (C), 1 for one-based (Fortran). `index_stride` is the Byte-stride used to walk the stride arrays (e.g., `sizeof(int)` for packed int arrays). `index_stride=0` selects constant-stride mode: each stride array points to a single element, and batch `i` uses offset `stride[0]*i` (equivalent to the strided batch pattern). Shape, alpha, beta, datatype, and transpose info come from `config->shape`.

```C
void libxs_gemm_index_task(
  const void* a, const int stride_a[],
  const void* b, const int stride_b[],
        void* c, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);
```

Per-thread form of `libxs_gemm_index`.
