# GEMM: Matrix Multiplication

Header: `libxs_gemm.h`

Batched general matrix-matrix multiplication (GEMM). Operations are expressed as C := alpha * op(A) * op(B) + beta * C, where op() is an optional transpose. The caller can supply BLAS or MKL JIT kernels; otherwise a built-in default kernel (auto-vectorized) is used.

## Types

```C
typedef void (*libxs_dgemm_blas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc);
```

Standard Fortran BLAS dgemm signature (e.g., `dgemm_`).

```C
typedef void (*libxs_sgemm_blas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc);
```

Standard Fortran BLAS sgemm signature (e.g., `sgemm_`).

```C
typedef void (*libxs_dgemm_jit_t)(void* jitter,
  const double* a, const double* b, double* c);
typedef void (*libxs_sgemm_jit_t)(void* jitter,
  const float* a, const float* b, float* c);
```

MKL JIT kernel signatures. Shape, alpha, beta, and transpose info are baked into the jitter handle.

```C
typedef enum libxs_gemm_flags_t {
  LIBXS_GEMM_FLAGS_DEFAULT = 0,
  LIBXS_GEMM_FLAG_NOLOCK = 1
} libxs_gemm_flags_t;
```

Flags controlling batch synchronization. By default, `_task` variants synchronize C-matrix updates. Set `LIBXS_GEMM_FLAG_NOLOCK` when no duplicate C pointers exist.

```C
typedef struct libxs_gemm_config_t {
  libxs_dgemm_blas_t dgemm_blas;
  libxs_sgemm_blas_t sgemm_blas;
  libxs_dgemm_jit_t dgemm_jit;
  libxs_sgemm_jit_t sgemm_jit;
  void* jitter;
  libxs_gemm_flags_t flags;
} libxs_gemm_config_t;
```

Configuration supplying GEMM kernels. Pass NULL to batch functions to use the built-in default kernel. Kernel selection priority: (1) JIT kernel if non-NULL, (2) BLAS kernel if non-NULL, (3) built-in default. Only the function pointers matching the datatype need to be set.

## Functions

### Strided Batch

```C
void libxs_gemm_strided(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, int stride_a,
                     const void* b, int ldb, int stride_b,
  const void* beta,        void* c, int ldc, int stride_c,
  int batchsize, const libxs_gemm_config_t* config);
```

Process a batch of GEMMs with strided access (constant offsets between matrices). Matrices are at A + i\*stride\_a\*elemsize, etc. Pass `config=NULL` to use the built-in kernel.

```C
void libxs_gemm_strided_task(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a, int lda, int stride_a,
                     const void* b, int ldb, int stride_b,
  const void* beta,        void* c, int ldc, int stride_c,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);
```

Per-thread form of `libxs_gemm_strided`. The caller distributes work by passing its `tid` and `ntasks`.

### Pointer-Array Batch

```C
void libxs_gemm_batch(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a_array[], int lda,
                     const void* b_array[], int ldb,
  const void* beta,        void* c_array[], int ldc,
  int batchsize, const libxs_gemm_config_t* config);
```

Process a batch of GEMMs given arrays of pointers to matrices.

```C
void libxs_gemm_batch_task(
  libxs_data_t datatype, const char* transa, const char* transb,
  int m, int n, int k,
  const void* alpha, const void* a_array[], int lda,
                     const void* b_array[], int ldb,
  const void* beta,        void* c_array[], int ldc,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);
```

Per-thread form of `libxs_gemm_batch`.

### Grouped Batch

```C
void libxs_gemm_groups(
  libxs_data_t datatype,
  const char transa_array[], const char transb_array[],
  const int m_array[], const int n_array[], const int k_array[],
  const void* alpha_array, const void* a_array[], const int lda_array[],
                           const void* b_array[], const int ldb_array[],
  const void* beta_array,        void* c_array[], const int ldc_array[],
  int ngroups, const int batchsize[],
  const libxs_gemm_config_t* config);
```

Process groups of batched GEMMs with varying parameters. Each group has its own transa, transb, m, n, k, lda, ldb, ldc, and batchsize. The pointer arrays are concatenated across groups. `alpha_array` and `beta_array` are arrays of `ngroups` scalars (each `LIBXS_TYPESIZE` bytes).
