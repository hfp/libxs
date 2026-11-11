# SYRK: Symmetric Rank-k Updates

Symmetric rank-2k and rank-k updates (`libxs_syrk.h`) built on top of
[GEMM](libxs_gemm.md) dispatch, i.e., the config, backend selection,
and registry are those of GEMM.

For small problems (n within LIBXS_GEMM_BM and LIBXS_GEMM_BN, and
k <= LIBXS_GEMM_BK), the dispatched kernel handles the full GEMM in one
call. Larger problems are split into column panels (see
LIBXS_GEMM_PANEL and LIBXS_SYRK_BLAS). Scratch memory is thread-local.

## Dispatch

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
[libxs_gemm_dispatch](libxs_gemm.md#dispatch). The `_cpy` flavors populate a caller-owned config
(see [Config ownership](libxs_gemm.md#config-ownership-registry-pointer-or-caller-copy)).

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

## Call

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

```C
int libxs_syrk_ntasks(const libxs_gemm_config_t* config, int nthreads);
```

Number of tasks worth splitting `libxs_syrk_task` or
`libxs_syr2k_task` into, given nthreads (1 means unsplit). Fortran:
`libxs_syrk_ntasks(config, nthreads)`.

In Fortran, `libxs_syrk` and `libxs_syr2k` also have a direct form that
dispatches and calls in one step:

```fortran
CALL libxs_syrk(config, uplo, alpha, beta, a, c)
CALL libxs_syrk(datatype, n, k, lda, ldc, uplo, alpha, beta, a, c)
CALL libxs_syr2k(config, uplo, alpha, beta, a, b, c)
CALL libxs_syr2k(datatype, n, k, lda, ldb, ldc, uplo, alpha, beta,
     &  a, b, c)
```

The direct form takes an OPTIONAL `registry` but no backend arguments.
When the Fortran module is compiled with OpenMP, both forms run over an
OpenMP team sized by `libxs_syrk_ntasks`, unless called from within a
parallel region.

## Compile-Time Tuning

The block sizes used for tiled SYRK/SYR2K can be overridden at
compile time via preprocessor defines:

    LIBXS_GEMM_BM        Row block size    (default: 192)
    LIBXS_GEMM_BN        Column block size (default: 32)
    LIBXS_GEMM_BK        K-direction block (default: 48)

Problems fitting within these limits use a single specialized
kernel call (MKL JIT or LIBXSMM when available). The same values form
the kernel shape that LIBXS_GEMM_JIT_MAX is applied to.

    LIBXS_GEMM_PANEL     Smallest SYRK/SYR2K column panel
                         (default: 0, i.e., LIBXS_GEMM_BN)
    LIBXS_SYRK_MINTASKS  Fewest threads that split a SYRK/SYR2K
                         against a sequential BLAS SYRK (default: 4)

## Environment Variables

    LIBXS_GEMM_BM=N         Row block size for tiled SYRK/SYR2K
                            (default: 192).
    LIBXS_GEMM_BN=N         Column block size (default: 32).
    LIBXS_GEMM_BK=N         K-direction block size (default: 48).
    LIBXS_GEMM_PANEL=N      Smallest column panel of a split SYRK or
                            SYR2K; zero (default) takes
                            LIBXS_GEMM_BN. A larger value yields
                            fewer and wider panels (fewer tasks).
                            The tile sizes above then apply to the
                            diagonal block alone, and only where no
                            BLAS SYRK is available.
    LIBXS_SYRK_PRINT=1      Print every double-precision BLAS SYR2K
                            call taken (see LIBXS_SYRK_BLAS) to stderr
                            (compile-time gate, enabled by default).
    LIBXS_SYRK_BLAS=0|1     Resolve the BLAS SYRK/SYR2K entry points
                            (double and single precision) and prefer them for any shape wider than
                            one tile; such a shape then compiles no
                            kernel and takes no registry entry. Set to
                            0 to tile (default: 1).

The GEMM variables (LIBXS_GEMM_BACKEND, LIBXS_GEMM_JIT_MAX, etc.) apply
as well, see [GEMM](libxs_gemm.md#environment-variables).

## Example

```C
#include <libxs/libxs_syrk.h>

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
