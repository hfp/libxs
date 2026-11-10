# GEMM: Matrix Multiplication

Header: `libxs_gemm.h`
Fortran: `USE LIBXS` (libxs/libxs.f)

Batched general matrix-matrix multiplication (GEMM) and symmetric
rank-k/2k updates. Operations are expressed as
C := alpha * op(A) * op(B) + beta * C, where op() is an optional
transpose. Kernels are dispatched via MKL JIT, LIBXSMM, or BLAS;
a built-in default kernel (auto-vectorized) is used as fallback.

## Kernel Providers

LIBXS does not link a kernel provider. MKL JIT, LIBXSMM, and BLAS are
linked by the application, and LIBXS receives their entry points as
function pointers (`libxs_gemm_backend_t`). One LIBXS build hence
serves applications with or without MKL or LIBXSMM, and the provider's
version is the one the application links.

**C.** The dispatch functions are header inlines. They are compiled in
the application's translation unit and take the address of what the
application's headers declare (`libxs_gemm_backend_init`):

| Provider | Enabled by |
| :--- | :--- |
| MKL JIT and BLAS | `-D__MKL`, or `mkl.h` included before `libxs_gemm.h` |
| LIBXSMM | `-D__LIBXSMM`, or `libxsmm.h` included before `libxs_gemm.h` |
| BLAS | `-D__BLAS` (Fortran `dgemm_`/`sgemm_`) |

```C
#include <mkl.h> /* or libxsmm.h */
#include <libxs/libxs_gemm.h>
```

`LIBXS_GEMM_JIT` is defined if a generator was found. To pass pointers
explicitly, fill a `libxs_gemm_backend_t` and call one of the `_rt`
forms.

**Fortran.** `libxs/libxs_jit.F` plays the role of the header inlines.
It is not part of the library: the application compiles it with its
own sources and flags (preprocessed, hence `.F`), and uses
`USE LIBXS_JIT` in place of `USE LIBXS`. The module passes the
provider's entry points as procedure pointers (`C_FUNLOC`) to the
dispatch of module LIBXS, where they are OPTIONAL `C_FUNPTR` arguments.

```bash
$FC -D__LIBXS [-D__MKL] [-D__LIBXSMM] -I<libxs>/libxs \
  -c <libxs>/libxs/libxs_jit.F
```

`-D__LIBXS` is the application's flag for using LIBXS; without it, the
module falls back to plain BLAS, so that call sites need not change
when LIBXS is optional. BLAS is required at link time.

**Runtime.** Where dynamic linking is supported, the library also looks
up `dgemm_`, `sgemm_`, the MKL JIT entry points, and
`libxsmm_dispatch_gemm` among the symbols already loaded by the
process, and uses them for any backend pointer left NULL. Linking a
provider into the application is then sufficient, even without the
compile-time detection above.

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
configurations. Dispatch builds its own key from the members, hence a
caller-supplied shape need not be zero-initialized.

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
3. BLAS kernel (dgemm_blas/sgemm_blas) — always non-NULL after
   dispatch (falls back to built-in auto-vectorized C code).

warmup counts dispatches of this shape and is maintained by
dispatch (see LIBXS_GEMM_JIT_WARMUP).

```C
typedef enum libxs_gemm_flags_t {
  LIBXS_GEMM_FLAGS_DEFAULT = 0,
  LIBXS_GEMM_FLAG_NOLOCK = 1,
  LIBXS_GEMM_FLAG_OWNJIT = 2,
  LIBXS_GEMM_FLAG_STREAM = 4,
  LIBXS_GEMM_FLAG_REUSE = 8
} libxs_gemm_flags_t;
```

Flags controlling batch synchronization. Set LIBXS_GEMM_FLAG_NOLOCK
when no duplicate C pointers exist across the batch.
LIBXS_GEMM_FLAG_OWNJIT is set by dispatch and marks the single config
owning `jitter`; aliasing configs (double-dispatch, and every caller
copy) leave it clear so the handle is released exactly once.

LIBXS_GEMM_FLAG_STREAM and LIBXS_GEMM_FLAG_REUSE are dispatch hints.
The copy flavors (`_cpy`) read them from the caller's config before
populating it, and do not keep them:

| Hint | Effect |
| :--- | :--- |
| STREAM | a generated kernel may serve operands that are windows into larger matrices (leading dimension beyond the kernel's extent), which the automatic backend otherwise leaves to BLAS |
| REUSE | the JIT warm-up is skipped, since the caller calls the kernel often (e.g., a batch) |

A kernel is shared per shape, whichever hints generated it: a later
dispatch of the same shape receives it with or without hints.

```C
typedef enum libxs_gemm_kind_t {
  LIBXS_GEMM_KIND_DEFAULT = 0,
  LIBXS_GEMM_KIND_BLAS = 1,
  LIBXS_GEMM_KIND_JIT = 2
} libxs_gemm_kind_t;
```

Kind of kernel serving a GEMM: built-in code, external BLAS, or a
generated kernel (MKL JIT or LIBXSMM); see
[Backend Query](#backend-query).

Function-pointer and helper types used by the config and the backend:

```C
typedef void (*libxs_gemm_dblas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc,
  size_t transa_len, size_t transb_len);
typedef void (*libxs_gemm_sblas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc,
  size_t transa_len, size_t transb_len);
typedef void (*libxs_gemm_djit_t)(void* jitter,
  const double* a, const double* b, double* c);
typedef void (*libxs_gemm_sjit_t)(void* jitter,
  const float* a, const float* b, float* c);
typedef struct libxs_gemm_param_t {
  void* op[4]; const void* a[6]; const void* b[6]; void* c[6];
} libxs_gemm_param_t;
typedef void (*libxs_gemm_xfn_t)(const void*);
typedef struct libxs_xgemm_shape_t {
  int m, n, k, lda, ldb, ldc;
  int a_in_type, b_in_type, out_type, comp_type;
} libxs_xgemm_shape_t;
typedef int (*libxs_blas_nthreads_t)(void);
```

- `libxs_gemm_dblas_t`/`libxs_gemm_sblas_t`: Fortran BLAS `dgemm_`/`sgemm_` including the hidden lengths of transa and transb.
- `libxs_gemm_djit_t`/`libxs_gemm_sjit_t`: MKL JIT kernel (`mkl_jit_get_dgemm_ptr`/`mkl_jit_get_sgemm_ptr`); shape and scalars are part of the jitter.
- `libxs_gemm_param_t`: argument of an XGEMM kernel (`libxs_gemm_xfn_t`), layout-compatible with `libxsmm_gemm_param`; only the primary pointers `a[0]`, `b[0]`, `c[0]` are used.
- `libxs_xgemm_shape_t`: shape passed to `xgemm_dispatch`, layout-compatible with `libxsmm_gemm_shape` (`libxsmm_blasint=int`).
- `libxs_blas_nthreads_t`: number of threads the BLAS uses by itself, e.g., `MKL_Get_Max_Threads` (1 means sequential).

## Dispatch

### C (inline, compile-time backend selection)

```C
libxs_gemm_config_t* libxs_gemm_dispatch(
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta,
  void* registry /* = NULL */);
```

Inline function that selects the backend at compile time (see
[Kernel Providers](#kernel-providers)). On registry hit, returns a pointer to the cached config (hash
probe only). On miss, dispatches a new kernel and stores it.
If registry is NULL, an internal registry is used.

The detection itself is exposed, so a caller assembling shapes by
hand can reach `libxs_gemm_dispatch_rt` without repeating it:

```C
void libxs_gemm_backend_init(libxs_gemm_backend_t* backend);
```

It zero-initializes backend and fills the pointers available at the
caller's compile time. Unavailable backends are left NULL, i.e.,
dispatch takes what the library found at runtime, and the built-in
default kernel otherwise.

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
backend at the caller's compile time (see
[Kernel Providers](#kernel-providers)). To supply backend pointers
explicitly, use the `_rt` forms, which are the only dispatch symbols
exported by the library itself:

| Operation | Explicit backend (library symbol) | Caller copy |
| :--- | :--- | :--- |
| GEMM | `libxs_gemm_dispatch_rt` (takes shape structs) | `libxs_gemm_dispatch_cpy_rt` |
| SYR2K | `libxs_syr2k_dispatch_rt` | `libxs_syr2k_dispatch_cpy_rt` |
| SYRK | `libxs_syrk_dispatch_rt` | `libxs_syrk_dispatch_cpy_rt` |

The `_cpy_rt` forms take a config as first argument and return nonzero
if it was populated, like the `_cpy` inlines.

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
  libxs_blas_nthreads_t blas_nthreads;
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
backend: function pointers for backends. A NULL backend, or a NULL
member, takes what the library found at runtime, and the built-in
default otherwise. blas_nthreads tells whether the BLAS is threaded
(e.g., `MKL_Get_Max_Threads`) where the library cannot find out
itself (static link). Same registry semantics as above. The BLAS pointers
take the two hidden lengths of transa and transb as trailing
`size_t` arguments (always 1), which a BLAS compiled from Fortran
expects; a C BLAS ignores them.
`LIBXS_GEMM_BACKEND` can restrict the starting point of the
runtime fallback chain: 0 = automatic/default, 1 = MKL JIT,
2 = LIBXSMM, 3 = BLAS/MKL, 4 = built-in fallback. Choices 1-3
still fall through to lower-priority backends if the requested
backend is not supplied or cannot dispatch the shape.

Automatic selection takes a generated kernel only for resident
operands, i.e., leading dimensions that match the kernel's own extent;
a larger leading dimension reaches BLAS. `LIBXS_GEMM_BACKEND=1` or `=2`
requests the respective generator regardless.

Among the generators, automatic selection tries the MKL JIT first and
LIBXSMM second, which in addition requires `alpha` of one and `beta` of
zero or one. A shape that neither generator serves reaches BLAS.

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
but fills in BLAS, and MKL JIT or LIBXSMM when compiled with
`-D__MKL` or `-D__LIBXSMM`. The application compiles `libxs_jit.F`
itself (see [Kernel Providers](#kernel-providers)).

**Without `-D__LIBXS`**, `libxs_jit.F` compiles to a BLAS-only module
LIBXS_JIT that needs no LIBXS library. It offers `libxs_init`,
`libxs_finalize`, `libxs_gemm_dispatch` (config flavor),
`libxs_gemm_call` (config and one-shot form), `libxs_syrk`, and
`libxs_syr2k`. These call DGEMM, DSYRK, and DSYR2K (double precision
only), and the SYRK/SYR2K forms take the shape directly:

```fortran
CALL libxs_syrk(datatype, n, k, lda, ldc, uplo, alpha, beta, a, c)
CALL libxs_syr2k(datatype, n, k, lda, ldb, ldc,
     &  uplo, alpha, beta, a, b, c)
```

### Fortran (LIBXS, explicit backends)

```fortran
USE LIBXS
rc = libxs_gemm_dispatch(config, datatype, transa, transb,
     &  m, n, k, lda, ldb, ldc, alpha, beta,
     &  jit_create_dgemm=..., jit_get_dgemm=...,
     &  dgemm_blas=..., registry=...)
```

All backend arguments are OPTIONAL C_FUNPTR (named arguments):
`jit_create_dgemm`, `jit_get_dgemm`, `jit_create_sgemm`,
`jit_get_sgemm`, `xgemm_dispatch`, `dgemm_blas`, `sgemm_blas`, and
`blas_nthreads`; `registry` is an OPTIONAL C_PTR. Returns nonzero on
success (dispatch produced a callable config). The config is populated
from the registry-owned copy.

```fortran
rc = libxs_gemm_config_cpy(config, cached) ! TYPE(C_PTR) :: cached
```

Converts a pointer obtained from a pointer-flavor dispatch into a
caller-owned config, like `libxs_gemm_config_cpy` in C.

## Backend Query

```C
libxs_gemm_kind_t libxs_gemm_backend_kind(
  const libxs_gemm_backend_t* backend /* = NULL */,
  const libxs_gemm_shape_t* shape /* = NULL */);
```

Tells which kind of kernel a dispatch settles on once warmed up:
`LIBXS_GEMM_KIND_JIT` (MKL JIT or LIBXSMM), `LIBXS_GEMM_KIND_BLAS`
(external BLAS), or `LIBXS_GEMM_KIND_DEFAULT` (built-in code). The
query honors `LIBXS_GEMM_BACKEND` and `LIBXS_GEMM_JIT_MAX`, but it
neither generates a kernel nor consults a registry, hence it is cheap
enough to decide per call. Without a shape, it answers for any shape;
without a backend, for the library's own. JIT means a generator is
tried, which may still refuse a particular shape. For shapes with
resident operands (leading dimensions equal to the extents), JIT for
the largest shape holds for all smaller ones.

```C
libxs_gemm_backend_t backend;
libxs_gemm_backend_init(&backend);
if (LIBXS_GEMM_KIND_JIT == libxs_gemm_backend_kind(&backend, NULL)) {
  /* generated kernels are available */
}
```

At compile time, `LIBXS_GEMM_JIT` is defined if
`libxs_gemm_backend_init` provides a generator, which likewise
requires mkl.h or libxsmm.h to be included before libxs_gemm.h.

```fortran
USE LIBXS_JIT
kind = libxs_gemm_backend_kind() ! any shape
kind = libxs_gemm_backend_kind(LIBXS_DATATYPE_F64, 'N', 'N',
     &  m, n, k, lda, ldb, ldc, alpha, beta)
```

With `USE LIBXS`, the backend pointers are OPTIONAL named arguments
as for the dispatch.

## Single-Kernel Call

```C
void libxs_gemm_call(
  const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c);
```

Call the dispatched GEMM kernel (JIT > XGEMM > BLAS fallback).
The caller must ensure config is non-NULL (dispatch succeeded).

```fortran
CALL libxs_gemm_call(config, a, b, c)
CALL libxs_gemm_call(datatype, transa, transb, m, n, k,
     &  lda, ldb, ldc, alpha, beta, a, b, c)
```

In Fortran, the second form dispatches and calls in one step
(`libxs_gemm_call_direct`); it accepts the OPTIONAL backend arguments
and `registry` of the dispatch. `a`, `b`, and `c` are `TYPE(C_PTR)`.

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

```C
int libxs_gemm_ntasks(const libxs_gemm_config_t* config,
  int batchsize, int nthreads);
```

Number of tasks worth splitting `libxs_gemm_batch_task` or
`libxs_gemm_index_task` into, given nthreads (1 means unsplit); never
exceeds batchsize. Unless config->flags carries
`LIBXS_GEMM_FLAG_NOLOCK`, a split serializes updates of the same C.
Fortran: `libxs_gemm_ntasks(config, batchsize, nthreads)`.

**Fortran and OpenMP.** When the application compiles the Fortran
module with OpenMP, `libxs_gemm_batch` and `libxs_gemm_index` run over
an OpenMP team sized by `libxs_gemm_ntasks`, unless called from within
a parallel region. The `_task` forms remain for explicit splitting.

## SYR2K / SYRK

Symmetric rank-2k and rank-k updates built on top of GEMM dispatch.

For small problems (n within LIBXS_GEMM_BM and LIBXS_GEMM_BN, and
k <= LIBXS_GEMM_BK), the dispatched kernel handles the full GEMM in one
call. Larger problems are split into column panels (see
LIBXS_GEMM_PANEL and LIBXS_SYRK_BLAS). Scratch memory is thread-local.

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

    LIBXS_GEMM_NWARMUP   JIT warm-up counters (default: 4096)

Number of slots for the per-shape reuse counters (must be a power of
two). Fewer slots than distinct shapes in flight can delay JIT, but
never yield a wrong kernel.

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
    LIBXS_GEMM_BACKEND=N    Select runtime backend chain start:
                            0 auto (default), 1 MKL JIT,
                            2 LIBXSMM, 3 BLAS, 4 built-in fallback.
                            Auto takes a generator only for resident
                            operands, the MKL JIT before LIBXSMM;
                            1 and 2 take the respective one always.
    LIBXS_GEMM_JIT_MAX=N    Arithmetic-intensity threshold for JIT
                            dispatch. JIT/LIBXSMM kernels are only
                            generated when the kernel shape's AI
                            (flops/bytes) is below N (default: 7).
                            Set to 0 to disable JIT entirely.
    LIBXS_GEMM_JIT_WARMUP=N Number of calls a shape must
                            accumulate before JIT compilation is
                            attempted. Shapes called fewer than N
                            times use the BLAS fallback. Reuse is
                            counted per dispatch; the first shape of
                            an empty registry is compiled immediately.
                            Counted in a fixed-size table, not in the
                            registry (see LIBXS_GEMM_NWARMUP); once a
                            shape's JIT has been attempted it is never
                            attempted again, whether it succeeded or
                            was refused by LIBXS_GEMM_JIT_MAX.
                            Set to 0 or 1 to compile immediately
                            on first miss (default: 8, clamped to 253).
    LIBXS_GEMM_PRINT=N      Print dispatch info every N-th call
                            to stderr (compile-time gate, enabled
                            by default).
                            Set to 0 for a summary at teardown instead
                            (registry, histogram, and warm-up slots).
                            Every line carries the origin of the
                            process, "LIBXS INFO[rid]:" with rid as
                            libxs_rid,
                            hence the local rank where it is known and
                            the process id otherwise.
    LIBXS_SYRK_PRINT=1      Print every double-precision BLAS SYR2K
                            call taken (see LIBXS_SYRK_BLAS) to stderr
                            (compile-time gate, enabled by default).
    LIBXS_SYRK_BLAS=0|1     Resolve the BLAS SYRK/SYR2K entry points
                            (double and single precision) and prefer them for any shape wider than
                            one tile; such a shape then compiles no
                            kernel and takes no registry entry. Set to
                            0 to tile (default: 1).

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
