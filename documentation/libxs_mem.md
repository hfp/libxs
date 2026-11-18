# Memory Operations

Header: `libxs_mem.h`

Byte-level memory macros, pointer alignment queries,
buffer comparison, and matrix copy/transposition.

## Memory Macros

The `LIBXS_MEM*` macros expand to an inline byte loop rather than a
library call, which suits small sizes; prefer `memset`/`memcpy` for
large buffers.

```C
LIBXS_MEMSET(DST, SRC, SIZE)
```

Set SIZE bytes at DST to the value SRC.

```C
LIBXS_MEMZERO(DST)
```

Zero all bytes of *DST (size derived from sizeof(*DST)).

```C
LIBXS_MEMCPY(DST, SRC, SIZE)
```

Copy SIZE bytes from SRC to DST.

```C
LIBXS_ASSIGN(DST, SRC)
```

Copy sizeof(*SRC) bytes from SRC to DST. Statically asserts
sizeof(*SRC) <= sizeof(*DST).

```C
LIBXS_MEMSWP(PTR_A, PTR_B, SIZE)
```

Swap SIZE bytes between PTR_A and PTR_B. Asserts
PTR_A != PTR_B.

```C
LIBXS_VALUE_ASSIGN(DST, SRC)
LIBXS_VALUE_SWAP(A, B)
```

Operate on L-values directly (address-of is taken internally).
VALUE_ASSIGN can cast away const qualifiers. VALUE_SWAP statically
asserts equal size.

## Memory Information

```C
int libxs_mem_info(size_t* mem_free, size_t* mem_total);
```

Host memory in bytes: `mem_total` is what the process may use (a cgroup limit
applies where present), `mem_free` what is unused. Either pointer may be NULL.
Device memory is not considered. Returns `EXIT_SUCCESS` if a size was
determined; otherwise the outputs are left untouched.

## Offset and Alignment

```C
size_t libxs_offset(size_t ndims, const size_t offset[],
  const size_t shape[], size_t* size);
```

Compute the linear offset of an n-dimensional coordinate within
a shape. Optionally returns the total linear size of the shape.

```C
int libxs_aligned(const void* ptr, const size_t* inc,
  int* alignment);
```

Check whether ptr is SIMD-aligned (optionally considering the
next access at ptr + *inc). Optionally writes the pointer
alignment in bytes to *alignment.

## Comparison

```C
unsigned char libxs_diff(const void* a, const void* b,
  unsigned char size);
```

Compare two short buffers. Returns zero when equal, non-zero
otherwise. Uses SSE/AVX2/AVX-512 when available.

```C
unsigned int libxs_diff_n(const void* a, const void* bn,
  unsigned char elemsize, unsigned char stride,
  unsigned int hint, unsigned int count);
```

Compare a against count elements in bn. Returns the first
index that matches a, or count if none match. hint supplies
the initial search position.

```C
int libxs_memcmp(const void* a, const void* b, size_t size);
```

Boolean variant of the C memcmp. Returns zero when equal.
Uses SSE/AVX2/AVX-512 when available.

```C
void libxs_memcpy_nt(void* dst, const void* src, size_t size);
void libxs_memcpy_nt_task(void* dst, const void* src, size_t size,
  int tid, int ntasks);
```

Copy like memcpy, but with streaming (non-temporal) stores: for a
destination that is not read again soon, e.g., a buffer a device
transfers from. The stores are visible on return, so the caller's own
synchronization (a barrier, a flag) suffices. The _task variant copies
every ntasks-th block of 32 KB starting at block tid, i.e., the caller's
threads share the copy. Uses SSE/AVX2/AVX-512 when available and plain
memcpy otherwise (including non-x86).

## Matrix Copy and Transposition

```C
void libxs_matcopy(void* out, const void* in,
  unsigned int typesize, int m, int n, int ldi, int ldo);
void libxs_matcopy_task(void* out, const void* in,
  unsigned int typesize, int m, int n, int ldi, int ldo,
  int tid, int ntasks);
```

Copy an m-by-n matrix from in to out with leading dimensions
ldi and ldo. If in is NULL the destination is zeroed. The
_task variant distributes work across ntasks threads (caller
passes its tid).

```C
void libxs_otrans(void* out, const void* in,
  unsigned int typesize, int m, int n, int ldi, int ldo);
void libxs_otrans_task(void* out, const void* in,
  unsigned int typesize, int m, int n, int ldi, int ldo,
  int tid, int ntasks);
```

Out-of-place matrix transposition. The _task variant
distributes work across ntasks threads.

```C
void libxs_itrans(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo);
void libxs_itrans_task(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo,
  int tid, int ntasks);
```

In-place matrix transposition. A shape other than square with
`ldi == ldo` needs a buffer of `m*n` elements, taken from the LIBXS
memory pool; `libxs_itrans_batch` amortizes it over the batch.

```C
void libxs_itrans_batch(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo,
  int index_base, int index_stride,
  const int stride[], int batchsize,
  int tid, int ntasks);
```

Batch of in-place matrix transpositions (per-thread form).

```C
typedef enum libxs_mem_op_t {
  LIBXS_MEM_OP_MATCOPY = 0, /* libxs_matcopy_task with a source */
  LIBXS_MEM_OP_MATZERO = 1, /* libxs_matcopy_task without source (zeroing) */
  LIBXS_MEM_OP_OTRANS = 2,  /* libxs_otrans_task */
  LIBXS_MEM_OP_ITRANS = 3   /* libxs_itrans_task (square with ldi == ldo) */
} libxs_mem_op_t;

int libxs_mem_ntasks(libxs_mem_op_t op, int m, int n,
  unsigned int typesize, int nthreads);
```

Number of tasks worth splitting a `_task` call for `op` into, given
`nthreads` available: 1 below a minimum size of the operation, otherwise
one task per so many bytes, never more than `nthreads`. An `itrans` of
another shape runs on task 0 only. In Fortran, compiled with OpenMP,
`libxs_matcopy`, `libxs_otrans`, and `libxs_itrans` use it to run over an
OpenMP team unless called from within a parallel region. These Fortran
routines are not `PURE`; their `_task` forms are.
