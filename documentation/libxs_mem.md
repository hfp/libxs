# Memory Operations

Header: `libxs_mem.h`

Byte-level memory macros, pointer alignment queries,
buffer comparison, and matrix copy/transposition.

## Memory Macros

```C
LIBXS_MEMSET(DST, SRC, SIZE)
```

Set SIZE bytes at DST to the value SRC, unrolled at compile-time.

```C
LIBXS_MEMZERO(DST)
```

Zero all bytes of *DST (size derived from sizeof(*DST)).

```C
LIBXS_MEMCPY(DST, SRC, SIZE)
```

Copy SIZE bytes from SRC to DST, unrolled at compile time.

```C
LIBXS_ASSIGN(DST, SRC)
```

Copy sizeof(*SRC) bytes from SRC to DST. Asserts
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
VALUE_ASSIGN can cast away const qualifiers. VALUE_SWAP asserts
equal size.

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
  int m, int n, int ldi, int ldo, void* scratch);
void libxs_itrans_task(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo, void* scratch,
  int tid, int ntasks);
```

In-place matrix transposition (square or via scratch buffer).
scratch can be NULL (auto-allocated).

```C
void libxs_itrans_batch(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo,
  int index_base, int index_stride,
  const int stride[], int batchsize,
  int tid, int ntasks);
```

Batch of in-place matrix transpositions (per-thread form).
