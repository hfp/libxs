# Permutation and Sorting

Header: `libxs_perm.h`

Permutation-based data reordering: deterministic co-prime shuffling
(in-place and out-of-place with optional SIMD gather) and
smoothness-optimized row permutations for matrices.

## Shuffling

The shuffle functions use a co-prime stride C to produce a fixed,
deterministic permutation of count elements. The mapping is affine:

    dst[k] = src[(N-1) - ((C*k + offset) mod N)]

The stride must be co-prime to count; passing NULL selects
libxs_coprime2(count). The offset parameter extends the basic
co-prime permutation into an affine family, expanding the number
of available permutations from ~phi(N)/2 to ~N*phi(N)/2.
Pass offset=0 for the standard (non-affine) shuffle.

```C
int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  const size_t* shuffle, size_t offset,
  const size_t* nrepeat);
```

In-place shuffle of count elements of elemsize bytes each.
Uses cycle following with a bit vector (N/8 bytes auxiliary).
nrepeat controls the number of successive permutation
applications (NULL or pointing to 1 means one pass). Returns
EXIT_SUCCESS or EXIT_FAILURE.

```C
int libxs_shuffle2(void* dst, const void* src, size_t elemsize,
  size_t count, const size_t* shuffle, size_t offset,
  const size_t* nrepeat);
```

Out-of-place shuffle from src to dst. dst and src must not
overlap. If \*nrepeat is zero an ordinary copy is performed.
Uses AVX2/AVX-512 gather instructions when available for 4-
and 8-byte elements (offset=0 path).

```C
size_t libxs_unshuffle(size_t count, const size_t* shuffle);
```

Return the number of libxs_shuffle2 applications needed to
restore the original element order for the given count and
co-prime stride. The cycle length depends only on C and N,
not on the offset.

```C
int libxs_unshuffle2(void* dst, const void* src, size_t elemsize,
  size_t count, const size_t* shuffle, size_t offset,
  const size_t* nrepeat);
```

Single-pass inverse of libxs_shuffle2. Computes the modular
inverse C_inv = C^{-1} mod N via the extended Euclidean
algorithm, then gathers elements using the inverse permutation:

    dst[m] = src[C_inv * (N-1-offset-m) mod N]

This restores the original order in one O(N) pass rather than
the R-1 repeated applications that libxs_unshuffle would
require. The loop structure (sequential write, strided read)
is identical to the forward shuffle and amenable to the same
SIMD gather optimizations. Supports multi-pass inversion
(nrepeat > 1) by iterating f^{-1}.

## General-Purpose Sort

```C
typedef int (*libxs_sort_cmp_t)(
  const void* a, const void* b, void* ctx);

void libxs_sort(void* base, int n, size_t size,
  libxs_sort_cmp_t cmp, void* ctx);
```

Sort n elements of the given byte size. The comparator returns
negative, zero, or positive (tristate, like qsort_r). The ctx
pointer is forwarded to the comparator unchanged.

Built-in comparators:

```C
int libxs_cmp_f64(const void* a, const void* b, void* ctx);
int libxs_cmp_f32(const void* a, const void* b, void* ctx);
int libxs_cmp_i32(const void* a, const void* b, void* ctx);
int libxs_cmp_u32(const void* a, const void* b, void* ctx);
```

Built-in comparators enable an O(n) radix sort fast path.
Custom comparators use O(n log n) heap sort.

With a built-in comparator, ctx controls the mode:

  ctx = NULL   Sort base in-place.
  ctx != NULL  Read from ctx, write sorted result to base
               (out-of-place, source unchanged).

With a custom comparator, ctx is user state passed to cmp
and base is sorted in-place.

Examples:

  libxs_sort(data, n, sizeof(double), libxs_cmp_f64, NULL);
  libxs_sort(dst, n, sizeof(double), libxs_cmp_f64, src);
  libxs_sort(perm, n, sizeof(int), my_indirect_cmp, keys);

## Space-Filling Curves

```C
uint64_t libxs_hilbert(const unsigned int coords[], int ndims);
```

N-dimensional Hilbert curve index. Maps ndims coordinates to a
64-bit key with strong locality guarantees. Each coordinate is
quantized to floor(64/ndims) bits. coords[k] must be in
[0, 2^bits_per_dim).

```C
uint64_t libxs_morton(const unsigned int coords[], int ndims);
```

N-dimensional Morton code (Z-order curve). Bit-interleaves ndims
coordinates into a 64-bit key. Each coordinate is quantized to
floor(64/ndims) bits.

## k-d Tree

```C
void libxs_kdtree_build(const double* pts, int* idx, int n,
  int ndims, int stride);
int libxs_kdtree_nearest(const double* pts, const int* idx,
  const unsigned char* used, int n, int ndims, int stride,
  const double* query, double max_dist2);
```

Dense-array k-d tree for nearest-neighbor queries in arbitrary
dimensions. Points are stored row-major: pts[i*stride + k] is
coordinate k of point i. stride >= ndims allows padding or
interleaved auxiliary data. The index array idx[0..n-1] is
rearranged into implicit tree structure during build.

Query: finds the nearest point to query[0..ndims-1] within
squared Euclidean distance max_dist2. Returns the point index
or -1. The used array (may be NULL) marks consumed points
(non-zero entries are skipped).

Convenience wrappers for 2D (interleaved x,y layout):

```C
void libxs_kdtree2d_build(double* pts, int* idx, int n);
int libxs_kdtree2d_nearest(const double* pts, const int* idx,
  const unsigned char* used, int n,
  double x, double y, double max_dist2);
```

## Smooth Sorting

```C
typedef enum libxs_sort_t {
  LIBXS_SORT_NONE     = 0,
  LIBXS_SORT_IDENTITY = 1,
  LIBXS_SORT_NORM     = 2,
  LIBXS_SORT_MEAN     = 3,
  LIBXS_SORT_GREEDY   = 4,
  LIBXS_SORT_MORTON   = 5,
  LIBXS_SORT_HILBERT  = 6
} libxs_sort_t;

int libxs_sort_smooth(libxs_sort_t method, int m, int n,
  const void* mat, int ld, libxs_data_t datatype, int* perm);
```

Compute a row permutation that reorders the rows of an m-by-n
column-major matrix for smoothness (decaying forward
differences between consecutive rows).

Parameters:

  method    Sorting strategy (see constants above).
  m, n      Matrix dimensions (m rows, n columns).
  mat       Column-major matrix data (read-only).
  ld        Leading dimension (ld >= m).
  datatype  Element type (F64, F32, I32, U32, I16, U16, I8, U8).
  perm      Output array of m ints; perm[i] = source row for
            position i after reordering.

Methods:

  NONE      No permutation (early return).
  IDENTITY  Identity permutation (perm[i] = i).
  NORM      Sort rows by ascending L1-norm.
  MEAN      Sort rows by ascending column mean.
  GREEDY    Nearest-neighbor chain: starting from row 0,
            each step picks the unvisited row with the
            smallest Euclidean distance to the current row.
  MORTON    Sort rows by Morton (Z-order) key computed from
            quantized column values.
  HILBERT   Sort rows by Hilbert curve key computed from
            quantized column values.

Returns EXIT_SUCCESS or EXIT_FAILURE.
