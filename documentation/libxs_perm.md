# Permutation and Sorting

Header: `libxs_perm.h`

Permutation-based data reordering: deterministic co-prime shuffling
(in-place and out-of-place with optional SIMD gather) and
smoothness-optimized row permutations for matrices.

## Shuffling

All three functions use a co-prime stride to produce a fixed,
deterministic permutation of count elements. The stride must be
co-prime to count; passing NULL selects libxs_coprime2(count).

```C
int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat);
```

In-place shuffle of count elements of elemsize bytes each.
nrepeat controls the number of successive permutation
applications (NULL or pointing to 1 means one pass). Returns
EXIT_SUCCESS or EXIT_FAILURE.

```C
int libxs_shuffle2(void* dst, const void* src, size_t elemsize,
  size_t count, const size_t* shuffle, const size_t* nrepeat);
```

Out-of-place shuffle from src to dst. dst and src must not
overlap. If \*nrepeat is zero an ordinary copy is performed.
Uses AVX2/AVX-512 gather instructions when available for 4-
and 8-byte elements.

```C
size_t libxs_unshuffle(size_t count, const size_t* shuffle);
```

Return the number of libxs_shuffle2 applications needed to
restore the original element order for the given count and
co-prime stride.

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

## Hilbert Curve

```C
unsigned int libxs_hilbert2d(
  unsigned int x, unsigned int y, int order);
```

Maps 2D grid coordinates (x, y) to a locality-preserving 1D
index on the Hilbert curve. Order is bits per axis (1..16),
producing a key of 2*order bits. Coordinates must be in
[0, 2^order). Adjacent positions on the curve are grid-adjacent.

## 2D k-d Tree

```C
void libxs_kdtree2d_build(double* pts, int* idx, int n);

int libxs_kdtree2d_nearest(const double* pts, const int* idx,
  const unsigned char* used, int n,
  double x, double y, double max_dist2);
```

Dense-array k-d tree for 2D nearest-neighbor queries.

Build: pts holds n interleaved (x, y) pairs. idx is an
identity-initialized index array [0..n-1] that gets rearranged
into implicit tree structure. pts is unchanged after build.

Query: finds the nearest point to (x, y) within squared
Euclidean distance max_dist2. Returns the point index or -1.
The used array (may be NULL) marks consumed points (non-zero
entries are skipped), enabling repeated queries with 1-to-1
consumption.

## Smooth Sorting

```C
typedef enum libxs_sort_t {
  LIBXS_SORT_NONE     = 0,
  LIBXS_SORT_IDENTITY = 1,
  LIBXS_SORT_NORM     = 2,
  LIBXS_SORT_MEAN     = 3,
  LIBXS_SORT_GREEDY   = 4
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

Returns EXIT_SUCCESS or EXIT_FAILURE.
