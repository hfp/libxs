# Permutation and Sorting

Header: `libxs_perm.h`

Permutation-based data reordering: deterministic co-prime shuffling
(in-place and out-of-place), sorting, space-filling curves, k-d trees,
and smoothness-optimized row permutations for matrices.

## Shuffling

The shuffle functions use a co-prime stride C to produce a fixed,
deterministic permutation of count elements. The mapping is affine:

    dst[k] = src[(N-1) - ((C*k + offset) mod N)]

The stride must be co-prime to count; passing NULL selects
libxs_coprime2(count). A nonzero offset selects another member of
the affine family. Pass offset=0 for the standard (non-affine) shuffle.

```C
int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  const size_t* shuffle, size_t offset,
  const size_t* nrepeat);
```

In-place shuffle of count elements of elemsize bytes each.
Needs N/8 bytes of scratch memory.
nrepeat controls the number of successive permutation
applications (NULL or pointing to 1 means one pass). Returns
EXIT_SUCCESS, or EXIT_FAILURE for invalid arguments or if the
auxiliary memory cannot be allocated (inout is then unchanged).

```C
int libxs_shuffle2(void* dst, const void* src, size_t elemsize,
  size_t count, const size_t* shuffle, size_t offset,
  const size_t* nrepeat);
```

Out-of-place shuffle from src to dst. dst and src must not
overlap. If \*nrepeat is zero an ordinary copy is performed.

```C
size_t libxs_unshuffle(size_t count, const size_t* shuffle);
```

Return the number of libxs_shuffle2 applications needed to
restore the original element order for the given count and
co-prime stride, assuming offset=0. A nonzero offset can change
the cycle length unless C+1 is co-prime to N (never for even N);
libxs_unshuffle2 inverts any offset in one pass.

```C
int libxs_unshuffle2(void* dst, const void* src, size_t elemsize,
  size_t count, const size_t* shuffle, size_t offset,
  const size_t* nrepeat);
```

Inverse of libxs_shuffle2 for the same shuffle, offset, and nrepeat,
in a single pass per repetition (no need to repeat libxs_shuffle2).

```C
#define LIBXS_SHUFFLE_INDEX(I, N, C, B)
#define LIBXS_UNSHUFFLE_INDEX(J, N, CINV, B)
```

Index maps for shuffling without moving data. `LIBXS_SHUFFLE_INDEX(I, N, C,
B)` is the index libxs_shuffle2 reads for destination I (stride C, offset B).
`LIBXS_UNSHUFFLE_INDEX(J, N, CINV, B)` is its inverse, the index
libxs_unshuffle2 reads for destination J, with CINV =
`libxs_mod_inverse(C, N)`.

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

Prefer the built-in comparators: they enable a radix-sort fast path,
whereas custom comparators fall back to a comparison sort.

Index comparators order an array of indices (base) by keys that ctx
points to, and leave the keys untouched (e.g., to permute a payload):

```C
int libxs_cmp_f64_idx(const void* a, const void* b, void* ctx);
int libxs_cmp_i32_idx(const void* a, const void* b, void* ctx);
```

With a built-in value comparator, ctx controls the mode:

- `ctx == NULL`: sort base in-place.
- `ctx != NULL`: read from ctx, write the sorted result to base
  (out-of-place, source unchanged).

With a custom comparator, ctx is user state passed to cmp
and base is sorted in-place.

Examples:

```C
libxs_sort(data, n, sizeof(double), libxs_cmp_f64, NULL);
libxs_sort(dst, n, sizeof(double), libxs_cmp_f64, src);
libxs_sort(perm, n, sizeof(int), libxs_cmp_i32_idx, keys);
libxs_sort(perm, n, sizeof(int), my_indirect_cmp, keys);
```

## Space-Filling Curves

```C
uint64_t libxs_hilbert(const unsigned int coords[], int ndims);
void libxs_hilbert_decode(uint64_t code, unsigned int coords[], int ndims);
```

N-dimensional Hilbert curve index. Maps ndims coordinates to a
64-bit key with strong locality guarantees. Each coordinate is
quantized to floor(64/ndims) bits. coords[k] must be in
[0, 2^bits_per_dim). The decode routine maps such a key back
to ndims coordinates with the same bit budget.

```C
uint64_t libxs_morton(const unsigned int coords[], int ndims);
void libxs_morton_decode(uint64_t code, unsigned int coords[], int ndims);
```

N-dimensional Morton code (Z-order curve). Bit-interleaves ndims
coordinates into a 64-bit key. Each coordinate is quantized to
floor(64/ndims) bits. The decode routine deinterleaves the key
back to ndims coordinates.

```C
uint64_t libxs_hilbert_bits(
  const unsigned int coords[], int ndims, int bits_per_dim);
void libxs_hilbert_decode_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim);
uint64_t libxs_morton_bits(
  const unsigned int coords[], int ndims, int bits_per_dim);
void libxs_morton_decode_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim);
```

Same curves with an explicit number of bits per coordinate
(1 to 32, with ndims·bits_per_dim at most 64); the encoders return
zero for arguments out of range.

```C
int libxs_stratify_morton(
  const unsigned int src_coords[], int src_ndims,
  unsigned int dst_coords[], int dst_ndims);
int libxs_stratify_hilbert(
  const unsigned int src_coords[], int src_ndims,
  unsigned int dst_coords[], int dst_ndims);
```

Stratify higher-dimensional coordinates into a lower-dimensional
layout without slicing. The source coordinates are encoded using
the selected source-dimensional space-filling curve, and the
resulting key is decoded as target-dimensional coordinates using
the same curve family. This gives deterministic rank-preserving
embeddings such as 3D to 2D:

    code = curve(src_coords, 3)          /* 21 bits per coordinate */
    dst_coords = inverse_curve(code, 2)  /* 32 bits per coordinate */

Each source coordinate keeps floor(64/src_ndims) bits, and each target
coordinate receives ceil(src_ndims·floor(64/src_ndims)/dst_ndims) bits,
which must not exceed 32. The target dimensionality must be smaller than
the source dimensionality. The functions return EXIT_SUCCESS or
EXIT_FAILURE.

```C
int libxs_stratify_morton_bits(
  const unsigned int src_coords[], int src_ndims, int src_bits,
  unsigned int dst_coords[], int dst_ndims, int dst_bits);
int libxs_stratify_hilbert_bits(
  const unsigned int src_coords[], int src_ndims, int src_bits,
  unsigned int dst_coords[], int dst_ndims, int dst_bits);
```

Same with explicit bits per source and target coordinate (each 1 to 32).
Requires dst_ndims < src_ndims, src_ndims·src_bits <= 64, and
dst_ndims·dst_bits >= src_ndims·src_bits; otherwise EXIT_FAILURE.

## k-d Tree

```C
typedef int (*libxs_kdtree_split_t)(
  int* dim, int* pos, double* threshold, const double* pts, int* idx,
  int count, int depth, int nleaves, void* ctx);

typedef struct libxs_kdtree_node_t {
  double threshold;
  int dim, next;
} libxs_kdtree_node_t;

typedef struct libxs_kdtree_config_t {
  libxs_kdtree_split_t split;
  void* ctx;
  libxs_kdtree_node_t* nodes;
  int min_leaf;
} libxs_kdtree_config_t;
```

Split callback for custom tree construction. Writes chosen
split dimension into `*dim` and left-child count into `*pos`, and
reorders idx[0..count-1] to place left entries first. `*threshold`
arrives as `HUGE_VAL`: left so, the tree places it midway between
the two sides along `*dim`; a callback that splits by coordinates
other than `pts` writes its own. nleaves gives the number of leaves
already assigned (in preorder). Return 0 for a valid split, nonzero
to force a leaf.

Config struct: zero-initialize it (`libxs_kdtree_config_t config =
{ 0 }`) so that unset fields stay off. min_leaf controls minimum node
size (0 = split down to 1 element). split=NULL uses round-robin median
splits. nodes (partition only) receives the tree in preorder: a split
(`dim >= 0`) has its left child next in the array and its right child
at index `next`, entries below `threshold` went left; a leaf (`dim < 0`)
has its leaf id in `next`. The tree has `2*nleaves-1` nodes, at most
`2n-1`.

```C
int libxs_kdtree_split_median(int* dim, int* pos, const double* pts,
  int* idx, int count, int depth, int nleaves, void* ctx);
int libxs_kdtree_split_middle(int* dim, int* pos, const double* pts,
  int* idx, int count, int depth, int nleaves, void* ctx);
```

Split rules of the tree itself, passed as `split`: the tree recognizes
them and splits internally (called directly, they only return nonzero).
`libxs_kdtree_split_median` is the default (split=NULL), the tree that
the nearest-neighbor queries search. `libxs_kdtree_split_middle` splits
at the middle of the range of the axis with the largest extent, which
suits grid coordinates: points below the middle go left, and a node
without extent stays a leaf.

```C
void libxs_kdtree_build(const double* pts, int* idx, int n,
  int ndims, int stride, const libxs_kdtree_config_t* config);
```

Build a k-d tree in-place. Points are stored row-major:
pts[i*stride + k] is coordinate k of point i. The index array
idx[0..n-1] must hold the identity {0, 1, ..., n-1} on entry and is
rearranged into implicit tree structure. config may be NULL
(round-robin median, leaf at 1 element).

```C
int libxs_kdtree_partition(const double* pts, int* idx, int n,
  int ndims, int stride, int* assignments,
  const libxs_kdtree_config_t* config);
```

Partition n points into leaves and write leaf IDs (in preorder) into
assignments[0..n-1]. Returns the number of leaves. Same tree
logic as build, but produces cluster assignments instead of a
query-tree index. As for build, idx must hold the identity on entry.

```C
int libxs_kdtree_descend(const libxs_kdtree_node_t nodes[],
  const double point[]);
```

The leaf id a point descends to in the nodes partition wrote
(`config.nodes`): below a split's threshold left, otherwise right. The
point is in the coordinates of the thresholds (those of `pts`, or of the
split callback that wrote them). Returns -1 for NULL arguments, or where
a node does not lead forward.

```C
int libxs_kdtree_nearest(const double* pts, const int* idx,
  const unsigned char* used, int n, int ndims, int stride,
  const double* query, double max_dist2);
```

Find the nearest point to query[0..ndims-1] within squared
Euclidean distance max_dist2. Returns the point index or -1.
The used array (may be NULL) marks consumed points.

```C
int libxs_kdtree_knearest(const double* pts, const int* idx,
  const unsigned char* used, int n, int ndims, int stride,
  const double* query, double max_dist2,
  int out_idx[], double out_dist2[], int k);
```

Find the k nearest points, writing indices to out_idx and their
squared distances to out_dist2, both sorted ascending, and
returning how many were written (0..k; fewer when n or max_dist2
bounds it).

The result is exact: the same k points a linear scan would pick,
in the same order.

Both searches require a tree built with `config == NULL`, or with a
config whose `split` is NULL or `libxs_kdtree_split_median` and whose
`min_leaf` is at most 1; other trees cannot be searched.

The tree beats a linear scan while n is large against 2^ndims. Measured
with k = 8, uniformly random points need n of about 64·2^ndims before
the tree is faster. Structured data (a lower intrinsic dimension) pays
earlier: 1.26x faster at 37 dimensions and 60,000 points, whereas data
without such structure was 3.4x slower than a scan at 28 dimensions and
100,000 points. Whether to prefer a scan depends on the data rather than
on ndims alone: time a few queries both ways.

```C
int libxs_kdtree_knearest_budget(const double* pts, const int* idx,
  const unsigned char* used, int n, int ndims, int stride,
  const double* query, double max_dist2,
  int out_idx[], double out_dist2[], int k, int budget);
```

As `libxs_kdtree_knearest` (same tree and output), but searched best bin
first: the waiting subtree with the smallest lower bound of its distance
next, until none can hold a nearer point or `budget` points have been
examined (one distance each). With `budget <= 0` the result is exact;
with a budget it is the k nearest among the points examined. This is the
search for dimensions where the exact one visits most of the tree.
Measured with k = 8 (recall: the share of the exact k nearest found, ties
counted as found): at 37 dimensions and 60,000 structured points, a
budget of 1024 was 13.6x faster than a scan at a recall of 0.95 (4096:
3.5x, 0.99); at 28 dimensions and 100,000 points without such structure,
4096 was 3.9x faster at 0.73 (16384: as fast as a scan, 0.91). Without a
budget, `libxs_kdtree_knearest` is the faster exact search.

Convenience wrappers for 2D (interleaved x,y layout):

```C
void libxs_kdtree2d_build(double* pts, int* idx, int n);
int libxs_kdtree2d_nearest(const double* pts, const int* idx,
  const unsigned char* used, int n,
  double x, double y, double max_dist2);
```

## Spatial Index

```C
typedef struct libxs_spatial_t {
  uint64_t* codes;
  void** values;
  int n;
} libxs_spatial_t;

int libxs_spatial_build(libxs_spatial_t* sp,
  const libxs_registry_t* registry);
int libxs_spatial_build_codes(libxs_spatial_t* sp,
  const uint64_t codes[], void* const values[], int n);
int libxs_spatial_nearest(const libxs_spatial_t* sp,
  uint64_t query_code, int k, void** out_values);
void libxs_spatial_destroy(libxs_spatial_t* sp);
```

An index of values sorted by a 64-bit locality code, such as
`libxs_hilbert`. `libxs_spatial_build` takes the codes from the first
eight bytes of each key of a [registry](libxs_reg.md) (keys of at least
eight bytes), and
`libxs_spatial_build_codes` from parallel arrays, which it copies; use the
latter when the registry key is not a locality code. Both return
EXIT_SUCCESS, or EXIT_FAILURE for invalid arguments, an empty input, or a
failed allocation. The values are stored as pointers, not copied.

`libxs_spatial_nearest` writes up to k values whose codes are numerically
closest to query_code to out_values and returns how many were written.
`libxs_spatial_destroy` releases the index.

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

| Parameter  | Description                                          |
|------------|------------------------------------------------------|
| `method`   | Sorting strategy (see constants above)               |
| `m`, `n`   | Matrix dimensions (m rows, n columns)                |
| `mat`      | Column-major matrix data (read-only)                 |
| `ld`       | Leading dimension (ld ≥ m)                           |
| `datatype` | Element type (F64, F32, and all integer types)       |
| `perm`     | Output array of m ints; perm[i] = source row for position i after reordering |

| Method     | Description                                          |
|------------|------------------------------------------------------|
| `NONE`     | No permutation (early return, perm is not written)   |
| `IDENTITY` | Identity permutation (perm[i] = i)                   |
| `NORM`     | Sort rows by ascending L1-norm                       |
| `MEAN`     | Sort rows by ascending row mean (mean over the columns) |
| `GREEDY`   | Nearest-neighbor chain: starting from row 0, each step picks the unvisited row with the smallest Euclidean distance to the current row |
| `MORTON`   | Sort rows by Morton (Z-order) key computed from quantized column values (first 53 columns) |
| `HILBERT`  | Sort rows by Hilbert curve key computed from quantized column values (first 53 columns) |

Returns EXIT_SUCCESS or EXIT_FAILURE.
