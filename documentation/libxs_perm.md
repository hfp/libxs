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
