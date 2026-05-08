# Math Utilities

Header: `libxs_math.h`

## Matrix Difference

```C
typedef struct libxs_matdiff_t {
  double norm1_abs, norm1_rel;  /* one-norm */
  double normi_abs, normi_rel;  /* infinity-norm */
  double normf_rel;             /* Frobenius-norm (relative) */
  double linf_abs, linf_rel;    /* max difference (abs/rel at same element) */
  double l2_abs, l2_rel, rsq;   /* L2-norm and R-squared */
  double l1_ref, min_ref, max_ref, avg_ref, var_ref;  /* reference stats */
  double l1_tst, min_tst, max_tst, avg_tst, var_tst;  /* test stats */
  double diag_min_ref, diag_max_ref;   /* diagonal min/max (reference) */
  double diag_min_tst, diag_max_tst;   /* diagonal min/max (test) */
  double v_ref, v_tst;          /* values at max-diff location */
  double w;                     /* cumulative weight for online mean */
  int m, n, i, r;               /* location and reduction count */
} libxs_matdiff_t;
```

The fields linf_abs, linf_rel, and v_ref/v_tst always refer to the
same element. For complex types, statistics use modulus; differences
use complex absolute error.

The `w` field accumulates element count (m*n) across reductions.
`libxs_matdiff_reduce` uses weighted online mean (West/Welford) for
avg_ref/avg_tst: collapses to plain Welford when all matrices are
the same size; gives the exact element-weighted grand mean otherwise.

```C
int libxs_matdiff(libxs_matdiff_t* info,
  libxs_data_t datatype, int m, int n,
  const void* ref, const void* tst,
  const int* ldref, const int* ldtst);
```

Compute scalar differences between two matrices. Supports all real and integer `libxs_data_t` types, plus `LIBXS_DATATYPE_C64` and `LIBXS_DATATYPE_C32` (interleaved complex; dimensions refer to complex elements).

```C
double libxs_matdiff_epsilon(const libxs_matdiff_t* input);
```

Combined error margin from absolute and relative norms. Optionally logs to file via `LIBXS_MATDIFF` env var.

```C
int libxs_matdiff_combine(libxs_matdiff_t* output,
  const libxs_matdiff_t* input);
```

Combine two single-matrix infos (`ref=NULL`) into a meta-diff. Per-side statistics are exact; `linf_abs` is the mean shift, `l2_abs` a statistical bound.

```C
void libxs_matdiff_reduce(libxs_matdiff_t* output,
  const libxs_matdiff_t* input);
void libxs_matdiff_clear(libxs_matdiff_t* info);
```

Worst-case reduction of matdiff results. Initialize with `libxs_matdiff_clear`.

```C
double libxs_matdiff_posdef(const libxs_matdiff_t* info);  /* inline */
```

Returns the smallest test-side diagonal element (positive = necessary
condition for positive definiteness met). Zero if no diagonal data.

## Multiset Distance

Order-independent distance between two vectors treated as multisets.
Counts elements that cannot be matched 1-to-1 within the given
tolerance. The result satisfies metric properties: symmetry,
non-negativity, and identity of indiscernibles.

For real and integer types, both arrays are sorted and matched via
a two-pointer merge. For complex types (C64, C32), a 2D k-d tree
is built on one array and nearest-neighbor queries with consumption
(used-flags) provide 1-to-1 matching by Euclidean distance. The
tolerance threshold is inclusive (less-than-or-equal).

```C
int libxs_setdiff(libxs_data_t datatype,
  const void* a, int na,
  const void* b, int nb, double tol);
```

Supports all libxs_data_t types. Returns the number of unmatched
elements, or -1 for unsupported types. The distance is at least
abs(na - nb) when the vector lengths differ.

```C
int libxs_setdiff_min(libxs_data_t datatype,
  const void* a, int na,
  const void* b, int nb, double* tol);
```

Minimizes the multiset distance over all tolerances using Golden
Section Search (libxs_gss_min). Returns the minimum unmatched count.
The pointer tol (may be NULL) receives the smallest tolerance that
achieves this minimum. Supported types: F64, F32, C64, C32 only
(integer types are not meaningful here; returns max(na, nb) with
tol set to zero).

The distance as a function of tolerance is monotonically
non-increasing (a step function with discrete drops), which makes
it unimodal -- the prerequisite for Golden Section Search.

## Foeppl Polynomial Fingerprint

Structural fingerprint for n-dimensional data based on Foeppl (Macaulay)
bracket polynomials. For 1-D data, the fingerprint records per-derivative-order
norms of the forward finite differences, normalized to the unit interval
[0,1]. Higher dimensions are handled hierarchically: the innermost dimension
is fingerprinted first, each child fingerprint is collapsed to a Sobolev
self-norm scalar, and those scalars form the 1-D input for the next outer
dimension.

Because the fingerprint is normalized to the unit interval, datasets of
different lengths (or shapes) produce directly comparable fingerprints.
The derivative-order decomposition captures structural features at
multiple scales: order 0 measures value magnitude, order 1 measures
slope/trend, order 2 measures curvature, and so on.

```C
#define LIBXS_FPRINT_MAXORDER 8

typedef struct libxs_fprint_t {
  double l2[LIBXS_FPRINT_MAXORDER + 1];
  double l1[LIBXS_FPRINT_MAXORDER + 1];
  double linf[LIBXS_FPRINT_MAXORDER + 1];
  int order, n;
  libxs_data_t datatype;
} libxs_fprint_t;
```

Three norm families are computed per derivative order k = 0..order:

| Field     | Norm | Description                               |
|-----------|------|-------------------------------------------|
| `l2[k]`   | L2   | RMS of the k-th finite difference         |
| `l1[k]`   | L1   | Mean absolute value (total variation)     |
| `linf[k]` | Linf | Maximum absolute value (worst-case decay) |

All three are normalized to the unit interval (h = 1/(n-1)).
The `order` field records how many derivative orders were used;
`n` is the extent of the fingerprinted dimension. The `datatype`
field records the element type -- either the type passed by the
caller, or the type discovered by probing when the caller passes
LIBXS_DATATYPE_UNKNOWN (see "Type Discovery" below).

To recover raw (unnormalized) finite-difference magnitudes, use
the inline helper:

```C
double libxs_fprint_raw(const libxs_fprint_t* info,
  int k, double value);
```

This divides by (n-1)^k, undoing the unit-interval scaling.
For k == 0, the value is returned unchanged.

```C
double libxs_fprint_decay(const libxs_fprint_t* info);
```

Geometric-mean per-order decay ratio:

    r = (l2[K] / l2[0])^(1/K) / (n - 1)

The 1/(n-1) factor removes the unit-interval scaling so that
decay ratios are comparable across different element counts
(without it, fewer elements always appears to have lower decay).

Interpretation of the return value:

    r << 1   Structured data: forward differences shrink with
             derivative order. Data is smooth and compressible
             under Newton truncation.

    r ~= 1   Noise / random data: forward differences grow at the
             rate expected for uncorrelated sequences (~2x per
             order before normalization).

    r >> 1   Should not occur for well-formed data; indicates
             divergent finite differences (numerical overflow or
             pathological input).

    1e30     Returned when no valid ratio can be computed (e.g.,
             order == 0 or l2[0] == 0 or n < 2).

The decay ratio is the key discriminator for type discovery
(see below) and for the compressibility diagnostic described
in the Foeppl paper.

The L2 norms serve comparison via the Sobolev distance.
The Linf norms serve as a decay diagnostic: decaying linf[k]
indicates structurally smooth data (compressible under Newton
truncation), while growing linf[k] indicates unstructured data
(no exploitable smoothness). The L1 norms provide a
noise-robust middle ground between L2 and Linf.

```C
int libxs_fprint(libxs_fprint_t* info,
  libxs_data_t datatype, const void* data,
  int ndims, const size_t shape[], const size_t stride[],
  int order, int axis);
```

Build a fingerprint from data described by shape and stride arrays.
For 1-D data, pass ndims=1, shape={n}, stride=NULL. For higher
dimensions, shape[0] is the innermost extent and shape[ndims-1]
the outermost. When stride is NULL, contiguous storage is assumed
(stride[0]=1, stride[k]=product of shape[0..k-1]). The requested
order is clamped to min(order, extent-1, LIBXS_FPRINT_MAXORDER).

The axis parameter controls how multi-dimensional data is handled:

  axis < 0    Hierarchical mode (default). Sweeps the innermost
              dimension first and collapses each level into the
              next outer dimension via Sobolev self-norms.

  axis >= 0   Per-axis mode. Differentiates along dimension
              'axis' only and takes the per-order maximum of
              each norm (linf, l2, l1) across all positions in
              the remaining dimensions.

For 1-D data, axis is ignored (both modes are equivalent).

Supported types: F64, F32, and all integer libxs_data_t types
(I64, I32, U32, I16, U16, I8, U8 -- promoted to double internally).

When datatype is LIBXS_DATATYPE_UNKNOWN, the function performs
automatic type discovery (Level 0 of the hierarchical analysis
described below). In this mode, shape[0] is the byte count of
the opaque buffer. The function probes all concrete types whose
element width divides the byte count, fingerprints each, and
selects the interpretation with the smallest decay ratio
(libxs_fprint_decay). The discovered type is written to
info->datatype. If no candidate has r < 1, the function returns
EXIT_FAILURE. Ties are broken by preferring the smallest element
width. Float candidates whose maximum absolute value is below
1e-37 (subnormals) are rejected.

```C
double libxs_fprint_diff(
  const libxs_fprint_t* a, const libxs_fprint_t* b,
  const double* weights);
```

Weighted Sobolev distance between two fingerprints:
d = sqrt( sum over k of weights[k] \* (a->l2[k] - b->l2[k])^2 ).
The number of orders compared is min(a->order, b->order) + 1.
If weights is NULL, default weights w(k) = 1/k! are used, which
naturally dampens higher-order (noisier) derivatives.

The distance is a metric: symmetric, non-negative, zero if and
only if the fingerprints are identical. The same function serves
both as a distance measure and as a fingerprint comparator since
the fingerprint is the decomposed form of the Sobolev norm.

## Type Discovery for Opaque Data

When data arrives as a raw byte stream with no type metadata, the
LIBXS toolkit can be composed into a hierarchical analysis that
discovers the element type and structure. Each level uses a
different tool, exploiting the fact that each tool is sensitive
to a different axis of structure while being invariant to others.

    Tool              Invariant to         Sensitive to
    ---------------------------------------------------------------
    fprint (decay)    element count        consecutive-element
                      and scaling          correlation (local)
    setdiff           element order        value identity (global)
    matdiff           (positional)         magnitude, spread,
                                           extrema
    sort_smooth       row order            row-to-row proximity
    shuffle           --                   provides controlled
                                           reordering

The hierarchical procedure:

    Level 0 -- Decay Screening (libxs_fprint + libxs_fprint_decay)

      Interpret the byte stream under every candidate type whose
      element width divides the stream length. Compute the decay
      ratio r for each. Discard r >= 1 (noise). Short-list the
      best few candidates (within a factor 2 of the minimum r).

      This level is built into libxs_fprint when the caller passes
      LIBXS_DATATYPE_UNKNOWN. The discovered type is reported in
      the datatype field of the returned libxs_fprint_t.

    Level 1 -- Self-Consistency via Setdiff (libxs_setdiff_min)

      Split the stream into two halves (A, B) under each surviving
      candidate type. Compute d* = libxs_setdiff_min(A, B). The
      ratio d* / max(|A|, |B|) is a "novelty fraction": how many
      values in one half have no counterpart in the other. A correct
      type produces low novelty (values repeat or cluster); a wrong
      type scatters values across the range (high novelty).

      Synergy: fprint measures local correlation (adjacent elements);
      setdiff measures global value identity (ignoring order). A
      candidate that passes both has independent evidence on two
      orthogonal axes.

    Level 2 -- Smoothness-Sort Separability (libxs_sort_smooth)

      Reshape the stream as a matrix under each surviving candidate.
      Compute a GREEDY row permutation that minimizes row-to-row
      Euclidean distance. Fingerprint the permuted matrix along the
      row axis and measure the improvement:

          delta = r_before - r_after

      The candidate with the largest delta has the most exploitable
      row structure.

      Synergy: the GREEDY sort is O(m^2 n) -- too expensive for the
      initial sweep but affordable for 2-3 survivors. Its sensitivity
      (row-to-row proximity) is orthogonal to 1-D decay and to
      order-independent setdiff.

    Level 3 -- Shuffle Stability (libxs_shuffle + libxs_fprint_decay)

      Shuffle the interpreted elements via libxs_shuffle (coprime
      permutation). Fingerprint the shuffled data. If the decay
      ratio increases substantially, the original element order
      carries genuine structure. If it stays the same, the apparent
      decay was accidental.

      Synergy: this is the only test that probes whether the ORDER
      of elements matters. Decay (Level 0) measures local correlation
      but cannot distinguish true order from coincidence. Setdiff
      (Level 1) is explicitly order-invariant. The shuffle test
      closes this gap.

    Level 4 -- Statistical Plausibility (libxs_matdiff)

      Compute single-matrix statistics (ref=NULL) for each surviving
      candidate: mean, variance, min, max, diagonal range. Reject
      float interpretations with NaN, infinity, or subnormals.
      Flag integer interpretations whose value range spans less than
      1% of the type range. Compare two candidates via
      libxs_matdiff_combine (mean-shift and variance bound).

    Stride Sweep -- Record Layout Discovery

      After discovering the atomic element width w, sweep candidate
      strides s that are multiples of w. Reshape the stream as an
      (L/s) x (s/w) matrix. Repeat Levels 0-2 on the outer
      dimension (across "records"). The stride with the smallest
      decay and largest sort-improvement likely corresponds to the
      record boundary.

Why the composition is stronger than any single tool:

  - Decay alone is fooled by reinterpretation artifacts (e.g.,
    integer data producing subnormal floats that look smooth).
  - Setdiff alone cannot detect structure -- it only counts
    matches, and with large enough tolerance any data matches.
  - Sort-smooth alone is too expensive for a brute-force sweep
    and cannot discriminate types without a baseline decay.
  - Shuffle stability alone cannot distinguish true order from
    accidental order in short sequences.
  - Statistics alone reject only pathological cases (NaN,
    subnormals) and cannot resolve two plausible interpretations.

Each level eliminates a different class of false positives. A
candidate that survives all five has demonstrated structure along
five orthogonal axes: local smoothness, global value consistency,
row separability, order sensitivity, and statistical plausibility.
A false positive would require a simultaneous coincidence on all
five, which is combinatorially unlikely.

Limitations: the analysis tests fixed-width candidates and cannot
discover variable-length encodings, compression, or encryption.
For data that is genuinely random at every granularity, all
candidates are rejected at Level 0 and the framework reports
"no structure found." The stride sweep discovers fixed-size
records but not nested or recursive structures.

## Golden Section Search

```C
double libxs_gss_min(
  double (*fn)(double x, const void* data),
  const void* data,
  double x0, double x1, double* xmin, int maxiter);
```

Minimizes a unimodal function fn on the interval [x0, x1]. The
callback receives x and an opaque context pointer. Returns f(x\*)
where x\* is the minimizer; xmin (may be NULL) receives x\*.

The bracket shrinks by factor phi = (sqrt(5)-1)/2 per iteration,
reusing one evaluation from the previous step. Convergence is
reached when the bracket collapses to machine precision or maxiter
iterations are exhausted.

## Number Theory

```C
size_t libxs_gcd(size_t a, size_t b);
size_t libxs_lcm(size_t a, size_t b);
```

GCD (returns 1 for gcd(0,0)) and LCM.

```C
int libxs_primes_u32(unsigned int num,
  unsigned int num_factors_n32[], int num_factors_max);
```

Prime factors of `num` in ascending order. Returns factor count.

```C
size_t libxs_coprime(size_t n, size_t minco);
size_t libxs_coprime_bias(size_t n, double bias);
size_t libxs_coprime2(size_t n); /* inline: coprime_bias(n, 0) */
```

`libxs_coprime`: co-prime of `n` not exceeding `minco`.

`libxs_coprime_bias`: co-prime of `n` selected by bias in [-1, +1]
via a logarithmic mapping (target = n^((1+bias)/2)).
bias=-1 selects the smallest non-trivial coprime (maximum
displacement), bias=0 selects near sqrt(n) (balanced, same as
libxs_coprime2), bias=+1 selects near n/2 (near-alternation).
Monotonic: larger bias always yields a larger (or equal) coprime.

`libxs_coprime2`: inline convenience for `libxs_coprime_bias(n, 0)`.

```C
unsigned int libxs_remainder(unsigned int a, unsigned int b,
  const unsigned int* limit, const unsigned int* remainder);
```

Smallest multiple of `b` minimizing remainder mod `a`.

```C
unsigned int libxs_product_limit(unsigned int product,
  unsigned int limit, int is_lower);
```

Sub-product of prime factors within `limit` (0/1-Knapsack).

## Scalar Utilities

```C
double libxs_kahan_sum(double value, double* accumulator,
  double* compensation);
```

Kahan compensated summation.

```C
unsigned int libxs_isqrt_u64(unsigned long long x);
unsigned int libxs_isqrt_u32(unsigned int x);
unsigned int libxs_isqrt2_u32(unsigned int x);
```

Integer square root. The isqrt2 variant returns a factor of x.

```C
double libxs_pow2(int n);
```

2^n via IEEE-754 exponent. Valid for n in [-1022, 1023]. Returns 0.0 for underflow (subnormals flushed to zero) and +Inf for overflow.

## Modular Arithmetic

```C
unsigned int libxs_mod_inverse_u32(unsigned int a, unsigned int m);
```

Modular inverse (extended Euclidean). Requires gcd(a, m) = 1.

```C
unsigned int libxs_barrett_rcp(unsigned int p);
unsigned int libxs_barrett_pow18(unsigned int p);
unsigned int libxs_barrett_pow36(unsigned int p);
```

Barrett reduction constants for modulus `p`.

```C
unsigned int libxs_mod_u32(uint32_t x, unsigned int p,
  unsigned int rcp);                           /* inline */
unsigned int libxs_mod_u64(uint64_t x, unsigned int p,
  unsigned int rcp, unsigned int pow18,
  unsigned int pow36);                         /* inline */
```

Fast modular reduction via Barrett. The 64-bit variant uses a radix-2^18 split.

## BF16 Conversion

```C
typedef uint16_t libxs_bf16_t;
libxs_bf16_t libxs_round_bf16(double x);   /* inline */
double libxs_bf16_to_f64(libxs_bf16_t v);  /* inline */
```

Round to BF16 (round-to-nearest-even) and expand to double. Uses native `__bf16` when available (`LIBXS_BF16`), otherwise portable bit manipulation.
