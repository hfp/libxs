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
  double grade;                 /* componentwise grade (matdiff_grade) */
  double w;                     /* cumulative weight for online mean */
  int m, n, i, r;               /* location and reduction count */
} libxs_matdiff_t;
```

The fields linf_abs, linf_rel, and v_ref/v_tst always refer to the
same element. For complex types, statistics use modulus; differences
use complex absolute error.

The `w` field accumulates the element count (m·n) across reductions, so
avg_ref/avg_tst of a reduction are the element-weighted means.

```C
int libxs_matdiff(libxs_matdiff_t* info,
  libxs_data_t datatype, int m, int n,
  const void* ref, const void* tst,
  const int* ldref, const int* ldtst);
```

Compute scalar differences between two matrices. Supports `F64`, `F32`, all
integer types, and `LIBXS_DATATYPE_C64` and `LIBXS_DATATYPE_C32` (interleaved
complex; dimensions refer to complex elements).

```C
int libxs_matdiff_grade(libxs_matdiff_t* info,
  libxs_data_t datatype, int m, int n,
  const void* ref, const void* tst, const void* bound,
  const int* ldref, const int* ldtst, const int* ldbnd);
```

Everything `libxs_matdiff` calculates, plus `grade`, which `libxs_matdiff`
leaves zero. The caller supplies `bound`, the elementwise upper bound on the
exact result (for a GEMM: `|alpha||A||B| + |beta||C|`). The grade is
dimensionless (units of roundoff), so the threshold stays with the caller:
a componentwise-stable implementation stays at or below a modest multiple
of the number of terms summed. `F64`, `F32`, `C64`, and `C32`: for complex
data, the difference is the modulus of the complex difference, and `bound`
is real (of the same precision; `ldbnd` counts its elements). Any other type
is reported rather than silently left ungraded.

```C
double libxs_matdiff_epsilon(const libxs_matdiff_t* input);
```

Combined error margin from absolute and relative norms. If the environment
variable `LIBXS_MATDIFF` names a file, the margin is appended to it as one line;
if it names a directory, the line goes to `libxs_matdiff.log` inside it. Further
space-separated words in `LIBXS_MATDIFF` are appended to the line, followed by
the command line.

```C
int libxs_matdiff_combine(libxs_matdiff_t* output,
  const libxs_matdiff_t* input);
```

Combine two single-matrix infos (`ref=NULL`) into a meta-diff. Per-side
statistics are exact; `linf_abs` is the mean shift, `l2_abs` a statistical
bound.

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

Environment variables read by `libxs_matdiff`:

- `LIBXS_DUMP`: unset, empty, or `0` disables it (default). Otherwise the
  reference and test matrices are written as MHD files
  `<prefix>-<address>-ref.mhd` and `<prefix>-<address>-tst.mhd`. The prefix
  is the value itself unless it starts with a digit or `-`, in which case it
  is `libxs_dump`. A value starting with `-` writes only matrices that differ,
  and `-2` (or higher) additionally prints a passed/failed line per
  comparison. A number whose magnitude exceeds 1 (e.g., `2`) writes the data
  reshaped into a near-square two-dimensional array. Files are written only
  in builds with MHD support (the default configuration).
- `LIBXS_MATDIFF_SHUFFLE`: a nonzero integer (or an empty value) visits the
  elements in a co-prime shuffled order rather than column-major order;
  unset or `0` keeps the default order.

From Fortran (`USE libxs`), `libxs_matdiff(info, datatype, m[, n, ref, tst,
ldref, ldtst, bound, ldbnd, rc])` takes `TYPE(C_PTR)` matrices and optional
leading dimensions; passing `bound` selects `libxs_matdiff_grade`, and `rc`
receives the status. `libxs_matdiff_reduce`, `libxs_matdiff_clear`,
`libxs_matdiff_epsilon`, and `libxs_matdiff_combine` are bound directly.

## Multiset Distance

Order-independent distance between two vectors treated as multisets.
Counts elements that cannot be matched 1-to-1 within the given
tolerance. The result satisfies metric properties: symmetry,
non-negativity, and identity of indiscernibles.

Complex values (C64, C32) are matched by the modulus of their difference.
The tolerance is inclusive (less-than-or-equal).

```C
int libxs_setdiff(libxs_data_t datatype,
  const void* a, int na,
  const void* b, int nb, double tol);
```

Supports all `libxs_data_t` types. Returns the number of
unmatched elements, or -1 for unsupported types. The distance is at least
abs(na - nb) when the vector lengths differ.

```C
int libxs_setdiff_min(libxs_data_t datatype,
  const void* a, int na,
  const void* b, int nb, double* tol);
```

Minimizes the multiset distance over all tolerances and returns the minimum
unmatched count. The pointer tol (may be NULL) receives the smallest
tolerance that achieves this minimum. Supported types: F64, F32,
C64, C32 only (integer types are not meaningful here; returns max(na, nb)
with tol set to zero).

## Polynomial Fingerprint

Structural fingerprint for n-dimensional data based on Macaulay bracket
(truncated-power) polynomials. For 1-D data, the fingerprint records
per-derivative-order norms of the forward finite differences, normalized to
the unit interval [0,1]. Higher dimensions are handled hierarchically: the
innermost dimension is fingerprinted first, each child fingerprint is
collapsed to a Sobolev self-norm scalar, and those scalars form the 1-D input
for the next outer dimension.

Because of this normalization, datasets of different lengths (or shapes)
produce directly comparable fingerprints. Order 0 reflects value magnitude,
order 1 slope, order 2 curvature, and so on.

Uses: compare datasets across resolutions or shapes (`libxs_fprint_diff`),
judge smoothness and compressibility (`libxs_fprint_decay`), discover the
element type of opaque data (`LIBXS_DATATYPE_UNKNOWN`), fingerprint streams
or merge partial results (`libxs_fprint_partial`, `libxs_fprint_join`), rate
roughness against a smooth reference (`libxs_fprint_bernstein`), and measure
an exact distance between two sample sets (`libxs_fprint_bernstein_diff`).

```C
#define LIBXS_FPRINT_MAXORDER 8

typedef struct libxs_fprint_t {
  double l2[LIBXS_FPRINT_MAXORDER + 1];
  double l1[LIBXS_FPRINT_MAXORDER + 1];
  double linf[LIBXS_FPRINT_MAXORDER + 1];
  double mean[LIBXS_FPRINT_MAXORDER + 1];
  double acc_sq[LIBXS_FPRINT_MAXORDER + 1];
  double acc_abs[LIBXS_FPRINT_MAXORDER + 1];
  double acc_sum[LIBXS_FPRINT_MAXORDER + 1];
  double tail[LIBXS_FPRINT_MAXORDER + 1];
  double bernstein[LIBXS_FPRINT_MAXORDER + 1];
  int order, n;
  int nk[LIBXS_FPRINT_MAXORDER + 1];
  int nbernstein;
  libxs_data_t datatype;
} libxs_fprint_t;
```

Three norm families and a signed mean are computed per derivative
order k = 0..order:

| Field       | Description                                    |
|-------------|------------------------------------------------|
| `l2[k]`    | L2 norm of the k-th finite difference          |
| `l1[k]`    | Mean absolute value (total variation)          |
| `linf[k]`  | Maximum absolute value (worst-case decay)      |
| `mean[k]`  | Signed mean (sum / count, preserves sign/phase)|
| `acc_sq[k]` | Accumulator: sum of raw squared differences   |
| `acc_abs[k]`| Accumulator: sum of raw absolute differences  |
| `acc_sum[k]`| Accumulator: sum of raw signed differences    |
| `tail[k]`  | Last value at derivative level k (for partial) |
| `nk[k]`    | Count of differences accumulated at order k    |
| `bernstein[k]` | Bernstein derivative norm (if requested)   |
| `nbernstein` | Orders held in bernstein (0: none)           |

All norms are normalized to the unit interval (h = 1/(n-1)).

The `order` field records how many derivative orders were used;
`n` is the extent of the fingerprinted dimension. The `datatype`
field records the element type — either the type passed by the
caller, or the type discovered by probing when the caller passes
LIBXS_DATATYPE_UNKNOWN (see Type Discovery below).

To recover raw (unnormalized) finite-difference magnitudes, use
libxs_fprint_raw (see Generalized Binomial and Distance section).

```C
double libxs_fprint_decay(const libxs_fprint_t* info);
```

Geometric-mean per-order decay ratio:

    r = (l2[K] / l2[0])^(1/K) / (n - 1)

A ratio well below 1 indicates smooth (compressible) data, and a ratio
between 1 and 2 indicates noise; 1e30 is returned when no ratio can be
computed (order == 0, l2[0] == 0, or n < 2).

```C
typedef enum libxs_fprint_flags_t {
  LIBXS_FPRINT_DEFAULT  = 0,
  LIBXS_FPRINT_SORT     = 1,
  LIBXS_FPRINT_AUTOCORR = 2,
  LIBXS_FPRINT_PERAXIS  = 4,
  LIBXS_FPRINT_WITH_BERNSTEIN = 8
} libxs_fprint_flags_t;

int libxs_fprint(libxs_fprint_t* info,
  libxs_data_t datatype, const void* data,
  int ndims, const size_t shape[], const size_t stride[],
  int order, int axis, int smooth, unsigned int flags);
```

Build a fingerprint from data described by shape and stride arrays.
For 1-D data, pass ndims=1, shape={n}, stride=NULL. For higher
dimensions, shape[0] is the innermost extent and shape[ndims-1]
the outermost. When stride is NULL, contiguous storage is assumed
(stride[0]=1, stride[k]=product of shape[0..k-1]). The requested
order is clamped to min(order, extent-1, LIBXS_FPRINT_MAXORDER).

The flags parameter is a bitwise OR of preprocessing options:

- `LIBXS_FPRINT_SORT`: sort values before differencing
  (permutation-invariant fingerprint).
- `LIBXS_FPRINT_AUTOCORR`: replace the signal with its autocorrelation
  before differencing (shift-invariant).
- `LIBXS_FPRINT_PERAXIS`: per-axis mode. Differentiates along dimension
  `axis` only and takes the per-order maximum of each norm across all
  positions in the remaining dimensions.
- `LIBXS_FPRINT_WITH_BERNSTEIN`: also store the libxs_fprint_bernstein
  norms of the samples as differenced (1-D only, at the cost of that call);
  libxs_fprint_diff then compares them as well.

Without LIBXS_FPRINT_PERAXIS, hierarchical mode is used (sweeps
the innermost dimension first, collapses via Sobolev self-norms).
The axis parameter is only consulted when LIBXS_FPRINT_PERAXIS
is set.

The smooth parameter specifies a box-filter radius applied before
differencing (0 = off). Each output sample becomes the average of
the 2*smooth+1 nearest input samples (clamped at boundaries).

For 1-D data without LIBXS_FPRINT_PERAXIS, axis is ignored.

Supported types: F64, F32, and all integer libxs_data_t types
(I64, U64, I32, U32, I16, U16, I8, U8 — promoted to double internally).

With datatype LIBXS_DATATYPE_UNKNOWN, the element type is discovered
(see Type Discovery below).

```C
int libxs_fprint_partial(libxs_fprint_t* info,
  libxs_data_t datatype, const void* data, int n, int order);
```

Streaming (incremental) fingerprint accumulation. Processes data
one chunk at a time and maintains exact junction derivatives across
calls via the tail buffer. First call: pass a zero-initialized info.
Subsequent calls: pass the same info with the next contiguous chunk.
Only 1-D contiguous data is supported. The normalized fields
(l2, l1, linf, mean) are updated after each call so that
libxs_fprint_diff can be used at any intermediate point.

For a single chunk of n elements, produces the same result as
libxs_fprint with ndims=1, shape={n}, stride=NULL.

```C
int libxs_fprint_join(libxs_fprint_t* output,
  const libxs_fprint_t* a, const libxs_fprint_t* b);
```

Approximate merge of two finalized fingerprints. Combines a and b
into output using the raw accumulators (acc_sq, acc_abs, acc_sum).
Junction derivatives between the two regions are not recomputed,
so the result is approximate with error proportional to
order / min(a->n, b->n). The linf field is the maximum of the
un-scaled maxima from both sides, re-scaled to the combined length.
Returns EXIT_FAILURE if the combined n is below 2.

```C
double libxs_fprint_diff(
  const libxs_fprint_t* a, const libxs_fprint_t* b,
  const double* weights);
```

Weighted Sobolev distance between two fingerprints:

```text
d = sqrt( sum_k weights[k] * [(a->l2[k] - b->l2[k])^2 + (a->mean[k] - b->mean[k])^2] )
```

The number of orders compared is min(a->order, b->order) + 1.
If weights is NULL, default weights w(k) = 1/k! are used. Where both
fingerprints hold Bernstein norms, w(k) \* (a->bernstein[k] -
b->bernstein[k])^2 is added for those orders; streamed and joined
fingerprints hold none.

```C
#define LIBXS_FPRINT_BANDED 1024

int libxs_fprint_bernstein(double norms[], libxs_data_t datatype,
  const void* data, int n, int order);
```

L2 norms on [0,1] of the derivatives of the Bernstein approximant of the n
samples: norms[k] = ‖B⁽ᵏ⁾‖ for k = 0..order. The order is clamped to
LIBXS_FPRINT_MAXORDER, and orders beyond the degree n − 1 give zero. For any
data, norms[k] ≤ l2[k] of `libxs_fprint` on the same samples, so the ratio
norms[k] / l2[k] rates roughness independently of the amplitude: near 1 for
smooth data, small for edges and noise.

The call costs far more than `libxs_fprint` for large n. Above
LIBXS_FPRINT_BANDED samples, weights below 10⁻¹⁷ of their largest are
skipped. The Fortran generic
`libxs_fprint_bernstein(norms, data[, order])` takes REAL(C_DOUBLE) or
REAL(C_FLOAT) data and fills norms(0:order).

```C
int libxs_fprint_bernstein_diff(double* dist, libxs_data_t datatype,
  const void* a, int na, const void* b, int nb, int order,
  const double weights[]);
```

Exact Sobolev distance between the Bernstein approximants of two sample
sets of any lengths: dist = sqrt(Σₖ w(k) ‖(B_a − B_b)⁽ᵏ⁾‖²) for
k = 0..order, with w(k) = 1/k! if weights is NULL. Unlike
`libxs_fprint_diff`, which compares summaries, it is a metric on the data
themselves (zero only for equal samples), and it needs both sample sets.
The Fortran generic
`libxs_fprint_bernstein_diff(dist, a, b[, order][, weights])` takes
REAL(C_DOUBLE) or REAL(C_FLOAT) samples.

From Fortran (`USE libxs`), the generic `libxs_fprint(info, data[, order,
axis, smooth, flags])` takes a one-dimensional REAL(C_DOUBLE), REAL(C_FLOAT),
INTEGER(C_INT), or INTEGER(C_INT8_T) array, and `libxs_fprint(info, datatype,
data, ndims, shape[, stride, order, axis, smooth, flags])` a `TYPE(C_PTR)`
with an explicit shape; order defaults to LIBXS_FPRINT_MAXORDER and the other
options to zero. `libxs_fprint_partial(info, data[, order])` takes the same
four array types, `libxs_fprint_diff(a, b[, weights])` returns the distance,
and `libxs_fprint_join(output, a, b)` is bound directly.

## Type Discovery for Opaque Data

To identify the element type of an untyped buffer, call `libxs_fprint` with
`LIBXS_DATATYPE_UNKNOWN`, ndims=1, and shape[0] set to the byte count. Every
type whose width divides the byte count is fingerprinted, and the one with
the smallest decay ratio is written to info->datatype; ties go to the
smaller width, and float interpretations whose largest magnitude is below
1e-37 are skipped. EXIT_FAILURE is returned if no candidate yields a finite
ratio. The choice is a best guess: if `libxs_fprint_decay` of the result is
above 1, the buffer showed no structure under any type.

## Golden Section Search

```C
typedef enum libxs_gss_flags_t {
  LIBXS_GSS_DEFAULT        = 0,
  LIBXS_GSS_EVAL_ENDPOINTS = 1
} libxs_gss_flags_t;

typedef enum libxs_gss_status_t {
  LIBXS_GSS_STATUS_DEFAULT      = 0,
  LIBXS_GSS_STATUS_ENDPOINT_MIN = 1,
  LIBXS_GSS_STATUS_FLAT_MIN     = 2,
  LIBXS_GSS_STATUS_LEFT_REFINED = 4,
  LIBXS_GSS_STATUS_MAXITER      = 8,
  LIBXS_GSS_STATUS_NO_BRACKET   = 16
} libxs_gss_status_t;

typedef struct libxs_gss_info_t {
  double unimodality;
  double xmin, fmin;
  double x0, x1;
  int evaluations;
  int iterations;
  int status;
} libxs_gss_info_t;
```

| Status                          | Meaning                                         |
|---------------------------------|-------------------------------------------------|
| `LIBXS_GSS_STATUS_ENDPOINT_MIN` | The minimum lies at x0 or x1                    |
| `LIBXS_GSS_STATUS_FLAT_MIN`     | Another sample is within ftol of the minimum    |
| `LIBXS_GSS_STATUS_LEFT_REFINED` | libxs_bisect_min refined the left edge          |
| `LIBXS_GSS_STATUS_MAXITER`      | maxiter iterations were exhausted               |
| `LIBXS_GSS_STATUS_NO_BRACKET`   | libxs_bisect_min: neither endpoint reaches fmin |

Both searches require x0 <= x1 and maxiter > 0. Function values that differ
by at most ftol count as equal; with ftol <= 0, only identical values do.

```C
double libxs_gss_min(
  double (*fn)(double x, const void* data),
  const void* data,
  double x0, double x1, double* xmin, int maxiter,
  int flags, double ftol,
  libxs_gss_info_t* info);
```

Minimizes a unimodal function fn on the interval [x0, x1]. The
callback receives x and an opaque context pointer. Returns f(x\*)
where x\* is the minimizer; xmin (may be NULL) receives x\*.
The search stops when the bracket collapses to machine precision or
after maxiter iterations. `LIBXS_GSS_EVAL_ENDPOINTS` also evaluates fn at
x0 and x1. For
step-like objectives such as tolerance thresholds, use
libxs_bisect_min instead.

The optional info pointer (may be NULL) receives status flags,
iteration and evaluation counts, the final x/f pair, the final
bracket, and a unimodality score in [0, 1] (1.0: samples consistent
with a single valley; lower: irregular or multimodal).

```C
double libxs_bisect_min(
  double (*fn)(double x, const void* data),
  const void* data,
  double x0, double x1, double fmin, double* xmin, int maxiter,
  double ftol, libxs_gss_info_t* info);
```

Bisects the interval [x0, x1] to find the left edge of a known
minimum level fmin, for monotone threshold or plateau-edge searches. The
right endpoint is expected to reach fmin; otherwise info->status includes
LIBXS_GSS_STATUS_NO_BRACKET. Returns fn at the found point, and xmin (may
be NULL) receives that point.

## Generalized Binomial and Distance

```C
double libxs_binom(double t, int k);
```

Generalized binomial coefficient C(t, k) for real-valued t:
C(t, k) = t * (t-1) * ... * (t-k+1) / k!. Evaluates the Newton
basis polynomial at position t. For integer t >= k >= 0, this
equals the standard binomial coefficient.

```C
double libxs_dist2(const double* a, const double* b, int n);
```

Squared Euclidean distance between two n-dimensional vectors.
Returns the sum of squared component differences.

```C
double libxs_fprint_raw(const libxs_fprint_t* info,
  int k, double value);
```

Recover unnormalized finite-difference magnitude by dividing by
(n-1)^k, undoing the unit-interval scaling. For k == 0, the
value is returned unchanged.

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

Prime factors of `num` in ascending order, repeated by multiplicity
(12 yields [2, 2, 3]). Returns the factor count, which may exceed
`num_factors_max` (only that many are written; zero queries the count
without writing). Returns zero if `num` is a prime other than two.

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

## Compensated Arithmetic

```C
double libxs_neumaier_sum(double value, double* accumulator, double* compensation);
double libxs_sum2(const double* x, int n);
double libxs_sum2_err(const double* x, int n, double* err);
double libxs_dot2(const double* x, const double* y, int n);
double libxs_dot2_err(const double* x, const double* y, int n, double* err);
```

| Function             | Result                                                                |
|----------------------|-----------------------------------------------------------------------|
| `libxs_neumaier_sum` | Like `libxs_kahan_sum`, robust for summands of any magnitude order    |
| `libxs_sum2`         | Sum of x[n] as if computed in twice the working precision             |
| `libxs_sum2_err`     | As `libxs_sum2`; err (may be NULL) bounds the error: the exact sum is in [res − err, res + err] |
| `libxs_dot2`         | Dot product of x[n] and y[n] as if computed in twice the working precision |
| `libxs_dot2_err`     | As `libxs_dot2`, with an error bound as for `libxs_sum2_err`          |

```C
int libxs_dot2_gemm(libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, const void* alpha, const void* a, int lda,
  const void* b, int ldb, const void* beta, void* c, int ldc);
```

Reference GEMM, C = alpha · op(A) · op(B) + beta · C (column-major, BLAS
semantics, trans one of `'N'`, `'T'`, `'C'`), with every element computed
by `libxs_dot2`. Use it to grade a GEMM computed in working precision.
Supports F64, F32, C64, and C32. It is serial: to grade a large product,
pass a few rows (m=1 per row). Returns EXIT_FAILURE in a build whose
compensation was optimized away (see `LIBXS_PRAGMA_FASTMATH_OFF`).

```C
double libxs_two_sum(double a, double b, double* err);      /* inline */
double libxs_fast_two_sum(double a, double b, double* err); /* inline */
double libxs_two_product(double a, double b, double* err);  /* inline */
```

Error-free transformations: each returns the rounded result x and writes the
rounding error to err, so that a + b == x + *err (or a · b == x + *err).
`libxs_fast_two_sum` is exact only if |a| >= |b|. `libxs_two_product` is
exact unless the product underflows.

```C
unsigned int libxs_isqrt_u64(unsigned long long x);
unsigned int libxs_isqrt_u32(unsigned int x);
unsigned int libxs_isqrt2_u32(unsigned int x);
```

Exact integer square root, floor(sqrt(x)), without floating point.
The isqrt2 variant returns a factor of x.

```C
double libxs_pow2(int n);
```

2^n via IEEE-754 exponent. Valid for n in [-1022, 1023]. Returns 0.0 for
underflow (subnormals flushed to zero) and +Inf for overflow.

## Modular Arithmetic

```C
unsigned int libxs_mod_inverse_u32(unsigned int a, unsigned int m);
size_t libxs_mod_inverse(size_t a, size_t m);
```

Modular inverse a⁻¹ mod m.
Requires gcd(a, m) = 1 and 0 < a, 1 < m. The size_t variant
supports counts up to 2^63 - 1.

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

Fast modular reduction via Barrett. The 32-bit variant is valid for any
`x`; the 64-bit variant uses a radix-2^18 split and is valid for x < 2^54
and p < 8192.

## BF16 and FP16 Conversion

```C
typedef uint16_t libxs_bf16_t;
typedef uint16_t libxs_f16_t;
libxs_bf16_t libxs_round_bf16_f32(float x);    /* inline */
float libxs_bf16_to_f32(libxs_bf16_t v);       /* inline */
libxs_bf16_t libxs_round_bf16(double x);       /* inline */
double libxs_bf16_to_f64(libxs_bf16_t v);      /* inline */
libxs_bf16_t libxs_u8_to_bf16(unsigned int v); /* inline */
libxs_f16_t libxs_round_f16_f32(float x);      /* inline */
float libxs_f16_to_f32(libxs_f16_t v);         /* inline */
libxs_f16_t libxs_round_f16(double x);         /* inline */
double libxs_f16_to_f64(libxs_f16_t v);        /* inline */
```

`libxs_bf16_t` and `libxs_f16_t` hold the raw encodings of BF16 and IEEE
FP16. The `round` functions round to nearest even, and the `to` functions
expand exactly to float or double. `libxs_u8_to_bf16` converts an 8-bit
unsigned integer exactly. Native `__bf16` (`LIBXS_BF16`) and `_Float16`
(`LIBXS_F16`) are used when the compiler provides them, otherwise portable
bit manipulation.

```C
void libxs_bf16_to_f32_block(float* dst,
  const libxs_bf16_t* src, size_t n);          /* inline */
void libxs_round_bf16_f32_block(libxs_bf16_t* dst,
  const float* src, size_t n);                 /* inline */
void libxs_f16_to_f32_block(float* dst,
  const libxs_f16_t* src, size_t n);           /* inline */
void libxs_round_f16_f32_block(libxs_f16_t* dst,
  const float* src, size_t n);                 /* inline */
void libxs_u8_to_bf16_block(libxs_bf16_t* dst,
  const unsigned char* src, size_t n);         /* inline */
```

Convert n contiguous elements like the scalar functions above, e.g., into
scratch ahead of a compute loop.

```C
void libxs_dekker_bf16(double val, int ndigits,
  libxs_bf16_t* dst);                       /* inline */
void libxs_dekker_bf16_f32(float val, int ndigits,
  libxs_bf16_t* dst);                       /* inline */
```

Dekker split: decompose a value into `ndigits` BF16 residuals via
iterative subtraction. `dst[0]` holds the dominant digit (the BF16
rounding of the input); `dst[1]` holds the residual after subtracting
the first digit's contribution, rounded to BF16; and so on. The sum
`libxs_bf16_to_f64(dst[0]) + ... + libxs_bf16_to_f64(dst[ndigits-1])`
reconstructs the original to about `8 * ndigits` significant bits, so
seven digits cover a double. BF16 has the exponent range of
float: a value beyond it overflows, and digits below it are lost. The
`_f32` variant keeps the residual arithmetic in single precision for
float storage/conversion paths.
