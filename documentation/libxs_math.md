# Math Utilities

Header: `libxs_math.h`

Matrix comparison, number-theoretic helpers, modular arithmetic, and BF16 conversion functions.

## Matrix Difference

```C
typedef struct libxs_matdiff_t {
  double norm1_abs, norm1_rel;   /* one-norm */
  double normi_abs, normi_rel;   /* infinity-norm */
  double normf_rel;              /* Frobenius-norm (relative) */
  double linf_abs, linf_rel;    /* max element-wise difference */
  double l2_abs, l2_rel, rsq;   /* L2-norm and R-squared */
  double l1_ref, min_ref, max_ref, avg_ref, var_ref;  /* reference stats */
  double l1_tst, min_tst, max_tst, avg_tst, var_tst;  /* test stats */
  double diag_min_ref, diag_max_ref;                  /* diagonal ref */
  double diag_min_tst, diag_max_tst;                  /* diagonal tst */
  double v_ref, v_tst;          /* values at location of max diff */
  int m, n, i, r;               /* location and reduction count */
} libxs_matdiff_t;
```

```C
int libxs_matdiff(libxs_matdiff_t* info,
  libxs_data_t datatype, int m, int n,
  const void* ref, const void* tst,
  const int* ldref, const int* ldtst);
```

Compute a collection of scalar differences between two matrices. Supports all `libxs_data_t` element types. The location of the largest absolute difference is recorded.

```C
double libxs_matdiff_epsilon(const libxs_matdiff_t* input);
```

Combine absolute and relative norms into a single margin value. Can optionally log to a calibration file via the `LIBXS_MATDIFF` environment variable.

```C
int libxs_matdiff_combine(libxs_matdiff_t* output, const libxs_matdiff_t* input);
```

Combine two single-matrix infos (each from `libxs_matdiff` with `ref=NULL`) into a meta-diff. The output supplies the reference side, the input the test side. Per-side statistics (l1, min, max, avg, var) are exact. Difference norms are summary bounds: `linf_abs` is the mean shift, `l2_abs` is a statistical bound from pooled variance. Element-wise norms are set to zero. No-op if neither side is a single-matrix info. Returns `EXIT_FAILURE` if only one side is single-matrix. Also called implicitly by `libxs_matdiff_reduce` when the input is a single-matrix info.

```C
void libxs_matdiff_reduce(libxs_matdiff_t* output,
  const libxs_matdiff_t* input);
void libxs_matdiff_clear(libxs_matdiff_t* info);
```

Thread-safe reduction (max) of matdiff results. Initialize `output` with `libxs_matdiff_clear` before the first reduction.

```C
double libxs_matdiff_posdef(const libxs_matdiff_t* info);
```

Necessary condition for positive definiteness: returns the smallest diagonal element of the test-side matrix. A positive return value means all diagonal elements are positive; the magnitude indicates the margin. Returns zero if no diagonal data is available. For single-matrix info (`ref=NULL`), the matrix is on the test-side after the internal swap.

## Number Theory

```C
size_t libxs_gcd(size_t a, size_t b);
size_t libxs_lcm(size_t a, size_t b);
```

Greatest common divisor (GCD of 0 and 0 is 1) and least common multiple.

```C
int libxs_primes_u32(unsigned int num,
  unsigned int num_factors_n32[], int num_factors_max);
```

Find prime factors of `num` in ascending order. Returns the number of factors found (zero if `num` is prime and not 2).

```C
size_t libxs_coprime(size_t n, size_t minco);
size_t libxs_coprime2(size_t n);
```

Find a co-prime of `n` not exceeding `minco` (or sqrt(n) for `coprime2`).

```C
unsigned int libxs_remainder(unsigned int a, unsigned int b,
  const unsigned int* limit, const unsigned int* remainder);
```

Find the smallest multiple of `b` whose remainder mod `a` is minimized, subject to optional limit and remainder constraints.

```C
unsigned int libxs_product_limit(unsigned int product,
  unsigned int limit, int is_lower);
```

Select prime factors of `product` such that the resulting sub-product stays within `limit` (0/1-Knapsack).

## Scalar Utilities

```C
double libxs_kahan_sum(double value, double* accumulator,
  double* compensation);
```

Kahan compensated summation: adds `value` to `*accumulator` while tracking rounding error in `*compensation`.

```C
unsigned int libxs_isqrt_u64(unsigned long long x);
unsigned int libxs_isqrt_u32(unsigned int x);
unsigned int libxs_isqrt2_u32(unsigned int x);
```

Integer square root (Newton's method). `libxs_isqrt2_u32` returns a result that is also a factor of `x`.

```C
double libxs_pow2(int n);
```

Compute 2^n by direct IEEE-754 exponent manipulation. Valid for n in [-1022, 1023]; returns 0 for underflow, +Inf for overflow.

## Modular Arithmetic

```C
unsigned int libxs_mod_inverse_u32(unsigned int a, unsigned int m);
```

Modular inverse via the extended Euclidean algorithm. Requires gcd(a, m) = 1.

```C
unsigned int libxs_barrett_rcp(unsigned int p);
unsigned int libxs_barrett_pow18(unsigned int p);
unsigned int libxs_barrett_pow36(unsigned int p);
```

Precompute Barrett reduction constants for modulus `p`.

```C
unsigned int libxs_mod_u32(uint32_t x, unsigned int p,
  unsigned int rcp);                           /* inline */
unsigned int libxs_mod_u64(uint64_t x, unsigned int p,
  unsigned int rcp, unsigned int pow18,
  unsigned int pow36);                         /* inline */
```

Fast modular reduction using precomputed Barrett constants. `libxs_mod_u64` uses a radix-2^18 split to avoid 64-bit division.

## BF16 Conversion

```C
typedef uint16_t libxs_bf16_t;
```

Storage type for BF16 values (1 sign + 8 exponent + 7 fraction bits).

```C
libxs_bf16_t libxs_round_bf16(double x);   /* inline */
double libxs_bf16_to_f64(libxs_bf16_t v);  /* inline */
```

Round a double to BF16 (round-to-nearest-even) and expand BF16 back to double (exact). When the compiler provides native `__bf16` support (`LIBXS_BF16` defined), the compiler's built-in conversions are used; otherwise a portable bit-manipulation fallback is used.
