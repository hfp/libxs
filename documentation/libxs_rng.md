# Random Number Generator

Header: `libxs_rng.h`

Thread-safe pseudo-random number generation (SplitMix64 with per-thread TLS state).

## Matrix Initialization Macros

```C
LIBXS_MATRNG(INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
LIBXS_MATRNG_SEQ(INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
LIBXS_MATRNG_OMP(INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
```

Fill a column-major matrix (`NROWS` x `NCOLS`, leading dimension `LD`) with deterministic values. When `SEED != 0` the fill is linear; when `SEED == 0` a shuffle-based initialization normalizes values into [-SCALE, +SCALE]. The `_OMP` variant parallelizes with OpenMP; `_SEQ` is serial.

## Functions

```C
void libxs_rng_set_seed(unsigned int seed);
```

Set the seed of the calling thread's PRNG state. Each thread maintains independent state via TLS. Unseeded threads start with a deterministic default (seed = 1).

```C
unsigned int libxs_rng_u32(unsigned int n);
```

Return a pseudo-random value in [0, n) with uniform distribution (Lemire's nearly-divisionless method). Thread-safe.

```C
double libxs_rng_f64(void);
```

Return a double-precision value in [0, 1) with full 53-bit mantissa resolution. Thread-safe.

```C
void libxs_rng_seq(void* data, size_t nbytes);
```

Fill a buffer with pseudo-random bytes. Thread-safe.
