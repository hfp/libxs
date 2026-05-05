# Histogram

Header: `libxs_hist.h`

Thread-safe histogram with running statistics. Buckets by `vals[0]`;
additional values per entry track user-defined statistics.

## Types

```C
typedef struct libxs_hist_t libxs_hist_t;  /* opaque */

typedef void (*libxs_hist_update_t)(double* dst, const double* src, int count);
```

```C
typedef struct libxs_hist_info_t {
  const int* buckets;   /* per-bucket counts [nbuckets] */
  const double* vals;   /* per-bucket values [nbuckets * nvals] */
  double range[2];      /* [min, max] of vals[0] */
  int nbuckets, nvals, nsamples;
} libxs_hist_info_t;
```

## Functions

```C
libxs_hist_t* libxs_hist_create(int nbuckets, int nvals,
  const libxs_hist_update_t update[]);
void libxs_hist_destroy(libxs_hist_t* hist);
```

Create/destroy. `update[nvals]` specifies per-slot accumulation
(NULL defaults to avg). Destroy accepts NULL.

```C
void libxs_hist_push(libxs_lock_t* lock,
  libxs_hist_t* hist, const double vals[]);
```

Insert one sample. Re-bins automatically if `vals[0]` exceeds range.

```C
void libxs_hist_query(libxs_lock_t* lock,
  const libxs_hist_t* hist, libxs_hist_info_t* info);
```

Query statistics. Lazy-commits queued items on first call.

```C
void libxs_hist_query_percentile(libxs_lock_t* lock,
  const libxs_hist_t* hist, double vals[], double percentile);
void libxs_hist_query_median(libxs_lock_t* lock,
  const libxs_hist_t* hist, double vals[]);
```

Interpolated values at percentile [0..1] or median (0.5).

```C
void libxs_hist_print(FILE* ostream, const libxs_hist_t* hist,
  const int prec[], const char fmt[], ...);
```

Print to stream. `prec[k]` controls precision; negative skips output.
NULL ostream/fmt accepted.

## Update Functions

```C
void libxs_hist_update_avg(double* dst, const double* src, int count);
void libxs_hist_update_add(double* dst, const double* src, int count);
void libxs_hist_update_min(double* dst, const double* src, int count);
void libxs_hist_update_max(double* dst, const double* src, int count);
```

- `avg` -- Welford online mean: `*dst += (*src - *dst) / count`
- `add` -- sum: `*dst += *src`
- `min` -- `*dst = min(*dst, *src)`
- `max` -- `*dst = max(*dst, *src)`

Custom callbacks use the same signature; `count` enables online
algorithms without external state.
