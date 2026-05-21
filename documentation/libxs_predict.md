# Parameter Prediction

Header: `libxs_predict.h`

Predict output parameters from input parameters using
fingerprint-guided polynomial interpolation and kNN
classification. Supports incremental training, model
persistence, confidence-gated evaluation, output transforms,
and thread-safe operation.

## Mode Flags

```C
typedef enum libxs_predict_mode_t {
  LIBXS_PREDICT_AUTO        = 0,
  LIBXS_PREDICT_INTERPOLATE = 1,
  LIBXS_PREDICT_CLASSIFY    = 2,
  LIBXS_PREDICT_EXTRAPOLATE = 4
} libxs_predict_mode_t;
```

ORable flags controlling prediction behavior:
- AUTO (0): fingerprint decides per-output (default).
- INTERPOLATE: force polynomial for all outputs.
- CLASSIFY: force kNN vote for all outputs.
- EXTRAPOLATE: timeseries mode -- enables recency weighting,
  adaptive multi-cluster blending on low confidence,
  fingerprint-based cluster weighting, local coherence
  smoothing, and continuous output (no snap-to-nearest).

Example: `LIBXS_PREDICT_CLASSIFY | LIBXS_PREDICT_EXTRAPOLATE`
forces kNN-weighted projection forward.

## Output Transforms

```C
typedef enum libxs_predict_transform_t {
  LIBXS_PREDICT_IDENTITY = 0,
  LIBXS_PREDICT_LOG      = 1,
  LIBXS_PREDICT_SQRT     = 2
} libxs_predict_transform_t;
```

Per-output transforms applied transparently:
- IDENTITY (0): no transform (default).
- LOG: forward = log(x+1), inverse = exp(x)-1.
- SQRT: forward = sqrt(x), inverse = x*x.

The forward transform is applied during push, the inverse
during eval. The fingerprint and kNN operate in transformed
space where heavy-tailed data is smoother.

## Create and Destroy

```C
libxs_predict_t* libxs_predict_create(int ninputs, int noutputs);
void libxs_predict_destroy(libxs_predict_t* model);
libxs_lock_t* libxs_predict_lock(libxs_predict_t* model);
```

Create a model for the given input/output dimensionality.
Returns NULL on invalid arguments. Destroy accepts NULL.
The lock accessor returns a pointer to the model's internal
lock for use with push/eval (NULL if model is NULL).

## Configuration

```C
void libxs_predict_set_mode(libxs_predict_t* model, int mode);
```

Set prediction mode (ORable flags from libxs_predict_mode_t).
Default is LIBXS_PREDICT_AUTO.

```C
void libxs_predict_set_weights(libxs_predict_t* model,
  const double weights[]);
```

Set per-dimension input weights for distance computation.
Larger weight means the dimension contributes more.
Pass NULL to reset to uniform weighting.
Must be called before libxs_predict_build.

```C
void libxs_predict_set_transform(libxs_predict_t* model,
  int output, int transform);
```

Set per-output transform. The output parameter is 0-based,
or -1 to set all outputs. Must be called before pushing
entries. Transforms are persisted in save/load.

## Training

```C
int libxs_predict_push(libxs_lock_t* lock,
  libxs_predict_t* model,
  const double inputs[], const double outputs[]);
```

Push one training entry. Output values are transformed
internally if a transform was set. The lock is optional
(NULL if single-threaded). Returns EXIT_SUCCESS or
EXIT_FAILURE.

```C
int libxs_predict_build(libxs_predict_t* model,
  int nclusters, int order);
```

Build the model from pushed entries. A value of nclusters=0
selects sqrt(n) clusters automatically. The order parameter
controls maximum polynomial order for interpolation:
order > 0 uses at most that order, order = 0 auto-optimizes
via exhaustive scan over [1..MAXORDER], order < 0 scans up
to |order|. May be called again after pushing more entries
(normalization is recomputed from all entries).

```C
int libxs_predict_build_task(libxs_lock_t* lock,
  libxs_predict_t* model, int nclusters, int order,
  int tid, int ntasks);
```

Per-thread collective form. All threads call with same
model/nclusters/order. tid=0 performs the build, others
spin-wait. The lock is optional (NULL is accepted).

## Evaluation

```C
void libxs_predict_eval(libxs_lock_t* lock,
  const libxs_predict_t* model,
  const double inputs[], double outputs[],
  libxs_predict_info_t* info, int nblend);
```

Predict outputs for given inputs. Output values are
inverse-transformed if a transform was set (caller receives
original-scale values). The nblend parameter controls
multi-cluster blending: 1=nearest only, 0=auto. The info
pointer (optional) receives per-output confidence, error
bounds, and mode flags.

```C
void libxs_predict_eval_batch(
  const libxs_predict_t* model,
  const double inputs_batch[], double outputs_batch[],
  int count, int nblend);

void libxs_predict_eval_batch_task(
  const libxs_predict_t* model,
  const double inputs_batch[], double outputs_batch[],
  int count, int nblend, int tid, int ntasks);
```

Batch prediction. The input array holds count*M values
(row-major), the output array receives count*N values.
The task variant distributes queries across threads by slice.

## Query and Access

```C
void libxs_predict_query(const libxs_predict_t* model,
  libxs_predict_query_t* info);

void libxs_predict_get(const libxs_predict_t* model,
  int index, double inputs[], double outputs[]);
```

The query function fills a statistics struct with cluster
count, entry count, compression ratio, polynomial order used,
and order-scan iterations. The get function retrieves the
i-th pushed entry (0-based); inputs and outputs may
independently be NULL.

## Persistence

```C
int libxs_predict_save(const libxs_predict_t* model,
  void* buffer, size_t* size);
libxs_predict_t* libxs_predict_load(
  const void* buffer, size_t size);
```

Save: pass buffer=NULL to query required size, then allocate
and call again. Load: returns a ready-to-eval model or NULL.
The loaded model does not reference the source buffer.
Weights and transforms are persisted in the binary format.

## CSV Import

```C
int libxs_predict_load_csv(libxs_predict_t* model,
  const char filename[], const char delims[],
  const char* inputs[], int ninputs,
  const char* outputs[], int noutputs);
```

Load delimited text and push entries. Setting delims to NULL
auto-detects the separator. Lines starting with '#' are
skipped as comments. Each column identifier is matched
case-insensitively against the header line; if no match, it
is parsed as a numeric index (0-based). Column names not found
in the header are assigned to trailing unlabeled data columns
in order. Rows with non-numeric values at selected columns
are skipped. Returns the number of entries pushed, or -1 on
I/O error.

## Structures

```C
typedef struct libxs_predict_info_t {
  const double* values;
  const double* error;
  const double* confidence;
  const int* interpolated;
  int noutputs;
  int cluster;
  double distance;
} libxs_predict_info_t;
```

Populated by libxs_predict_eval when info is non-NULL.
The error array holds per-output truncation error bounds.
The confidence array holds per-output kNN vote fraction
(0..1): the weighted agreement among k nearest neighbors.
The interpolated array is non-zero for outputs where
polynomial interpolation was used. The cluster field gives
the assigned cluster index (-1 if blended). The distance
field gives the normalized distance to the nearest cluster
centroid relative to its radius (0 = at centroid, >1 =
outside training region).

```C
typedef struct libxs_predict_query_t {
  double compression;
  int order;
  int nclusters;
  int nentries;
  int iterations;
} libxs_predict_query_t;
```

Populated by libxs_predict_query. The compression field
gives the ratio of raw data size to model size. The order
field holds the polynomial order used (after auto-optimization
if the build parameter was <= 0). The iterations field reports
the number of orders scanned (0 if order > 0 was given
directly).
