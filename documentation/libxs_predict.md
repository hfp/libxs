# Parameter Prediction

<a href="../predict/" target="_blank">Start Presentation</a>

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
  LIBXS_PREDICT_TEMPORAL    = 4
} libxs_predict_mode_t;
```

ORable flags controlling prediction behavior:
- AUTO (0): fingerprint decides per-output (default).
- INTERPOLATE: force polynomial for all outputs.
- CLASSIFY: force kNN vote for all outputs.
- TEMPORAL: timeseries mode -- enables recency weighting
  (recent neighbors preferred), continuous output (no
  snap-to-nearest), and local coherence smoothing across
  horizon steps. Only effective when set_series was called
  (nseries > 0). For timeseries models without this flag,
  temporal heuristics auto-enable only when the query falls
  outside the training bounding box.

Example: `LIBXS_PREDICT_CLASSIFY | LIBXS_PREDICT_TEMPORAL`
forces kNN-weighted projection forward with temporal heuristics.

## Input Decomposition and Feature Selection

```C
typedef enum libxs_predict_decompose_t {
  LIBXS_PREDICT_RAW     = 0,
  LIBXS_PREDICT_SPREAD  = 1,
  LIBXS_PREDICT_PCA     = 2,
  LIBXS_PREDICT_SETDIFF = 3,
  LIBXS_PREDICT_FISHER  = 4,
  LIBXS_PREDICT_RF      = 5,
  LIBXS_PREDICT_HKNN    = 6
} libxs_predict_decompose_t;
```

Controls input processing and prediction strategy:

- RAW (0): no decomposition, standard kNN (default).
- SPREAD: sum/diff modes for anti-correlated pairs (2 series).
  Forward: sum = (A+B)/2, diff = (A-B)/2.
  Inverse: A = sum+diff, B = sum-diff.
  Only effective when nseries == 2.
- PCA: principal component rotation of the input space.
  At build time, computes eigenvectors of the input covariance
  matrix (Jacobi iteration), stores the rotation matrix, and
  transforms all entries into PC space. Weights are auto-set
  to zero out PCs below the 95% cumulative variance threshold
  (dimensionality reduction). Applied via libxs_gemm.
- SETDIFF: automatic feature selection using libxs_setdiff.
  At build time, scores each input dimension by class
  separability (per-class-pair setdiff with 5% tolerance,
  range-normalized). Zeroes weights for features below the
  median score. Supervised (uses output labels).
- FISHER: automatic feature selection using Fisher's
  discriminant ratio (between-class / within-class variance).
  At build time, computes per-feature Fisher score, applies
  sqrt weighting above median, zeroes below. Supervised.
  Generally best for kNN classification.
- RF: Random Forest classification. At build time, trains an
  ensemble of 100 decision trees per output (bootstrap
  sampling, sqrt(ninputs) random features per split, Gini
  impurity). At eval time, returns majority vote across
  trees. Per-output confidence = vote fraction. Excels on
  high-dimensional classification (37+ features). Persisted
  in save/load (compact on-disk format: 15 bytes/node).
- HKNN: hierarchical kNN. At build time, partitions data
  using a Fisher-guided kd-tree (output-aware dimension
  selection, adaptive balance penalty) followed by Lloyd
  refinement. Uses Gini impurity when a single categorical
  output is detected. A soft cap based on the Fisher/Gini
  score suppresses marginal splits near target cluster count.
  Inference uses standard kNN within each partition. Best for
  structured parameter prediction where output-aware spatial
  partitioning outperforms geometric k-means.

PCA/SPREAD decomposition is applied at build time to stored
entries and at eval time to user queries. Inverse prediction
returns inputs in raw (user) space. Feature selection modes
(SETDIFF, FISHER) only set weights at build time. RF replaces
the kNN eval path entirely. HKNN replaces only the clustering
step (kNN inference is unchanged).

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

```C
void libxs_predict_set_refine(libxs_predict_t* model,
  int iterations);
```

Set the number of forward-inverse-forward refinement
iterations applied during eval. Default (0): iterate
automatically when per-output confidence drops below 0.9.
When set to >0, always perform the given number of iterations
regardless of confidence.

```C
void libxs_predict_set_smooth(libxs_predict_t* model,
  double amount);
```

Control multi-cluster blending at eval time.
- amount<0: auto-determine blending radius from model structure.
- amount=0: no blending (default after manual assignment).
- amount>0: blend predictions from clusters within
  (1+amount)*nearest_distance radius. Only smooth-mode outputs
  are blended; categorical outputs remain unblended.

```C
void libxs_predict_set_consistency(libxs_predict_t* model,
  double amount);
```

Set round-trip consistency penalty (0..1). During the
refinement loop, the prediction is inverse-mapped back to
input space. If the reconstructed input differs from the
original query by more than the cluster diameter, the
prediction is geometrically inconsistent.

- amount=0 (default): inconsistency only skips refinement
  (no confidence change).
- amount>0: confidence is actively penalized proportional
  to the round-trip distance:
  `conf = quality + conf / (1 + amount * rt_dist / dmax)`
- amount=1: halves confidence above the quality floor when
  the round-trip distance equals the cluster diameter.

This catches predictions where the model contradicts itself:
the output maps back to a different input region than the
query came from.

```C
void libxs_predict_set_quantile(libxs_predict_t* model,
  double quantile);
```

Set quantile level for prediction intervals (0..0.5).
When enabled, `info->lower` and `info->upper` are filled
with the q-th and (1-q)-th distance-weighted quantiles of
the k nearest neighbors.

- quantile=0 (default): intervals not computed,
  `info->lower` and `info->upper` are NULL.
- quantile>0 (e.g. 0.1): 10th/90th percentile bounds,
  scaled by `1/confidence` so sparse regions produce wider
  intervals and dense regions stay close to the raw
  neighbor spread.

The interval captures "plausible range given the neighbors"
while confidence captures "should I trust this at all."
Narrow interval + high confidence = strong prediction.

## Timeseries Configuration

```C
void libxs_predict_set_series(libxs_predict_t* model,
  int nseries, int window);
void libxs_predict_set_target(libxs_predict_t* model,
  int target);
void libxs_predict_set_decompose(libxs_predict_t* model,
  int decompose);
void libxs_predict_set_diff(libxs_predict_t* model,
  int order);
```

Declare timeseries structure for automatic sliding-window
construction. The model's ninputs must equal nseries * window;
noutputs is the forecast horizon.

When set_series has been called, push(lock, model, values, NULL)
accumulates one timestep (nseries values per call). The build
step then constructs all valid sliding windows internally:
input = concatenated windows across all series, output = the
next horizon values of the target series.

set_target selects which series to predict (0-based, default 0).
set_decompose selects input processing mode (see above).

set_diff enables auto-differencing for non-stationary series:
- `set_diff(model, 0)`: auto-detect order from fingerprint decay.
- `set_diff(model, d)`: explicit order d (1 = linear detrend).
- Default is disabled (-1).

Single-series example (sunspots):
```C
model = libxs_predict_create(WINDOW, HORIZON);
libxs_predict_set_series(model, 1, WINDOW);
for (t = 0; t < n; ++t)
    libxs_predict_push(NULL, model, &series[t], NULL);
libxs_predict_build(model, 0, 2, 0);
```

Multi-series example (anti-correlated pair):
```C
model = libxs_predict_create(WINDOW * 2, HORIZON);
libxs_predict_set_series(model, 2, WINDOW);
libxs_predict_set_target(model, 0);
libxs_predict_set_decompose(model, LIBXS_PREDICT_SPREAD);
for (t = 0; t < n; ++t) {
    double vals[2] = {A[t], B[t]};
    libxs_predict_push(NULL, model, vals, NULL);
}
libxs_predict_build(model, 0, 2, 0);
```

Non-stationary example (trending stock prices):
```C
model = libxs_predict_create(WINDOW * 2, HORIZON);
libxs_predict_set_mode(model, LIBXS_PREDICT_TEMPORAL);
libxs_predict_set_series(model, 2, WINDOW);
libxs_predict_set_target(model, 0);
libxs_predict_set_decompose(model, LIBXS_PREDICT_SPREAD);
libxs_predict_set_diff(model, 0);
for (t = 0; t < n; ++t) {
    double vals[2] = {price_A[t], price_B[t]};
    libxs_predict_push(NULL, model, vals, NULL);
}
libxs_predict_build(model, 0, 2, 0);
```

## Training

```C
int libxs_predict_push(libxs_lock_t* lock,
  libxs_predict_t* model,
  const double inputs[], const double outputs[]);
```

Push one training entry. When set_series was called and
outputs is NULL, inputs holds nseries values representing
one timestep; sliding windows are constructed at build time.
Returns EXIT_SUCCESS or EXIT_FAILURE.

```C
int libxs_predict_build(libxs_predict_t* model,
  int nclusters, int order, double quality);
```

Build the model from pushed entries. nclusters=0 selects
sqrt(n) clusters automatically. order>0 uses at most that
order, order=0 auto-optimizes, order<0 scans up to |order|.

The quality parameter (0..1) controls model compression and
confidence scaling:
- quality=0: no compression, no coverage scaling (default).
- quality>0: leave-one-out pass removes entries that are
  perfectly predictable by their neighbors. Additionally,
  confidence is scaled by coverage at eval time:
  `conf = quality + coverage * (raw_conf - quality)`
  where coverage is the product of two factors:
  - inter-cluster: min(cluster_entries / expected_entries, 1).
    Penalizes underpopulated clusters.
  - intra-cluster: 1/(1 + query_dist/cluster_dmax).
    Penalizes queries far from the cluster center.
  The quality value acts as a confidence floor -- predictions
  in sparse regions approach quality but never go below it.
  The quality is persisted in save/load.

```C
int libxs_predict_build_task(libxs_lock_t* lock,
  libxs_predict_t* model, int nclusters, int order,
  double quality, int tid, int ntasks);
```

Per-thread collective form. All threads call with same
parameters. tid=0 performs the build, others spin-wait.

## Evaluation

```C
void libxs_predict_eval(libxs_lock_t* lock,
  const libxs_predict_t* model,
  const double inputs[], double outputs[],
  libxs_predict_info_t* info, int nblend);
```

Predict outputs for given inputs. nblend controls
multi-cluster blending: 1=nearest only, 0=auto.
The info pointer (optional) receives per-output confidence,
variance, error bounds, and mode flags.

```C
void libxs_predict_inverse(libxs_lock_t* lock,
  const libxs_predict_t* model,
  const double target_outputs[], double inputs[],
  libxs_predict_info_t* info);
```

Inverse prediction: find inputs that produce desired outputs.

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

Batch prediction. The task variant distributes queries
across threads by slice.

## Query and Access

```C
void libxs_predict_query(const libxs_predict_t* model,
  libxs_predict_query_t* info);

void libxs_predict_get(const libxs_predict_t* model,
  int index, double inputs[], double outputs[]);
```

query fills a statistics struct (cluster count, entry count,
compression ratio, polynomial order, diff_order).
get retrieves the i-th pushed entry (0-based).

## Persistence

```C
int libxs_predict_save(const libxs_predict_t* model,
  void* buffer, size_t* size);
libxs_predict_t* libxs_predict_load(
  const void* buffer, size_t size);
```

Save: pass buffer=NULL to query required size, then allocate
and call again. Load: returns a ready-to-eval model or NULL.

## CSV Import

```C
int libxs_predict_load_csv(libxs_predict_t* model,
  const char filename[], const char delims[],
  const char* inputs[], int ninputs,
  const char* outputs[], int noutputs);
```

Load delimited text and push entries. delims=NULL auto-detects
separator. Returns number of entries pushed, or -1 on error.

## Structures

```C
typedef struct libxs_predict_info_t {
  const double* values;
  const double* error;
  const double* confidence;
  const double* variance;
  const double* lower;
  const double* upper;
  const int* interpolated;
  int noutputs;
  int cluster;
  double distance;
} libxs_predict_info_t;
```

Populated by eval when info is non-NULL. confidence holds
per-output confidence (0..1), incorporating:
- variety: kNN vote agreement (classify) or 1.0 (interpolation),
- coverage: cluster population and query centrality (when
  quality>0, see libxs_predict_build),
- consistency: round-trip penalty (when set_consistency>0).
lower/upper hold per-output prediction intervals (NULL when
set_quantile is 0, i.e. disabled). When enabled, bounds are
the distance-weighted quantiles from k neighbors scaled by
1/confidence. variance holds per-output neighbor disagreement.
cluster gives the assigned cluster index (-1 if blended).
distance gives normalized distance to nearest centroid.

```C
typedef struct libxs_predict_query_t {
  double compression;
  int order;
  int nclusters;
  int nentries;
  int iterations;
  int diff_order;
} libxs_predict_query_t;
```

Populated by query. compression = raw/model size ratio.
order = polynomial order used. diff_order = differencing
order (0 if disabled).
