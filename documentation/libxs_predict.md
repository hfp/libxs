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
- TEMPORAL: timeseries mode — enables recency weighting
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
  LIBXS_PREDICT_AUTO_DECOMPOSE = -1,
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

- AUTO_DECOMPOSE (-1): choose among the applicable modes at
  build by trial (each is built on part of the corpus and scored
  on a held-back part). Read the chosen mode from
  `libxs_predict_query_t::decompose`.
- RAW (0): no decomposition, standard kNN (default).
- SPREAD: sum/diff modes for anti-correlated pairs (2 series).
  Forward: sum = (A+B)/2, diff = (A-B)/2.
  Inverse: A = sum+diff, B = sum-diff.
  Only effective when nseries == 2.
- PCA: principal component rotation of the input space with
  dimensionality reduction.
- SETDIFF: supervised feature selection by class separability
  (libxs_setdiff).
- FISHER: supervised feature weighting by Fisher's
  discriminant ratio.
- RF: Random Forest per output, 64 trees by default (see
  `libxs_predict_set_forest`). An integral output spanning
  at most 128 levels is classified, any other output is
  regressed. Persisted in save/load.
- HKNN: hierarchical kNN. An output-aware partition replaces
  the clustering step; inference is kNN within each partition.

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
Returns NULL on invalid arguments or allocation failure.
Destroy accepts NULL.
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
iterations applied during eval.

- iterations=0 (default): off, except for a single pass when
  set_consistency>0 and a per-output confidence is below 0.9.
- iterations<0: a single pass when a per-output confidence is
  below 0.9.
- iterations>0: always perform that many iterations
  regardless of confidence.

```C
void libxs_predict_set_smooth(libxs_predict_t* model,
  double amount);
```

Control multi-cluster blending at eval time.

- amount<0: auto-determine blending radius from model structure.
- amount=0 (default): no blending.
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
  `conf = floor + (conf - floor) / (1 + amount * rt_dist / dmax)`
  where floor is set by `libxs_predict_set_floor` (default 0).
- amount=1: halves confidence above the floor when
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

```C
void libxs_predict_set_central(libxs_predict_t* model, int mode);
void libxs_predict_set_neighbors(libxs_predict_t* model, int k);
void libxs_predict_set_reach(libxs_predict_t* model, int nclusters);
void libxs_predict_set_coordinate(libxs_predict_t* model,
  int coordinate);
void libxs_predict_set_floor(libxs_predict_t* model, double floor);
void libxs_predict_set_missing(libxs_predict_t* model, int enable);
void libxs_predict_set_forest(libxs_predict_t* model,
  int ntrees, int max_depth);
```

| Setter        | Values                                                                 |
| ------------- | ---------------------------------------------------------------------- |
| `set_central` | Vote of a many-valued output: 1 median, 2 mean, 0 (default) chosen per output at build; negative also prints the choice. |
| `set_neighbors` | Neighbors the vote reads: k>0 pins it, 0 (default) derives it, k<0 selects it per output at build by trial. Persisted. |
| `set_reach`   | Further clusters a flat vote gathers from, scanned by the nearest any member can be and stopped once none can hold one of the k nearest: n>0 at most n, n<0 decided at build by trial (with or without at most \|n\|), 0 (default) that trial at 4, `LIBXS_PREDICT_REACH_NONE` none. Set before build; persisted. |
| `set_coordinate` | Coordinate an input contributes to the distance (`libxs_predict_coord_t`, default `LIBXS_PREDICT_AUTO_COORD`). |
| `set_floor`   | Floor (0..1) the confidence rescalings pull toward; 0 (default) disables them. Persisted. |
| `set_missing` | Non-zero: `load_csv` admits rows with an empty or unparsable input field as absent (NaN) instead of skipping them (default 0). Call before push. |
| `set_forest`  | `LIBXS_PREDICT_RF` trees per output (ntrees>0, default 64) and depth (max_depth>0 fixed, 0 (default) derived from the entry count, <0 selected per output at build). Call before build; not persisted. |

```C
typedef enum libxs_predict_coord_t {
  LIBXS_PREDICT_AUTO_COORD = 0,
  LIBXS_PREDICT_RANK       = 1,
  LIBXS_PREDICT_SCORE      = 2
} libxs_predict_coord_t;
```

- AUTO_COORD (0): RANK or SCORE chosen at build by trial, from 8 inputs on;
  with fewer inputs, RANK.
- RANK: an input is placed by its rank among the stored entries.
- SCORE: an input is placed by what its value says about the output. Needs one
  discrete output and no series, and applies to RAW and HKNN only. A loaded
  model built this way reports no corpus.

Absent inputs (NaN, pushed directly or admitted by `set_missing`) are omitted
from the distance. Interpolation abstains for a query with an absent input, and
build fails under PCA, SPREAD, RF or HKNN when an absence is present.

## Timeseries Configuration

```C
#define LIBXS_PREDICT_AUTO_WINDOW 0

void libxs_predict_set_series(libxs_predict_t* model,
  int nseries, int window);
void libxs_predict_set_series_deriv(libxs_predict_t* model,
  int nderiv);
void libxs_predict_set_series_aux(libxs_predict_t* model,
  int naux);
void libxs_predict_set_series_bank(libxs_predict_t* model,
  int nbank);
void libxs_predict_set_target(libxs_predict_t* model,
  int target);
void libxs_predict_set_decompose(libxs_predict_t* model,
  int decompose);
void libxs_predict_set_diff(libxs_predict_t* model,
  int order);
```

Declare timeseries structure for automatic sliding-window
construction. For a positive window, the model's ninputs must
equal nseries * window + nderiv + naux (see
`libxs_predict_set_series_deriv` and `libxs_predict_set_series_aux`);
noutputs is the forecast horizon. With window =
`LIBXS_PREDICT_AUTO_WINDOW` (0), the window is selected at build,
and a negative window selects it by trial; in both cases ninputs
is an upper bound and `libxs_predict_query_t::window` reports the
window in force. A negative window builds candidate windows and
keeps the one with the least held-back error over the horizon; its
magnitude, when at least 4, is one more candidate. This costs
several builds (see `LIBXS_PREDICT_WINDOW_FOLDS`).

- set_series_deriv: append nderiv terminal first-differences of
  the target window as inputs (default 0). Call before push.
- set_series_aux: naux exogenous per-timestep features (e.g.
  day-of-year), pushed after the series values and neither windowed
  nor transformed (default 0). Call before push.
- set_series_bank: average the forecast over nbank views of the
  window, each halving the lags of the one before (default 1 =
  off). Call before build; `libxs_predict_query_t::nbank` reports
  the views built. `libxs_predict_inverse` abstains on a loaded
  model with views.

When set_series has been called, push(lock, model, values, NULL)
accumulates one timestep (nseries + naux values per call: the
series first, then the auxiliary features). The build
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
outputs is NULL, inputs holds nseries + naux values representing
one timestep; sliding windows are constructed at build time.
At eval, such a model takes the raw window followed by the naux
auxiliary values (nseries * window + naux values).
Returns EXIT_SUCCESS or EXIT_FAILURE.

```C
int libxs_predict_push_weighted(libxs_lock_t* lock,
  libxs_predict_t* model, const double inputs[], const double outputs[],
  double weight);
```

Push one entry whose kNN vote counts weight times (weight > 0;
1 is exactly push). A forest draws its bootstraps by weight
(rounded to whole units) and counts its calibration by weight.
The polynomial fit, the partition and the probability API do
not read it. In series mode only weight 1 is accepted. Weights
are persisted.

```C
int libxs_predict_build(libxs_predict_t* model,
  int nclusters, int order, double quality);
```

Build the model from pushed entries. nclusters=0 selects
sqrt(n) clusters automatically. order>0 uses at most that
order, order=0 auto-optimizes, order<0 scans up to |order|.
Entries with equal inputs stay together in every hold-out the
build draws (mode trial, forest bags, boosting, calibration).

The quality parameter (0..1) controls model compression:

- quality=0: no compression (default).
- quality>0: a leave-one-out pass removes the entries that
  every output recovers exactly from a unanimous
  neighborhood. For a classify output every positive value
  yields the same drop set; an interpolating output reads the
  magnitude as a residual tolerance. The quality is persisted
  in save/load.

Quality does not affect confidence. With
`libxs_predict_set_floor` > 0, confidence is discounted for
underpopulated clusters and for queries far from the cluster
center, approaching the floor but never going below it. The
floor is persisted in save/load.

```C
int libxs_predict_build_task(libxs_predict_t* model,
  int nclusters, int order, double quality, int tid, int ntasks);
```

Per-thread collective form. All tasks call with the same parameters, and every
task must enter: a task that does not enter stops the build. The model owns the
barrier, so no lock or barrier is passed. Two models may be built by two teams at
once; the same model may not be built concurrently.

## Evaluation

```C
void libxs_predict_eval(libxs_lock_t* lock,
  const libxs_predict_t* model,
  const double inputs[], double outputs[],
  libxs_predict_info_t* info, int nblend);
```

Predict outputs for given inputs. nblend controls
multi-cluster blending: nblend>=2 averages that many clusters,
0 and 1 are automatic (the nearest cluster answers unless its
confidence calls for more evidence), and nblend<0 pins the
answer to the nearest cluster.
The info pointer (optional) receives per-output confidence,
variance, error bounds, and mode flags.

```C
int libxs_predict_shape(const libxs_predict_t* model,
  int* ninputs, int* noutputs);
void libxs_predict_eval_flat(const libxs_predict_t* model,
  const double inputs[], double outputs[], double confidence[],
  double variance[], double lower[], double upper[]);
```

shape reports the model's ninputs and noutputs (either pointer
may be NULL; EXIT_FAILURE for a NULL model). eval_flat is eval
with the per-output fields copied into caller arrays of noutputs
each, every one optional (NULL skips it): the form for a
foreign-function caller (ctypes, Fortran) that should not mirror
`libxs_predict_info_t`. Without a quantile level, lower and upper
receive the prediction.

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

double libxs_predict_get_weight(const libxs_predict_t* model,
  int index);
```

query fills a statistics struct (see `libxs_predict_query_t`
below).
get retrieves the i-th pushed entry (0-based), and get_weight
its weight (1 unless set; 0 if there is no such entry).

`nscan` is the number of candidates a query scans in the worst
case (the largest cluster); a model built with a single cluster
scans every entry per query.

## Probability

```C
void* libxs_predict_prob_create(const libxs_predict_t* model);
void libxs_predict_prob_destroy(void* context);
int libxs_predict_prob_commit(libxs_predict_t* model,
  const void* context);

void libxs_predict_prob(libxs_lock_t* lock,
  const libxs_predict_t* model, void* context,
  const double inputs[], const double candidate[], double prob[],
  libxs_predict_prob_info_t* info, int vocabulary, int nblend);
int libxs_predict_prob_observe(libxs_lock_t* lock,
  const libxs_predict_t* model, void* context,
  const double inputs[], int output, const double* candidate,
  double values[], double probs[], int capacity, double* novel,
  libxs_predict_prob_info_t* info, int vocabulary, int nblend);
int libxs_predict_prob_support(const libxs_predict_t* model,
  int output, double values[], int capacity);
```

These score a value the caller supplies, including one the model
would never pick. `prob` is a point query for P(y|x);
`prob_observe` reports the distribution over one discrete output
and then optionally observes the outcome.

- `context != NULL` — adaptive; the context's weights adapt over
  the stream and the model is not modified.
- `context == NULL` — frozen; the model's stored weights are used
  and never written, so results do not depend on call order.

`prob_commit` copies a context's weights into the model. Call it
before frozen scoring; without it, frozen scoring uses a uniform
prior and scores poorly.

`prob_support` returns the attested support of a discrete output
(distinct values, sorted ascending), writing up to `capacity`
entries and nothing at all when the support is larger.

## Calibration

```C
int libxs_predict_recalibrate_prob(libxs_predict_t* model,
  const double* inputs, const double* outputs, int nentries);
int libxs_predict_recalibrate_interval(libxs_predict_t* model,
  const double* inputs, const double* outputs, int nentries);
int libxs_predict_interval(const libxs_predict_t* model, int output,
  double lower, double upper, double* lower_out, double* upper_out);
int libxs_predict_probability(const libxs_predict_t* model,
  int output, double confidence, double* probability);
```

A confidence ranks queries; it is not a probability. A forest
carries a calibration curve fitted at build.

- probability: translates a confidence from eval into the
  probability that the reported RF class is correct. Returns
  EXIT_SUCCESS when a curve was applied, EXIT_FAILURE when the
  confidence passed through unchanged (still a ranking).
- recalibrate_prob: replaces a forest's curve with one fitted on
  rows the model was not built from (inputs: nentries * ninputs,
  outputs: nentries * noutputs, laid out as push takes them), e.g.
  after the deployment distribution changed. Fails for a model
  without a forest.
- recalibrate_interval: fits a correction of the prediction
  interval on such held-out rows. Requires a model built with
  `set_quantile` > 0; applies to any model kind.
- interval: applies that correction to an interval from eval.
  lower_out/upper_out always receive a value; EXIT_SUCCESS when
  corrected, EXIT_FAILURE when passed through unchanged.

## Persistence

```C
int libxs_predict_save(const libxs_predict_t* model,
  void* buffer, size_t* size);
libxs_predict_t* libxs_predict_load(
  const void* buffer, size_t size);
```

Save: pass buffer=NULL to query required size, then allocate
and call again. Load: returns a ready-to-eval model or NULL.

Load accepts every released format version (currently 1 and 2)
and defaults fields an older file does not carry. A v1.0.0 flat
model stores no global entry order, so it abstains from
`libxs_predict_inverse` and from recency weighting, also after
saving it again. Version 2 adds a CRC32 trailer, so a truncated
or corrupted file is rejected.

## CSV Import

```C
int libxs_predict_load_csv(libxs_predict_t* model,
  const char filename[], const char delims[],
  const char inputs[], const char outputs[],
  char header[], int header_size, char* delim_out);
```

Load delimited text and push entries. delims=NULL auto-detects
separator. inputs and outputs are comma-separated column names
(matched case-insensitively against the header line) or 0-based
indices, as many as the model's ninputs and noutputs; NULL
selects sequential columns. header (optional) receives the
header line, delim_out (optional) the delimiter in use. Returns
number of entries pushed, or -1 on error.

```C
int libxs_predict_load_csv_opts(libxs_predict_t* model,
  const char filename[], const libxs_predict_csv_t* opts);

typedef struct libxs_predict_csv_t {
  const char* delims;
  const char* inputs;
  const char* outputs;
  char* header;
  int header_size;
  char* delim_out;
  int nrows;
  int stride;
  int offset;
  int unique;
} libxs_predict_csv_t;
```

load_csv with the arguments in a struct, for loading a subset of
a large file. A zero-initialized struct (or opts=NULL) loads like
load_csv with every argument NULL or zero.

- nrows: stop after that many entries were pushed (0: whole file).
- stride: push every stride-th admissible row (0 or 1: every row).
- offset: admissible rows to skip before the first one taken.
- unique: 1 drops a row whose inputs and outputs equal an earlier
  row's (−0 equals 0, an absent input equals an absent one); 2
  adds one to the weight of the entry it repeats instead (a series
  model drops it). Rows that share only the inputs are kept.

With stride and offset, one file splits into disjoint parts that
each span the whole file. Unparsable and comment rows are not
admissible and do not shift the selection, and neither is a
repeat under unique, so no two parts hold the same row (under 2,
the part holding the first occurrence holds every copy).

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
  the floor is >0, see libxs_predict_build),
- consistency: round-trip penalty (when set_consistency>0).
lower/upper hold per-output prediction intervals (NULL when
set_quantile is 0, i.e. disabled). When enabled, bounds are
the distance-weighted quantiles from k neighbors scaled by
1/confidence. variance holds per-output neighbor disagreement.
cluster gives the assigned cluster index (-1 if blended).
distance gives normalized distance to nearest centroid.
Confidence is a ranking; see Calibration for a probability.

```C
typedef enum libxs_predict_pkind_t {
  LIBXS_PREDICT_PNONE = 0,
  LIBXS_PREDICT_PMASS = 1,
  LIBXS_PREDICT_PDENSITY = 2
} libxs_predict_pkind_t;

typedef struct libxs_predict_prob_info_t {
  const double* prob;
  const double* logprob;
  const double* zscore;
  const int* kind;
  const int* attested;
  const int* support;
  const double* novel;
  double total_logprob;
  int noutputs;
  int cluster;
  double entropy;
} libxs_predict_prob_info_t;
```

Populated by `libxs_predict_prob` and `libxs_predict_prob_observe`,
per output where not noted otherwise:

- prob: P(y|x), a mass or a density according to kind
  (PMASS: discrete output, PDENSITY: continuous output, PNONE: no
  result, e.g. a blended or RF output).
- logprob: base-2 log of prob; read it rather than prob, which
  underflows to zero for an improbable candidate.
- zscore: (y - yhat)/h in density mode, 0 in mass mode.
- attested: non-zero if the candidate is in the local
  neighborhood. support: distinct values in the local support;
  novel: mass reserved for values outside it (mass mode).
- total_logprob: sum of logprob over outputs, assuming the outputs
  are conditionally independent (not a calibrated joint
  likelihood).
- cluster: cluster serving the query (-1 if blended).
- entropy: entropy in bits of the scoring weights, 0 once
  committed and log2(nexperts) at the uniform prior; tells whether
  the weights are still settling.

```C
typedef struct libxs_predict_query_t {
  double compression;
  int ninputs;
  int noutputs;
  int version;    /* serialization version (current if built here) */
  int neighbors;  /* smallest resolved neighbor count, 0 if none */
  int order;
  int nclusters;
  int nentries;
  int corpus;     /* 0 if the entries cannot be rebuilt from */
  int iterations;
  int diff_order;
  int window;     /* effective sliding window (0 if series mode off) */
  int nbank;      /* window views built (1 when disabled) */
  int nscan;      /* entries in the LARGEST cluster */
  double escan;   /* entries an average query scans */
  int decompose;  /* decomposition mode in force */
  int coordinate; /* RANK, SCORE, or 0 if only scaled */
  int ncoord;     /* coordinates per stored point */
} libxs_predict_query_t;
```

Populated by query. compression = raw/model size ratio.
ninputs/noutputs = the model's shape, which a caller evaluating
into its own buffer sizes it from. order = polynomial order used.
diff_order = differencing order (0 if disabled).

## Environment Variables

All are read at build and default to the behavior without them.

| Variable                        | Effect                                                              |
| ------------------------------- | ------------------------------------------------------------------- |
| `LIBXS_PREDICT_WINDOW_FOLDS`    | Held-back splits scored per candidate window when `set_series` is given a negative window (default 3, at least 1). |
| `LIBXS_PREDICT_DECOMPOSE_FOLDS` | Folds scored per candidate when the decomposition or the coordinate is chosen by trial (default 3 for a series, 1 otherwise; 0 or less keeps the default). |
| `LIBXS_PREDICT_REFINE`          | Overrides `libxs_predict_set_refine` (same values). |
| `LIBXS_PREDICT_REACH`           | Overrides `libxs_predict_set_reach` at build (same values, `none` for `LIBXS_PREDICT_REACH_NONE`). |
| `LIBXS_PREDICT_RF_FRONTIER`     | N > 0 builds each forest tree in up to N (at most 64) frontier parts; the trees are the same. Off by default. |
| `LIBXS_PREDICT_RF_RATE`         | Shrinkage of the forest's boosted corrections (default 0.1, applied to regression outputs). Setting it also enables corrections for classification outputs; 0 or less disables them. |
