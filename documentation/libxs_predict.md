# Parameter Prediction

Header: `libxs_predict.h`

Predict output parameters from input parameters using
fingerprint-guided polynomial interpolation and kNN
classification. Supports incremental training, model
persistence, and thread-safe evaluation.

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

Set prediction mode for all outputs.
-1 = auto (use build-time fingerprint decision, default).
 0 = force interpolation for all outputs.
 1 = force classify (kNN majority vote) for all outputs.

The mode can also be overridden per eval call via negative
nblend (forces classify for that call only).

## Training

```C
int libxs_predict_push(libxs_lock_t* lock,
  libxs_predict_t* model,
  const double inputs[], const double outputs[]);
```

Push one training entry. Lock is optional (NULL if
single-threaded, or libxs_predict_lock(model) for
thread-safe access). Returns EXIT_SUCCESS or EXIT_FAILURE.

```C
int libxs_predict_build(libxs_predict_t* model,
  int nclusters, double quality);
```

Build the model from pushed entries. A value of nclusters=0
selects sqrt(n) clusters automatically. The quality parameter
controls truncation in [0,1]: 0=maximum compression,
1=maximum fidelity. A negative value auto-optimizes via GSS
(-1 converges to precision, -N uses exactly N iterations).
May be called again after pushing additional entries.

```C
int libxs_predict_build_task(libxs_lock_t* lock,
  libxs_predict_t* model, int nclusters, double quality,
  int tid, int ntasks);
```

Per-thread collective form. All threads call with same
model/nclusters/quality. tid=0 performs the build, others
spin-wait. Lock is optional (NULL is accepted).

## Evaluation

```C
void libxs_predict_eval(libxs_lock_t* lock,
  const libxs_predict_t* model,
  const double inputs[], double outputs[],
  libxs_predict_info_t* info, int nblend);
```

Predict outputs for given inputs. outputs may be NULL if
only info is needed. nblend controls multi-cluster blending:
1=nearest only, 0=auto. Negative nblend forces classify mode
(absolute value is used as blend count). info (optional)
receives per-output error bounds and mode flags.

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

Query fills a statistics struct with cluster count, entry
count, compression ratio, quality used, and GSS iterations.
Get retrieves the i-th pushed entry (0-based); inputs and
outputs may independently be NULL.

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

## CSV Import

```C
int libxs_predict_load_csv(libxs_predict_t* model,
  const char filename[], const char delims[],
  const char* inputs[], int ninputs,
  const char* outputs[], int noutputs);
```

Load delimited text and push entries. delims=NULL auto-detects
the separator. Each column identifier is matched
case-insensitively against the header line; if no match,
it is parsed as a numeric index (0-based). Rows with
non-numeric values at selected columns are skipped. Returns
the number of entries pushed, or -1 on I/O error.

## Structures

```C
typedef struct libxs_predict_info_t {
  const double* values;
  const double* error;
  const int* interpolated;
  int noutputs;
  int cluster;
} libxs_predict_info_t;
```

Populated by libxs_predict_eval when info is non-NULL.
The error array holds per-output truncation error bounds.
The interpolated array is non-zero for outputs where
polynomial interpolation was used. The cluster field
gives the assigned cluster index (-1 if blended).

```C
typedef struct libxs_predict_query_t {
  double compression;
  double quality;
  int nclusters;
  int nentries;
  int iterations;
} libxs_predict_query_t;
```

Populated by libxs_predict_query. The compression field
gives the ratio of raw data size to model size. The quality
field holds the value used (after auto-optimization if the
build parameter was negative). The iterations field reports
the number of GSS steps performed (0 if quality >= 0).
