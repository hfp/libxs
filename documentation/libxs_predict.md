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

Build the model from pushed entries. nclusters=0 selects
sqrt(n) clusters automatically. quality in [0,1] controls
truncation: 0=maximum compression, 1=maximum fidelity.
quality=-1 auto-optimizes via GSS (converges to precision).
quality=-N uses exactly N GSS iterations (N > 1).
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
1=nearest only, 0=auto. info (optional) receives per-output
error bounds and reliability flags.

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

Batch prediction. inputs_batch is count*M values (row-major),
outputs_batch receives count*N values. The task variant
distributes queries across threads by slice.

## Query and Access

```C
void libxs_predict_query(const libxs_predict_t* model,
  int* nclusters, int* nentries, double* compression);

void libxs_predict_get(const libxs_predict_t* model,
  int index, double inputs[], double outputs[]);
```

Query retrieves model statistics. Any output pointer may
be NULL. Get retrieves the i-th pushed entry (0-based);
inputs and outputs may independently be NULL.

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

## Info Structure

```C
typedef struct libxs_predict_info_t {
  const double* values;
  const double* error;
  const int* reliable;
  int noutputs;
  int cluster;
} libxs_predict_info_t;
```

Populated by libxs_predict_eval when info is non-NULL.
error[j] is the per-output truncation error bound.
reliable[j] is non-zero when the output uses polynomial
interpolation (decaying fingerprint). cluster is the
assigned cluster index (-1 if blended).
