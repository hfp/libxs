# Predict Sample

Train a prediction model from a CSV file and save it for later use.
Finds the optimal training fraction and polynomial order automatically.
Reports validation quality on a held-out subset.


## Build

    make

Or from the LIBXS root:

    make GNU=1 samples/predict


## Usage

    ./predict.x [fraction] [auto|cat|interp] [-N] <csvfile> [modelfile]

    fraction   Validation split 0..1 for quality report (default: 0.8).
               The full model always trains on all entries.
    auto       Auto-detect mode per output (default).
    cat        Force categorical (kNN) for all outputs.
    interp     Force interpolation for all outputs.
    -N         Max polynomial order for final build (default: 0 = auto).
    csvfile    Delimited text file (semicolons, commas, or tabs).
               The first line may be a header (auto-skipped if non-numeric).
    modelfile  Output path for the binary model.
               Default: derived from CSV basename (e.g., data.csv -> data.bin).


## Example

    ./predict.x ../../samples/smm/params/tune_multiply_PVC.csv

    Loaded 1339 entries from ...tune_multiply_PVC.csv
    Optimal fraction: 1.00 (1338/1339 entries, unimodality=1.00)
    Built: 37 clusters, 14.9x compression, order=2 (8 iter)
    Validation (1339 samples):
      param   avg-err   max-err  avg-bound
      BS     6.74e-01  5.70e+01   0.00e+00
      BM     8.35e-01  3.40e+01   0.00e+00
      ...
    Saved model to tune_multiply_PVC.bin (212132 bytes)

Parameters marked with * use polynomial interpolation; others use
kNN majority vote (auto-detected from fingerprint decay order).


## How It Works

1. libxs_predict_load_csv loads the file using column names ("M", "N",
   "K" as inputs; "BS", "BM", etc. as outputs). Names are matched
   case-insensitively against the CSV header.

2. libxs_shuffle produces a deterministic permutation of entries.
   libxs_gss_min finds the training fraction that minimizes total
   prediction error across all samples. The unimodality score reports
   how well-behaved the error landscape is.

3. libxs_predict_build (with order=0) scans all polynomial orders
   [1..MAXORDER] and picks the best. Per output, the fingerprint
   consecutive-decay-order decides:
   - Two or more consecutive decaying norms: polynomial interpolation
     along a Hilbert-ordered input-space sequence.
   - Fewer than two or few distinct values: kNN majority vote from
     k=nc/3 nearest neighbors via the k-d tree.

4. Input normalization maps each dimension to [0,1] for uniform
   distance weighting across dimensions with different ranges.

5. The model is saved as a compact binary (centroids, k-d tree points,
   raw outputs for kNN, polynomial coefficients for interpolation).

6. A separate validation model is built on the specified fraction
   (default 0.8) and evaluated against all entries for honest
   generalization reporting.


## Column Layout

The sample uses column names matching the SMM tuning CSV header:

    Inputs (3):   M  N  K
    Outputs (15): BS  BM  BN  BK  WS  WG  LU  NZ
                  AL  TB  TC  AP  AA  AB  AC

Column names are resolved against the header line. To use a different
CSV layout, adjust input_names and output_names in predict.c.


## Mode Override

The mode argument controls how all outputs are predicted:

    auto     Fingerprint-guided per-output decision (default).
    cat      Force kNN majority vote everywhere (fastest).
    interp   Force polynomial interpolation everywhere.

The mode can also be set programmatically via libxs_predict_set_mode,
or overridden per eval call via negative nblend (forces classify).


## Parallelism

When built with OMP=1, the sample uses OpenMP for:
- Batch evaluation during the fraction GSS (libxs_predict_eval_batch_task)
- Validation quality report

The library itself is thread-safe: use libxs_predict_lock(model)
for the internal lock, or pass NULL for single-threaded operation.
Concurrent eval calls without a lock use thread-local scratch space.


## Output

The "Optimal fraction" line reports the GSS result and the
unimodality score (1.0 = well-behaved single-valley landscape).

The "Built" line reports:
- Cluster count and compression ratio
- Polynomial order found by exhaustive scan
- Number of orders evaluated

The validation table shows per-output:

- avg-err / max-err: prediction error against all samples
- avg-bound: model's own error estimate (from fingerprint)
- asterisk marker: polynomial interpolation was used
