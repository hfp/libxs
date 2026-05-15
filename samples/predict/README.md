# Predict Sample

Train a prediction model from a CSV file and save it for later use.
Finds the optimal training fraction and quality automatically.


## Build

    make

Or from the LIBXS root:

    make GNU=1 samples/predict


## Usage

    ./predict.x [auto|cat|interp] [-N] <csvfile> [modelfile]

    auto       Auto-detect mode per output (default).
    cat        Force categorical (kNN) for all outputs.
    interp     Force interpolation for all outputs.
    -N         Quality GSS iterations for final build (default: -1 = converge).
               Use -5 for faster, -20 for more thorough optimization.
    csvfile    Delimited text file (semicolons, commas, or tabs).
               The first line may be a header (auto-skipped if non-numeric).
    modelfile  Output path for the binary model.
               Default: derived from CSV basename (e.g., data.csv -> data.bin).


## Example

    ./predict.x ../../samples/smm/params/tune_multiply_H100.csv

    Loaded 295 entries from ...tune_multiply_H100.csv
    Optimal fraction: 0.66 (195/295 entries)
    Built: 14 clusters, 14.6x compression, quality=0.10 (20 iter)
    Quality (295 samples):
      param   avg-err   max-err  avg-bound
      BS *   5.92e+00  5.70e+01   0.00e+00
      BM *   2.01e+00  5.20e+01   7.40e-01
      BN     7.52e-01  6.30e+01   0.00e+00
      ...
    Saved model to tune_multiply_H100.bin (43123 bytes)

Parameters marked with * use polynomial interpolation; others use
kNN majority vote (auto-detected from fingerprint decay).


## How It Works

1. libxs_predict_load_csv loads the file using column names ("M", "N",
   "K" as inputs; "BM", "BN", etc. as outputs). Names are matched
   case-insensitively against the CSV header.

2. libxs_shuffle produces a deterministic permutation of entries.
   libxs_gss_min finds the training fraction that minimizes total
   prediction error across all samples.

3. libxs_predict_build (with quality=-1) auto-optimizes the truncation
   quality via an internal GSS. Per output, the fingerprint decides:
   - Decaying norms (decay < 0.5): polynomial interpolation along a
     Morton (Z-order) input-space ordering.
   - Non-decaying or few distinct values: kNN majority vote from
     k=5 nearest neighbors via the k-d tree.

4. The model is saved as a compact binary (centroids, k-d tree points,
   raw outputs for kNN, polynomial coefficients for interpolation).


## Column Layout

The sample uses column names matching the SMM tuning CSV header:

    Inputs (3):   M  N  K
    Outputs (14): BM  BN  BK  WS  WG  LU  NZ
                  AL  TB  TC  AP  AA  AB  AC

Column names are resolved against the header line. To use a different
CSV layout, adjust input_names and output_names in predict.c.


## Mode Override

The mode argument controls how all outputs are predicted:

    auto     Fingerprint-guided per-output decision (default).
    cat      Force kNN majority vote everywhere (fastest, no polynomial).
    interp   Force polynomial interpolation everywhere.

The mode can also be set programmatically via libxs_predict_set_mode,
or overridden per eval call via negative nblend (forces classify).


## Parallelism

When built with OMP=1, the sample uses OpenMP for:
- Batch evaluation during the fraction GSS (libxs_predict_eval_batch_task)
- Final quality report

The library itself is thread-safe: use libxs_predict_lock(model)
for the internal lock, or pass NULL for single-threaded operation.


## Output

The "Built" line reports:
- Cluster count and compression ratio
- Quality value found by GSS (0=aggressive, 1=conservative)
- Number of GSS iterations performed

The quality table shows per-output:
- avg-err / max-err: prediction error across all samples
- avg-bound: model's own error estimate (from fingerprint)
- * marker: polynomial interpolation was used
