# Predict Sample

Train a prediction model from a CSV file and save it for later use.
Finds the optimal training fraction and quality automatically.


## Build

    make

Or from the LIBXS root:

    make GNU=1 samples/predict


## Usage

    ./predict.x [-N] <csvfile> [modelfile]

    -N         Quality GSS iterations for final build (default: -1 = converge).
               Use -5 for faster, -20 for more thorough optimization.
    csvfile    Delimited text file (semicolons, commas, or tabs).
               The first line may be a header (auto-skipped if non-numeric).
    modelfile  Output path for the binary model.
               Default: derived from CSV basename (e.g., data.csv -> data.bin).


## Example

    ./predict.x ../../samples/smm/params/tune_multiply_H100.csv

    Loaded 295 entries from ...tune_multiply_H100.csv
    Optimal fraction: 0.76 (224/295 entries)
    Built: 15 clusters, 15.2x compression
    Quality (295 samples):
      param   avg-err   max-err  avg-bound
      BM *   2.01e+00  5.20e+01   7.40e-01
      BN     7.52e-01  6.30e+01   0.00e+00
      BK *   5.13e+00  5.50e+01   6.40e-01
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
     per-output smooth ordering (libxs_sort_smooth).
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


## Parallelism

When built with OMP=1, the sample uses OpenMP for:
- Batch evaluation during the fraction GSS (libxs_predict_eval_batch_task)
- Final quality report

The library itself is thread-safe: use libxs_predict_lock(model)
for the internal lock, or pass NULL for single-threaded operation.
