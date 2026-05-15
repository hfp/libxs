# Predict Sample

Train a prediction model from a CSV file and save it for later use.
Optionally hold out a fraction of entries for validation.
## Build

    make

Or from the LIBXS root:

    make GNU=1 samples/predict
## Usage

    ./predict.x <csvfile> [modelfile] [fraction]

    csvfile    Delimited text file (semicolons, commas, or tabs).
               The first line may be a header (auto-skipped if non-numeric).
    modelfile  Output path for the binary model (default: predict.bin).
    fraction   Training split in 0..1 (default: 1.0 = use all entries).
               Values below 1.0 hold out the rest for validation.
## Examples

Full model (no validation):

    ./predict.x ../../samples/smm/params/tune_multiply_H100.csv

    Loaded 295 entries from ...tune_multiply_H100.csv
    Training: 295, held out: 0
    Built: 17 clusters, 8.3x compression
    Saved model to predict.bin (4720 bytes)

80/20 train-test split:

    ./predict.x ../../samples/smm/params/tune_multiply_H100.csv model.bin 0.8

    Loaded 295 entries from ...tune_multiply_H100.csv
    Training: 236, held out: 59
    Built: 15 clusters, 6.9x compression
    Validation (59 samples):
      col  avg-err  max-err
        8     1.42     6.00
        9     0.31     2.00
       10     0.85     4.00
       ...
## Column Layout

The sample uses fixed column indices matching the SMM tuning CSV format:

    Inputs (3):   M=2  N=3  K=4
    Outputs (14): BM=8  BN=9  BK=10  WS=11  WG=12  LU=13  NZ=14
                  AL=15  TB=16  TC=17  AP=18  AA=19  AB=20  AC=21

Column indices are 0-based. Adjust the arrays in predict.c to match
other CSV layouts.
## Validation

When fraction < 1.0, a random subset is selected for training and
the remaining entries are predicted. For each output column, the
average and maximum absolute error are reported. This allows tuning
the quality parameter or cluster count before deploying the model.
## Parameters

The build call uses:

    libxs_predict_build(model, 0, 0.8)

    nclusters = 0    Auto-determine (sqrt of entry count).
    quality   = 0.8  Favor fidelity over compression.
                     0.0 = aggressive truncation, 1.0 = minimal truncation.
