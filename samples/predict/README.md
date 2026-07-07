# Predict Samples

Eight executables demonstrating fingerprint-guided prediction:

- **predict_params** -- Parameter prediction from structured CSV
  (GPU kernel tuning, configuration databases).
- **predict_sunspots** -- Timeseries forecasting via sliding-window kNN
  (monthly sunspot numbers, 1749-present).
- **predict_earthquakes** -- Spatial prediction of earthquake magnitude
  from location and depth (USGS catalog).
- **predict_discharge** -- River discharge forecasting via sliding-window
  kNN with day-of-year seasonality (USGS NWIS daily streamflow).
- **predict_soi** -- Southern Oscillation Index prediction from
  anti-correlated Tahiti/Darwin sea level pressure using SPREAD
  decomposition (sum/diff modes).
- **predict_stock** -- Paired-stock timeseries prediction from CSV
  using SPREAD decomposition on two correlated price series.
- **predict_crystal** -- Crystal system classification from composition
  features (AFLOW ICSD, 7 classes, 60K entries).
- **predict_ett** -- ETT (Electricity Transformer Temperature) hourly
  forecasting with univariate, multivariate, PCA, and local-attention
  modes (ETTh1, standard benchmark for timeseries LLMs).


## Build

    make

Or from the LIBXS root:

    make GNU=1 samples/predict


## predict_params

Train a prediction model from a CSV file and save it for later use.
Reports validation quality on a held-out subset.

### Usage

    ./predict_params.x [fraction] [auto|cat|compress[Q]|interp|rf|hknn] [-N] <csvfile> [modelfile [confidence-prefix]]

    fraction   Validation split 0..1 for quality report (default: 0.8).
    auto       Auto-detect mode per output (default).
    cat        Force categorical (kNN) for all outputs.
    compress   Drop redundant entries (Q: threshold, default 0.9).
    interp     Force interpolation for all outputs.
    rf         Random Forest classification.
    hknn       Hierarchical kNN (Fisher-guided partition).
    -N         Max polynomial order (default: 0 = auto).
    csvfile    Delimited text file.
    modelfile  Output path for the binary model.
    confidence-prefix  Optional prefix for confidence map MHD files.

### Example

    ./predict_params.x ../../samples/smm/params/tune_multiply_PVC.csv
    ./predict_params.x 0.8 hknn tune_multiply_PVC.csv model.bin


## predict_sunspots

Timeseries forecasting using sliding-window nearest-neighbor prediction.

### Usage

    ./predict_sunspots.x <csvfile> [train_fraction] [compress[Q]] [hknn|rf]

### Example

    ./predict_sunspots.x predict_sunspots.csv 0.8

### Data Source

Monthly mean total sunspot number from
SILSO (World Data Center, Royal Observatory of Belgium).
Semicolon-delimited: year, month, decimal_year, sunspot_number.


## predict_earthquakes

Predict earthquake magnitude from geographic location and depth.

### Usage

    ./predict_earthquakes.x <usgs_csv> [train_fraction] [compress[Q]] [hknn|rf]

### Example

    ./predict_earthquakes.x predict_earthquakes.csv

### Data Source

USGS Earthquake Hazards Program (public domain).
Comma-delimited: time, latitude, longitude, depth, mag, ...


## predict_discharge

River discharge forecasting with day-of-year seasonality and
log-transform on outputs for heavy-tailed data.

### Usage

    ./predict_discharge.x <discharge_tsv> [train_fraction] [compress[Q]] [hknn|rf]

### Example

    ./predict_discharge.x predict_discharge.tsv

### Data Source

USGS National Water Information System (public domain).
Colorado River at Lees Ferry, site 09380000.
Tab-delimited RDB format.


## predict_soi

Southern Oscillation Index prediction from anti-correlated sea level
pressure at Tahiti and Darwin using SPREAD decomposition.

### Usage

    ./predict_soi.x <tahiti_file> <darwin_file> [train_fraction] [compress[Q]] [hknn|rf]

### Example

    ./predict_soi.x predict_soi_tahiti.dat predict_soi_darwin.dat

### Data Source

NOAA Climate Prediction Center (public domain).
Fixed-width monthly sea level pressure (mb above 1000 mb).


## predict_stock

Multi-stock timeseries prediction with auto-differencing and
PCA/SPREAD decomposition.

### Usage

    ./predict_stock.x <csv_file> [columns] [train_fraction] [compress[Q]] [hknn|rf]

    columns    Comma-separated 0-based column indices (default: 1,2).

### Example

    ./predict_stock.x stocks.csv 1,2,3


## predict_crystal

Crystal system prediction (7-class classification) from chemical
composition features.

### Usage

    ./predict_crystal.x <crystal_csv> [train_fraction] [order] [nclusters] [compress[Q]] [fisher|hknn|setdiff|rf|none]

    fisher     Fisher discriminant feature weighting.
    hknn       Hierarchical kNN (Gini-guided partition).
    setdiff    Setdiff feature selection.
    rf         Random Forest classification (default).
    none       Raw kNN without feature processing.

### Example

    ./predict_crystal.x predict_crystal.csv

### Data Source

AFLOW ICSD catalog (free for academic use).
60,386 entries with Magpie-style composition features
(37 features). Crystal systems: triclinic(1), monoclinic(2),
orthorhombic(3), tetragonal(4), trigonal(5), hexagonal(6),
cubic(7).


## predict_ett

ETT (Electricity Transformer Temperature) hourly forecasting.
Supports univariate and multivariate modes with PCA decomposition
and per-query local-correlation attention.  Standard benchmark
for comparison against transformer-based timeseries models.

### Usage

    ./predict_ett.x <ett_csv> [nseries=1..7] [attend|spread|pca|hknn|rf|nocompress]

    nseries    Number of input channels (1=OT only, 7=all).
    attend     Per-query local-correlation channel weighting.
    spread     Sum/diff decomposition across channels.
    pca        PCA rotation of multi-channel input space.

### Examples

    ./predict_ett.x predict_ett.csv              # univariate OT
    ./predict_ett.x predict_ett.csv 2 pca        # 2ch with PCA (best MSE)
    ./predict_ett.x predict_ett.csv 7 attend     # 7ch with local-attention
    ./predict_ett.x predict_ett.csv 7            # 7ch raw (baseline)

### Data Source

ETTh1 (Electricity Transformer Temperature, hourly) from
Zhou et al. 2021 (Informer).  17,420 hourly readings of an
oil-filled electrical transformer (July 2016 - June 2018).
Seven channels: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT.
Target: OT (oil temperature).  Standard split: train months
1-12, test months 17-24, horizon H=96 steps (4 days).

Download: github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv
Rename to predict_ett.csv (CRLF line endings accepted).
