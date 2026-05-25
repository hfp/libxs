# Predict Samples

Six executables demonstrating fingerprint-guided prediction:

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
- **predict_crystal** -- Crystal system classification from composition
  features (AFLOW ICSD, 7 classes, 60K entries).
## Build

    make

Or from the LIBXS root:

    make GNU=1 samples/predict
## predict_params

Train a prediction model from a CSV file and save it for later use.
Finds the optimal training fraction and polynomial order automatically.
Reports validation quality on a held-out subset.

### Usage

    ./predict_params.x [fraction] [auto|cat|interp] [-N] <csvfile> [modelfile]

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

### Example

    ./predict_params.x ../../samples/smm/params/tune_multiply_PVC.csv
## predict_sunspots

Timeseries forecasting using sliding-window nearest-neighbor prediction.
The recent history (window of W values) serves as input; the next H values
are predicted as output. The kNN confidence indicates whether similar
patterns were seen in training.

### Usage

    ./predict_sunspots.x <csvfile> [train_fraction]

    csvfile         Semicolon-delimited timeseries (SILSO sunspot format).
    train_fraction  Fraction of data used for training (default: 0.8).

### Example

    ./predict_sunspots.x predict_sunspots.csv 0.8

    Loaded 3328 monthly sunspot values from predict_sunspots.csv
    Window=12, Horizon=6, Train=2650, Test=666
    Built: 51 clusters, 32.0x compression, order=2
    Forecast quality (661 test windows):
      step   avg-err   max-err
      t+1      17.58     88.10
      t+2      19.48    115.00
      t+3      21.01    107.10
      t+4      21.76    114.50
      t+5      22.84    118.40
      t+6      24.26    153.20
      avg confidence: 1.000

### Data Source

Monthly mean total sunspot number from
[SILSO](https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv)
(World Data Center, Royal Observatory of Belgium).
Semicolon-delimited: year, month, decimal_year, sunspot_number,
std_dev, obs_count, marker.
"Source: WDC-SILSO, Royal Observatory of Belgium, Brussels"
## predict_earthquakes

Predict earthquake magnitude from geographic location and depth.
This is a spatial prediction problem (not timeseries): given where
an earthquake occurs, what magnitude is expected based on historical
patterns at nearby locations?

### Usage

    ./predict_earthquakes.x <usgs_csv> [train_fraction]

    usgs_csv        USGS earthquake catalog CSV (comma-delimited).
    train_fraction  Fraction of data for training (default: 0.8).

### Example

    ./predict_earthquakes.x predict_earthquakes.csv

    Loaded 19619 earthquake events from predict_earthquakes.csv
    Inputs: latitude, longitude, depth -> Output: magnitude
    Train=15695, Test=3924
    Built: 125 clusters, 83.9x compression, order=2
    Prediction quality (3924 test events):
      avg magnitude error: 0.272
      max magnitude error: 2.700
      avg confidence: 0.649

### Data Source

[USGS Earthquake Hazards Program](https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2022-04-01&endtime=2025-01-01&minmagnitude=4.5&limit=20000)
(public domain, US Government).
Comma-delimited: time, latitude, longitude, depth, mag, magType, ...
## predict_discharge

River discharge (streamflow) forecasting using sliding-window kNN
with day-of-year as an additional input dimension to capture
seasonality. Uses log-transform on outputs (via API) for
heavy-tailed data. Predicts the next 7 days from the previous
14 days + rate-of-change derivatives.

### Usage

    ./predict_discharge.x <discharge_tsv> [train_fraction]

    discharge_tsv   USGS NWIS daily discharge (tab-delimited RDB format).
    train_fraction  Fraction of data for training (default: 0.8).

### Example

    ./predict_discharge.x predict_discharge.tsv

    Loaded 9135 daily discharge values from predict_discharge.tsv
    Window=14 (+3 diffs +day-of-year), Horizon=7, Train=7294, Test=1827
    Built: 85 clusters, 58.4x compression, order=2
    Forecast quality (1821 test windows):
      step   avg-err   max-err
      t+1      644.5   17726.6
      t+2      769.0   21363.5
      t+3      877.9   23199.8
      t+4      963.7   24193.4
      t+5     1044.1   25086.3
      t+6     1131.3   25735.3
      t+7     1225.9   26729.3
      avg confidence: 1.000

### Data Source

[USGS National Water Information System](https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=09380000&parameterCd=00060&startDT=2000-01-01&endDT=2025-01-01)
(public domain, US Government). Colorado River at Lees Ferry, site 09380000.
Tab-delimited RDB, comment lines start with #, data columns:
agency_cd, site_no, datetime, discharge_value, qualification_code.
## predict_soi

Southern Oscillation Index prediction from anti-correlated sea level
pressure at Tahiti and Darwin. Demonstrates cross-series decomposition
via `libxs_predict_set_decompose(LIBXS_PREDICT_SPREAD)`: the sum/diff
modes separate the common trend from the anti-correlated signal,
making the spread (which *is* the SOI) easier to predict.

### Usage

    ./predict_soi.x <tahiti_file> <darwin_file> [train_fraction]

    tahiti_file     NOAA CPC monthly Tahiti SLP (fixed-width).
    darwin_file     NOAA CPC monthly Darwin SLP (fixed-width).
    train_fraction  Fraction of data for training (default: 0.8).

### Example

    ./predict_soi.x predict_soi_tahiti.dat predict_soi_darwin.dat

### Data Source

[NOAA Climate Prediction Center](https://www.cpc.ncep.noaa.gov/data/indices/)
(public domain, US Government).
- Tahiti: https://www.cpc.ncep.noaa.gov/data/indices/tahiti
- Darwin: https://www.cpc.ncep.noaa.gov/data/indices/darwin

Fixed-width: YEAR followed by 12 monthly sea level pressure values
(mb above 1000 mb). Data from 1951 to present.
## predict_crystal

Crystal system prediction (7-class classification) from chemical
composition features. Demonstrates `LIBXS_PREDICT_FISHER` for
automatic feature weighting via Fisher's discriminant criterion,
or `LIBXS_PREDICT_RF` for Random Forest classification.

### Usage

    ./predict_crystal.x <crystal_csv> [train_fraction] [order] [nclusters]

    crystal_csv     CSV with composition features + crystal_system label.
    train_fraction  Fraction of data for training (default: 0.8).

### Example

    ./predict_crystal.x predict_crystal.csv

    Loaded 60386 entries (37 features) from predict_crystal.csv
    Train=48309, Test=12077
    Built: 220 clusters, 208.7x compression, order=2
    Accuracy: 9625/12077 = 79.7%
    Confidence-gated (>=0.9): 6167/6502 = 94.8% (coverage 53.8%)
    Avg confidence: 0.820

### Data Source

[AFLOW ICSD catalog](https://aflow.org/API/aflux/) (free for academic use).
60,386 entries with Magpie-style composition features (7 elemental
properties x 5 statistics + 2 counts = 37 features). Crystal systems:
triclinic(1), monoclinic(2), orthorhombic(3), tetragonal(4),
trigonal(5), hexagonal(6), cubic(7).
Data preparation: `prepare_crystal.py`.
## How It Works

All samples share the same prediction library (libxs_predict):

1. **predict_params**: Each CSV row is an independent (inputs, outputs)
   pair. The model learns spatial relationships in the input space
   and predicts outputs for unseen input combinations.

2. **predict_sunspots**: Uses `libxs_predict_set_series` to declare
   timeseries structure; the framework constructs sliding windows
   internally from accumulated timesteps. The model finds historically
   similar windows and predicts the continuation. The kNN confidence
   reflects how well the current pattern matches training history.

3. **predict_earthquakes**: Each earthquake event provides
   (lat, lon, depth) as inputs and magnitude as output. The model
   finds geographically similar past events and predicts expected
   magnitude for new locations.

4. **predict_discharge**: Combines temporal sliding-window (14 days)
   with day-of-year seasonality as an extra input dimension.
   Log-transform on outputs (via `libxs_predict_set_transform`)
   handles heavy-tailed discharge data transparently.

5. **predict_soi**: Two anti-correlated series (Tahiti and Darwin
   sea level pressure) feed a single model via `set_series(2, W)`.
   SPREAD decomposition transforms the stacked windows into sum/diff
   modes before kNN matching, exploiting the anti-correlation structure
   that defines the Southern Oscillation Index.

6. **predict_crystal**: 37 composition features predict one of 7
   crystal systems. Uses `LIBXS_PREDICT_RF` (Random Forest) for
   79.7% accuracy, or `LIBXS_PREDICT_FISHER` (kNN with automatic
   feature weighting) for 70.7%. Confidence gating raises accuracy
   to 95%+ on the reliable subset.

The fingerprint automatically determines per-output whether polynomial
interpolation or distance-weighted kNN voting is more appropriate.
Per-output confidence and variance scores enable the caller to gate
predictions and fall back to safe defaults when the model is uncertain.

When confidence is low (<0.7), the framework automatically expands to
multi-cluster blending, improving predictions by averaging over
distinct regimes (e.g., -4% MAE on earthquake magnitude prediction).

Timeseries samples use `LIBXS_PREDICT_TEMPORAL` mode which enables
recency weighting (recent neighbors preferred), continuous-valued
output without snap-to-nearest discretization, and local coherence
smoothing across horizon steps. These heuristics also auto-enable
for any timeseries model (nseries > 0) when the query falls outside
the training bounding box.

The sunspot sample uses `libxs_predict_set_series` to declare
timeseries structure: instead of manually constructing sliding windows,
the caller pushes one timestep at a time (with outputs=NULL) and the
framework builds all valid windows at build time. For multiple
co-observed series, `libxs_predict_set_decompose(LIBXS_PREDICT_SPREAD)`
transforms the stacked windows into sum/diff modes, exploiting
anti-correlation between series (e.g., one's gain is the other's loss).

The sunspot sample additionally demonstrates forward-inverse-forward
iteration: predicting outputs, finding the canonical historical window
via `libxs_predict_inverse`, then re-predicting from that pattern.
This reduces worst-case errors (-17% max error) at the cost of slight
average regression -- a variance-bias tradeoff useful for applications
where avoiding catastrophic predictions matters most.
