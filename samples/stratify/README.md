# Stratify

Samples for folding higher-dimensional data into lower-dimensional layouts
using libxs space-filling-curve stratification. The folder is intended to
hold related data-preparation and framework-adapter examples that share the
same mapping primitive.

The current dense 3D sample folds a regular voxel grid into a 2D sheet. It is
intentionally a data-preparation example: the mapping can be computed once
for a fixed detector or simulation grid, then reused by a Python, PyTorch,
TensorFlow, or file-based pipeline.

Sample names include the subject after the folder prefix, following the style
used by other multi-sample folders such as `samples/predict`: regular dense
3D-grid tools are named `stratify_dense3d.py` and
`stratify_dense3d_metrics.py`. Future medical-imaging or detector-specific
adapters can use the same pattern, for example `stratify_medmnist3d.py` or
`stratify_mhd.c` if a dependency-light C reader is useful.

Python is the preferred language for dataset adapters because formats such as
HDF5, NPZ, NIfTI, and framework dataloaders are easiest to keep optional there.
C samples remain appropriate for small dependency-free demonstrations,
performance checks, or formats already supported in C, such as MHD volume data.

The transform is not a stack of slices. Each source voxel coordinate is
encoded as a 3D Hilbert or Morton key and decoded as a 2D coordinate:

```text
3D coordinate -> 3D curve rank -> inverse 2D curve coordinate
```

This preserves the deterministic curve order while giving downstream code
a 2D tensor layout that can be consumed by optimized 2D convolution kernels.

The samples distinguish the curve order from the 2D frame. The default
`compact` frame streams voxels in curve-rank order into a dense near-square
sheet, avoiding unused cells when an exact factor pair is available. The
`canonical` frame decodes the finite 3D curve rank through the corresponding
2D curve, preserving the canonical destination curve coordinate at the cost of
possible empty cells.

## Usage

Build libxs first so that `lib/libxs.so` exists:

```bash
cd ../..
make GNU=1 PEDANTIC=2
cd samples/stratify
make
```

Optional HDF5 and MedMNIST3D adapters need `numpy` and `h5py`. In an isolated
environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The default target runs the dense 3D sample. It can also be invoked
explicitly:

```bash
make dense3d
make dense3d FRAME=canonical
```

The older `make volume` spelling remains as a compatibility alias.

Direct invocation:

```bash
python3 stratify_dense3d.py --shape 8 16 16 --curve hilbert \
  --frame compact --out sheet.pgm
python3 stratify_dense3d.py --shape 8 16 16 --curve morton \
  --frame canonical --map-csv map.csv
```

HDF5 input is optional and requires `h5py` and `numpy`. The input dataset may
be a single `D,H,W` volume, a batched `N,D,H,W` dataset, a channelled
`N,C,D,H,W` or `N,D,H,W,C` dataset, or a flattened `N,V` dataset when
`--hdf5-reshape D H W` is supplied. The default `auto` layout recognizes the
3DGAN Caffe sample layout `N,C,D,H,W`:

```bash
python3 stratify_dense3d.py --hdf5 /tmp/3Dgan-risk-audit/caffe/train.h5 \
  --hdf5-dataset ECAL --hdf5-event 0 --hdf5-channel 0 \
  --curve hilbert --frame compact --out sheet.pgm --out-hdf5 sheet.h5
```

The Makefile exposes the same path without making the default target depend on
external data:

```bash
make hdf5 HDF5_FILE=/tmp/3Dgan-risk-audit/caffe/train.h5 FRAME=compact
```

For flattened HDF5 files, pass the dataset name, layout, and target 3D shape:

```bash
make hdf5 HDF5_FILE=dataset_2_1.hdf5 HDF5_DATASET=showers \
  HDF5_LAYOUT=flat HDF5_RESHAPE="45 16 9"
```

Public calorimeter HDF5 datasets with a documented download path are available
from the CaloChallenge project. The challenge HDF5 files contain
`incident_energies` and flattened `showers` datasets, so use `--hdf5-dataset
showers` and `--hdf5-reshape D H W` with the geometry stated for the selected
dataset. Useful starting points are:

| Source | HDF5 format |
| ------ | ----------- |
| [CaloChallenge homepage](https://calochallenge.github.io/homepage/) | Dataset overview, geometry, and evaluation format. |
| [Dataset 1 DOI](https://doi.org/10.5281/zenodo.8099322) | ATLAS photons/pions, flattened showers with dataset-specific voxel counts. |
| [Dataset 2 DOI](https://doi.org/10.5281/zenodo.6366270) | Electrons, reshape `showers` to `45 16 9`. |
| [Dataset 3 DOI](https://doi.org/10.5281/zenodo.6366323) | Higher-granularity electrons, reshape `showers` to `45 50 18`. |
| [Legacy dataset 1 photons](https://doi.org/10.5281/zenodo.15961728), [dataset 1 pions](https://doi.org/10.5281/zenodo.15961924), [dataset 2](https://doi.org/10.5281/zenodo.15962050), [dataset 3](https://doi.org/10.5281/zenodo.15962527) | Submitted model outputs and samples for reproducing published figures. |

Arguments:

| Option | Description |
| ------ | ----------- |
| `--shape D H W` | Source volume shape. Default: `8 16 16`. |
| `--curve` | `hilbert` or `morton`. Default: `hilbert`. |
| `--frame` | `compact` or `canonical`. Default: `compact`. |
| `--libxs` | Explicit path to the libxs shared library. |
| `--hdf5` | Read the source volume from an HDF5 file. |
| `--hdf5-dataset` | Dataset to read from the HDF5 file. Default: `ECAL`. |
| `--hdf5-layout` | `auto`, `dhw`, `ndhw`, `ncdhw`, `ndhwc`, or `flat`. |
| `--hdf5-reshape D H W` | Reshape selected flat HDF5 data to a 3D volume. |
| `--hdf5-event` | Event index for batched HDF5 layouts. Default: `0`. |
| `--hdf5-channel` | Channel index for channelled HDF5 layouts. Default: `0`. |
| `--out` | Write an 8-bit PGM image of the stratified sheet. |
| `--map-csv` | Write `src,z,y,x,dst,v,u` mapping rows. |
| `--out-hdf5` | Write `sheet` and `map` datasets to an HDF5 file. |

The script prints source shape, resulting sheet shape, density, mapping
time, and deposition sums before and after stratification.

## MedMNIST3D

The `stratify_medmnist3d.py` sample reads one standardized MedMNIST3D volume
from an `.npz` file and applies the same 3D-to-2D stratification primitive. It
is a dataset adapter rather than a training script, which keeps the dependency
surface small: only `numpy` is required to parse the MedMNIST file. The sample
does not require PyTorch or the `medmnist` Python package unless you use those
tools separately to download the data.

The official MedMNIST distribution is available from the project page and
Zenodo: [MedMNIST](https://medmnist.com/) and
[Zenodo DOI](https://doi.org/10.5281/zenodo.10519652). The 3D subsets are
`organmnist3d`, `nodulemnist3d`, `adrenalmnist3d`, `fracturemnist3d`,
`vesselmnist3d`, and `synapsemnist3d`.

Direct invocation with an explicit NPZ file:

```bash
python3 stratify_medmnist3d.py --npz ~/.medmnist/organmnist3d.npz \
  --split train --index 0 --curve hilbert --frame compact \
  --out stratified_medmnist3d.pgm \
  --map-csv stratified_medmnist3d.csv \
  --label-csv stratified_medmnist3d_label.csv
```

The Makefile exposes the same path without making the default target depend on
external data:

```bash
make medmnist3d MEDMNIST3D_NPZ=~/.medmnist/organmnist3d.npz FRAME=compact
```

The same NPZ input can be used with the metrics script to report invariants,
locality distortion, and LIBXS Foeppl fingerprint distances for a selected
volume:

```bash
make medmnist3d-metrics MEDMNIST3D_NPZ=~/.medmnist/organmnist3d.npz \
  MEDMNIST3D_SPLIT=test MEDMNIST3D_INDEX=0 FRAME=canonical
```

If `MEDMNIST3D_NPZ` is omitted, the script looks for a dataset under
`MEDMNIST3D_ROOT` using `MEDMNIST3D_FLAG` and `MEDMNIST3D_SIZE`:

```bash
make medmnist3d MEDMNIST3D_ROOT=~/.medmnist MEDMNIST3D_FLAG=nodulemnist3d
```

This gives us a lightweight benchmark bridge: native MedMNIST3D models can use
the original `N,D,H,W` arrays, while 2D models can consume the generated
stratified sheet and keep the label from `stratified_medmnist3d_label.csv`.

## Metrics

The `stratify_dense3d_metrics.py` script reports invariants and locality distortion for
the same synthetic and HDF5 inputs:

```bash
make metrics
python3 stratify_dense3d_metrics.py --hdf5 /tmp/3Dgan-risk-audit/caffe/train.h5 \
  --hdf5-dataset ECAL --curve hilbert
python3 stratify_dense3d_metrics.py --hdf5 dataset_2_1.hdf5 --hdf5-dataset showers \
  --hdf5-layout flat --hdf5-reshape 45 16 9 --curve morton
```

The invariant metrics check that stratification is a lossless layout transform:
the total energy, reconstructed voxel values, and per-axis energy profiles match
after applying the inverse map. The distortion metrics measure what changes for
a convolutional model: how far source 3D grid neighbors move apart in the 2D
sheet, and how often adjacent sheet cells correspond to adjacent source voxels.

The script also reports LIBXS Foeppl fingerprint metrics. These are compact
multi-order L2 descriptors of value and finite-difference structure. The
`fprint.source_reconstructed.diff` value should be zero for a correct lossless
round trip, while `fprint.source_sheet.diff` summarizes how different the
stratified 2D sheet looks as a structured field. This is useful as an additional
quality marker in a paper, but it should be interpreted as a representation
roughness/shape descriptor, not as a detector-physics fidelity score.

These metrics do not prove physics fidelity. They expose the representation
tradeoff: scalar voxel values are preserved exactly, but the neighborhood graph
seen by a 2D convolution is different from the original 3D grid.

## Geometry

The current sample treats the source as a regular `D,H,W` index grid. This is
enough for dense 3DGAN-style arrays and for CaloChallenge datasets after their
flattened `showers` arrays are reshaped with the documented dimensions.

Supporting other detector geometries should be data-driven rather than encoded
as a list of known experiments. A general geometry adapter needs one of the
following descriptions:

| Geometry input | Role |
| -------------- | ---- |
| Integer logical coordinates per voxel, for example `z,alpha,r` or `z,y,x` | Direct input to Hilbert/Morton stratification. |
| Floating physical coordinates per voxel | Quantized or ranked into integer coordinates before stratification. |
| Optional adjacency edges between voxels | Used by metrics to measure physical-neighbor distortion independent of grid shape. |
| Optional voxel weights or cell volumes | Used by downstream physics metrics when bins have unequal size. |

With such a coordinate or adjacency table, the stratification primitive does not
need hard-coded knowledge of a detector. The values remain losslessly permuted;
the geometry file defines which neighborhoods and distances should be considered
meaningful when judging whether the 2D layout is a good substitute for native 3D
convolutions.

## Framework Use

For a fixed geometry, the CSV mapping or the in-memory index arrays can be
cached and reused. A framework integration does not need a custom operator
at first: use the mapping during dataset preparation, or use native gather
and scatter operations in the data loader. The training and inference hot
path can then use ordinary 2D convolution models.

The intended comparison is:

```text
3D volume -> 3D convolution baseline
3D volume -> naive 2D slicing or flattening -> 2D convolution control
3D volume -> Hilbert/Morton stratified sheet -> 2D convolution candidate
```

This separates the amortized layout cost from the model throughput and
quality measurements.

## Related Samples

Additional self-contained integrations can live in this folder when they reuse
the same stratification primitive. Framework-specific adapters should first be
kept as external patches or scripts until their data, dependency versions, and
runtime setup are reproducible. For example, a 3DGAN adapter should only move
into this directory once it can clone the reference model, prepare stratified
2D calorimeter sheets from documented HDF5 showers, and compare them against
the original 3D convolutional baseline without private paths or fragile legacy
packages.
