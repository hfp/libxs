# Spatial

Samples for spatial data structures and queries built on libxs primitives:
space-filling curves (Hilbert, Morton), kd-tree partition and search, and
spatial pair counting. Each topic uses a filename prefix to group related
files in this flat directory.

| Prefix       | Topic                                    |
|--------------|------------------------------------------|
| `stratify_*` | 3D-to-2D layout via space-filling curves |
| `paircnt_*`  | Spatial pair counting (planned)          |

Underlying libxs primitives (from `libxs_perm.h`):

- Hilbert and Morton curve encode/decode (N-D stratification)
- kd-tree build, partition, nearest-neighbor search
- Sort by spatial locality (various methods)

## Building

Build libxs first so that `lib/libxs.so` exists:

```bash
cd ../..
make GNU=1 PEDANTIC=2
cd samples/spatial
make
```

Optional Python adapters need `numpy` and `h5py`:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## Stratification (stratify\_\*)

Folding higher-dimensional data into lower-dimensional layouts using
space-filling-curve stratification. The mapping can be computed once for a
fixed detector or simulation grid and reused by any downstream pipeline.

The transform encodes each source voxel coordinate as a 3D Hilbert or Morton
key and decodes it as a 2D coordinate:

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

### Usage

The default target runs the dense 3D sample:

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

| Source                | Format / notes                       |
|-----------------------|--------------------------------------|
| [CaloChallenge][cc]   | Overview, geometry, evaluation.      |
| [Dataset 1][cc1]      | ATLAS photons/pions, flat showers.   |
| [Dataset 2][cc2]      | Electrons, reshape to `45 16 9`.     |
| [Dataset 3][cc3]      | Hi-gran electrons, `45 50 18`.       |
| [Legacy sets][ccl]    | Submitted outputs for repro figures. |

[cc]:  https://calochallenge.github.io/homepage/
[cc1]: https://doi.org/10.5281/zenodo.8099322
[cc2]: https://doi.org/10.5281/zenodo.6366270
[cc3]: https://doi.org/10.5281/zenodo.6366323
[ccl]: https://doi.org/10.5281/zenodo.15961728

Arguments:

| Option              | Description                          |
|---------------------|--------------------------------------|
| `--shape D H W`     | Volume shape (default `8 16 16`).    |
| `--curve`           | `hilbert` or `morton`.               |
| `--frame`           | `compact` or `canonical`.            |
| `--libxs`           | Path to libxs shared library.        |
| `--hdf5`            | Read volume from an HDF5 file.       |
| `--hdf5-dataset`    | HDF5 dataset name (default `ECAL`).  |
| `--hdf5-layout`     | `auto`/`dhw`/`ndhw`/`ncdhw`/`flat`.  |
| `--hdf5-reshape`    | Reshape flat data to `D H W`.        |
| `--hdf5-event`      | Event index (default `0`).           |
| `--hdf5-channel`    | Channel index (default `0`).         |
| `--out`             | Write 8-bit PGM of the sheet.        |
| `--map-csv`         | Write `src,z,y,x,dst,v,u` rows.      |
| `--out-hdf5`        | Write sheet+map to HDF5 file.        |

### MedMNIST3D

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

### Metrics

The `stratify_dense3d_metrics.py` script reports invariants and locality
distortion for the same synthetic and HDF5 inputs:

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
stratified 2D sheet looks as a structured field.

### Geometry

The current sample treats the source as a regular `D,H,W` index grid. This is
enough for dense 3DGAN-style arrays and for CaloChallenge datasets after their
flattened `showers` arrays are reshaped with the documented dimensions.

Supporting other detector geometries should be data-driven rather than encoded
as a list of known experiments. A general geometry adapter needs one of the
following descriptions:

| Geometry input              | Role                               |
|-----------------------------|------------------------------------|
| Integer logical coords      | Direct Hilbert/Morton input.       |
| Floating physical coords    | Quantized before stratification.   |
| Adjacency edges (optional)  | Neighbor-distortion metrics.       |
| Voxel weights (optional)    | Physics metrics for unequal bins.  |

---

## Pair Counting (paircnt\_\*) -- planned

Spatial pair counting inspired by Corrfunc: given a point catalog, bin all
pairs by separation distance to compute correlation functions. The approach
uses libxs kd-tree partition for cell decomposition and cell-pair pruning
(minimum-separation skip), with vectorized inner loops for the actual distance
computation and bin assignment.

Design goals:

- Leverage `libxs_kdtree_build` / `libxs_kdtree_partition` for spatial
  decomposition rather than a separate grid structure.
- Cell-pair pruning: skip cell pairs whose minimum separation exceeds the
  largest bin edge.
- Vectorized pair-counting kernel for the inner loop (SIMD where available).
- Periodic and non-periodic boundary conditions.
- Output: binned pair counts DD(r), optionally normalized to xi(r) via
  Landy-Szalay with a random catalog.
