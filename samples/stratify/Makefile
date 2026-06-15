PROJECT ?= libxs
PYTHON ?= python3
HDF5_FILE ?=
HDF5_DATASET ?= ECAL
HDF5_LAYOUT ?= auto
HDF5_RESHAPE ?=
HDF5_EVENT ?= 0
HDF5_CHANNEL ?= 0
HDF5_RESHAPE_ARG = $(if $(HDF5_RESHAPE),--hdf5-reshape $(HDF5_RESHAPE),)
MEDMNIST3D_NPZ ?=
MEDMNIST3D_FLAG ?= organmnist3d
MEDMNIST3D_ROOT ?= $(HOME)/.medmnist
MEDMNIST3D_SIZE ?= 28
MEDMNIST3D_SPLIT ?= train
MEDMNIST3D_INDEX ?= 0
FRAME ?= compact
MEDMNIST3D_INPUT_ARG = $(if $(MEDMNIST3D_NPZ),--npz "$(MEDMNIST3D_NPZ)",--root "$(MEDMNIST3D_ROOT)" --flag "$(MEDMNIST3D_FLAG)" --size "$(MEDMNIST3D_SIZE)")
MEDMNIST3D_METRICS_INPUT_ARG = $(if $(MEDMNIST3D_NPZ),--medmnist3d-npz "$(MEDMNIST3D_NPZ)",--medmnist3d --medmnist3d-root "$(MEDMNIST3D_ROOT)" --medmnist3d-flag "$(MEDMNIST3D_FLAG)" --medmnist3d-size "$(MEDMNIST3D_SIZE)")

.PHONY: all run dense3d volume hdf5 medmnist3d metrics medmnist3d-metrics clean realclean deepclean

all: run

run: dense3d

dense3d:
	$(PYTHON) stratify_dense3d.py --shape 8 16 16 --curve hilbert --frame "$(FRAME)" --out stratified_hilbert.pgm
	$(PYTHON) stratify_dense3d.py --shape 8 16 16 --curve morton --frame "$(FRAME)" --out stratified_morton.pgm

volume: dense3d

hdf5:
	@test -n "$(HDF5_FILE)" || { echo "set HDF5_FILE=/path/to/input.h5"; exit 1; }
	$(PYTHON) stratify_dense3d.py --hdf5 "$(HDF5_FILE)" --hdf5-dataset "$(HDF5_DATASET)" --hdf5-layout "$(HDF5_LAYOUT)" $(HDF5_RESHAPE_ARG) --hdf5-event "$(HDF5_EVENT)" --hdf5-channel "$(HDF5_CHANNEL)" --curve hilbert --frame "$(FRAME)" --out stratified_hdf5.pgm --out-hdf5 stratified_hdf5.h5

medmnist3d:
	$(PYTHON) stratify_medmnist3d.py $(MEDMNIST3D_INPUT_ARG) --split "$(MEDMNIST3D_SPLIT)" --index "$(MEDMNIST3D_INDEX)" --curve hilbert --frame "$(FRAME)" --out stratified_medmnist3d.pgm --map-csv stratified_medmnist3d.csv --label-csv stratified_medmnist3d_label.csv

metrics:
	$(PYTHON) stratify_dense3d_metrics.py --shape 8 16 16 --curve hilbert --frame "$(FRAME)" --csv stratified_metrics.csv

medmnist3d-metrics:
	$(PYTHON) stratify_dense3d_metrics.py $(MEDMNIST3D_METRICS_INPUT_ARG) --medmnist3d-split "$(MEDMNIST3D_SPLIT)" --medmnist3d-index "$(MEDMNIST3D_INDEX)" --curve hilbert --frame "$(FRAME)" --csv stratified_medmnist3d_metrics.csv

clean:
	@-rm -f stratified_*.pgm stratified_*.csv
	@-rm -f stratified_*.h5
	@-rm -rf __pycache__

realclean: clean

deepclean: realclean
