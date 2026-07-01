ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
PROJECT := libxs
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

BLAS = 0
OMP = 1
SYM = 1
SSE = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# header-only when library is not (yet) built
ifeq (,$(XLIB))
ifneq (,$(wildcard $(DEPDIR)/libxs/libxs_source.h))
  DFLAGS += -DLIBXS_SOURCE
  HEADERONLY = 1
endif
endif

IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/libxs)

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
PAIRCNT_NPOINTS ?= 10000
PAIRCNT_BOXSIZE ?= 420.0
PAIRCNT_RMIN ?= 0.1
PAIRCNT_RMAX ?= 25.0
PAIRCNT_NBINS ?= 20
MEDMNIST3D_INPUT_ARG = $(if $(MEDMNIST3D_NPZ),--npz "$(MEDMNIST3D_NPZ)",--root "$(MEDMNIST3D_ROOT)" --flag "$(MEDMNIST3D_FLAG)" --size "$(MEDMNIST3D_SIZE)")
MEDMNIST3D_METRICS_INPUT_ARG = $(if $(MEDMNIST3D_NPZ),--medmnist3d-npz "$(MEDMNIST3D_NPZ)",--medmnist3d --medmnist3d-root "$(MEDMNIST3D_ROOT)" --medmnist3d-flag "$(MEDMNIST3D_FLAG)" --medmnist3d-size "$(MEDMNIST3D_SIZE)")

HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(wildcard $(SRCDIR)/*.h) \
           $(DEPDIR)/libxs/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/paircnt_*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
SOURCES := $(CSOURCS)
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/paircnt_dd.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

$(OUTDIR)/paircnt_dd.x: $(OUTDIR)/.make $(BLDDIR)/paircnt_dd-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/paircnt_dd-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: dense3d volume hdf5 medmnist3d metrics medmnist3d-metrics

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

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJECTS)
endif
	@-rm -f stratified_*.pgm stratified_*.csv stratified_*.h5
	@-rm -rf __pycache__

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(XFILES)
endif

.PHONY: deepclean
deepclean: realclean
