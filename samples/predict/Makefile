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

# setup XGBoost as an optional dependency
XGB ?= 1
ifneq (0,$(XGB))
  # must be set before including Makefile.inc
  override PEDANTIC := 0
endif

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# header-only when library is not (yet) built
ifeq (,$(XLIB))
ifneq (,$(wildcard $(DEPDIR)/libxs/libxs_source.h))
  DFLAGS += -DLIBXS_SOURCE
  HEADERONLY = 1
endif
endif

# XGBoost for head-to-head comparison
ifneq (0,$(XGB))
  PKGCFG ?= $(call which,pkg-config)
  XGBOOST_ROOT ?= $(shell echo "$$XGBOOST_ROOT")
  # A source checkout carries the headers but no built library, so the library
  # is looked for under XGBOOST_ROOT rather than assumed, and XGBOOST_LIBDIR
  # names one held elsewhere (a directory of links to a library from a Python
  # wheel, say). Nothing is enabled without a library to link: a build that
  # links and then cannot load is worse than one without the comparison.
  XGBOOST_LIBDIR ?= $(shell echo "$$XGBOOST_LIBDIR")
  ifeq (,$(filter Darwin Windows_NT,$(UNAME)))
    XGB_OLDTAGS := $(call linkopt,--disable-new-dtags)
  endif
  ifneq (,$(wildcard $(XGBOOST_ROOT)/include/xgboost/c_api.h))
    ifeq (,$(XGBOOST_LIBDIR))
      XGBOOST_LIBDIR := $(patsubst %/,%,$(dir $(firstword $(wildcard \
        $(XGBOOST_ROOT)/lib64/libxgboost.* $(XGBOOST_ROOT)/lib/libxgboost.*))))
    endif
    # Nothing built under the root: ask the Python package where its own library
    # is. That library is not linkable as it stands, so what is named here is a
    # directory of links under the build tree rather than the package: the
    # library carries a SONAME that no file provides, and the dependencies it
    # vendored sit in a sibling directory. Both are supplied by the link farm
    # below, so nothing outside the build tree is written to.
    ifeq (,$(filter Darwin Windows_NT,$(UNAME)))
    ifeq (,$(XGBOOST_LIBDIR))
      XGBPY_LIB := $(shell $(PYTHON) -c \
        "import xgboost.core as c; print(c._LIB._name)" 2>/dev/null)
      ifneq (,$(wildcard $(XGBPY_LIB)))
        XGBPY_SOV := $(shell $(PYTHON) -c \
          "import xgboost; print(xgboost.__version__.split('.')[0])" 2>/dev/null)
        XGBPY_DEPS := $(shell $(PYTHON) -c "import os,glob,xgboost; \
          print(' '.join(glob.glob(os.path.dirname(xgboost.__file__)+'.libs/*')))" \
          2>/dev/null)
        # The links are made while this file is read, not by a rule: the link
        # flags are built here too, and the helper that builds them skips a
        # directory that does not exist yet. A rule runs too late, which left
        # the first build on a clean tree with a bare -lxgboost: it failed
        # outright where no system copy was installed, and quietly resolved
        # against one of a different version where there was.
        XGBFARM := $(BLDDIR)/xgboost
        XGBOK := $(shell $(MKDIR) -p $(XGBFARM) \
          && ln -sf $(XGBPY_LIB) $(XGBFARM)/libxgboost.$(ILIBEXT) \
          && ln -sf $(XGBPY_LIB) $(XGBFARM)/libxgboost.$(ILIBEXT).$(XGBPY_SOV) \
          && for DEP in $(XGBPY_DEPS); do ln -sf $$DEP $(XGBFARM)/; done \
          && echo ok)
        ifeq (ok,$(XGBOK))
          XGBOOST_LIBDIR := $(XGBFARM)
        endif
      endif
    endif
    endif
  endif
  ifneq (,$(and $(wildcard $(XGBOOST_ROOT)/include/xgboost/c_api.h),$(XGBOOST_LIBDIR)))
    IFLAGS += -I$(call quote,$(XGBOOST_ROOT)/include)
    DFLAGS += -D__XGBOOST
    # A library taken from a Python wheel is not self-contained: it names its
    # own dependencies (a vendored OpenMP among them) and finds them through its
    # own rpath, which the loader does not consult for a dependency of a
    # dependency. The old tag is inherited down the chain where RUNPATH is not,
    # so one directory resolves the library and whatever it drags in.
    XGBLDFLAGS := $(XGB_OLDTAGS) $(call abslibpath,$(XGBOOST_LIBDIR)) -lxgboost
  else ifneq (,$(shell $(PKGCFG) --exists xgboost 2>/dev/null && echo yes))
    IFLAGS += $(shell $(PKGCFG) --cflags xgboost)
    DFLAGS += -D__XGBOOST
    XGBLDFLAGS := $(shell $(PKGCFG) --libs xgboost)
  endif
endif

# include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/libxs)

HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(wildcard $(SRCDIR)/*.h) \
           $(wildcard $(DEPDIR)/libxs/*.h) \
           $(DEPDIR)/libxs/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
SOURCES := $(CSOURCS)
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/predict_params.x $(OUTDIR)/predict_sunspots.x \
          $(OUTDIR)/predict_earthquakes.x $(OUTDIR)/predict_discharge.x \
          $(OUTDIR)/predict_soi.x $(OUTDIR)/predict_stock.x \
          $(OUTDIR)/predict_crystal.x $(OUTDIR)/predict_ett.x \
          $(OUTDIR)/predict_prob.x $(OUTDIR)/predict_dist.x \
          $(OUTDIR)/predict_bits.x $(OUTDIR)/predict_higgs.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

$(OUTDIR)/predict_params.x: $(OUTDIR)/.make $(BLDDIR)/predict_params-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_params-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(XGBLDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_sunspots.x: $(OUTDIR)/.make $(BLDDIR)/predict_sunspots-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_sunspots-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_earthquakes.x: $(OUTDIR)/.make $(BLDDIR)/predict_earthquakes-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_earthquakes-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(XGBLDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_discharge.x: $(OUTDIR)/.make $(BLDDIR)/predict_discharge-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_discharge-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_soi.x: $(OUTDIR)/.make $(BLDDIR)/predict_soi-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_soi-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_stock.x: $(OUTDIR)/.make $(BLDDIR)/predict_stock-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_stock-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_crystal.x: $(OUTDIR)/.make $(BLDDIR)/predict_crystal-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_crystal-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(XGBLDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_ett.x: $(OUTDIR)/.make $(BLDDIR)/predict_ett-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_ett-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(XGBLDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_prob.x: $(OUTDIR)/.make $(BLDDIR)/predict_prob-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_prob-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_bits.x: $(OUTDIR)/.make $(BLDDIR)/predict_bits-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_bits-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_higgs.x: $(OUTDIR)/.make $(BLDDIR)/predict_higgs-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_higgs-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_dist.x: $(OUTDIR)/.make $(BLDDIR)/predict_dist-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_dist-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)


$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

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

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(XFILES) $(MODULES) *.bin
endif

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
