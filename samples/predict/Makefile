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

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# header-only when library is not (yet) built
ifeq (,$(XLIB))
ifneq (,$(wildcard $(DEPDIR)/libxs/libxs_source.h))
  DFLAGS += -DLIBXS_SOURCE
  HEADERONLY = 1
endif
endif

# include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/libxs)

HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(wildcard $(SRCDIR)/*.h) \
           $(DEPDIR)/libxs/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
SOURCES := $(CSOURCS)
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/predict_params.x $(OUTDIR)/predict_sunspots.x \
          $(OUTDIR)/predict_earthquakes.x $(OUTDIR)/predict_discharge.x \
          $(OUTDIR)/predict_soi.x $(OUTDIR)/predict_stock.x \
          $(OUTDIR)/predict_crystal.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

$(OUTDIR)/predict_params.x: $(OUTDIR)/.make $(BLDDIR)/predict_params-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_params-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_sunspots.x: $(OUTDIR)/.make $(BLDDIR)/predict_sunspots-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_sunspots-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_earthquakes.x: $(OUTDIR)/.make $(BLDDIR)/predict_earthquakes-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_earthquakes-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_discharge.x: $(OUTDIR)/.make $(BLDDIR)/predict_discharge-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_discharge-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_soi.x: $(OUTDIR)/.make $(BLDDIR)/predict_soi-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_soi-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_stock.x: $(OUTDIR)/.make $(BLDDIR)/predict_stock-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_stock-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_crystal.x: $(OUTDIR)/.make $(BLDDIR)/predict_crystal-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_crystal-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

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
