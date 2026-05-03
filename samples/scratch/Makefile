ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
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
ifneq (,$(wildcard $(DEPDIR)/include/libxs_source.h))
  DFLAGS += -DLIBXS_SOURCE
endif
endif

# include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/include)

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(wildcard $(SRCDIR)/*.h) \
           $(DEPDIR)/include/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
ifneq (,$(strip $(FC)))
FXXSRCS := $(wildcard $(SRCDIR)/*.f)
F90SRCS := $(wildcard $(SRCDIR)/*.f90) $(wildcard $(SRCDIR)/*.F90)
FXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(FXXSRCS:.f=-f.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90SRCS:.f90=-f90.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90OBJS:.F90=-f90.o)))
endif
SOURCES := $(CSOURCS)
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/$(OUTNAME).x $(OUTDIR)/$(OUTNAME)f.x
FTNSRCS := $(FXXSRCS) $(F90SRCS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS)))
FTNOBJS := $(FXXOBJS) $(F90OBJS)

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

ifneq (0,$(shell echo "$$((0!=$(INTEL) && 190001<=$(CC_VERSION_NUM)))"))
  QKMALLOC ?= 1
else
  QKMALLOC ?= 0
endif

ifneq (,$(strip $(FC)))
ifneq (0,$(shell echo "$$((3>$(XSMM_GCC) || 40600<=$(FC_VERSION_NUM)))"))
$(OUTDIR)/$(OUTNAME)f.x: $(OUTDIR)/.make $(FTNOBJS) $(XLIB)
	$(FLD) -o $@ $(FTNOBJS) $(XLIB) $(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
else
.PHONY: $(OUTDIR)/$(OUTNAME)f.x
endif
else
.PHONY: $(OUTDIR)/$(OUTNAME)f.x
endif

$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(XLIB)
ifneq (0,$(QKMALLOC))
	$(LD) -o $@ $(OBJECTS) $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) -lqkmalloc
else
	$(LD) -o $@ $(OBJECTS) $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
endif

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJECTS) $(FTNOBJS)
endif

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(XFILES) $(MODULES)
endif

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
