ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

BLAS = 1
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
FXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(FXXSRCS:.f=-f.o)))
endif
OBJECTS := $(COBJCTS)
FTNOBJS := $(FXXOBJS)
XFILES := $(OUTDIR)/$(OUTNAME).x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(FTNOBJS)

ifneq (,$(strip $(FC)))
ifneq (0,$(shell echo "$$((3>$(XSMM_GCC) || 40600<=$(FC_VERSION_NUM)))"))
$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(FTNOBJS) $(XLIB)
	$(FLD) -o $@ $(FTNOBJS) $(XLIB) $(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
else
.PHONY: $(OUTDIR)/$(OUTNAME).x
endif
else
.PHONY: $(OUTDIR)/$(OUTNAME).x
endif

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
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
	@-rm -f $(XFILES)
endif

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
