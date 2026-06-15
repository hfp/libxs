ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
PROJECT ?= libxs
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

BLAS_STATIC = 0
BLAS = 1
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

# include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/libxs)

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(wildcard $(SRCDIR)/*.h) \
           $(DEPDIR)/libxs/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
ifneq (1,$(HEADERONLY))
ifneq (,$(strip $(FC)))
F77SRCS := $(wildcard $(SRCDIR)/*.f)
F77OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.f=-f77.o)))
JITOBJ := $(BLDDIR)/libxs_jit-f77.o
endif
endif
OBJECTS := $(COBJCTS)
FTNOBJS := $(F77OBJS) $(JITOBJ)
XFILES := $(OUTDIR)/$(OUTNAME).x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

ifneq (1,$(HEADERONLY))
ifneq (,$(strip $(FC)))
ifneq (0,$(shell echo "$$((3>$(XSMM_GCC) || 40600<=$(FC_VERSION_NUM)))"))
$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(FTNOBJS) $(XLIB)
	$(FLD) -o $@ $(OBJECTS) $(FTNOBJS) $(XLIB) $(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
else
.PHONY: $(OUTDIR)/$(OUTNAME).x
endif
else
.PHONY: $(OUTDIR)/$(OUTNAME).x
endif
else
.PHONY: $(OUTDIR)/$(OUTNAME).x
endif

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/libxs_jit-f77.o: $(DEPDIR)/libxs/libxs_jit.F .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@ $(FMFLAGS) $(BLDDIR)

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make $(JITOBJ) Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@ $(FMFLAGS) $(BLDDIR)

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
