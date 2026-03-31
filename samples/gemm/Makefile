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

# include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/include)

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(SRCDIR)/*.h) \
           $(DEPDIR)/include/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/gemm_*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
ifneq (,$(strip $(FC)))
F77SRCS := $(wildcard $(SRCDIR)/gemm_*.F)
F77OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.F=-f77.o)))
endif
OBJECTS := $(COBJCTS)
FTNOBJS := $(F77OBJS)
XFILES := $(OUTDIR)/gemm_strided.x $(OUTDIR)/gemm_batch.x \
          $(OUTDIR)/gemm_groups.x $(OUTDIR)/gemm_index.x \
          $(OUTDIR)/gemm_indexf.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

$(OUTDIR)/gemm_strided.x: $(OUTDIR)/.make $(BLDDIR)/gemm_strided-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/gemm_strided-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/gemm_batch.x: $(OUTDIR)/.make $(BLDDIR)/gemm_batch-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/gemm_batch-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/gemm_groups.x: $(OUTDIR)/.make $(BLDDIR)/gemm_groups-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/gemm_groups-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/gemm_index.x: $(OUTDIR)/.make $(BLDDIR)/gemm_index-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/gemm_index-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

ifneq (,$(strip $(FC)))
ifneq (0,$(shell echo "$$((3>$(XSMM_GCC) || 40600<=$(FC_VERSION_NUM)))"))
$(OUTDIR)/gemm_indexf.x: $(OUTDIR)/.make $(BLDDIR)/gemm_index-f77.o $(XLIB)
	$(FLD) -o $@ $(BLDDIR)/gemm_index-f77.o $(XLIB) $(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
else
.PHONY: $(OUTDIR)/gemm_indexf.x
endif
else
.PHONY: $(OUTDIR)/gemm_indexf.x
endif

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
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
