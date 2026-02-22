ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

OMP = 1
SYM = 1

BLAS = $(if $(filter-out 0,$(OMP)),2,1)
#BLAS_STATIC = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR)) -I$(call quote,$(DEPDIR)/include)

# Wrap all four BLAS GEMM symbols (dgemm, sgemm, zgemm, cgemm)
ifneq (Darwin,$(UNAME))
  ifneq (0,$(BLAS_STATIC))
    WRAP_GEMM := -Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ -Wl,--wrap=zgemm_ -Wl,--wrap=cgemm_
  endif
else # macOS
  $(info ==========================================================)
  $(info The static link-time wrapper is not supported under macOS!)
  $(info ==========================================================)
endif

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(SRCDIR)/*.h) $(DEPDIR)/include/libxs_source.h
# Driver source (compiled once with default double precision)
CSOURCS := $(SRCDIR)/gemm.c
# Implementation sources: _WRAP + support (compiled for each precision)
IMPLSRCS := $(SRCDIR)/$(OUTNAME).c $(SRCDIR)/ozaki1.c $(SRCDIR)/ozaki2.c $(SRCDIR)/zgemm3m.c $(SRCDIR)/gemm-print.c
# Entry/dlsym source: _REAL + entry points (LD_PRELOAD path only)
ENTRYSRCS := $(SRCDIR)/wrap.c
COBJCTS := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-c.o,$(CSOURCS))
IMPLOBJ_D := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-d.o,$(IMPLSRCS))
IMPLOBJ_S := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-s.o,$(IMPLSRCS))
IMPLOBJS := $(IMPLOBJ_D) $(IMPLOBJ_S)
ENTRYOBJ_D := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-d.o,$(ENTRYSRCS))
ENTRYOBJ_S := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-s.o,$(ENTRYSRCS))
ENTRYOBJS := $(ENTRYOBJ_D) $(ENTRYOBJ_S)
OBJECTS := $(COBJCTS) $(IMPLOBJS) $(ENTRYOBJS)
XFILES := $(OUTDIR)/gemm-blas.x $(OUTDIR)/gemm-wrap.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

ifneq (,$(WRAP_GEMM))
$(OUTDIR)/libwrap.$(SLIBEXT): $(OUTDIR)/.make $(IMPLOBJS)
	$(MAKE_AR) $@ $(call tailwords,$^)

$(OUTDIR)/gemm-wrap.x: $(BLDDIR)/gemm-c.o $(OUTDIR)/libwrap.$(SLIBEXT) $(XSDEP) $(OUTDIR)/.make
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(OUTDIR)/libwrap.$(SLIBEXT) $(XSLIB)\
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(OUTDIR)/libwrap.$(SLIBEXT) $(XSLIB) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(OUTDIR)/libwrap.$(SLIBEXT) $(XSLIB) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
endif

.PHONY: $(OUTDIR)/libwrap.$(DLIBEXT)
.PHONY: $(OUTDIR)/gemm-blas.x
else
$(OUTDIR)/libwrap.$(DLIBEXT): $(IMPLOBJS) $(ENTRYOBJS) $(XSDEP) $(OUTDIR)/.make
	$(LIB_SOLD) $(call solink,$@) $(IMPLOBJS) $(ENTRYOBJS) \
		$(XSLIB) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))

$(OUTDIR)/gemm-blas.x: $(COBJCTS) $(BLDDIR)/gemm-print-d.o $(XSDEP) $(OUTDIR)/.make $(OUTDIR)/libwrap.$(DLIBEXT)
	$(LD) -o $@ $(COBJCTS) $(BLDDIR)/gemm-print-d.o $(XSLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

.PHONY: $(OUTDIR)/libwrap.$(SLIBEXT)
.PHONY: $(OUTDIR)/gemm-wrap.x
endif

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-d.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-s.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) -DGEMM_REAL_TYPE=float $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: test
test: $(ROOTDIR)/wrap-test.sh $(XFILES)
	@NREPEAT=$(TESTSIZE) $< gemm                               
	@NREPEAT=$(TESTSIZE) $< gemm  16  20 350 1 0  1 0.0 350 350 1000
	@NREPEAT=$(TESTSIZE) $< gemm  23  21  32 0 1 -1 0.5  32  32 1000
	@NREPEAT=$(TESTSIZE) $< gemm 200 200 256 1 1  1 0.0 256 256 1000

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
	@-rm -f $(XFILES)
endif
	@-rm -f $(OUTDIR)/libwrap.*

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
