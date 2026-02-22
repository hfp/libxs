ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

# Set GEMM_PRECISION=float for single-precision (default: double)
GEMM_PRECISION ?= double

OMP = 1
SYM = 1

BLAS = $(if $(filter-out 0,$(OMP)),2,1)
#BLAS_STATIC = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR)) -I$(call quote,$(DEPDIR)/include)

# Derive GEMM symbol and compiler flags from GEMM_PRECISION
ifneq (double,$(GEMM_PRECISION))
  GEMM_SYMBOL := sgemm_
  ZGEMM_SYMBOL := cgemm_
  DFLAGS += -DGEMM_REAL_TYPE=float
else
  GEMM_SYMBOL := dgemm_
  ZGEMM_SYMBOL := zgemm_
endif

ifneq (Darwin,$(UNAME))
  ifneq (0,$(BLAS_STATIC))
    WRAP_GEMM := -Wl,--wrap=$(GEMM_SYMBOL) -Wl,--wrap=$(ZGEMM_SYMBOL)
  endif
else # macOS
  $(info ==========================================================)
  $(info The static link-time wrapper is not supported under macOS!)
  $(info ==========================================================)
endif

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(SRCDIR)/*.h) $(DEPDIR)/include/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/gemm.c $(SRCDIR)/gemm-print.c)
WRPSRCS := $(wildcard $(SRCDIR)/wrap.c $(SRCDIR)/zgemm3m.c $(SRCDIR)/$(OUTNAME).c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
WRPOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(WRPSRCS:.c=-c.o)))
OBJECTS := $(COBJCTS) $(WRPOBJS)
XFILES := $(OUTDIR)/gemm-blas.x $(OUTDIR)/gemm-wrap.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

ifneq (,$(WRAP_GEMM))
$(OUTDIR)/libwrap.$(SLIBEXT): $(OUTDIR)/.make $(BLDDIR)/$(OUTNAME)-c.o $(BLDDIR)/gemm-print-c.o
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
$(OUTDIR)/libwrap.$(DLIBEXT): $(WRPOBJS) $(BLDDIR)/gemm-print-c.o $(XSDEP) $(OUTDIR)/.make
	$(LIB_SOLD) $(call solink,$@) $(WRPOBJS) $(BLDDIR)/gemm-print-c.o \
		$(XSLIB) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))

$(OUTDIR)/gemm-blas.x: $(COBJCTS) $(XSDEP) $(OUTDIR)/.make $(OUTDIR)/libwrap.$(DLIBEXT)
	$(LD) -o $@ $(COBJCTS) $(XSLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

.PHONY: $(OUTDIR)/libwrap.$(SLIBEXT)
.PHONY: $(OUTDIR)/gemm-wrap.x
endif

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

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
