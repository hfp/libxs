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

ifneq (Darwin,$(UNAME))
  ifneq (0,$(BLAS_STATIC))
    WRAP_GEMM := -Wl,--wrap=dgemm_
  endif
else # macOS
  $(info ==========================================================)
  $(info The static link-time wrapper is not supported under macOS!)
  $(info ==========================================================)
endif

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(SRCDIR)/*.h) $(DEPDIR)/include/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/gemm.c $(SRCDIR)/$(OUTNAME)-gemm.c)
WRPSRCS := $(wildcard $(SRCDIR)/$(OUTNAME).c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
WRPOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(WRPSRCS:.c=-c.o)))
OBJECTS := $(COBJCTS) $(WRPOBJS)
XFILES := $(OUTDIR)/gemm-blas.x $(OUTDIR)/gemm-wrap.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

$(BLDDIR)/$(OUTNAME)-c.o: $(SRCDIR)/$(OUTNAME).c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) -DLIBXS_BUILD $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(OUTDIR)/lib$(OUTNAME).$(DLIBEXT): $(OUTDIR)/.make $(WRPOBJS) $(BLDDIR)/$(OUTNAME)-gemm-c.o
	$(LIB_SOLD) $(call solink,$(OUTDIR)/lib$(OUTNAME).$(DLIBEXT)) \
		$(call tailwords,$^) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))

$(OUTDIR)/gemm-blas.x: $(BLDDIR)/gemm-c.o $(OUTDIR)/lib$(OUTNAME).$(DLIBEXT) $(XSDEP) $(OUTDIR)/.make
	$(LD) -o $@ $< $(XSLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

ifneq (,$(WRAP_GEMM))
$(OUTDIR)/lib$(OUTNAME).$(SLIBEXT): $(OUTDIR)/.make $(BLDDIR)/$(OUTNAME)-gemm-c.o
	$(MAKE_AR) $(OUTDIR)/lib$(OUTNAME).$(SLIBEXT) $(call tailwords,$^)

$(OUTDIR)/gemm-wrap.x: $(BLDDIR)/gemm-c.o $(OUTDIR)/lib$(OUTNAME).$(SLIBEXT) $(XSDEP) $(OUTDIR)/.make
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(XSLIB) $(OUTDIR)/lib$(OUTNAME).$(SLIBEXT) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(XSLIB) $(OUTDIR)/lib$(OUTNAME).$(SLIBEXT) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(XSLIB) $(OUTDIR)/lib$(OUTNAME).$(SLIBEXT) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
endif
else
.PHONY: $(OUTDIR)/gemm-wrap.x
endif

.PHONY: test
test: $(ROOTDIR)/$(OUTNAME)-test.sh $(XFILES)
	@$(ROOTDIR)/$(OUTNAME)-test.sh gemm                                $(shell echo $$(($(TESTSIZE) * 10)))
	@$(ROOTDIR)/$(OUTNAME)-test.sh gemm 350  16  20 350  35 350  1 0.0 $(shell echo $$(($(TESTSIZE) * 10)))
	@$(ROOTDIR)/$(OUTNAME)-test.sh gemm 200 200 200 256 256 256  1 0.0 $(shell echo $$(($(TESTSIZE) * 10)))
	@$(ROOTDIR)/$(OUTNAME)-test.sh gemm  24  23  21  32  32  32 -1 0.5 $(shell echo $$(($(TESTSIZE) * 10)))

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
	@-rm -f $(OUTDIR)/lib$(OUTNAME).*

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
