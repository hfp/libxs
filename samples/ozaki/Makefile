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
OCL = 2
SYM = 1

BLAS = $(if $(filter-out 0,$(OMP)),2,1)
#BLAS_STATIC = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# root directory of LIBXSTREAM
LIBXSTREAMROOT := $(wildcard $(DEPDIR)/../libxstream)
LIBXSTREAM_SL := $(wildcard $(LIBXSTREAMROOT)/lib/libxstream.$(SLIBEXT))
LIBXSTREAM_DL := $(wildcard $(LIBXSTREAMROOT)/lib/libxstream.$(DLIBEXT))
LIBXSTREAM := $(wildcard $(LIBXSTREAMROOT)/lib/libxstream.$(LIBEXT))
LIBXSTREAM := $(if $(LIBXSTREAM),$(LIBXSTREAM), \
  $(if $(LIBXSTREAM_SL),$(LIBXSTREAM_SL),$(LIBXSTREAM_DL)))
ifneq (,$(LIBXSTREAM))
  IFLAGS += -I$(call quote,$(LIBXSTREAMROOT)/include)
  IFLAGS += -I$(call quote,$(LIBXSTREAMROOT)/samples/ozaki)
  DFLAGS += -D__LIBXSTREAM
  GPUOBJS := $(BLDDIR)/ozaki_gpu-c.o $(BLDDIR)/ozaki_opencl-c.o
  SCRDIR := $(LIBXSTREAMROOT)/scripts
  GPUKERNELS := $(wildcard $(LIBXSTREAMROOT)/samples/ozaki/kernels/*.cl)
  GPUGENHDR := $(LIBXSTREAMROOT)/samples/ozaki/ozaki_kernels.h
endif

# include directories
IFLAGS += -I$(call quote,$(DEPDIR)/include)
IFLAGS += -I$(call quote,$(INCDIR))

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
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(SRCDIR)/*.h)
# Driver source (compiled once with default double precision)
CSOURCS := $(SRCDIR)/gemm.c
# Implementation sources: _WRAP + support (compiled for each precision)
IMPLSRCS := $(SRCDIR)/$(OUTNAME).c $(SRCDIR)/ozaki1_int8.c $(SRCDIR)/ozaki2_int8.c $(SRCDIR)/ozaki1_bf16.c $(SRCDIR)/ozaki2_bf16.c $(SRCDIR)/zgemm3m.c $(SRCDIR)/gemm-print.c
# Entry/dlsym source: _REAL + entry points (LD_PRELOAD path only)
ENTRYSRCS := $(SRCDIR)/wrap.c
COBJCTS := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-c.o,$(CSOURCS))
IMPLOBJ_D := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-d.o,$(IMPLSRCS))
IMPLOBJ_S := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-s.o,$(IMPLSRCS))
IMPLOBJS := $(IMPLOBJ_D) $(IMPLOBJ_S)
ENTRYOBJ_D := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-d.o,$(ENTRYSRCS))
ENTRYOBJ_S := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-s.o,$(ENTRYSRCS))
ENTRYOBJS := $(ENTRYOBJ_D) $(ENTRYOBJ_S)
OBJECTS := $(COBJCTS) $(IMPLOBJS) $(ENTRYOBJS) $(GPUOBJS)
XFILES := $(OUTDIR)/gemm-blas.x $(OUTDIR)/gemm-wrap.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

ifneq (,$(WRAP_GEMM))
$(OUTDIR)/libwrap.$(SLIBEXT): $(OUTDIR)/.make $(IMPLOBJS) $(GPUOBJS)
	$(MAKE_AR) $@ $(call tailwords,$^)

$(OUTDIR)/gemm-wrap.x: $(BLDDIR)/gemm-c.o $(OUTDIR)/libwrap.$(SLIBEXT) $(XLIB) $(LIBXSTREAM) $(OUTDIR)/.make
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(OUTDIR)/libwrap.$(SLIBEXT) $(LIBXSTREAM) $(XLIB) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(OUTDIR)/libwrap.$(SLIBEXT) $(LIBXSTREAM) $(XLIB) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(OUTDIR)/libwrap.$(SLIBEXT) $(LIBXSTREAM) $(XLIB) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
endif

.PHONY: $(OUTDIR)/libwrap.$(DLIBEXT)
.PHONY: $(OUTDIR)/gemm-blas.x
else
$(OUTDIR)/libwrap.$(DLIBEXT): $(IMPLOBJS) $(ENTRYOBJS) $(GPUOBJS) $(XLIB) $(LIBXSTREAM) $(OUTDIR)/.make
	$(LIB_SOLD) $(call solink,$@) $(IMPLOBJS) $(ENTRYOBJS) $(GPUOBJS) \
		$(XLIB) $(LIBXSTREAM) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))

$(OUTDIR)/gemm-blas.x: $(COBJCTS) $(BLDDIR)/gemm-print-d.o $(OUTDIR)/libwrap.$(DLIBEXT) $(XLIB) $(LIBXSTREAM) $(OUTDIR)/.make
	$(LD) -o $@ $(COBJCTS) $(BLDDIR)/gemm-print-d.o $(XLIB) $(LIBXSTREAM) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

.PHONY: $(OUTDIR)/libwrap.$(SLIBEXT)
.PHONY: $(OUTDIR)/gemm-wrap.x
endif

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

# GPU sources need __OPENCL for LIBXSTREAM OpenCL headers
ifneq (,$(GPUOBJS))
# Generate embedded kernel header from .cl sources
$(GPUGENHDR): $(GPUKERNELS) $(SCRDIR)/acc_opencl.sh
	$(SCRDIR)/acc_opencl.sh -p "" $(GPUKERNELS) $@

$(BLDDIR)/ozaki_gpu-c.o: $(SRCDIR)/ozaki_gpu.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) -D__OPENCL $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/ozaki_opencl-c.o: $(LIBXSTREAMROOT)/samples/ozaki/ozaki_opencl.c $(GPUGENHDR) .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) -D__OPENCL $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@
endif

$(BLDDIR)/%-d.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-s.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) -DGEMM_REAL_TYPE=float $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: test
test: test-wrap test-check

.PHONY: test-wrap
test-wrap: $(ROOTDIR)/test-wrap.sh $(XFILES)
	@NREPEAT=$(TESTSIZE) $< gemm                               
	@NREPEAT=$(TESTSIZE) $< gemm  16  20 350 1 0  1 0.0 350 350 1000
	@NREPEAT=$(TESTSIZE) $< gemm  23  21  32 0 1 -1 0.5  32  32 1000
	@NREPEAT=$(TESTSIZE) $< gemm 200 200 256 1 1  1 0.0 256 256 1000

.PHONY: test-check
test-check: $(ROOTDIR)/test-check.sh $(XFILES)
	@$< gemm                               
	@$< gemm  16  20 350 1 0  1 0.0 350 350 1000
	@$< gemm  23  21  32 0 1 -1 0.5  32  32 1000
	@$< gemm 200 200 256 1 1  1 0.0 256 256 1000

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
	@-rm -f $(GPUGENHDR)

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
