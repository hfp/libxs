ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

DNNL = 0
OMP = 1
OCL = 2
SYM = 1

BLAS = $(if $(filter-out 0,$(OMP)),2,1)
#BLAS_STATIC = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# root directory of LIBXSTREAM
ifneq (0,$(OCL))
ifneq (0,$(OPENCL))
ifneq (0,$(LIBXSTREAM))
  LIBXSTREAMROOT := $(wildcard $(DEPDIR)/../libxstream)
  XSTREAM_SLIB := $(wildcard $(LIBXSTREAMROOT)/lib/libxstream.$(SLIBEXT))
  XSTREAM_DLIB := $(wildcard $(LIBXSTREAMROOT)/lib/libxstream.$(DLIBEXT))
  XSTREAM_LIB := $(wildcard $(LIBXSTREAMROOT)/lib/libxstream.$(LIBEXT))
  XSTREAM_LIB := $(strip $(if $(XSTREAM_LIB),$(XSTREAM_LIB), \
    $(if $(XSTREAM_SLIB),$(XSTREAM_SLIB),$(XSTREAM_DLIB))))
  ifneq (,$(XSTREAM_LIB))
    IFLAGS += -I$(call quote,$(LIBXSTREAMROOT)/include)
    IFLAGS += -I$(call quote,$(LIBXSTREAMROOT)/samples/ozaki)
    DFLAGS += -D__LIBXSTREAM
    OCLOBJS := $(BLDDIR)/ozaki_ocl-c.o $(BLDDIR)/ozaki_opencl-c.o $(BLDDIR)/ozaki_gemm-c.o
    SCRDIR := $(LIBXSTREAMROOT)/scripts
    OCLKERNELS := $(wildcard $(LIBXSTREAMROOT)/samples/ozaki/kernels/*.cl)
    OCLGENHDR := $(LIBXSTREAMROOT)/samples/ozaki/ozaki_kernels.h
    OCLGENHDR_PROD := $(LIBXSTREAMROOT)/samples/ozaki/ozaki_tinytc.h
    OCLGENSCR_PROD := $(LIBXSTREAMROOT)/samples/ozaki/kernels/ozaki1.sh
  endif
endif
endif
endif

# oneDNN (DNNL) for int8 matmul acceleration
ifneq (0,$(DNNL))
  DNNLROOT ?= $(shell echo "$$DNNLROOT")
  ifneq (,$(wildcard $(DNNLROOT)/include/oneapi/dnnl/dnnl.h))
    IFLAGS += -I$(call quote,$(DNNLROOT)/include)
    DFLAGS += -D__DNNL
    LDFLAGS += -Wl,--rpath=$(DNNLROOT)/lib -L$(DNNLROOT)/lib -ldnnl
  endif
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
# Driver source (compiled for each precision)
CSOURCS := $(SRCDIR)/gemm.c
# Implementation sources: _WRAP + support (compiled for each precision)
IMPLSRCS := $(SRCDIR)/$(OUTNAME).c $(SRCDIR)/ozaki1_int8.c $(SRCDIR)/ozaki2_int8.c $(SRCDIR)/zgemm3m.c $(SRCDIR)/gemm-print.c
# Entry/dlsym source: _REAL + entry points (LD_PRELOAD path only)
ENTRYSRCS := $(SRCDIR)/wrap.c
COBJCTS_D := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-dc.o,$(CSOURCS))
COBJCTS_S := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-sc.o,$(CSOURCS))
IMPLOBJ_D := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-d.o,$(IMPLSRCS))
IMPLOBJ_S := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-s.o,$(IMPLSRCS))
IMPLOBJS := $(IMPLOBJ_D) $(IMPLOBJ_S)
ENTRYOBJ_D := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-d.o,$(ENTRYSRCS))
ENTRYOBJ_S := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%-s.o,$(ENTRYSRCS))
ENTRYOBJS := $(ENTRYOBJ_D) $(ENTRYOBJ_S)
OBJECTS := $(COBJCTS_D) $(COBJCTS_S) $(IMPLOBJS) $(ENTRYOBJS) $(OCLOBJS)
XFILES := $(OUTDIR)/dgemm-blas.x $(OUTDIR)/dgemm-wrap.x \
          $(OUTDIR)/sgemm-blas.x $(OUTDIR)/sgemm-wrap.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

# Link recipe for static-wrap executables (handles OMP variants)
ifneq (0,$(OMP))
  define LINK_WRAP
	$(LD) -o $@ $< $(OUTDIR)/libwrap.$(SLIBEXT) $(XSTREAM_LIB) $(XLIB) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
  endef
else ifneq (,$(strip $(OMPLIB)))
  define LINK_WRAP
	$(LD) -o $@ $< $(OUTDIR)/libwrap.$(SLIBEXT) $(XSTREAM_LIB) $(XLIB) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
  endef
else
  define LINK_WRAP
	$(LD) -o $@ $< $(OUTDIR)/libwrap.$(SLIBEXT) $(XSTREAM_LIB) $(XLIB) \
		$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
  endef
endif

ifneq (,$(WRAP_GEMM))
$(OUTDIR)/libwrap.$(SLIBEXT): $(OUTDIR)/.make $(IMPLOBJS) $(OCLOBJS)
	$(MAKE_AR) $@ $(call tailwords,$^)

$(OUTDIR)/dgemm-wrap.x: $(BLDDIR)/gemm-dc.o $(OUTDIR)/libwrap.$(SLIBEXT) $(XLIB) $(XSTREAM_LIB) $(OUTDIR)/.make
	$(LINK_WRAP)

$(OUTDIR)/sgemm-wrap.x: $(BLDDIR)/gemm-sc.o $(OUTDIR)/libwrap.$(SLIBEXT) $(XLIB) $(XSTREAM_LIB) $(OUTDIR)/.make
	$(LINK_WRAP)

.PHONY: $(OUTDIR)/libwrap.$(DLIBEXT)
.PHONY: $(OUTDIR)/dgemm-blas.x $(OUTDIR)/sgemm-blas.x
else
$(OUTDIR)/libwrap.$(DLIBEXT): $(IMPLOBJS) $(ENTRYOBJS) $(OCLOBJS) $(XLIB) $(XSTREAM_LIB) $(OUTDIR)/.make
	$(LIB_SOLD) $(call solink,$@) $(IMPLOBJS) $(ENTRYOBJS) $(OCLOBJS) \
		$(XLIB) $(XSTREAM_LIB) $(XLIB) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))

$(OUTDIR)/dgemm-blas.x: $(BLDDIR)/gemm-dc.o $(BLDDIR)/gemm-print-d.o $(OUTDIR)/libwrap.$(DLIBEXT) $(XLIB) $(XSTREAM_LIB) $(OUTDIR)/.make
	$(LD) -o $@ $(BLDDIR)/gemm-dc.o $(BLDDIR)/gemm-print-d.o $(XLIB) $(XSTREAM_LIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/sgemm-blas.x: $(BLDDIR)/gemm-sc.o $(BLDDIR)/gemm-print-s.o $(OUTDIR)/libwrap.$(DLIBEXT) $(XLIB) $(XSTREAM_LIB) $(OUTDIR)/.make
	$(LD) -o $@ $(BLDDIR)/gemm-sc.o $(BLDDIR)/gemm-print-s.o $(XLIB) $(XSTREAM_LIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

.PHONY: $(OUTDIR)/libwrap.$(SLIBEXT)
.PHONY: $(OUTDIR)/dgemm-wrap.x $(OUTDIR)/sgemm-wrap.x
endif

$(BLDDIR)/%-dc.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-sc.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) -DGEMM_REAL_TYPE=float $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

# OpenCL sources need __OPENCL for XSTREAM_LIB OpenCL headers
ifneq (,$(OCLOBJS))
# Generate embedded kernel header from .cl sources
$(OCLGENHDR): $(OCLKERNELS) $(SCRDIR)/tool_opencl.sh
	$(SCRDIR)/tool_opencl.sh -p "" $(OCLKERNELS) $@

$(OCLGENHDR_PROD): $(wildcard $(LIBXSTREAMROOT)/samples/ozaki/kernels/ozaki1_*_n*t*_tri.clx \
                              $(LIBXSTREAMROOT)/samples/ozaki/kernels/ozaki1_*_n*t*_sq.clx) \
                   $(OCLGENSCR_PROD)
	$(OCLGENSCR_PROD) -h $@

$(BLDDIR)/ozaki_ocl-c.o: $(SRCDIR)/ozaki_ocl.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) -D__OPENCL $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/ozaki_opencl-c.o: $(LIBXSTREAMROOT)/samples/ozaki/ozaki_opencl.c \
    $(OCLGENHDR) $(OCLGENHDR_PROD) .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) -D__OPENCL $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/ozaki_gemm-c.o: $(LIBXSTREAMROOT)/samples/ozaki/ozaki_gemm.c \
    $(OCLGENHDR) $(OCLGENHDR_PROD) .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
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
	@NREPEAT=$(TESTSIZE) $<
	@NREPEAT=$(TESTSIZE) $< ""  16  20 350 1 0  1 0.0 350 350 1000
	@NREPEAT=$(TESTSIZE) $< ""  23  21  32 0 1 -1 0.5  32  32 1000
	@NREPEAT=$(TESTSIZE) $< "" 200 200 256 1 1  1 0.0 256 256 1000

.PHONY: test-check
test-check: $(ROOTDIR)/test-check.sh $(XFILES)
	@$<
	@$< ""  16  20 350 1 0  1 0.0 350 350 1000
	@$< ""  23  21  32 0 1 -1 0.5  32  32 1000
	@$< "" 200 200 256 1 1  1 0.0 256 256 1000

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
	@-rm -f $(OCLGENHDR) $(OCLGENHDR_PROD)

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
