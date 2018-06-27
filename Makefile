# Export all variables to sub-make processes.
#.EXPORT_ALL_VARIABLES: #export

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explicitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(strip $(MAKE_PARALLEL)))
ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif
endif

ROOTDIR = $(abspath $(subst //,$(NULL),$(dir $(firstword $(MAKEFILE_LIST)))/))
SPLDIR = $(ROOTDIR)/samples
SCRDIR = $(ROOTDIR)/scripts
TSTDIR = $(ROOTDIR)/tests
SRCDIR = $(ROOTDIR)/src
INCDIR = include
BLDDIR = build
OUTDIR = lib
BINDIR = bin
DOCDIR = documentation

# subdirectories for prefix based installation
PINCDIR = $(INCDIR)
POUTDIR = $(OUTDIR)
PBINDIR = $(BINDIR)
PTSTDIR = tests
PDOCDIR = share/libxs
LICFILE = LICENSE.md

# initial default flags: RPM_OPT_FLAGS are usually NULL
CFLAGS = $(RPM_OPT_FLAGS)
CXXFLAGS := $(CFLAGS)
FCFLAGS := $(CFLAGS)
DFLAGS = -DLIBXS_BUILD
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(SRCDIR)

# THRESHOLD problem size (M x N x K) determining when to use BLAS
# A value of zero (0) populates a default threshold
THRESHOLD ?= 0

# Generates M,N,K-combinations for each comma separated group e.g., "1, 2, 3" generates (1,1,1), (2,2,2),
# and (3,3,3). This way a heterogeneous set can be generated e.g., "1 2, 3" generates (1,1,1), (1,1,2),
# (1,2,1), (1,2,2), (2,1,1), (2,1,2) (2,2,1) out of the first group, and a (3,3,3) for the second group
# To generate a series of square matrices one can specify e.g., make MNK=$(echo $(seq -s, 1 5))
# Alternative to MNK, index sets can be specified separately according to a loop nest relationship
# (M(N(K))) using M, N, and K separately. Please consult the documentation for further details.
MNK ?= 0

# Enable thread-local cache of recently dispatched kernels either
# 0: "disable", 1: "enable", or small power-of-two number.
CACHE ?= 1

# Issue software prefetch instructions (see end of section
# https://github.com/hfp/libxs/#generator-driver)
# Use the enumerator 1...16, or the exact strategy
# name pfsigonly...AL1_BL1_CL1.
#  1: auto-select
#  2: pfsigonly
#  3: BL2viaC
#  4: curAL2
#  7: curAL2_BL2viaC
#  5: AL2
#  6: AL2_BL2viaC
#  8: AL2jpst
#  9: AL2jpst_BL2viaC
# 10: AL1
# 11: BL1
# 12: CL1
# 13: AL1_BL1
# 14: BL1_CL1
# 15: AL1_CL1
# 16: AL1_BL1_CL1
PREFETCH ?= 1

# Preferred precision when registering statically generated code versions
# 0: SP and DP code versions to be registered
# 1: SP only
# 2: DP only
PRECISION ?= 0

# Specify the size of a cacheline (Bytes)
CACHELINE ?= 64

# Alpha argument of GEMM
# Supported: 1.0
ALPHA ?= 1
ifneq (1,$(ALPHA))
  $(error ALPHA needs to be 1)
endif

# Beta argument of GEMM
# Supported: 0.0, 1.0
# 0: C  = A * B
# 1: C += A * B
BETA ?= 1
ifneq (1,$(BETA))
ifneq (0,$(BETA))
  $(error BETA needs to be either 0 or 1)
endif
endif

# Determines if the library is thread-safe
THREADS ?= 1

# 0: shared libraries files suitable for dynamic linkage
# 1: library archives suitable for static linkage
STATIC ?= 1

# 0: build according to the value of STATIC
# 1: build according to STATIC=0 and STATIC=1
SHARED ?= 0

# Determines if the library can act as a wrapper-library (GEMM)
# 1: enables wrapping SGEMM/DGEMM, and GEMV (depends on "GEMM")
# 2: enables wrapping DGEMM only (DGEMV-wrap depends on "GEMM")
WRAP ?= 0

# Determines the kind of routine called for intercepted GEMMs
# odd: sequential and non-tiled (small problem sizes only)
# even: parallelized and tiled (all problem sizes)
# 3: GEMV is intercepted; small problem sizes
# 4: GEMV is intercepted; all problem sizes
GEMM ?= 1

# JIT backend is enabled by default
JIT ?= 1

# TRACE facility
INSTRUMENT ?= $(TRACE)

# target library for a broad range of systems
ifneq (0,$(JIT))
  SSE ?= 1
endif

# Profiling JIT code using Linux Perf
# PERF=0: disabled (default)
# PERF=1: enabled (without JITDUMP)
# PERF=2: enabled (with JITDUMP)
#
# Additional support for jitdump
# JITDUMP=0: disabled (default)
# JITDUMP=1: enabled
# PERF=2: enabled
#
ifneq (,$(PERF))
ifneq (0,$(PERF))
ifneq (1,$(PERF))
  JITDUMP ?= 1
endif
endif
endif
JITDUMP ?= 0

ifneq (0,$(JITDUMP))
  PERF ?= 1
endif

PERF ?= 0
ifneq (0,$(PERF))
  SYM ?= 1
endif

# OpenMP is disabled by default and LIBXS is
# always agnostic wrt the threading runtime
OMP ?= 0

ifneq (0,$(OMP))
  DFLAGS += -DLIBXS_OMP
endif

ifneq (,$(MKL))
ifneq (0,$(MKL))
  BLAS = $(MKL)
endif
endif

BLAS_WARNING ?= 0
ifeq (0,$(STATIC))
  ifeq (Windows_NT,$(OS)) # !UNAME
    BLAS_WARNING = 1
    BLAS ?= 2
  else ifeq (Darwin,$(shell uname))
    BLAS_WARNING = 1
    BLAS ?= 2
  endif
endif

ifneq (1,$(CACHE))
  DFLAGS += -DLIBXS_CAPACITY_CACHE=$(CACHE)
endif

# disable lazy initialization and rely on ctor attribute
ifeq (0,$(INIT))
  DFLAGS += -DLIBXS_CTOR
endif

# Kind of documentation (internal key)
DOCEXT = pdf

# state to be excluded from tracking the (re-)build state
EXCLUDE_STATE = BLAS_WARNING PREFIX DESTDIR

# include common Makefile artifacts
include $(ROOTDIR)/Makefile.inc

# Version numbers according to interface (version.txt)
VERSION_MAJOR ?= $(shell $(PYTHON) $(SCRDIR)/libxs_utilities.py 1)
VERSION_MINOR ?= $(shell $(PYTHON) $(SCRDIR)/libxs_utilities.py 2)
VERSION_UPDATE ?= $(shell $(PYTHON) $(SCRDIR)/libxs_utilities.py 3)
VERSION_API ?= $(VERSION_MAJOR)
VERSION_RELEASE ?= HEAD
VERSION_PACKAGE ?= 1

# target library for a broad range of systems
ifneq (0,$(JIT))
ifeq (file,$(origin AVX))
  AVX_STATIC = 0
endif
endif
AVX_STATIC ?= $(AVX)

ifeq (1,$(AVX_STATIC))
  GENTARGET = snb
else ifeq (2,$(AVX_STATIC))
  GENTARGET = hsw
else ifeq (3,$(AVX_STATIC))
  ifneq (0,$(MIC))
    ifeq (2,$(MIC))
      GENTARGET = knm
    else
      GENTARGET = knl
    endif
  else
    GENTARGET = skx
  endif
else ifneq (0,$(SSE))
  GENTARGET = wsm
else
  GENTARGET = noarch
endif

ifeq (0,$(STATIC))
  ifneq (Darwin,$(UNAME))
    GENGEMM = @$(ENV) \
      LD_LIBRARY_PATH=$(OUTDIR):$${LD_LIBRARY_PATH} \
      PATH=$(OUTDIR):$${PATH} \
    $(BINDIR)/libxs_gemm_generator
  else # osx
    GENGEMM = @$(ENV) \
      DYLD_LIBRARY_PATH=$(OUTDIR):$${DYLD_LIBRARY_PATH} \
      PATH=$(OUTDIR):$${PATH} \
    $(BINDIR)/libxs_gemm_generator
  endif
else
  GENGEMM = $(BINDIR)/libxs_gemm_generator
endif

INDICES ?= $(shell $(PYTHON) $(SCRDIR)/libxs_utilities.py -1 $(THRESHOLD) $(words $(MNK)) $(MNK) $(words $(M)) $(words $(N)) $(M) $(N) $(K))
NINDICES = $(words $(INDICES))

HEADERS = $(wildcard $(SRCDIR)/template/*.c) $(wildcard $(SRCDIR)/*.h) \
          $(SRCDIR)/libxs_hash.c $(SRCDIR)/libxs_gemm_diff.c \
          $(ROOTDIR)/include/libxs_bgemm.h \
          $(ROOTDIR)/include/libxs_cpuid.h \
          $(ROOTDIR)/include/libxs_dnn.h \
          $(ROOTDIR)/include/libxs_dnn_rnncell.h \
          $(ROOTDIR)/include/libxs_dnn_lstmcell.h \
          $(ROOTDIR)/include/libxs_frontend.h \
          $(ROOTDIR)/include/libxs_fsspmdm.h \
          $(ROOTDIR)/include/libxs_generator.h \
          $(ROOTDIR)/include/libxs_intrinsics_x86.h \
          $(ROOTDIR)/include/libxs_macros.h \
          $(ROOTDIR)/include/libxs_malloc.h \
          $(ROOTDIR)/include/libxs_math.h \
          $(ROOTDIR)/include/libxs_mhd.h \
          $(ROOTDIR)/include/libxs_spmdm.h \
          $(ROOTDIR)/include/libxs_sync.h \
          $(ROOTDIR)/include/libxs_timer.h \
          $(ROOTDIR)/include/libxs_typedefs.h
SRCFILES_LIB = $(patsubst %,$(SRCDIR)/%, \
          libxs_main.c libxs_cpuid_x86.c libxs_malloc.c libxs_math.c libxs_sync.c \
          libxs_python.c libxs_mhd.c libxs_timer.c libxs_perf.c \
          libxs_gemm.c libxs_trans.c libxs_bgemm.c \
          libxs_spmdm.c libxs_fsspmdm.c \
          libxs_dnn.c libxs_dnn_dryruns.c libxs_dnn_setup.c libxs_dnn_handle.c \
          libxs_dnn_elementwise.c libxs_dnn_rnncell.c libxs_dnn_lstmcell.c \
          libxs_dnn_convolution_forward.c \
          libxs_dnn_convolution_backward.c \
          libxs_dnn_convolution_weight_update.c \
          libxs_dnn_convolution_winograd_forward.c \
          libxs_dnn_convolution_winograd_backward.c \
          libxs_dnn_convolution_winograd_weight_update.o )

SRCFILES_KERNELS = $(patsubst %,$(BLDDIR)/mm_%.c,$(INDICES))
SRCFILES_GEN_LIB = $(patsubst %,$(SRCDIR)/%,$(wildcard $(SRCDIR)/generator_*.c) libxs_trace.c libxs_generator.c)
SRCFILES_GEN_GEMM_BIN = $(patsubst %,$(SRCDIR)/%,libxs_generator_gemm_driver.c)
SRCFILES_GEN_CONVWINO_BIN = $(patsubst %,$(SRCDIR)/%,libxs_generator_convolution_winograd_driver.c)
SRCFILES_GEN_CONV_BIN = $(patsubst %,$(SRCDIR)/%,libxs_generator_convolution_driver.c)
OBJFILES_GEN_LIB = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_LIB))))
OBJFILES_GEN_GEMM_BIN = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_GEMM_BIN))))
OBJFILES_GEN_CONVWINO_BIN = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_CONVWINO_BIN))))
OBJFILES_GEN_CONV_BIN = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_CONV_BIN))))
OBJFILES_GEN_LIB = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_LIB))))
OBJFILES_HST = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_LIB))))
OBJFILES_MIC = $(patsubst %,$(BLDDIR)/mic/%.o,$(basename $(notdir $(SRCFILES_LIB))))
KRNOBJS_HST  = $(patsubst %,$(BLDDIR)/intel64/mm_%.o,$(INDICES))
KRNOBJS_MIC  = $(patsubst %,$(BLDDIR)/mic/mm_%.o,$(INDICES))
EXTOBJS_HST  = $(BLDDIR)/intel64/libxs_ext.o \
               $(BLDDIR)/intel64/libxs_ext_trans.o \
               $(BLDDIR)/intel64/libxs_ext_bgemm.o \
               $(BLDDIR)/intel64/libxs_ext_gemm.o
EXTOBJS_MIC  = $(BLDDIR)/mic/libxs_ext.o \
               $(BLDDIR)/mic/libxs_ext_trans.o \
               $(BLDDIR)/mic/libxs_ext_bgemm.o \
               $(BLDDIR)/mic/libxs_ext_gemm.o
NOBLAS_HST   = $(BLDDIR)/intel64/libxs_noblas.o
NOBLAS_MIC   = $(BLDDIR)/mic/libxs_noblas.o

# list of object might be "incomplete" if not all code gen. FLAGS are supplied with clean target!
OBJECTS = $(OBJFILES_GEN_LIB) $(OBJFILES_GEN_GEMM_BIN) $(OBJFILES_GEN_CONV_BIN) $(OBJFILES_GEN_CONVWINO_BIN) $(OBJFILES_HST) $(OBJFILES_MIC) \
          $(KRNOBJS_HST) $(KRNOBJS_MIC) $(EXTOBJS_HST) $(EXTOBJS_MIC) $(NOBLAS_HST) $(NOBLAS_MIC)
ifneq (,$(strip $(FC)))
  FTNOBJS = $(BLDDIR)/intel64/libxs-mod.o $(BLDDIR)/mic/libxs-mod.o
endif

ifneq (0,$(JIT))
ifneq (0,$(SYM))
ifeq (,$(filter Darwin,$(UNAME)))
  ifneq (0,$(PERF))
    DFLAGS += -DLIBXS_PERF
    ifneq (0,$(JITDUMP))
      DFLAGS += -DLIBXS_PERF_JITDUMP
    endif
  endif
  VTUNEROOT = $(shell env | grep VTUNE_AMPLIFIER | grep -m1 _DIR | cut -d= -f2-)
  ifneq (,$(wildcard $(VTUNEROOT)/lib64/libjitprofiling.$(SLIBEXT)))
    LIBJITPROFILING = $(BLDDIR)/jitprofiling/libjitprofiling.$(SLIBEXT)
    OBJJITPROFILING = $(BLDDIR)/jitprofiling/*.o
    DFLAGS += -DLIBXS_VTUNE
    IFLAGS += -I$(VTUNEROOT)/include
    ifneq (0,$(INTEL))
      CXXFLAGS += -diag-disable 271
      CFLAGS += -diag-disable 271
    endif
  endif
endif
endif
endif

information = \
	$(info ================================================================================) \
	$(info LIBXS $(shell $(PYTHON) $(SCRDIR)/libxs_utilities.py) (STATIC=$(STATIC))) \
	$(info --------------------------------------------------------------------------------) \
	$(info $(GINFO)) \
	$(info $(CINFO)) \
	$(if $(strip $(FC)),$(info $(FINFO)),$(NULL)) \
	$(info --------------------------------------------------------------------------------) \
	$(if $(strip $(FC)),$(NULL),$(if $(strip $(FC_VERSION_STRING)), \
	$(info Fortran Compiler $(FC_VERSION_STRING) is outdated!), \
	$(info Fortran Compiler is missing: no Fortran interface is built!)) \
	$(info ================================================================================)) \
	$(if $(filter Windows_NT0,$(UNAME)$(STATIC)), \
	$(info The shared link-time wrapper (libxsext) is not supported under Windows/Cygwin!) \
	$(info ================================================================================), \
	$(NULL)) \
	$(if $(filter _0_,_$(BLAS_WARNING)_),$(NULL), \
	$(info Building a shared library requires to link against BLAS since there is) \
	$(info no runtime resolution/search for weak symbols implemented for this OS.)) \
	$(if $(filter _0_,_$(BLAS)_), \
	$(if $(filter _0_,_$(NOBLAS)_),$(NULL), \
	$(info LIBXS's link-time BLAS dependency is removed (fallback might be unavailable!)) \
	$(info ================================================================================)), \
	$(if $(filter _0_,_$(BLAS_WARNING)_), \
	$(info LIBXS is link-time agnostic with respect to BLAS/GEMM!) \
	$(info Linking a certain BLAS library may prevent users to decide.), \
	$(NULL)) \
	$(if $(filter _1_,_$(BLAS)_), \
	$(info LIBXS's THRESHOLD already prevents calling small GEMMs!) \
	$(info A sequential BLAS is superfluous with respect to LIBXS.), \
	$(NULL)) \
	$(info ================================================================================))

ifneq (,$(strip $(TEST)))
.PHONY: run-tests
run-tests: tests
endif

.PHONY: libxs
ifeq (0,$(COMPATIBLE))
ifeq (0,$(SHARED))
libxs: lib generator
else
libxs: libs generator
endif
else
ifeq (0,$(SHARED))
libxs: lib
else
libxs: libs
endif
endif
	$(information)
ifneq (,$(strip $(LIBJITPROFILING)))
	$(info Intel VTune Amplifier support has been incorporated.)
	$(info ================================================================================)
endif

.PHONY: lib
lib: headers drytest lib_hst lib_mic

.PHONY: libs
libs: lib
ifneq (0,$(STATIC))
	@$(MAKE) --no-print-directory lib STATIC=0
else
	@$(MAKE) --no-print-directory lib STATIC=1
endif

.PHONY: all
all: libxs samples

.PHONY: headers
headers: cheader cheader_only fheader

.PHONY: header-only
header-only: cheader_only

.PHONY: header_only
header_only: header-only

.PHONY: interface
interface: headers module

.PHONY: lib_mic
lib_mic: clib_mic flib_mic ext_mic noblas_mic

.PHONY: lib_hst
lib_hst: clib_hst flib_hst ext_hst noblas_hst

PREFETCH_UID = 0
PREFETCH_TYPE = 0
PREFETCH_SCHEME = nopf
ifneq (Windows_NT,$(UNAME)) # TODO: full support for Windows calling convention
  ifneq (0,$(shell echo $$((0 <= $(PREFETCH) && $(PREFETCH) <= 16))))
    PREFETCH_UID = $(PREFETCH)
  else ifneq (0,$(shell echo $$((0 > $(PREFETCH))))) # auto
    PREFETCH_UID = 1
  else ifeq (pfsigonly,$(PREFETCH))
    PREFETCH_UID = 2
  else ifeq (BL2viaC,$(PREFETCH))
    PREFETCH_UID = 3
  else ifeq (curAL2,$(PREFETCH))
    PREFETCH_UID = 4
  else ifeq (curAL2_BL2viaC,$(PREFETCH))
    PREFETCH_UID = 5
  else ifeq (AL2,$(PREFETCH))
    PREFETCH_UID = 6
  else ifeq (AL2_BL2viaC,$(PREFETCH))
    PREFETCH_UID = 7
  else ifeq (AL2jpst,$(PREFETCH))
    PREFETCH_UID = 8
  else ifeq (AL2jpst_BL2viaC,$(PREFETCH))
    PREFETCH_UID = 9
  else ifeq (AL1,$(PREFETCH))
    PREFETCH_UID = 10
  else ifeq (BL1,$(PREFETCH))
    PREFETCH_UID = 11
  else ifeq (CL1,$(PREFETCH))
    PREFETCH_UID = 12
  else ifeq (AL1_BL1,$(PREFETCH))
    PREFETCH_UID = 13
  else ifeq (BL1_CL1,$(PREFETCH))
    PREFETCH_UID = 14
  else ifeq (AL1_CL1,$(PREFETCH))
    PREFETCH_UID = 15
  else ifeq (AL1_BL1_CL1,$(PREFETCH))
    PREFETCH_UID = 16
  endif

  # Mapping build options to libxs_gemm_prefetch_type (see include/libxs_typedefs.h)
  ifeq (1,$(PREFETCH_UID))
    # Prefetch "auto" is a pseudo-strategy introduced by the frontend;
    # select "nopf" for statically generated code.
    PREFETCH_SCHEME = nopf
    PREFETCH_TYPE = -1
  else ifeq (2,$(PREFETCH_UID))
    PREFETCH_SCHEME = pfsigonly
    PREFETCH_TYPE = 1
  else ifeq (3,$(PREFETCH_UID))
    PREFETCH_SCHEME = BL2viaC
    PREFETCH_TYPE = 8
  else ifeq (4,$(PREFETCH_UID))
    PREFETCH_SCHEME = curAL2
    PREFETCH_TYPE = 16
  else ifeq (5,$(PREFETCH_UID))
    PREFETCH_SCHEME = curAL2_BL2viaC
    PREFETCH_TYPE = $(shell echo $$((8 | 16)))
  else ifeq (6,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL2
    PREFETCH_TYPE = 2
  else ifeq (7,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL2_BL2viaC
    PREFETCH_TYPE = $(shell echo $$((8 | 2)))
  else ifeq (8,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL2jpst
    PREFETCH_TYPE = 4
  else ifeq (9,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL2jpst_BL2viaC
    PREFETCH_TYPE = $(shell echo $$((8 | 4)))
  else ifeq (10,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL1
    PREFETCH_TYPE = 32
  else ifeq (11,$(PREFETCH_UID))
    PREFETCH_SCHEME = BL1
    PREFETCH_TYPE = 64
  else ifeq (12,$(PREFETCH_UID))
    PREFETCH_SCHEME = CL1
    PREFETCH_TYPE = 128
  else ifeq (13,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL1_BL1
    PREFETCH_TYPE = $(shell echo $$((32 | 64)))
  else ifeq (14,$(PREFETCH_UID))
    PREFETCH_SCHEME = BL1_CL1
    PREFETCH_TYPE = $(shell echo $$((64 | 128)))
  else ifeq (15,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL1_CL1
    PREFETCH_TYPE = $(shell echo $$((32 | 128)))
  else ifeq (16,$(PREFETCH_UID))
    PREFETCH_SCHEME = AL1_BL1_CL1
    PREFETCH_TYPE = $(shell echo $$((32 | 64 | 128)))
  endif
endif
ifeq (,$(PREFETCH_SCHEME_MIC)) # adopt host scheme
  PREFETCH_SCHEME_MIC = $(PREFETCH_SCHEME)
endif

# Mapping build options to libxs_gemm_flags (see include/libxs_typedefs.h)
#FLAGS = $(shell echo $$((((0==$(ALPHA))*4) | ((0>$(ALPHA))*8) | ((0==$(BETA))*16) | ((0>$(BETA))*32))))
FLAGS = 0

SUPPRESS_UNUSED_VARIABLE_WARNINGS = LIBXS_UNUSED(A); LIBXS_UNUSED(B); LIBXS_UNUSED(C);
ifneq (nopf,$(PREFETCH_SCHEME))
  #SUPPRESS_UNUSED_VARIABLE_WARNINGS += LIBXS_UNUSED(A_prefetch); LIBXS_UNUSED(B_prefetch);
  #SUPPRESS_UNUSED_PREFETCH_WARNINGS = $(NULL)  LIBXS_UNUSED(C_prefetch);~
  SUPPRESS_UNUSED_PREFETCH_WARNINGS = $(NULL)  LIBXS_UNUSED(A_prefetch); LIBXS_UNUSED(B_prefetch); LIBXS_UNUSED(C_prefetch);~
endif

.PHONY: config
config: $(INCDIR)/libxs_config.h
$(INCDIR)/libxs_config.h: $(INCDIR)/.make .state $(SRCDIR)/template/libxs_config.h \
                            $(SCRDIR)/libxs_config.py $(SCRDIR)/libxs_utilities.py \
                            $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc \
                            $(wildcard $(ROOTDIR)/.github/*) \
                            $(ROOTDIR)/version.txt
	$(information)
	@if [ -e $(ROOTDIR)/.github/install.sh ]; then \
		$(ROOTDIR)/.github/install.sh; \
	fi
	@$(CP) $(ROOTDIR)/include/libxs_bgemm.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_cpuid.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_dnn.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_dnn_rnncell.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_dnn_lstmcell.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_frontend.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_fsspmdm.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_generator.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_intrinsics_x86.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_macros.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_malloc.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_math.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_mhd.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_spmdm.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_sync.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_timer.h $(INCDIR) 2>/dev/null || true
	@$(CP) $(ROOTDIR)/include/libxs_typedefs.h $(INCDIR) 2>/dev/null || true
	@$(PYTHON) $(SCRDIR)/libxs_config.py $(SRCDIR)/template/libxs_config.h \
		$(MAKE_ILP64) $(OFFLOAD) $(CACHELINE) $(PRECISION) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) \
		$(shell echo $$(($(THREADS)+$(OMP)))) \
		$(JIT) $(FLAGS) $(ALPHA) $(BETA) $(GEMM) $(INDICES) > $@

.PHONY: cheader
cheader: $(INCDIR)/libxs.h
$(INCDIR)/libxs.h: $(SCRDIR)/libxs_interface.py \
                     $(SRCDIR)/template/libxs.h \
                     $(INCDIR)/libxs_config.h $(HEADERS)
	@$(PYTHON) $(SCRDIR)/libxs_interface.py $(SRCDIR)/template/libxs.h \
		$(PRECISION) $(PREFETCH_TYPE) $(INDICES) > $@

.PHONY: cheader_only
cheader_only: $(INCDIR)/libxs_source.h
$(INCDIR)/libxs_source.h: $(INCDIR)/.make $(SCRDIR)/libxs_source.sh $(INCDIR)/libxs.h
	@$(SCRDIR)/libxs_source.sh > $@

.PHONY: fheader
fheader: $(INCDIR)/libxs.f
$(INCDIR)/libxs.f: $(SCRDIR)/libxs_interface.py \
                     $(SRCDIR)/template/libxs.f \
                     $(INCDIR)/libxs_config.h
	@$(PYTHON) $(SCRDIR)/libxs_interface.py $(SRCDIR)/template/libxs.f \
		$(PRECISION) $(PREFETCH_TYPE) $(INDICES) | \
	$(PYTHON) $(SCRDIR)/libxs_config.py /dev/stdin \
		$(MAKE_ILP64) $(OFFLOAD) $(CACHELINE) $(PRECISION) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) \
		$(shell echo $$(($(THREADS)+$(OMP)))) \
		$(JIT) $(FLAGS) $(ALPHA) $(BETA) $(GEMM) $(INDICES) | \
	sed "/ATTRIBUTES OFFLOAD:MIC/d" > $@

.PHONY: sources
sources: $(SRCFILES_KERNELS) $(BLDDIR)/libxs_dispatch.h
$(BLDDIR)/libxs_dispatch.h: $(BLDDIR)/.make $(SCRDIR)/libxs_dispatch.py $(SRCFILES_KERNELS) \
                              $(INCDIR)/libxs.h
	@$(PYTHON) $(SCRDIR)/libxs_dispatch.py $(PRECISION) $(THRESHOLD) $(INDICES) > $@

$(BLDDIR)/%.c: $(BLDDIR)/.make $(INCDIR)/libxs.h $(BINDIR)/libxs_gemm_generator $(SCRDIR)/libxs_utilities.py $(SCRDIR)/libxs_specialized.py
ifneq (,$(strip $(SRCFILES_KERNELS)))
	$(eval MVALUE := $(shell echo $(basename $(notdir $@)) | cut -d_ -f2))
	$(eval NVALUE := $(shell echo $(basename $(notdir $@)) | cut -d_ -f3))
	$(eval KVALUE := $(shell echo $(basename $(notdir $@)) | cut -d_ -f4))
	$(eval MNVALUE := $(MVALUE))
	$(eval NMVALUE := $(NVALUE))
	@echo "#include <libxs.h>" > $@
	@echo >> $@
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (2,$(PRECISION))
	@echo "#define LIBXS_GENTARGET_knc_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXS_GENTARGET_knc_dp" >> $@
endif
endif
endif
ifeq (noarch,$(GENTARGET))
ifneq (,$(CTARGET))
ifneq (2,$(PRECISION))
	@echo "#define LIBXS_GENTARGET_knl_sp" >> $@
	@echo "#define LIBXS_GENTARGET_hsw_sp" >> $@
	@echo "#define LIBXS_GENTARGET_snb_sp" >> $@
	@echo "#define LIBXS_GENTARGET_wsm_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXS_GENTARGET_knl_dp" >> $@
	@echo "#define LIBXS_GENTARGET_hsw_dp" >> $@
	@echo "#define LIBXS_GENTARGET_snb_dp" >> $@
	@echo "#define LIBXS_GENTARGET_wsm_dp" >> $@
endif
	@echo >> $@
	@echo >> $@
ifneq (2,$(PRECISION))
	$(GENGEMM) dense $@ libxs_s$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 knl $(PREFETCH_SCHEME) SP
	$(GENGEMM) dense $@ libxs_s$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 hsw $(PREFETCH_SCHEME) SP
	$(GENGEMM) dense $@ libxs_s$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 snb $(PREFETCH_SCHEME) SP
	$(GENGEMM) dense $@ libxs_s$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 wsm $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENGEMM) dense $@ libxs_d$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 knl $(PREFETCH_SCHEME) DP
	$(GENGEMM) dense $@ libxs_d$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 hsw $(PREFETCH_SCHEME) DP
	$(GENGEMM) dense $@ libxs_d$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 snb $(PREFETCH_SCHEME) DP
	$(GENGEMM) dense $@ libxs_d$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 wsm $(PREFETCH_SCHEME) DP
endif
endif # target
else # noarch
ifneq (2,$(PRECISION))
	@echo "#define LIBXS_GENTARGET_$(GENTARGET)_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXS_GENTARGET_$(GENTARGET)_dp" >> $@
endif
	@echo >> $@
	@echo >> $@
ifneq (2,$(PRECISION))
	$(GENGEMM) dense $@ libxs_s$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 $(GENTARGET) $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENGEMM) dense $@ libxs_d$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 $(GENTARGET) $(PREFETCH_SCHEME) DP
endif
endif # noarch
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (2,$(PRECISION))
	$(GENGEMM) dense $@ libxs_s$(basename $(notdir $@))_knc $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 knc $(PREFETCH_SCHEME_MIC) SP
endif
ifneq (1,$(PRECISION))
	$(GENGEMM) dense $@ libxs_d$(basename $(notdir $@))_knc $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) 0 0 knc $(PREFETCH_SCHEME_MIC) DP
endif
endif
endif
	$(eval TMPFILE = $(shell $(MKTEMP) /tmp/.libxs_XXXXXX.mak))
	@cat $@ | sed \
		-e "s/void libxs_/LIBXS_INLINE LIBXS_RETARGETABLE void libxs_/" \
		-e "s/#ifndef NDEBUG/$(SUPPRESS_UNUSED_PREFETCH_WARNINGS)#ifdef LIBXS_NEVER_DEFINED/" \
		-e "s/#pragma message (\".*KERNEL COMPILATION ERROR in: \" __FILE__)/  $(SUPPRESS_UNUSED_VARIABLE_WARNINGS)/" \
		-e "/#error No kernel was compiled, lacking support for current architecture?/d" \
		-e "/#pragma message (\".*KERNEL COMPILATION WARNING: compiling ..* code on ..* or newer architecture: \" __FILE__)/d" \
		| tr "~" "\n" > $(TMPFILE)
	@$(PYTHON) $(SCRDIR)/libxs_specialized.py $(PRECISION) $(MVALUE) $(NVALUE) $(KVALUE) $(PREFETCH_TYPE) >> $(TMPFILE)
	@$(MV) $(TMPFILE) $@
endif

define DEFINE_COMPILE_RULE
$(1): $(2) $(3) $(dir $(1))/.make
	$(CC) $(4) -c $(2) -o $(1)
endef

EXTCFLAGS = -DLIBXS_BUILD_EXT
ifneq (0,$(STATIC))
ifneq (0,$(WRAP))
ifneq (,$(strip $(WRAP)))
  EXTCFLAGS += -DLIBXS_GEMM_WRAP=$(WRAP)
endif
endif
endif

ifeq (0,$(OMP))
ifeq (,$(filter environment% override command%,$(origin OMP)))
  EXTCFLAGS += $(OMPFLAG)
  EXTLDFLAGS += $(OMPFLAG)
endif
endif

ifneq (0,$(MIC))
ifneq (0,$(MPSS))
$(foreach OBJ,$(OBJFILES_MIC),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ), $(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h $(BLDDIR)/libxs_dispatch.h, \
  -mmic $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(KRNOBJS_MIC),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ), $(patsubst %.o,$(BLDDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h, \
  -mmic $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(EXTOBJS_MIC),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ), $(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h, \
  -mmic $(EXTCFLAGS) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(eval $(call DEFINE_COMPILE_RULE,$(NOBLAS_MIC),$(SRCDIR)/libxs_ext.c,$(INCDIR)/libxs.h, \
  -mmic $(NOBLAS_CFLAGS) $(NOBLAS_FLAGS) $(NOBLAS_IFLAGS) $(DNOBLAS)))
endif
endif

$(foreach OBJ,$(OBJFILES_HST),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h $(BLDDIR)/libxs_dispatch.h, \
  $(CTARGET) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(KRNOBJS_HST),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(BLDDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h, \
  $(CTARGET) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(EXTOBJS_HST),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h, \
  $(CTARGET) $(EXTCFLAGS) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(OBJFILES_GEN_LIB),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h, \
  $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(OBJFILES_GEN_GEMM_BIN),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h, \
  $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(OBJFILES_GEN_CONVWINO_BIN),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h, \
  $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(OBJFILES_GEN_CONV_BIN),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h, \
  $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(eval $(call DEFINE_COMPILE_RULE,$(NOBLAS_HST),$(SRCDIR)/libxs_ext.c,$(INCDIR)/libxs.h, \
  $(CTARGET) $(NOBLAS_CFLAGS) $(NOBLAS_FLAGS) $(NOBLAS_IFLAGS) $(DNOBLAS)))

.PHONY: compile_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
compile_mic:
$(BLDDIR)/mic/%.o: $(BLDDIR)/%.c $(BLDDIR)/mic/.make $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h $(BLDDIR)/libxs_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
endif
endif

.PHONY: compile_hst
compile_hst:
$(BLDDIR)/intel64/%.o: $(BLDDIR)/%.c $(BLDDIR)/intel64/.make $(INCDIR)/libxs.h $(INCDIR)/libxs_source.h $(BLDDIR)/libxs_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

.PHONY: module_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (,$(strip $(FC)))
module_mic: $(INCDIR)/mic/libxs.mod
$(BLDDIR)/mic/libxs-mod.o: $(BLDDIR)/mic/.make $(INCDIR)/mic/.make $(INCDIR)/libxs.f
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $(INCDIR)/libxs.f -o $@ $(FMFLAGS) $(INCDIR)/mic
$(INCDIR)/mic/libxs.mod: $(BLDDIR)/mic/libxs-mod.o
	@if [ -e $(BLDDIR)/mic/LIBXS.mod ]; then $(CP) $(BLDDIR)/mic/LIBXS.mod $(INCDIR); fi
	@if [ -e $(BLDDIR)/mic/libxs.mod ]; then $(CP) $(BLDDIR)/mic/libxs.mod $(INCDIR); fi
	@if [ -e LIBXS.mod ]; then $(MV) LIBXS.mod $(INCDIR); fi
	@if [ -e libxs.mod ]; then $(MV) libxs.mod $(INCDIR); fi
	@touch $@
else
.PHONY: $(BLDDIR)/mic/libxs-mod.o
.PHONY: $(INCDIR)/mic/libxs.mod
endif
else
.PHONY: $(BLDDIR)/mic/libxs-mod.o
.PHONY: $(INCDIR)/mic/libxs.mod
endif
else
.PHONY: $(BLDDIR)/mic/libxs-mod.o
.PHONY: $(INCDIR)/mic/libxs.mod
endif

.PHONY: module_hst
ifneq (,$(strip $(FC)))
module_hst: $(INCDIR)/libxs.mod
$(BLDDIR)/intel64/libxs-mod.o: $(BLDDIR)/intel64/.make $(INCDIR)/libxs.f
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $(INCDIR)/libxs.f -o $@ $(FMFLAGS) $(INCDIR)
$(INCDIR)/libxs.mod: $(BLDDIR)/intel64/libxs-mod.o
	@if [ -e $(BLDDIR)/intel64/LIBXS.mod ]; then $(CP) $(BLDDIR)/intel64/LIBXS.mod $(INCDIR); fi
	@if [ -e $(BLDDIR)/intel64/libxs.mod ]; then $(CP) $(BLDDIR)/intel64/libxs.mod $(INCDIR); fi
	@if [ -e LIBXS.mod ]; then $(MV) LIBXS.mod $(INCDIR); fi
	@if [ -e libxs.mod ]; then $(MV) libxs.mod $(INCDIR); fi
	@touch $@
else
.PHONY: $(BLDDIR)/intel64/libxs-mod.o
.PHONY: $(INCDIR)/libxs.mod
endif

.PHONY: module
module: module_hst module_mic

.PHONY: build_generator_lib
build_generator_lib: $(OUTDIR)/libxsgen.$(LIBEXT)
$(OUTDIR)/libxsgen.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_GEN_LIB)
ifeq (0,$(STATIC))
	$(LIB_LD) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) $(OBJFILES_GEN_LIB) $(LDFLAGS) $(CLDFLAGS) $(LIBRT)
else # static
	$(AR) -rs $@ $(OBJFILES_GEN_LIB)
endif

.PHONY: generator
generator: $(BINDIR)/libxs_gemm_generator $(BINDIR)/libxs_conv_generator $(BINDIR)/libxs_convwino_generator
$(BINDIR)/libxs_gemm_generator: $(BINDIR)/.make $(OBJFILES_GEN_GEMM_BIN) $(OUTDIR)/libxsgen.$(LIBEXT)
	$(LD) -o $@ $(OBJFILES_GEN_GEMM_BIN) $(call abslib,$(OUTDIR)/libxsgen.$(IMPEXT)) $(LDFLAGS) $(CLDFLAGS)
$(BINDIR)/libxs_conv_generator: $(BINDIR)/.make $(OBJFILES_GEN_CONV_BIN) $(OUTDIR)/libxsgen.$(LIBEXT)
	$(LD) -o $@ $(OBJFILES_GEN_CONV_BIN) $(call abslib,$(OUTDIR)/libxsgen.$(IMPEXT)) $(LDFLAGS) $(CLDFLAGS)
$(BINDIR)/libxs_convwino_generator: $(BINDIR)/.make $(OBJFILES_GEN_CONVWINO_BIN) $(OUTDIR)/libxsgen.$(LIBEXT)
	$(LD) -o $@ $(OBJFILES_GEN_CONVWINO_BIN) $(call abslib,$(OUTDIR)/libxsgen.$(IMPEXT)) $(LDFLAGS) $(CLDFLAGS)

ifneq (,$(strip $(LIBJITPROFILING)))
$(LIBJITPROFILING): $(BLDDIR)/jitprofiling/.make
	@$(CP) $(VTUNEROOT)/lib64/libjitprofiling.$(SLIBEXT) $(BLDDIR)/jitprofiling
	@cd $(BLDDIR)/jitprofiling; $(AR) x libjitprofiling.$(SLIBEXT)
endif

.PHONY: clib_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
clib_mic: $(OUTDIR)/mic/libxs.$(LIBEXT)
$(OUTDIR)/mic/libxs.$(LIBEXT): $(OUTDIR)/mic/.make $(OBJFILES_MIC) $(KRNOBJS_MIC)
ifeq (0,$(STATIC))
	$(LIB_LD) -mmic $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) $(OBJFILES_MIC) $(KRNOBJS_MIC) $(LDFLAGS) $(CLDFLAGS)
else # static
	$(AR) -rs $@ $(OBJFILES_MIC) $(KRNOBJS_MIC)
endif
endif
endif

.PHONY: clib_hst
clib_hst: $(OUTDIR)/libxs.$(LIBEXT)
$(OUTDIR)/libxs.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_HST) $(OBJFILES_GEN_LIB) $(KRNOBJS_HST) $(LIBJITPROFILING)
ifeq (0,$(STATIC))
	$(LIB_LD) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) $(OBJFILES_HST) $(OBJFILES_GEN_LIB) $(KRNOBJS_HST) $(LIBJITPROFILING) $(LDFLAGS) $(CLDFLAGS)
else # static
	$(AR) -rs $@ $(OBJFILES_HST) $(OBJFILES_GEN_LIB) $(KRNOBJS_HST) $(OBJJITPROFILING)
endif

.PHONY: flib_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (,$(strip $(FC)))
flib_mic: $(OUTDIR)/mic/libxsf.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/mic/libxsf.$(LIBEXT): $(INCDIR)/mic/libxs.mod $(OUTDIR)/mic/libxs.$(LIBEXT)
	$(LIB_FLD) -mmic $(FCMTFLAGS) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(BLDDIR)/mic/libxs-mod.o $(call abslib,$(OUTDIR)/mic/libxs.$(IMPEXT)) $(LDFLAGS) $(FLDFLAGS)
else # static
$(OUTDIR)/mic/libxsf.$(LIBEXT): $(INCDIR)/mic/libxs.mod $(OUTDIR)/mic/.make
	$(AR) -rs $@ $(BLDDIR)/mic/libxs-mod.o
endif
else
.PHONY: $(OUTDIR)/mic/libxsf.$(LIBEXT)
endif
endif
endif

.PHONY: flib_hst
ifneq (,$(strip $(FC)))
flib_hst: $(OUTDIR)/libxsf.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/libxsf.$(LIBEXT): $(INCDIR)/libxs.mod $(OUTDIR)/libxs.$(LIBEXT)
	$(LIB_FLD) $(FCMTFLAGS) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(BLDDIR)/intel64/libxs-mod.o $(call abslib,$(OUTDIR)/libxs.$(IMPEXT)) $(LDFLAGS) $(FLDFLAGS)
else # static
$(OUTDIR)/libxsf.$(LIBEXT): $(INCDIR)/libxs.mod $(OUTDIR)/.make
	$(AR) -rs $@ $(BLDDIR)/intel64/libxs-mod.o
endif
else
.PHONY: $(OUTDIR)/libxsf.$(LIBEXT)
endif

.PHONY: ext_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ext_mic: $(OUTDIR)/mic/libxsext.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/mic/libxsext.$(LIBEXT): $(OUTDIR)/mic/.make $(EXTOBJS_MIC) $(OUTDIR)/mic/libxs.$(LIBEXT)
	$(LIB_LD) -mmic $(EXTLDFLAGS) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(EXTOBJS_MIC) $(call abslib,$(OUTDIR)/mic/libxs.$(IMPEXT)) $(LDFLAGS) $(CLDFLAGS)
else # static
$(OUTDIR)/mic/libxsext.$(LIBEXT): $(OUTDIR)/mic/.make $(EXTOBJS_MIC)
	$(AR) -rs $@ $(EXTOBJS_MIC)
endif
endif
endif

.PHONY: ext_hst
ext_hst: $(OUTDIR)/libxsext.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/libxsext.$(LIBEXT): $(OUTDIR)/.make $(EXTOBJS_HST) $(OUTDIR)/libxs.$(LIBEXT)
ifneq (Darwin,$(UNAME))
	$(LIB_LD) $(EXTLDFLAGS) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
		$(EXTOBJS_HST)  $(call abslib,$(OUTDIR)/libxs.$(IMPEXT)) $(LDFLAGS) $(CLDFLAGS)
else ifneq (0,$(INTEL)) # intel @ osx
	$(LIB_LD) $(EXTLDFLAGS) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
			$(EXTOBJS_HST)  $(call abslib,$(OUTDIR)/libxs.$(IMPEXT)) $(LDFLAGS) $(CLDFLAGS)
else # osx
	$(LIB_LD)               $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) \
			$(EXTOBJS_HST)  $(call abslib,$(OUTDIR)/libxs.$(IMPEXT)) $(LDFLAGS) $(CLDFLAGS)
endif
else # static
$(OUTDIR)/libxsext.$(LIBEXT): $(OUTDIR)/.make $(EXTOBJS_HST)
	$(AR) -rs $@ $(EXTOBJS_HST)
endif

.PHONY: noblas_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
noblas_mic: $(OUTDIR)/mic/libxsnoblas.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/mic/libxsnoblas.$(LIBEXT): $(OUTDIR)/mic/.make $(NOBLAS_MIC)
	$(LIB_LD) -mmic $(EXTLDFLAGS) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) $(NOBLAS_MIC) $(LDFLAGS) $(CLDFLAGS)
else # static
$(OUTDIR)/mic/libxsnoblas.$(LIBEXT): $(OUTDIR)/mic/.make $(NOBLAS_MIC)
	$(AR) -rs $@ $(NOBLAS_MIC)
endif
endif
endif

.PHONY: noblas_hst
noblas_hst: $(OUTDIR)/libxsnoblas.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/libxsnoblas.$(LIBEXT): $(OUTDIR)/.make $(NOBLAS_HST)
ifneq (Darwin,$(UNAME))
	$(LIB_LD) $(EXTLDFLAGS) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) $(NOBLAS_HST) $(LDFLAGS) $(CLDFLAGS)
else ifneq (0,$(INTEL)) # intel @ osx
	$(LIB_LD) $(EXTLDFLAGS) $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) $(NOBLAS_HST) $(LDFLAGS) $(CLDFLAGS)
else # osx
	$(LIB_LD)               $(call soname,$@,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API)) $(NOBLAS_HST) $(LDFLAGS) $(CLDFLAGS)
endif
else # static
$(OUTDIR)/libxsnoblas.$(LIBEXT): $(OUTDIR)/.make $(NOBLAS_HST)
	$(AR) -rs $@ $(NOBLAS_HST)
endif

.PHONY: samples
samples: lib_hst
	@find $(SPLDIR) -type f -name Makefile | grep -v /pyfr/ | grep -v /lstmcell/ | grep -v /gxm/ \
		$(patsubst %, | grep -v /%/,$^) | xargs -I {} $(FLOCK) {} "$(MAKE) DEPSTATIC=$(STATIC)"

.PHONY: cp2k
cp2k: lib_hst
	@$(FLOCK) $(SPLDIR)/cp2k "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"

.PHONY: cp2k_mic
cp2k_mic: lib_mic
	@$(FLOCK) $(SPLDIR)/cp2k "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1"

.PHONY: wrap
wrap: lib_hst
	@$(FLOCK) $(SPLDIR)/wrap "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) TRACE=0"

.PHONY: wrap_mic
wrap_mic: lib_mic
	@$(FLOCK) $(SPLDIR)/wrap "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1 TRACE=0"

.PHONY: nek
nek: lib_hst
	@$(FLOCK) $(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"

.PHONY: nek_mic
nek_mic: lib_mic
	@$(FLOCK) $(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1"

.PHONY: smm
smm: lib_hst
	@$(FLOCK) $(SPLDIR)/smm "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"

.PHONY: smm_mic
smm_mic: lib_mic
	@$(FLOCK) $(SPLDIR)/smm "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1"

# added for specfem sample
# will need option: make MNK="5 25" ..
.PHONY: specfem
specfem: lib_hst
	@$(FLOCK) $(SPLDIR)/specfem "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"

.PHONY: specfem_mic
specfem_mic: lib_mic
	@$(FLOCK) $(SPLDIR)/specfem "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) KNC=1"

.PHONY: drytest
drytest: $(SPLDIR)/cp2k/cp2k-perf.sh $(SPLDIR)/smm/smmf-perf.sh \
	$(SPLDIR)/nek/axhm-perf.sh $(SPLDIR)/nek/grad-perf.sh $(SPLDIR)/nek/rstr-perf.sh

$(SPLDIR)/cp2k/cp2k-perf.sh: $(SPLDIR)/cp2k/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=cp2k-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23 4_6_9 13_5_7 24_3_36\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  SIZE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "else" >> $@
	@echo "  SIZE=0" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/cp2k.sh \$${MVALUE} \$${SIZE} 0 \$${NVALUE} \$${KVALUE} >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/smm/smmf-perf.sh: $(SPLDIR)/smm/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=\$${HERE}/smmf-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23 4_6_9 13_5_7 24_3_36\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/smm.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/axhm-perf.sh: $(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=\$${HERE}/axhm-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_6_9 8_8_8 13_13_13 16_8_13\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/axhm.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/grad-perf.sh: $(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=\$${HERE}/grad-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_6_9 8_8_8 13_13_13 16_8_13\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/grad.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/rstr-perf.sh: $(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=\$${HERE}/rstr-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
	@echo "RUNT=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_4_4 8_8_8\"" >> $@
	@echo "RUNT=\"7_7_7 10_10_10\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NRUNS=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "NRUNT=\$$(\$${ECHO} \$${RUNT} | wc -w)" >> $@
	@echo "NMAX=\$$((NRUNS*NRUNT))" >> $@
	@echo "for RUN1 in \$${RUNS} ; do" >> $@
	@echo "  for RUN2 in \$${RUNT} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN1} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN1} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN1} | cut -d_ -f3)" >> $@
	@echo "  MMVALUE=\$$(\$${ECHO} \$${RUN2} | cut -d_ -f1)" >> $@
	@echo "  NNVALUE=\$$(\$${ECHO} \$${RUN2} | cut -d_ -f2)" >> $@
	@echo "  KKVALUE=\$$(\$${ECHO} \$${RUN2} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/rstr.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$${MMVALUE} \$${NNVALUE} \$${KKVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

.PHONY: test
test: tests

.PHONY: perf
perf: perf-cp2k

.PHONY: test-all
test-all: tests test-cp2k test-smm test-nek test-wrap

.PHONY: build-tests
build-tests: lib_hst
	@$(FLOCK) $(TSTDIR) "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC)"

.PHONY: tests
tests: lib_hst
	@$(FLOCK) $(TSTDIR) "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) test"

.PHONY: cpp-test
cpp-test: test-cpp

.PHONY: test-cpp
test-cpp: $(INCDIR)/libxs_source.h
	@$(FLOCK) $(SPLDIR)/cp2k "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) TRACE=0 \
		ECXXFLAGS='-DUSE_HEADER_ONLY $(ECXXFLAGS)' clean compile"

.PHONY: test-cp2k
test-cp2k: $(SPLDIR)/cp2k/cp2k-test.txt
$(SPLDIR)/cp2k/cp2k-test.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_hst
	$(info ========================)
	$(info Running CP2K Code Sample)
	$(info ========================)
	@$(FLOCK) $(SPLDIR)/cp2k "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) cp2k"
	@$(FLOCK) $(SPLDIR)/cp2k "./cp2k-perf.sh $(notdir $@) $(shell echo $$(($(TESTSIZE) * 128)))"

.PHONY: perf-cp2k
perf-cp2k: $(SPLDIR)/cp2k/cp2k-perf.txt
$(SPLDIR)/cp2k/cp2k-perf.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_hst
	@$(FLOCK) $(SPLDIR)/cp2k "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) cp2k"
	@$(FLOCK) $(SPLDIR)/cp2k "./cp2k-perf.sh $(notdir $@)"

.PHONY: test-wrap
test-wrap: wrap
	@$(FLOCK) $(SPLDIR)/wrap "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) TRACE=0 test"

.PHONY: test-smm
ifneq (,$(strip $(FC)))
test-smm: $(SPLDIR)/smm/smm-test.txt
$(SPLDIR)/smm/smm-test.txt: $(SPLDIR)/smm/smmf-perf.sh lib_hst
	$(info =======================)
	$(info Running SMM Code Sample)
	$(info =======================)
	@$(FLOCK) $(SPLDIR)/smm "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) smm"
	@$(FLOCK) $(SPLDIR)/smm "./smmf-perf.sh $(notdir $@) $(shell echo $$(($(TESTSIZE) * -128)))"
endif

.PHONY: perf-smm
ifneq (,$(strip $(FC)))
perf-smm: $(SPLDIR)/smm/smmf-perf.txt
$(SPLDIR)/smm/smmf-perf.txt: $(SPLDIR)/smm/smmf-perf.sh lib_hst
	@$(FLOCK) $(SPLDIR)/smm "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) smm"
	@$(FLOCK) $(SPLDIR)/smm "./smmf-perf.sh $(notdir $@)"
endif

.PHONY: test-nek
ifneq (,$(strip $(FC)))
test-nek: $(SPLDIR)/nek/axhm-perf.txt $(SPLDIR)/nek/grad-perf.txt $(SPLDIR)/nek/rstr-perf.txt
$(SPLDIR)/nek/axhm-perf.txt: $(SPLDIR)/nek/axhm-perf.sh lib_hst
	$(info =======================)
	$(info Running NEK/AXHM Sample)
	$(info =======================)
	@$(FLOCK) $(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) axhm"
	@$(FLOCK) $(SPLDIR)/nek "./axhm-perf.sh $(notdir $@) $(shell echo $$(($(TESTSIZE) * -128)))"
$(SPLDIR)/nek/grad-perf.txt: $(SPLDIR)/nek/grad-perf.sh lib_hst
	$(info =======================)
	$(info Running NEK/GRAD Sample)
	$(info =======================)
	@$(FLOCK) $(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) grad"
	@$(FLOCK) $(SPLDIR)/nek "./grad-perf.sh $(notdir $@) $(shell echo $$(($(TESTSIZE) * -128)))"
$(SPLDIR)/nek/rstr-perf.txt: $(SPLDIR)/nek/rstr-perf.sh lib_hst
	$(info =======================)
	$(info Running NEK/RSTR Sample)
	$(info =======================)
	@$(FLOCK) $(SPLDIR)/nek "$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) rstr"
	@$(FLOCK) $(SPLDIR)/nek "./rstr-perf.sh $(notdir $@) $(shell echo $$(($(TESTSIZE) * -128)))"
endif

$(DOCDIR)/index.md: $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/README.md
	@sed $(ROOTDIR)/README.md \
		-e 's/\[!\[..*\](..*)\](..*)//g' \
		-e 's/\[\[..*\](..*)\]//g' \
		-e "s/](${DOCDIR}\//](/g" \
		> $@

$(DOCDIR)/libxs.$(DOCEXT): $(DOCDIR)/.make $(ROOTDIR)/documentation/index.md \
$(ROOTDIR)/documentation/libxs_mm.md $(ROOTDIR)/documentation/libxs_dl.md $(ROOTDIR)/documentation/libxs_aux.md \
$(ROOTDIR)/documentation/libxs_prof.md $(ROOTDIR)/documentation/libxs_tune.md $(ROOTDIR)/documentation/libxs_be.md
	$(eval TMPFILE = $(shell $(MKTEMP) $(ROOTDIR)/documentation/.libxs_XXXXXX.tex))
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		> $(TMPFILE)
	@cd $(ROOTDIR)/documentation && ( \
		iconv -t utf-8 index.md && echo && \
		echo "# LIBXS Domains" && \
		iconv -t utf-8 libxs_mm.md && echo && \
		iconv -t utf-8 libxs_dl.md && echo && \
		iconv -t utf-8 libxs_aux.md && echo && \
		iconv -t utf-8 libxs_prof.md && echo && \
		iconv -t utf-8 libxs_tune.md && echo && \
		iconv -t utf-8 libxs_be.md && echo && \
		echo "# Appendix" && \
		echo "## Compatibility" && \
		wget -q -O - https://raw.githubusercontent.com/wiki/hfp/libxs/Compatibility.md 2>/dev/null && echo && \
		echo "## Validation" && \
		wget -q -O - https://raw.githubusercontent.com/wiki/hfp/libxs/Validation.md 2>/dev/null; ) \
	| sed \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--template=$(notdir $(TMPFILE)) --listings \
		-f markdown_github+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="LIBXS Documentation" \
		-V author-meta="Hans Pabst, Alexander Heinecke" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $(notdir $@)
	@rm $(TMPFILE)

$(DOCDIR)/libxs_samples.md: $(ROOTDIR)/Makefile $(SPLDIR)/*/README.md $(SPLDIR)/deeplearning/*/README.md
	@cat $(SPLDIR)/*/README.md $(SPLDIR)/deeplearning/*/README.md \
	| sed \
		-e 's/^#/##/' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
		-e '1s/^/# [LIBXS Samples](https:\/\/github.com\/hfp\/libxs\/raw\/master\/documentation\/libxs_samples.pdf)\n\n/' \
		> $@

$(DOCDIR)/libxs_samples.$(DOCEXT): $(ROOTDIR)/documentation/libxs_samples.md
	$(eval TMPFILE = $(shell $(MKTEMP) .libxs_XXXXXX.tex))
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		> $(TMPFILE)
	@iconv -t utf-8 $(ROOTDIR)/documentation/libxs_samples.md \
	| pandoc \
		--template=$(TMPFILE) --listings \
		-f markdown_github+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="LIBXS Sample Code Summary" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TMPFILE)

$(DOCDIR)/cp2k.$(DOCEXT): $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/documentation/cp2k.md
	$(eval TMPFILE = $(shell $(MKTEMP) $(ROOTDIR)/documentation/.libxs_XXXXXX.tex))
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		> $(TMPFILE)
	@cd $(ROOTDIR)/documentation && iconv -t utf-8 cp2k.md \
	| sed \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--template=$(notdir $(TMPFILE)) --listings \
		-f markdown_github+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="CP2K with LIBXS" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $(notdir $@)
	@rm $(TMPFILE)

$(DOCDIR)/tensorflow.$(DOCEXT): $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/documentation/tensorflow.md
	$(eval TMPFILE = $(shell $(MKTEMP) $(ROOTDIR)/documentation/.libxs_XXXXXX.tex))
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		> $(TMPFILE)
	@cd $(ROOTDIR)/documentation && iconv -t utf-8 tensorflow.md \
	| sed \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--template=$(notdir $(TMPFILE)) --listings \
		-f markdown_github+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="TensorFlow with LIBXS" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $(notdir $@)
	@rm $(TMPFILE)

$(DOCDIR)/tfserving.$(DOCEXT): $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/documentation/tfserving.md
	$(eval TMPFILE = $(shell $(MKTEMP) $(ROOTDIR)/documentation/.libxs_XXXXXX.tex))
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		> $(TMPFILE)
	@cd $(ROOTDIR)/documentation && iconv -t utf-8 tfserving.md \
	| sed \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--template=$(notdir $(TMPFILE)) --listings \
		-f markdown_github+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="TensorFlow Serving with LIBXS" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $(notdir $@)
	@rm $(TMPFILE)

.PHONY: documentation
documentation: \
$(DOCDIR)/libxs.$(DOCEXT) \
$(DOCDIR)/libxs_samples.$(DOCEXT) \
$(DOCDIR)/cp2k.$(DOCEXT) \
$(DOCDIR)/tensorflow.$(DOCEXT) \
$(DOCDIR)/tfserving.$(DOCEXT)

.PHONY: mkdocs
mkdocs: $(ROOTDIR)/documentation/index.md $(ROOTDIR)/documentation/libxs_samples.md
	@mkdocs build --clean
	@mkdocs serve

.PHONY: clean
clean:
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@rm -f $(OBJECTS) $(FTNOBJS) $(SRCFILES_KERNELS) $(BLDDIR)/libxs_dispatch.h
	@rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif
	@find . -type f \( -name .make -or -name .state \) -exec rm {} \;
	@rm -f $(SCRDIR)/libxs_utilities.pyc
	@rm -rf $(SCRDIR)/__pycache__

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@rm -f $(OUTDIR)/libxs.$(LIBEXT)* $(OUTDIR)/mic/libxs.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsf.$(LIBEXT)* $(OUTDIR)/mic/libxsf.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsext.$(LIBEXT)* $(OUTDIR)/mic/libxsext.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsnoblas.$(LIBEXT)* $(OUTDIR)/mic/libxsnoblas.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsgen.$(LIBEXT)*
endif
ifneq ($(abspath $(BINDIR)),$(ROOTDIR))
ifneq ($(abspath $(BINDIR)),$(abspath .))
	@rm -rf $(BINDIR)
endif
endif
ifneq (,$(wildcard $(BINDIR))) # still exists
	@rm -f $(BINDIR)/libxs_*_generator
endif
	@rm -f $(SPLDIR)/cp2k/cp2k-perf.sh
	@rm -f $(SPLDIR)/smm/smmf-perf.sh
	@rm -f $(SPLDIR)/nek/grad-perf.sh
	@rm -f $(SPLDIR)/nek/axhm-perf.sh
	@rm -f $(SPLDIR)/nek/rstr-perf.sh
	@rm -f $(INCDIR)/libxs_config.h
	@rm -f $(INCDIR)/libxs_source.h
	@rm -f $(INCDIR)/libxs.modmic
	@rm -f $(INCDIR)/libxs.mod
	@rm -f $(INCDIR)/libxs.f
	@rm -f $(INCDIR)/libxs.h

.PHONY: clean-all
clean-all: clean
	@find $(ROOTDIR) -type f -name Makefile -exec $(FLOCK) {} \
		"$(MAKE) --no-print-directory clean 2>/dev/null || true" \;

.PHONY: realclean-all
realclean-all: realclean
	@find $(ROOTDIR) -type f -name Makefile -exec $(FLOCK) {} \
		"$(MAKE) --no-print-directory realclean 2>/dev/null || true" \;

.PHONY: distclean
distclean: realclean-all

ifneq (,$(strip $(PREFIX)))
INSTALL_ROOT = $(PREFIX)
else
INSTALL_ROOT = .
endif
ifeq ($(dir .),$(dir $(INSTALL_ROOT)))
ifneq (,$(strip $(DESTDIR)))
INSTALL_ROOT := $(abspath $(DESTDIR)/$(INSTALL_ROOT))
else
INSTALL_ROOT := $(abspath $(INSTALL_ROOT))
endif
endif

.PHONY: install-minimal
install-minimal: libxs
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXS installing binaries..."
	@mkdir -p $(INSTALL_ROOT)/$(POUTDIR) $(INSTALL_ROOT)/$(PBINDIR) $(INSTALL_ROOT)/$(PINCDIR)
	@$(CP) -v $(OUTDIR)/libxsnoblas.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v $(OUTDIR)/libxsnoblas.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v $(OUTDIR)/libxsgen.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v $(OUTDIR)/libxsgen.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v $(OUTDIR)/libxsext.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v $(OUTDIR)/libxsext.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v $(OUTDIR)/libxsf.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v $(OUTDIR)/libxsf.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v $(OUTDIR)/libxs.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v $(OUTDIR)/libxs.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2>/dev/null || true
	@if [ -e $(OUTDIR)/mic/libxsnoblas.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsnoblas.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsnoblas.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsnoblas.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsext.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsext.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsext.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsext.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsf.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsf.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsf.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxsf.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxs.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxs.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxs.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		$(CP) -v $(OUTDIR)/mic/libxs.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@echo
	@echo "LIBXS installing interface..."
	@$(CP) -v $(BINDIR)/libxs_*_generator $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(INCDIR)/*.mod* $(INSTALL_ROOT)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -v $(INCDIR)/libxs*.h $(INSTALL_ROOT)/$(PINCDIR)
	@$(CP) -v $(INCDIR)/libxs.f $(INSTALL_ROOT)/$(PINCDIR)
	@echo
	@echo "LIBXS installing stand-alone generators..."
	@$(CP) -v $(BINDIR)/libxs_*_generator $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
endif

.PHONY: install
install: install-minimal
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXS installing documentation..."
	@mkdir -p $(INSTALL_ROOT)/$(PDOCDIR)
	@$(CP) -v $(ROOTDIR)/$(DOCDIR)/*.pdf $(INSTALL_ROOT)/$(PDOCDIR)
	@$(CP) -v $(ROOTDIR)/$(DOCDIR)/*.md $(INSTALL_ROOT)/$(PDOCDIR)
	@$(CP) -v $(ROOTDIR)/version.txt $(INSTALL_ROOT)/$(PDOCDIR)
	@$(CP) -v $(ROOTDIR)/CONTRIBUTING.md $(INSTALL_ROOT)/$(PDOCDIR)
	@$(CP) -v $(ROOTDIR)/LICENSE.md $(INSTALL_ROOT)/$(PDOCDIR)/$(LICFILE)
endif

.PHONY: install-all
install-all: install samples
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXS installing samples..."
	@$(CP) -v $(addprefix $(SPLDIR)/cp2k/,cp2k cp2k.sh cp2k-perf* cp2k-plot.sh) $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(SPLDIR)/wrap/,dgemm-blas dgemm-blas.sh dgemm-wrap dgemm-wrap.sh dgemm-test.sh) $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(SPLDIR)/dispatch/,dispatch dispatch.sh) $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(SPLDIR)/nek/,axhm grad rstr *.sh) $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(SPLDIR)/smm/,smm smm.sh smm-perf* smmf-perf.sh smm-plot.sh) $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(SPLDIR)/smm/,specialized specialized.sh) $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(SPLDIR)/smm/,dispatched dispatched.sh) $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(SPLDIR)/smm/,inlined inlined.sh) $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(SPLDIR)/smm/,blas blas.sh) $(INSTALL_ROOT)/$(PBINDIR) 2>/dev/null || true
endif

.PHONY: install-dev
install-dev: install-all build-tests
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXS installing tests..."
	@mkdir -p $(INSTALL_ROOT)/$(PTSTDIR)
	@$(CP) -v $(basename $(wildcard ${TSTDIR}/*.c)) $(INSTALL_ROOT)/$(PTSTDIR) 2>/dev/null || true
endif

.PHONY: install-artifacts
install-artifacts: install-dev
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXS installing artifacts..."
	@mkdir -p $(INSTALL_ROOT)/$(PDOCDIR)/artifacts
	@$(CP) -v .state $(INSTALL_ROOT)/$(PDOCDIR)/artifacts/make.txt
endif

.PHONY: deb
deb:
	@if [ "" != "$$(which git)" ]; then \
		VERSION_ARCHIVE=$$(git describe --tags --abbrev=0 2>/dev/null); \
	fi; \
	if [ "" != "$${VERSION_ARCHIVE}" ]; then \
		ARCHIVE_AUTHOR_NAME="$$(git config user.name)"; \
		ARCHIVE_AUTHOR_MAIL="$$(git config user.email)"; \
		ARCHIVE_DATE="$$(LANG=C date -R)"; \
		if [ "" != "$${ARCHIVE_AUTHOR_NAME}" ] && [ "" != "$${ARCHIVE_AUTHOR_MAIL}" ]; then \
			ARCHIVE_AUTHOR="$${ARCHIVE_AUTHOR_NAME} <$${ARCHIVE_AUTHOR_MAIL}>"; \
		else \
			echo "Warning: Please git-config user.name and user.email!"; \
			if [ "" != "$${ARCHIVE_AUTHOR_NAME}" ] || [ "" != "$${ARCHIVE_AUTHOR_MAIL}" ]; then \
				ARCHIVE_AUTHOR="$${ARCHIVE_AUTHOR_NAME}$${ARCHIVE_AUTHOR_MAIL}"; \
			fi \
		fi; \
		if ! [ -e libxs_$${VERSION_ARCHIVE}.orig.tar.gz ]; then \
			git archive --prefix libxs-$${VERSION_ARCHIVE}/ \
				-o libxs_$${VERSION_ARCHIVE}.orig.tar.gz $(VERSION_RELEASE); \
		fi; \
		tar xf libxs_$${VERSION_ARCHIVE}.orig.tar.gz; \
		cd libxs-$${VERSION_ARCHIVE}; \
		mkdir -p debian/source; cd debian/source; \
		echo "3.0 (quilt)" > format; \
		cd ..; \
		echo "Source: libxs" > control; \
		echo "Section: libs" >> control; \
		echo "Homepage: https://github.com/hfp/libxs" >> control; \
		echo "Vcs-Git: https://github.com/hfp/libxs/libxs.git" >> control; \
		echo "Maintainer: $${ARCHIVE_AUTHOR}" >> control; \
		echo "Priority: optional" >> control; \
		echo "Build-Depends: debhelper (>= 9)" >> control; \
		echo "Standards-Version: 3.9.8" >> control; \
		echo >> control; \
		echo "Package: libxs" >> control; \
		echo "Section: libs" >> control; \
		echo "Architecture: amd64" >> control; \
		echo "Depends: \$${shlibs:Depends}, \$${misc:Depends}" >> control; \
		echo "Description: Matrix operations and deep learning primitives" >> control; \
		wget -qO- https://api.github.com/repos/hfp/libxs \
		| sed -n 's/ *\"description\": \"\(..*\)\".*/\1/p' \
		| fold -s -w 79 | sed -e 's/^/ /' -e 's/\s\s*$$//' >> control; \
		echo "libxs ($${VERSION_ARCHIVE}-$(VERSION_PACKAGE)) UNRELEASED; urgency=low" > changelog; \
		echo >> changelog; \
		wget -qO- https://api.github.com/repos/hfp/libxs/releases/tags/$${VERSION_ARCHIVE} \
		| sed -n 's/ *\"body\": \"\(..*\)\".*/\1/p' \
		| sed -e 's/\\r\\n/\n/g' -e 's/\\"/"/g' -e 's/\[\([^]]*\)\]([^)]*)/\1/g' \
		| sed -n 's/^\* \(..*\)/\* \1/p' \
		| fold -s -w 78 | sed -e 's/^/  /g' -e 's/^  \* /\* /' -e 's/^/  /' -e 's/\s\s*$$//' >> changelog; \
		echo >> changelog; \
		echo " -- $${ARCHIVE_AUTHOR}  $${ARCHIVE_DATE}" >> changelog; \
		echo "#!/usr/bin/make -f" > rules; \
		echo "export DH_VERBOSE = 1" >> rules; \
		echo >> rules; \
		echo "%:" >> rules; \
		echo "	dh \$$@" >> rules; \
		echo >> rules; \
		echo "override_dh_auto_install:" >> rules; \
        echo "	dh_auto_install -- prefix=/usr" >> rules; \
		echo >> rules; \
		echo "9" > compat; \
		chmod +x rules; \
		debuild \
			-e PREFIX=debian/libxs/usr \
			-e PDOCDIR=share/doc/libxs \
			-e LICFILE=copyright \
			-e SHARED=1 \
			-e SYM=1 \
			-us -uc; \
	else \
		echo "Error: Git is unavailable or make-deb runs outside of cloned repository!"; \
	fi
