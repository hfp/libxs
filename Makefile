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

# Python interpreter
PYTHON ?= python

# Use ROW_MAJOR matrix representation if set to 1, COL_MAJOR otherwise
ROW_MAJOR ?= 0

# Generates M,N,K-combinations for each comma separated group e.g., "1, 2, 3" gnerates (1,1,1), (2,2,2),
# and (3,3,3). This way a heterogeneous set can be generated e.g., "1 2, 3" generates (1,1,1), (1,1,2),
# (1,2,1), (1,2,2), (2,1,1), (2,1,2) (2,2,1) out of the first group, and a (3,3,3) for the second group
# To generate a series of square matrices one can specify e.g., make MNK=$(echo $(seq -s, 1 5))
# Alternative to MNK, index sets can be specified separately according to a loop nest relationship
# (M(N(K))) using M, N, and K separately. Please consult the documentation for further details.
MNK ?= 0

# Preferred precision when registering statically generated code versions
# 0: SP and DP code versions to be registered
# 1: SP only
# 2: DP only
PRECISION ?= 0

# Specify an alignment (Bytes)
ALIGNMENT ?= 64

# Generate prefetches
PREFETCH ?= 0

# THRESHOLD problem size (M x N x K) determining when to use BLAS; can be zero
THRESHOLD ?= $(shell echo $$((80 * 80 * 80)))

# Generate code using aligned Load/Store instructions
# !=0: enable if lda/ldc (m) is a multiple of ALIGNMENT
# ==0: disable emitting aligned Load/Store instructions
ALIGNED_STORES ?= 0
ALIGNED_LOADS ?= 0

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
ifneq (0,$(BETA))
ifneq (1,$(BETA))
$(error BETA needs to be eiter 0 or 1)
endif
endif

ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
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

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -DLIBXS_BUILD
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(SRCDIR)

PTHREAD ?= 1
STATIC ?= 1

# JIT backend is enabled by default
JIT ?= 1
ifneq (0,$(JIT))
	AVX ?= 0
	SSE ?= 1
endif

BLAS_WARNING ?= 0
ifeq (0,$(STATIC))
	ifeq (Windows_NT,$(OS))
		BLAS_WARNING = 1
		BLAS ?= 2
	else ifeq (Darwin,$(shell uname))
		BLAS_WARNING = 1
		BLAS ?= 2
	endif
endif

# include common Makefile artifacts
include $(ROOTDIR)/Makefile.inc

# Number of repeated calls (tests)
TESTSIZE ?= 1000

ifeq (1,$(AVX))
	GENTARGET = snb
else ifeq (2,$(AVX))
	GENTARGET = hsw
else ifeq (3,$(AVX))
	GENTARGET = knl
else ifneq (0,$(SSE))
	GENTARGET = wsm
else
	GENTARGET = noarch
endif

ifeq (0,$(STATIC))
	GENERATOR = @$(ENV) \
		LD_LIBRARY_PATH="$(OUTDIR):$(LD_LIBRARY_PATH)" \
		PATH="$(OUTDIR):$(PATH)" \
	$(BINDIR)/libxs_gemm_generator
else
	GENERATOR = $(BINDIR)/libxs_gemm_generator
endif

INDICES ?= $(shell $(PYTHON) $(SCRDIR)/libxs_utilities.py -1 $(THRESHOLD) $(words $(MNK)) $(MNK) $(words $(M)) $(words $(N)) $(M) $(N) $(K))
NINDICES = $(words $(INDICES))

HEADERS = $(shell ls -1 $(SRCDIR)/*.h 2> /dev/null | tr "\n" " ") \
          $(shell ls -1 $(SRCDIR)/*.c 2> /dev/null | tr "\n" " ") \
          $(ROOTDIR)/include/libxs_frontend.h \
          $(ROOTDIR)/include/libxs_generator.h \
          $(ROOTDIR)/include/libxs_macros.h \
          $(ROOTDIR)/include/libxs_timer.h \
          $(ROOTDIR)/include/libxs_typedefs.h

SRCFILES = $(patsubst %,$(BLDDIR)/mm_%.c,$(INDICES))
SRCFILES_GEN_LIB = $(patsubst %,$(SRCDIR)/%,$(wildcard $(SRCDIR)/generator_*.c) libxs_timer.c libxs_trace.c)
SRCFILES_GEN_GEMM_BIN = $(patsubst %,$(SRCDIR)/%,libxs_generator_dense_gemm_driver.c)
OBJFILES_GEN_LIB = $(patsubst %,$(BLDDIR)/%.o,$(basename $(notdir $(SRCFILES_GEN_LIB))))
OBJFILES_GEN_GEMM_BIN = $(patsubst %,$(BLDDIR)/%.o,$(basename $(notdir $(SRCFILES_GEN_GEMM_BIN))))
OBJFILES_HST = $(patsubst %,$(BLDDIR)/intel64/mm_%.o,$(INDICES)) \
               $(BLDDIR)/intel64/libxs.o $(BLDDIR)/intel64/libxs_gemm.o
OBJFILES_MIC = $(patsubst %,$(BLDDIR)/mic/mm_%.o,$(INDICES)) \
               $(BLDDIR)/mic/libxs.o $(BLDDIR)/mic/libxs_gemm.o \
               $(BLDDIR)/mic/libxs_trace.o $(BLDDIR)/mic/libxs_timer.o
WRAPOBJS_HST = $(BLDDIR)/intel64/libxs_gemm_wrap.o
WRAPOBJS_MIC = $(BLDDIR)/mic/libxs_gemm_wrap.o

# list of object might be "incomplete" if not all code gen. FLAGS are supplied with clean target!
OBJECTS = $(OBJFILES_GEN_LIB) $(OBJFILES_GEN_GEMM_BIN) $(OBJFILES_HST) $(OBJFILES_MIC) $(WRAPOBJS_HST) $(WRAPOBJS_MIC)
FTNOBJS = $(BLDDIR)/intel64/libxs-mod.o $(BLDDIR)/mic/libxs-mod.o

.PHONY: libxs
libxs: lib

.PHONY: lib
lib: headers drytest lib_hst lib_mic

.PHONY: all
all: lib samples

.PHONY: headers
headers: cheader fheader

.PHONY: interface
interface: headers

.PHONY: lib_mic
lib_mic: clib_mic flib_mic wrap_mic

.PHONY: lib_hst
lib_hst: clib_hst flib_hst wrap_hst

PREFETCH_ID = 0
PREFETCH_SCHEME = nopf
PREFETCH_TYPE = 0

ifneq (0,$(shell echo $$((2 <= $(PREFETCH) && $(PREFETCH) <= 9))))
	PREFETCH_ID = $(PREFETCH)
else ifeq (1,$(PREFETCH)) # AL2_BL2viaC
	PREFETCH_ID = 6
else ifeq (pfsigonly,$(PREFETCH))
	PREFETCH_ID = 2
else ifeq (BL2viaC,$(PREFETCH))
	PREFETCH_ID = 3
else ifeq (AL2,$(PREFETCH))
	PREFETCH_ID = 4
else ifeq (curAL2,$(PREFETCH))
	PREFETCH_ID = 5
else ifeq (AL2_BL2viaC,$(PREFETCH))
	PREFETCH_ID = 6
else ifeq (curAL2_BL2viaC,$(PREFETCH))
	PREFETCH_ID = 7
else ifeq (AL2jpst,$(PREFETCH))
	PREFETCH_ID = 8
else ifeq (AL2jpst_BL2viaC,$(PREFETCH))
	PREFETCH_ID = 9
endif

# Mapping build options to libxs_prefetch_type (see include/libxs_typedefs.h)
ifeq (2,$(PREFETCH_ID))
	PREFETCH_SCHEME = pfsigonly
	PREFETCH_TYPE = 1
else ifeq (3,$(PREFETCH_ID))
	PREFETCH_SCHEME = BL2viaC
	PREFETCH_TYPE = 8
else ifeq (4,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2
	PREFETCH_TYPE = 2
else ifeq (5,$(PREFETCH_ID))
	PREFETCH_SCHEME = curAL2
	PREFETCH_TYPE = 16
else ifeq (8,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2jpst
	PREFETCH_TYPE = 4
else ifeq (6,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2_BL2viaC
	PREFETCH_TYPE = $(shell echo $$((8 | 2)))
else ifeq (7,$(PREFETCH_ID))
	PREFETCH_SCHEME = curAL2_BL2viaC
	PREFETCH_TYPE = $(shell echo $$((8 | 16)))
else ifeq (9,$(PREFETCH_ID))
	PREFETCH_SCHEME = AL2jpst_BL2viaC
	PREFETCH_TYPE = $(shell echo $$((8 | 4)))
endif

# Mapping build options to libxs_gemm_flags (see include/libxs_typedefs.h)
FLAGS = $(shell echo $$((((0!=$(ALIGNED_LOADS))*4) | ((0!=$(ALIGNED_STORES))*8))))

SUPPRESS_UNUSED_VARIABLE_WARNINGS = LIBXS_UNUSED(A); LIBXS_UNUSED(B); LIBXS_UNUSED(C);
ifneq (nopf,$(PREFETCH_SCHEME))
	SUPPRESS_UNUSED_VARIABLE_WARNINGS += LIBXS_UNUSED(A_prefetch); LIBXS_UNUSED(B_prefetch);
	SUPPRESS_UNUSED_PREFETCH_WARNINGS = $(NULL)  LIBXS_UNUSED(C_prefetch);~
endif

.PHONY: cheader
cheader: $(INCDIR)/libxs.h
$(INCDIR)/libxs.h: .state $(INCDIR)/.make \
                     $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc \
                     $(ROOTDIR)/.hooks/install.sh $(ROOTDIR)/version.txt \
                     $(HEADERS)
	@$(ROOTDIR)/.hooks/install.sh
	@cp $(ROOTDIR)/include/libxs_macros.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxs_typedefs.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxs_frontend.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxs_generator.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxs_timer.h $(INCDIR) 2> /dev/null || true
	@$(PYTHON) $(SCRDIR)/libxs_interface.py $(SRCDIR)/libxs.template.h \
		$(PRECISION) $(MAKE_ILP64) $(OFFLOAD) $(ALIGNMENT) $(ROW_MAJOR) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) $(shell echo $$(($(PTHREAD)+$(OMP)))) \
		$(JIT) $(FLAGS) $(ALPHA) $(BETA) $(INDICES) > $@
	$(info ================================================================================)
	$(info $(INFO))
	$(info ================================================================================)
ifeq (,$(strip $(FC)))
ifeq (,$(strip $(FC_VERSION_STRING)))
	$(info Fortran Compiler is missing: building without Fortran support!)
else
	$(info Fortran Compiler $(FC_VERSION_STRING) is outdated!)
endif
	$(info ================================================================================)
endif
ifeq (0,$(STATIC))
ifeq (Windows_NT,$(UNAME))
	$(info The shared link-time wrapper (libxsld) is not supported under Windows/Cygwin!)
	$(info ================================================================================)
endif
endif
ifneq (0,$(OMP))
	$(info LIBXS is agnostic with respect to the threading runtime!)
	$(info Enabling OpenMP suppresses using OS primitives (PThreads).)
	$(info ================================================================================)
endif
ifneq (0,$(BLAS_WARNING))
	$(info Building a shared library requires to link against BLAS since there is)
	$(info no runtime resolution/search for weak symbols implemented for this OS.)
endif
ifneq (0,$(BLAS))
ifneq (Windows_NT,$(UNAME))
	$(info LIBXS is link-time agnostic with respect to BLAS/GEMM!)
	$(info Linking it now may prevent users to make an own decision.)
endif
ifeq (1,$(BLAS))
	$(info LIBXS's THRESHOLD already prevents calling small GEMMs!)
	$(info A sequential BLAS is superfluous with respect to LIBXS.)
endif
	$(info ================================================================================)
endif

.PHONY: fheader
fheader: $(INCDIR)/libxs.f
$(INCDIR)/libxs.f: .state $(INCDIR)/.make $(BLDDIR)/.make \
                     $(SRCDIR)/libxs.template.f $(ROOTDIR)/.hooks/install.sh $(ROOTDIR)/version.txt \
                     $(SCRDIR)/libxs_interface.py $(SCRDIR)/libxs_utilities.py \
                     $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@$(ROOTDIR)/.hooks/install.sh
ifeq (0,$(OFFLOAD))
	@$(PYTHON) $(SCRDIR)/libxs_interface.py $(SRCDIR)/libxs.template.f \
		$(PRECISION) $(MAKE_ILP64) $(OFFLOAD) $(ALIGNMENT) $(ROW_MAJOR) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) $(shell echo $$(($(PTHREAD)+$(OMP)))) \
		$(JIT) $(FLAGS) $(ALPHA) $(BETA) $(INDICES) | \
	sed '/ATTRIBUTES OFFLOAD:MIC/d' > $@
else
	@$(PYTHON) $(SCRDIR)/libxs_interface.py $(SRCDIR)/libxs.template.f \
		$(PRECISION) $(MAKE_ILP64) $(OFFLOAD) $(ALIGNMENT) $(ROW_MAJOR) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) $(shell echo $$(($(PTHREAD)+$(OMP)))) \
		$(JIT) $(FLAGS) $(ALPHA) $(BETA) $(INDICES) > $@
endif

.PHONY: compile_generator_lib
compile_generator_lib: $(OBJFILES_GEN_LIB)
$(BLDDIR)/%.o: $(SRCDIR)/%.c $(BLDDIR)/.make $(INCDIR)/libxs.h $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -c $< -o $@
.PHONY: build_generator_lib
build_generator_lib: $(OUTDIR)/libxsgen.$(LIBEXT)
$(OUTDIR)/libxsgen.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_GEN_LIB)
ifeq (0,$(STATIC))
	$(LD) -o $@ $(OBJFILES_GEN_LIB) -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $(OBJFILES_GEN_LIB)
endif

.PHONY: compile_gemm_generator
compile_gemm_generator: $(OBJFILES_GEN_GEMM_BIN)
$(BLDDIR)/%.o: $(SRCDIR)/%.c $(BLDDIR)/.make $(INCDIR)/libxs.h $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -c $< -o $@
.PHONY: generator
generator: $(BINDIR)/libxs_gemm_generator
$(BINDIR)/libxs_gemm_generator: $(BINDIR)/.make $(OBJFILES_GEN_GEMM_BIN) $(OUTDIR)/libxsgen.$(LIBEXT) $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	$(CC) $(OBJFILES_GEN_GEMM_BIN) $(call libdir,$(OUTDIR)/libxsgen.$(LIBEXT)) $(LDFLAGS) $(CLDFLAGS) -o $@

$(BLDDIR)/libxs_dispatch.h: $(BLDDIR)/.make $(SCRDIR)/libxs_dispatch.py
	@$(PYTHON) $(SCRDIR)/libxs_dispatch.py $(PRECISION) $(THRESHOLD) $(INDICES) > $@

.PHONY: sources
sources: $(SRCFILES) $(BLDDIR)/libxs_dispatch.h
$(BLDDIR)/%.c: $(BLDDIR)/.make $(INCDIR)/libxs.h $(BINDIR)/libxs_gemm_generator $(SCRDIR)/libxs_utilities.py $(SCRDIR)/libxs_specialized.py
ifneq (,$(strip $(SRCFILES)))
	$(eval MVALUE := $(shell echo $(basename $@) | cut -d_ -f2))
	$(eval NVALUE := $(shell echo $(basename $@) | cut -d_ -f3))
	$(eval KVALUE := $(shell echo $(basename $@) | cut -d_ -f4))
ifneq (0,$(ROW_MAJOR)) # row-major
	$(eval MNVALUE := $(NVALUE))
	$(eval NMVALUE := $(MVALUE))
else # column-major
	$(eval MNVALUE := $(MVALUE))
	$(eval NMVALUE := $(NVALUE))
endif
	$(eval ASTSP := $(shell echo $$((0!=$(ALIGNED_STORES)&&0==($(MNVALUE)*4)%$(ALIGNMENT)))))
	$(eval ASTDP := $(shell echo $$((0!=$(ALIGNED_STORES)&&0==($(MNVALUE)*8)%$(ALIGNMENT)))))
	$(eval ALDSP := $(shell echo $$((0!=$(ALIGNED_LOADS)&&0==($(MNVALUE)*4)%$(ALIGNMENT)))))
	$(eval ALDDP := $(shell echo $$((0!=$(ALIGNED_LOADS)&&0==($(MNVALUE)*8)%$(ALIGNMENT)))))
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
	$(GENERATOR) dense $@ libxs_s$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) knl $(PREFETCH_SCHEME) SP
	$(GENERATOR) dense $@ libxs_s$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) hsw $(PREFETCH_SCHEME) SP
	$(GENERATOR) dense $@ libxs_s$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) snb $(PREFETCH_SCHEME) SP
	$(GENERATOR) dense $@ libxs_s$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) wsm $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENERATOR) dense $@ libxs_d$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) knl $(PREFETCH_SCHEME) DP
	$(GENERATOR) dense $@ libxs_d$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) hsw $(PREFETCH_SCHEME) DP
	$(GENERATOR) dense $@ libxs_d$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) snb $(PREFETCH_SCHEME) DP
	$(GENERATOR) dense $@ libxs_d$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) wsm $(PREFETCH_SCHEME) DP
endif
else
ifneq (2,$(PRECISION))
	@echo "#define LIBXS_GENTARGET_$(GENTARGET)_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXS_GENTARGET_$(GENTARGET)_dp" >> $@
endif
	@echo >> $@
	@echo >> $@
ifneq (2,$(PRECISION))
	$(GENERATOR) dense $@ libxs_s$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) $(GENTARGET) $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENERATOR) dense $@ libxs_d$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) $(GENTARGET) $(PREFETCH_SCHEME) DP
endif
endif
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (2,$(PRECISION))
	$(GENERATOR) dense $@ libxs_s$(basename $(notdir $@))_knc $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTDP) knc $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENERATOR) dense $@ libxs_d$(basename $(notdir $@))_knc $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTDP) knc $(PREFETCH_SCHEME) DP
endif
endif
endif
	$(eval TMPFILE = $(shell mktemp /tmp/fileXXXXXX))
	@cat $@ | sed \
		-e 's/void libxs_/LIBXS_INLINE LIBXS_RETARGETABLE void libxs_/' \
		-e 's/#ifndef NDEBUG/$(SUPPRESS_UNUSED_PREFETCH_WARNINGS)#ifdef LIBXS_NEVER_DEFINED/' \
		-e 's/#pragma message (".*KERNEL COMPILATION ERROR in: " __FILE__)/  $(SUPPRESS_UNUSED_VARIABLE_WARNINGS)/' \
		-e '/#error No kernel was compiled, lacking support for current architecture?/d' \
		-e '/#pragma message (".*KERNEL COMPILATION WARNING: compiling ..* code on ..* or newer architecture: " __FILE__)/d' \
		| tr '~' '\n' > $(TMPFILE)
	@$(PYTHON) $(SCRDIR)/libxs_specialized.py $(PRECISION) $(MVALUE) $(NVALUE) $(KVALUE) $(PREFETCH_TYPE) >> $(TMPFILE)
	@mv $(TMPFILE) $@
endif

.PHONY: compile_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
compile_mic: $(OBJFILES_MIC)
$(BLDDIR)/mic/%.o: $(SRCDIR)/%.c $(BLDDIR)/mic/.make $(INCDIR)/libxs.h $(BLDDIR)/libxs_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
$(BLDDIR)/mic/%.o: $(BLDDIR)/%.c $(BLDDIR)/mic/.make $(INCDIR)/libxs.h $(BLDDIR)/libxs_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
endif
endif

.PHONY: compile_hst
compile_hst: $(OBJFILES_HST)
$(BLDDIR)/intel64/%.o: $(SRCDIR)/%.c $(BLDDIR)/intel64/.make $(INCDIR)/libxs.h $(BLDDIR)/libxs_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@
$(BLDDIR)/intel64/%.o: $(BLDDIR)/%.c $(BLDDIR)/intel64/.make $(INCDIR)/libxs.h $(BLDDIR)/libxs_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

.PHONY: compile_mic_mod
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (,$(strip $(FC)))
compile_mic_mod: $(BLDDIR)/mic/libxs-mod.o
$(BLDDIR)/mic/libxs-mod.o: $(BLDDIR)/mic/.make $(INCDIR)/mic/.make $(INCDIR)/libxs.f
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $(INCDIR)/libxs.f -o $(BLDDIR)/mic/libxs-mod.o $(FMFLAGS) $(INCDIR)/mic
endif
endif
endif

.PHONY: compile_hst_mod
ifneq (,$(strip $(FC)))
compile_hst_mod: $(BLDDIR)/intel64/libxs-mod.o
$(BLDDIR)/intel64/libxs-mod.o: $(BLDDIR)/intel64/.make $(INCDIR)/libxs.f
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $(INCDIR)/libxs.f -o $(BLDDIR)/intel64/libxs-mod.o $(FMFLAGS) $(INCDIR)
endif

.PHONY: clib_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
clib_mic: $(OUTDIR)/mic/libxs.$(LIBEXT)
$(OUTDIR)/mic/libxs.$(LIBEXT): $(OUTDIR)/mic/.make $(OBJFILES_MIC)
ifeq (0,$(STATIC))
	$(LD) -o $@ $(OBJFILES_MIC) -mmic -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $(OBJFILES_MIC)
endif
endif
endif

.PHONY: clib_hst
clib_hst: $(OUTDIR)/libxs.$(LIBEXT)
$(OUTDIR)/libxs.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_HST) $(OBJFILES_GEN_LIB)
ifeq (0,$(STATIC))
	$(LD) -o $@ $(OBJFILES_HST) $(OBJFILES_GEN_LIB) -shared $(LDFLAGS) $(CLDFLAGS)
else
	$(AR) -rs $@ $(OBJFILES_HST) $(OBJFILES_GEN_LIB)
endif

.PHONY: flib_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (,$(strip $(FC)))
flib_hst: $(OUTDIR)/mic/libxsf.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/mic/libxsf.$(LIBEXT): $(BLDDIR)/mic/libxs-mod.o $(OUTDIR)/mic/libxs.$(LIBEXT)
	$(FC) -o $@ $(BLDDIR)/mic/libxs-mod.o $(call libdir,$(OUTDIR)/mic/libxs.$(LIBEXT)) -mmic -shared $(FCMTFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
else
$(OUTDIR)/mic/libxsf.$(LIBEXT): $(BLDDIR)/mic/libxs-mod.o $(OUTDIR)/mic/.make
	$(AR) -rs $@ $(BLDDIR)/mic/libxs-mod.o
endif
endif
endif
endif

.PHONY: flib_hst
ifneq (,$(strip $(FC)))
flib_hst: $(OUTDIR)/libxsf.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/libxsf.$(LIBEXT): $(BLDDIR)/intel64/libxs-mod.o $(OUTDIR)/libxs.$(LIBEXT)
	$(FC) -o $@ $(BLDDIR)/intel64/libxs-mod.o $(call libdir,$(OUTDIR)/libxs.$(LIBEXT)) -shared $(FCMTFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
else
$(OUTDIR)/libxsf.$(LIBEXT): $(BLDDIR)/intel64/libxs-mod.o $(OUTDIR)/.make
	$(AR) -rs $@ $(BLDDIR)/intel64/libxs-mod.o
endif
endif

.PHONY: wrap_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifeq (0,$(STATIC))
wrap_mic: $(OUTDIR)/mic/libxsld.$(DLIBEXT)
$(OUTDIR)/mic/libxsld.$(DLIBEXT): $(OUTDIR)/mic/.make $(WRAPOBJS_MIC) $(OUTDIR)/mic/libxs.$(DLIBEXT)
	$(LD) -o $@ $(WRAPOBJS_MIC) $(call libdir,$(OUTDIR)/mic/libxs.$(DLIBEXT)) -mmic -shared $(LDFLAGS) $(CLDFLAGS)
endif
endif
endif

.PHONY: wrap_hst
ifeq (0,$(STATIC))
ifneq (Windows_NT,$(UNAME))
wrap_hst: $(OUTDIR)/libxsld.$(DLIBEXT)
$(OUTDIR)/libxsld.$(DLIBEXT): $(OUTDIR)/.make $(WRAPOBJS_HST) $(OUTDIR)/libxs.$(DLIBEXT)
	$(LD) -o $@ $(WRAPOBJS_HST) $(call libdir,$(OUTDIR)/libxs.$(DLIBEXT)) -shared $(LDFLAGS) $(CLDFLAGS)
endif
endif

.PHONY: samples
samples: cp2k smm nek

.PHONY: cp2k
cp2k: lib_hst
	@cd $(SPLDIR)/cp2k && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: cp2k_mic
cp2k_mic: lib_mic
	@cd $(SPLDIR)/cp2k && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=1 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: smm
smm: lib_hst
	@cd $(SPLDIR)/smm && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: smm_mic
smm_mic: lib_mic
	@cd $(SPLDIR)/smm && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=1 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: nek
nek: lib_hst
	@cd $(SPLDIR)/nek && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: nek_mic
nek_mic: lib_mic
	@cd $(SPLDIR)/nek && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) MIC=1 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

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
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/smm.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$${SIZE} >> \$${FILE}; } 2>&1)" >> $@
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
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/axhm.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$${SIZE} >> \$${FILE}; } 2>&1)" >> $@
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
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/grad.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$${SIZE} >> \$${FILE}; } 2>&1)" >> $@
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
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  SIZE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "else" >> $@
	@echo "  SIZE=0" >> $@
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
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/rstr.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$${MMVALUE} \$${NNVALUE} \$${KKVALUE} \$${SIZE} >> \$${FILE}; } 2>&1)" >> $@
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
test: test-cp2k

.PHONY: perf
perf: perf-cp2k

.PHONY: test-all
test-all: tests test-cp2k test-wrap test-nek test-smm

.PHONY: build-tests
build-tests: lib_hst
	@cd $(TSTDIR) && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: tests
tests: build-tests
	@cd $(TSTDIR) && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) test

.PHONY: test-cp2k
test-cp2k: $(SPLDIR)/cp2k/cp2k-test.txt
$(SPLDIR)/cp2k/cp2k-test.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_hst
	@cd $(SPLDIR)/cp2k && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) cp2k
	@$(SPLDIR)/cp2k/cp2k-perf.sh $@ $(TESTSIZE)

.PHONY: perf-cp2k
perf-cp2k: $(SPLDIR)/cp2k/cp2k-perf.txt
$(SPLDIR)/cp2k/cp2k-perf.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_hst
	@cd $(SPLDIR)/cp2k && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) cp2k
	@$(SPLDIR)/cp2k/cp2k-perf.sh $@

.PHONY: test-dgemm
test-dgemm: lib_hst
	@cd $(SPLDIR)/dgemm && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) test

.PHONY: test-wrap
test-wrap: test-dgemm

.PHONY: test-smm
test-smm: $(SPLDIR)/smm/smm-test.txt
$(SPLDIR)/smm/smm-test.txt: $(SPLDIR)/smm/smmf-perf.sh lib_hst
	@cd $(SPLDIR)/smm && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) smm
	@$(SPLDIR)/smm/smmf-perf.sh $@ $(TESTSIZE)

.PHONY: perf-smm
perf-smm: $(SPLDIR)/smm/smmf-perf.txt
$(SPLDIR)/smm/smmf-perf.txt: $(SPLDIR)/smm/smmf-perf.sh lib_hst
	@cd $(SPLDIR)/smm && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) smm
	@$(SPLDIR)/smm/smmf-perf.sh $@

.PHONY: test-nek
test-nek: $(SPLDIR)/nek/axhm-perf.txt $(SPLDIR)/nek/grad-perf.txt $(SPLDIR)/nek/rstr-perf.txt
$(SPLDIR)/nek/axhm-perf.txt: $(SPLDIR)/nek/axhm-perf.sh lib_hst
	@cd $(SPLDIR)/nek && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) axhm
	@$(SPLDIR)/nek/axhm-perf.sh $@ $(TESTSIZE)
$(SPLDIR)/nek/grad-perf.txt: $(SPLDIR)/nek/grad-perf.sh lib_hst
	@cd $(SPLDIR)/nek && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) grad
	@$(SPLDIR)/nek/grad-perf.sh $@ $(TESTSIZE)
$(SPLDIR)/nek/rstr-perf.txt: $(SPLDIR)/nek/rstr-perf.sh lib_hst
	@cd $(SPLDIR)/nek && \
	$(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) rstr
	@$(SPLDIR)/nek/rstr-perf.sh $@ $(TESTSIZE)

$(DOCDIR)/libxs.pdf: $(DOCDIR)/.make $(ROOTDIR)/README.md
	$(eval TMPFILE = $(shell mktemp fileXXXXXX))
	@mv $(TMPFILE) $(TMPFILE).tex
	@pandoc -D latex | sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' > \
		$(TMPFILE).tex
	@sed \
		-e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxs\/master\///' \
		-e 's/\[!\[..*\](https:\/\/travis-ci.org\/hfp\/libxs.svg?branch=..*)\](..*)//' \
		-e 's/\[\[..*\](..*)\]//' -e '/!\[..*\](..*)/{n;d}' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		$(ROOTDIR)/README.md | \
	pandoc \
		--latex-engine=xelatex --template=$(TMPFILE).tex --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="LIBXS Documentation" \
		-V author-meta="Hans Pabst, Alexander Heinecke" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TMPFILE).tex

$(DOCDIR)/cp2k.pdf: $(DOCDIR)/.make $(ROOTDIR)/documentation/cp2k.md
	$(eval TMPFILE = $(shell mktemp fileXXXXXX))
	@mv $(TMPFILE) $(TMPFILE).tex
	@pandoc -D latex | sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' \
		$(TMPFILE).tex
	@sed \
		-e 's/https:\/\/raw\.githubusercontent\.com\/hfp\/libxs\/master\///' \
		-e 's/\[!\[..*\](https:\/\/travis-ci.org\/hfp\/libxs.svg?branch=..*)\](..*)//' \
		-e 's/\[\[..*\](..*)\]//' -e '/!\[..*\](..*)/{n;d}' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		$(ROOTDIR)/documentation/cp2k.md | \
	pandoc \
		--latex-engine=xelatex --template=$(TMPFILE).tex --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="CP2K with LIBXS" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TMPFILE).tex

.PHONY: documentation
documentation: $(DOCDIR)/libxs.pdf $(DOCDIR)/cp2k.pdf

.PHONY: clean-minimal
clean-minimal:
	@rm -f $(SCRDIR)/libxs_utilities.pyc
	@rm -rf $(SCRDIR)/__pycache__
	@touch $(SPLDIR)/cp2k/.make
	@touch $(SPLDIR)/smm/.make
	@touch $(SPLDIR)/nek/.make
	@touch $(INCDIR)/.make

.PHONY: clean
clean: clean-minimal
	@rm -f $(OBJECTS) $(FTNOBJS) $(SRCFILES)
	@rm -f $(BLDDIR)/libxs_dispatch.h

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
endif
endif
ifneq ($(abspath $(BINDIR)),$(ROOTDIR))
ifneq ($(abspath $(BINDIR)),$(abspath .))
	@rm -rf $(BINDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR)))
	@rm -f $(OUTDIR)/libxs.$(LIBEXT) $(OUTDIR)/mic/libxs.$(LIBEXT)
	@rm -f $(OUTDIR)/libxsf.$(LIBEXT) $(OUTDIR)/mic/libxsf.$(LIBEXT)
	@rm -f $(OUTDIR)/libxsld.$(LIBEXT) $(OUTDIR)/mic/libxsld.$(LIBEXT)
	@rm -f $(OUTDIR)/libxsgen.$(LIBEXT)
endif
ifneq (,$(wildcard $(BINDIR)))
	@rm -f $(BINDIR)/libxs_gemm_generator
endif
	@rm -f *.gcno *.gcda *.gcov
	@rm -f $(SPLDIR)/cp2k/cp2k-perf.sh
	@rm -f $(SPLDIR)/smm/smmf-perf.sh
	@rm -f $(SPLDIR)/nek/grad-perf.sh
	@rm -f $(SPLDIR)/nek/axhm-perf.sh
	@rm -f $(SPLDIR)/nek/rstr-perf.sh
	@rm -f $(INCDIR)/libxs.modmic
	@rm -f $(INCDIR)/libxs.mod
	@rm -f $(INCDIR)/libxs.f
	@rm -f $(INCDIR)/libxs.h
	@rm -f $(INCDIR)/.make
	@rm -f $(DOCDIR)/.make
	@rm -f .state

.PHONY: clean-all
clean-all: clean
	@cd $(TSTDIR) && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) clean-minimal
	@cd $(SPLDIR)/cp2k && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) clean-minimal
	@cd $(SPLDIR)/dgemm && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) clean-minimal
	@cd $(SPLDIR)/dispatch && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) clean-minimal
	@cd $(SPLDIR)/nek && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) clean-minimal
	@cd $(SPLDIR)/smm && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) clean-minimal

.PHONY: realclean-all
realclean-all: realclean
	@cd $(TSTDIR) && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) realclean
	@cd $(SPLDIR)/cp2k && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) realclean
	@cd $(SPLDIR)/dgemm && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) realclean
	@cd $(SPLDIR)/dispatch && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) realclean
	@cd $(SPLDIR)/nek && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) realclean
	@cd $(SPLDIR)/smm && $(MAKE) --no-print-directory DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) OFFLOAD=$(OFFLOAD) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) realclean

# Dummy prefix
ifneq (,$(strip $(PREFIX)))
INSTALL_ROOT = $(PREFIX)
else
INSTALL_ROOT = .
endif

.PHONY: install-minimal
install-minimal: lib generator
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXS installing binaries..."
	@mkdir -p $(INSTALL_ROOT)/$(POUTDIR) $(INSTALL_ROOT)/$(PBINDIR) $(INSTALL_ROOT)/$(PINCDIR)
	@cp -v $(OUTDIR)/libxsgen.$(DLIBEXT) $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsgen.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxs.$(DLIBEXT) $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxs.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsf.$(DLIBEXT) $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsf.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@if [ -e $(OUTDIR)/mic/libxs.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic ; \
		cp -uv $(OUTDIR)/mic/libxs.$(DLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic ; \
	fi
	@if [ -e $(OUTDIR)/mic/libxs.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic ; \
		cp -uv $(OUTDIR)/mic/libxs.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic ; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsf.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic ; \
		cp -uv $(OUTDIR)/mic/libxsf.$(DLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic ; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsf.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic ; \
		cp -uv $(OUTDIR)/mic/libxsf.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic ; \
	fi
	@cp -v $(BINDIR)/libxs_gemm_generator $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(INCDIR)/libxs*.h $(INSTALL_ROOT)/$(PINCDIR)
	@cp -v $(INCDIR)/libxs.f $(INSTALL_ROOT)/$(PINCDIR)
	@cp -v $(INCDIR)/*.mod* $(INSTALL_ROOT)/$(PINCDIR)
endif

.PHONY: install
install: install-minimal
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXS installing documentation..."
	@mkdir -p $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/$(DOCDIR)/*.pdf $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/$(DOCDIR)/*.md $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/version.txt $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/README.md $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/LICENSE $(INSTALL_ROOT)/$(PDOCDIR)
endif

.PHONY: install-all
install-all: install samples
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXS installing samples..."
	@cp -v $(addprefix $(SPLDIR)/cp2k/,cp2k cp2k.sh cp2k-perf* cp2k-plot.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/dgemm/,dgemm-blas dgemm-blas.sh dgemm-wrap dgemm-wrap.sh dgemm-test.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/dispatch/,dispatch dispatch.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/nek/,axhm grad rstr *.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,smm smm.sh smm-perf* smmf-perf.sh smm-plot.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,specialized specialized.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,dispatched dispatched.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,inlined inlined.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,blas blas.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
endif

.PHONY: install-dev
install-dev: install-all build-tests
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXS installing tests..."
	@mkdir -p $(INSTALL_ROOT)/$(PTSTDIR)
	@cp -v $(basename $(shell ls -1 ${TSTDIR}/*.c 2> /dev/null | tr "\n" " ")) $(INSTALL_ROOT)/$(PTSTDIR) 2> /dev/null || true
endif

# enable tracking last build
.PHONY: state
state: .state

