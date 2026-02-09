# ROOTDIR avoid abspath to match Makefile targets
ROOTDIR := $(subst //,,$(dir $(firstword $(MAKEFILE_LIST)))/)
# Source and scripts locations
ROOTINC := $(ROOTDIR)/include
ROOTSCR := $(ROOTDIR)/scripts
ROOTSRC := $(ROOTDIR)/src
# Project directory structure
INCDIR := include
SCRDIR := scripts
TSTDIR := tests
BLDDIR := obj
SRCDIR := src
OUTDIR := lib
BINDIR := bin
SPLDIR := samples
UTLDIR := $(SPLDIR)/utilities
DOCDIR := documentation

# subdirectories (relative) to PREFIX (install targets)
PINCDIR ?= $(INCDIR)
PSRCDIR ?= libxs
POUTDIR ?= $(OUTDIR)
PPKGDIR ?= $(OUTDIR)
PMODDIR ?= $(OUTDIR)
PBINDIR ?= $(BINDIR)
PTSTDIR ?= $(TSTDIR)
PSHRDIR ?= share/libxs
PDOCDIR ?= $(PSHRDIR)
LICFDIR ?= $(PDOCDIR)
LICFILE ?= LICENSE.md

# initial default flags: RPM_OPT_FLAGS are usually NULL
CFLAGS := $(RPM_OPT_FLAGS)
CXXFLAGS := $(RPM_OPT_FLAGS)
FCFLAGS := $(RPM_OPT_FLAGS)

# THRESHOLD problem size (M x N x K) determining when to use BLAS
# A value of zero (0) populates a default threshold
THRESHOLD ?= 0

# Generates M,N,K-combinations for each comma separated group, e.g., "1, 2, 3" generates (1,1,1), (2,2,2),
# and (3,3,3). This way a heterogeneous set can be generated, e.g., "1 2, 3" generates (1,1,1), (1,1,2),
# (1,2,1), (1,2,2), (2,1,1), (2,1,2) (2,2,1) out of the first group, and a (3,3,3) for the second group
# To generate a series of square matrices one can specify, e.g., make MNK=$(echo $(seq -s, 1 5))
# Alternative to MNK, index sets can be specified separately according to a loop nest relationship
# (M(N(K))) using M, N, and K separately. Please consult the documentation for further details.
MNK ?= 0

# Enable thread-local cache (registry)
# 0: "disable", 1: "enable", or small power-of-two number.
CACHE ?= 1

# Specify the size of a cacheline (Bytes)
CACHELINE ?= 64

# Determines if the library is thread-safe
THREADS ?= 1

# 0: link all dependencies as specified for the target
# 1: attempt to avoid dependencies if not referenced
ASNEEDED ?= 0

# Attempts to pin OpenMP based threads
AUTOPIN ?= 0
ifneq (0,$(AUTOPIN))
  DFLAGS += -DLIBXS_AUTOPIN
endif

# OpenMP is disabled by default and LIBXS is
# always agnostic wrt the threading runtime
OMP ?= 0

ifneq (1,$(CACHE))
  DFLAGS += -DLIBXS_CAPACITY_CACHE=$(CACHE)
endif

# disable lazy initialization and rely on ctor attribute
ifeq (0,$(INIT))
  DFLAGS += -DLIBXS_CTOR
endif

# Kind of documentation (internal key)
DOCEXT := pdf

# Timeout when downloading documentation parts
TIMEOUT := 30

# state to be excluded from tracking the (re-)build state
EXCLUDE_STATE := \
  DESTDIR PREFIX BINDIR CURDIR DOCDIR DOCEXT INCDIR LICFDIR OUTDIR TSTDIR TIMEOUT \
  PBINDIR PINCDIR POUTDIR PPKGDIR PMODDIR PSRCDIR PTSTDIR PSHRDIR PDOCDIR SCRDIR \
  SPLDIR UTLDIR SRCDIR TEST VERSION_STRING ALIAS_% BLAS %_TARGET %ROOT

# fixed .state file directory (included by source)
DIRSTATE := $(OUTDIR)/..

# avoid to link with C++ standard library
FORCE_CXX := 0

# enable additional/compile-time warnings
WCHECK := 1

# include common Makefile artifacts
include $(ROOTDIR)/Makefile.inc

# 0: static, 1: shared, 2: static and shared
ifneq (,$(filter-out file,$(origin STATIC)))
  ifneq (0,$(STATIC))
    BUILD := 0
  else # shared
    BUILD := 1
  endif
else # default
  BUILD := 2
endif

# target library for a broad range of systems
SSE ?= 1

ifneq (,$(MKL))
ifneq (0,$(MKL))
  BLAS := $(MKL)
endif
endif

ifneq (,$(MAXTARGET))
  DFLAGS += -DLIBXS_MAXTARGET=$(MAXTARGET)
endif

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(ROOTSRC))

ifeq (,$(PYTHON))
  $(info --------------------------------------------------------------------------------)
  $(error No Python interpreter found)
endif

# Version numbers according to interface (version.txt)
VERSION_MAJOR ?= $(shell $(ROOTSCR)/libxs_version.sh 1)
VERSION_MINOR ?= $(shell $(ROOTSCR)/libxs_version.sh 2)
VERSION_UPDATE ?= $(shell $(ROOTSCR)/libxs_version.sh 3)
VERSION_STRING ?= $(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_UPDATE)
VERSION_ALL ?= $(shell $(ROOTSCR)/libxs_version.sh 0)
VERSION_API ?= $(VERSION_MAJOR)
VERSION_RELEASED ?= $(if $(shell $(ROOTSCR)/libxs_version.sh 4),0,1)
VERSION_RELEASE ?= HEAD
VERSION_PACKAGE ?= 1

# Link shared library with correct version stamp
solink_version = $(call solink,$1,$(VERSION_MAJOR),$(VERSION_MINOR),$(VERSION_UPDATE),$(VERSION_API))

ifeq (0,$(BLAS))
ifneq (0,$(LNKSOFT))
ifeq (Darwin,$(UNAME))
  LDFLAGS += $(call linkopt,-U,_dgemm_)
  LDFLAGS += $(call linkopt,-U,_sgemm_)
  LDFLAGS += $(call linkopt,-U,_dgemv_)
  LDFLAGS += $(call linkopt,-U,_sgemv_)
endif
endif
endif

# target library for a broad range of systems
ifeq (file,$(origin AVX))
  AVX_STATIC := 0
endif
AVX_STATIC ?= $(AVX)

HEADERS_MAIN := \
          $(ROOTINC)/libxs.h \
          $(ROOTINC)/libxs_cpuid.h \
          $(ROOTINC)/libxs_hist.h \
          $(ROOTINC)/libxs_macros.h \
          $(ROOTINC)/libxs_malloc.h \
          $(ROOTINC)/libxs_math.h \
          $(ROOTINC)/libxs_mem.h \
          $(ROOTINC)/libxs_mhd.h \
          $(ROOTINC)/libxs_reg.h \
          $(ROOTINC)/libxs_rng.h \
          $(ROOTINC)/libxs_sync.h \
          $(ROOTINC)/libxs_timer.h \
          $(ROOTINC)/libxs_utils.h \
          $(NULL)
HEADERS_SRC := $(wildcard $(ROOTSRC)/*.h)
HEADERS := $(HEADERS_SRC) $(HEADERS_MAIN)
SRCFILES := $(patsubst %,$(ROOTSRC)/%, \
          libxs_cpuid_arm.c libxs_cpuid_rv64.c libxs_cpuid_x86.c \
          libxs_hash.c libxs_hist.c libxs_main.c libxs_malloc.c \
          libxs_math.c libxs_mem.c libxs_mhd.c libxs_reg.c \
          libxs_rng.c libxs_sync.c libxs_timer.c libxs_utils.c)

OBJFILES := $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES))))

ifneq (,$(strip $(FC)))
  FTNOBJS := $(BLDDIR)/intel64/libxs-mod.o
endif

# no warning conversion for released versions
ifneq (0,$(VERSION_RELEASED))
  WERROR := 0
endif
# no warning conversion for non-x86
#ifneq (x86_64,$(MNAME))
#  WERROR := 0
#endif
# no warning conversion
ifneq (,$(filter-out 0 1,$(INTEL)))
  WERROR := 0
endif

information = \
  $(info ================================================================================) \
  $(info LIBXS $(VERSION_ALL) ($(UNAME)$(if $(filter-out 0,$(LIBXS_TARGET_HIDDEN)),$(NULL),$(if $(HOSTNAME),@$(HOSTNAME))))) \
  $(info --------------------------------------------------------------------------------) \
  $(info $(GINFO)) \
  $(info $(CINFO)) \
  $(if $(strip $(FC)),$(info $(FINFO))) \
  $(if $(strip $(FC)),$(NULL), \
  $(if $(strip $(FC_VERSION)), \
  $(info Fortran Compiler $(FC_VERSION) is outdated!), \
  $(info Fortran Compiler is disabled or missing: no Fortran interface is built!))) \
  $(info --------------------------------------------------------------------------------) \
  $(if $(ENVSTATE),$(info Environment: $(ENVSTATE)) \
  $(info --------------------------------------------------------------------------------))

ifneq (,$(strip $(TEST)))
.PHONY: run-tests
run-tests: tests
endif

.PHONY: libxs
libxs: lib
	$(information)
ifneq (,$(filter _0_,_$(LNKSOFT)_))
ifeq (0,$(STATIC))
	$(info Building a shared library requires to link against BLAS)
	$(info since a deferred choice is not implemented for this OS.)
	$(info --------------------------------------------------------------------------------)
endif
endif

.PHONY: libs
libs: clib flib

.PHONY: lib
lib: libs

.PHONY: all
all: libxs

.PHONY: realall
realall: all samples

.PHONY: headers
headers: cheader fheader

.PHONY: header-only
header-only: cheader

.PHONY: interface
interface: headers module

.PHONY: winterface
winterface: headers sources

.PHONY: config
config: $(INCDIR)/libxs_version.h

$(INCDIR)/libxs_version.h: $(INCDIR)/.make $(ROOTSCR)/libxs_version.sh
	$(information)
	$(info --- LIBXS build log)
	@$(CP) -r $(ROOTSCR) . 2>/dev/null || true
	@$(CP) $(ROOTDIR)/Makefile.inc . 2>/dev/null || true
	@$(CP) $(ROOTDIR)/.mktmp.sh . 2>/dev/null || true
	@$(CP) $(ROOTDIR)/.flock.sh . 2>/dev/null || true
	@$(CP) $(ROOTDIR)/.state.sh . 2>/dev/null || true
	@$(CP) $(HEADERS_MAIN) $(INCDIR) 2>/dev/null || true
	@$(CP) $(SRCFILES) $(HEADERS_SRC) $(SRCDIR) 2>/dev/null || true
	@$(ROOTSCR)/libxs_version.sh -1 >$@

.PHONY: cheader
cheader: $(INCDIR)/libxs_source.h $(INCDIR)/libxs_version.h
$(INCDIR)/libxs_source.h: $(INCDIR)/.make $(ROOTSCR)/libxs_source.sh $(HEADERS_SRC) $(SRCFILES)
	@$(ROOTSCR)/libxs_source.sh >$@

define DEFINE_COMPILE_RULE
$(1): $(2) $(3) $(dir $(1))/.make
# @-rm -f $(1)
	-$(CC) $(if $(filter 0,$(WERROR)),$(4),$(filter-out $(WERROR_CFLAG),$(4)) $(WERROR_CFLAG)) -c $(2) -o $(1)
endef

ifneq (0,$(GLIBC))
  DFLAGS += -DLIBXS_BUILD=2
else
  DFLAGS += -DLIBXS_BUILD=1
endif

# build rules that include target flags
ifeq (0,$(CRAY))
$(foreach OBJ,$(OBJFILES),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(ROOTSRC)/%.c,$(notdir $(OBJ))), \
  $(HEADERS_MAIN) $(INCDIR)/libxs_version.h $(INCDIR)/libxs_source.h, \
  $(DFLAGS) $(IFLAGS) $(call applyif,1,libxs_main,$(OBJ),-I$(BLDDIR)) $(CTARGET) $(CFLAGS))))
else
$(foreach OBJ,$(filter-out $(BLDDIR)/intel64/libxs_mhd.o,$(OBJFILES)),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(ROOTSRC)/%.c,$(notdir $(OBJ))), \
  $(HEADERS_MAIN) $(INCDIR)/libxs_version.h $(INCDIR)/libxs_source.h, \
  $(DFLAGS) $(IFLAGS) $(call applyif,1,libxs_main,$(OBJ),-I$(BLDDIR)) $(CTARGET) $(CFLAGS))))
$(foreach OBJ,$(BLDDIR)/intel64/libxs_mhd.o,$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(ROOTSRC)/%.c,$(notdir $(OBJ))), \
  $(HEADERS_MAIN) $(INCDIR)/libxs_version.h $(INCDIR)/libxs_source.h, \
  $(DFLAGS) $(IFLAGS) $(CTARGET) $(patsubst $(OPTFLAGS),$(OPTFLAG1),$(CFLAGS)))))
endif

.PHONY: module
#ifneq (,$(strip $(FC)))
#module: $(INCDIR)/libxs.mod
#$(BLDDIR)/intel64/libxs-mod.o: $(BLDDIR)/intel64/.make $(INCDIR)/libxs.f
#	$(FC) $(DFLAGS) $(IFLAGS) $(FCMTFLAGS) $(filter-out $(FFORM_FLAG),$(FCFLAGS)) $(FTARGET) \
#		-c $(INCDIR)/libxs.f -o $@ $(FMFLAGS) $(INCDIR)
#$(INCDIR)/libxs.mod: $(BLDDIR)/intel64/libxs-mod.o
#	@if [ -e $(BLDDIR)/intel64/LIBXS.mod ]; then $(CP) $(BLDDIR)/intel64/LIBXS.mod $(INCDIR); fi
#	@if [ -e $(BLDDIR)/intel64/libxs.mod ]; then $(CP) $(BLDDIR)/intel64/libxs.mod $(INCDIR); fi
#	@if [ -e LIBXS.mod ]; then $(MV) LIBXS.mod $(INCDIR); fi
#	@if [ -e libxs.mod ]; then $(MV) libxs.mod $(INCDIR); fi
#	@-touch $@
#else
.PHONY: $(BLDDIR)/intel64/libxs-mod.o
.PHONY: $(INCDIR)/libxs.mod
#endif

.PHONY: clib
clib: $(OUTDIR)/libxs-static.pc $(OUTDIR)/libxs-shared.pc
ifeq (,$(filter-out 0 2,$(BUILD)))
$(OUTDIR)/libxs.$(SLIBEXT): $(OUTDIR)/.make $(OBJFILES)
	$(MAKE_AR) $(OUTDIR)/libxs.$(SLIBEXT) $(call tailwords,$^)
else
.PHONY: $(OUTDIR)/libxs.$(SLIBEXT)
endif
ifeq (0,$(filter-out 1 2,$(BUILD))$(ANALYZE))
$(OUTDIR)/libxs.$(DLIBEXT): $(OUTDIR)/.make $(OBJFILES)
	$(LIB_SOLD) $(call solink_version,$(OUTDIR)/libxs.$(DLIBEXT)) \
		$(call tailwords,$^) $(call cleanld,$(LDFLAGS) $(CLDFLAGS))
else
.PHONY: $(OUTDIR)/libxs.$(DLIBEXT)
endif

.PHONY: flib
#ifneq (,$(strip $(FC)))
#flib: $(OUTDIR)/libxsf-static.pc $(OUTDIR)/libxsf-shared.pc
#ifeq (,$(filter-out 0 2,$(BUILD)))
#$(OUTDIR)/libxsf.$(SLIBEXT): $(INCDIR)/libxs.mod $(OUTDIR)/libxs.$(DLIBEXT)
#	$(MAKE_AR) $(OUTDIR)/libxsf.$(SLIBEXT) $(BLDDIR)/intel64/libxs-mod.o
#else
#.PHONY: $(OUTDIR)/libxsf.$(SLIBEXT)
#endif
#ifeq (0,$(filter-out 1 2,$(BUILD))$(ANALYZE))
#$(OUTDIR)/libxsf.$(DLIBEXT): $(INCDIR)/libxs.mod $(OUTDIR)/libxs.$(DLIBEXT)
#ifneq (Darwin,$(UNAME))
#	$(LIB_SFLD) $(FCMTFLAGS) $(call solink_version,$(OUTDIR)/libxsf.$(DLIBEXT)) \
#		$(BLDDIR)/intel64/libxs-mod.o $(call abslib,$(OUTDIR)/libxs.$(ILIBEXT)) \
#		$(call cleanld,$(LDFLAGS) $(FLDFLAGS))
#else ifneq (0,$(LNKSOFT)) # macOS
#	$(LIB_SFLD) $(FCMTFLAGS) $(call solink_version,$(OUTDIR)/libxsf.$(DLIBEXT)) \
#		$(BLDDIR)/intel64/libxs-mod.o $(call abslib,$(OUTDIR)/libxs.$(ILIBEXT)) \
#		$(call cleanld,$(LDFLAGS) $(FLDFLAGS)) $(call linkopt,-U,_libxs_gemm_batch_omp_)
#else # macOS
#	$(LIB_SFLD) $(FCMTFLAGS) $(call solink_version,$(OUTDIR)/libxsf.$(DLIBEXT)) \
#		$(BLDDIR)/intel64/libxs-mod.o $(call abslib,$(OUTDIR)/libxs.$(ILIBEXT)) \
#		$(call cleanld,$(LDFLAGS) $(FLDFLAGS))
#endif
#else
.PHONY: $(OUTDIR)/libxsf.$(DLIBEXT)
#endif
#else
.PHONY: $(OUTDIR)/libxsf.$(SLIBEXT) $(OUTDIR)/libxsf.$(DLIBEXT)
#endif

# use dir not qdir to avoid quotes; also $(ROOTDIR)/$(SPLDIR) is relative
DIRS_SAMPLES := $(dir $(shell find $(ROOTDIR)/$(SPLDIR) -type f -name Makefile \
	$(NULL)))

.PHONY: samples $(DIRS_SAMPLES)
samples: $(DIRS_SAMPLES)
$(DIRS_SAMPLES): libs
	@$(FLOCK) $@ "$(MAKE)"

.PHONY: cp2k
cp2k: libs
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/cp2k "$(MAKE) --no-print-directory"

.PHONY: specfem
specfem: libs
	@$(FLOCK) $(ROOTDIR)/$(SPLDIR)/specfem "$(MAKE) --no-print-directory"

.PHONY: test-all
test-all: tests

.PHONY: test
test: tests

.PHONY: drytest
drytest: build-tests

.PHONY: build-tests
build-tests: libs
	@$(FLOCK) $(ROOTDIR)/$(TSTDIR) "$(MAKE) --no-print-directory"

.PHONY: tests
tests: libs
	@$(FLOCK) $(ROOTDIR)/$(TSTDIR) "$(MAKE) --no-print-directory test"

.PHONY: test-cp2k
test-cp2k: $(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-test.txt
$(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-test.txt: $(ROOTDIR)/$(SPLDIR)/cp2k/cp2k-perf.sh libs cp2k
	@$(FLOCK) $(call qdir,$@) "./cp2k-perf.sh $(call qndir,$@) $(shell echo $$(($(TESTSIZE)*128)))"

.PHONY: test-smm
ifneq (,$(strip $(FC)))
test-smm: $(ROOTDIR)/$(UTLDIR)/smmbench/smm-test.txt
$(ROOTDIR)/$(UTLDIR)/smmbench/smm-test.txt: $(ROOTDIR)/$(UTLDIR)/smmbench/smmf-perf.sh libs smm
	@$(FLOCK) $(call qdir,$@) "./smmf-perf.sh $(call qndir,$@) $(shell echo $$(($(TESTSIZE)*-128)))"
endif

$(DOCDIR)/index.md: $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/README.md
	@$(SED) $(ROOTDIR)/README.md \
		-e 's/\[!\[..*\](..*)\](..*)//g' \
		-e 's/\[\[..*\](..*)\]//g' \
		-e "s/](${DOCDIR}\//](/g" \
		-e 'N;/^\n$$/d;P;D' \
		>$@

$(DOCDIR)/libxs_scripts.md: $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTSCR)/README.md
	@$(SED) $(ROOTSCR)/README.md \
		-e 's/\[!\[..*\](..*)\](..*)//g' \
		-e 's/\[\[..*\](..*)\]//g' \
		-e "s/](${DOCDIR}\//](/g" \
		-e 'N;/^\n$$/d;P;D' \
		>$@

$(DOCDIR)/libxs_compat.md: $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/version.txt
	@wget -T $(TIMEOUT) -q -O $@ "https://raw.githubusercontent.com/wiki/libxs/libxs/Compatibility.md"
	@echo >>$@

$(DOCDIR)/libxs_valid.md: $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/version.txt
	@wget -T $(TIMEOUT) -q -O $@ "https://raw.githubusercontent.com/wiki/libxs/libxs/Validation.md"
	@echo >>$@

$(DOCDIR)/libxs_qna.md: $(DOCDIR)/.make $(ROOTDIR)/Makefile $(ROOTDIR)/version.txt
	@wget -T $(TIMEOUT) -q -O $@ "https://raw.githubusercontent.com/wiki/libxs/libxs/Q&A.md"
	@echo >>$@

$(DOCDIR)/libxs.$(DOCEXT): $(DOCDIR)/.make $(ROOTDIR)/$(DOCDIR)/index.md \
$(ROOTDIR)/$(DOCDIR)/libxs_mm.md $(ROOTDIR)/$(DOCDIR)/libxs_aux.md $(ROOTDIR)/$(DOCDIR)/libxs_prof.md \
$(ROOTDIR)/$(DOCDIR)/libxs_tune.md $(ROOTDIR)/$(DOCDIR)/libxs_be.md $(ROOTDIR)/$(DOCDIR)/libxs_scripts.md \
$(ROOTDIR)/$(DOCDIR)/libxs_compat.md $(ROOTDIR)/$(DOCDIR)/libxs_valid.md $(ROOTDIR)/$(DOCDIR)/libxs_qna.md
	$(eval TMPFILE = $(shell $(MKTEMP) $(ROOTDIR)/$(DOCDIR)/.libxs_XXXXXX.tex))
	@pandoc -D latex \
	| $(SED) \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily,showstringspaces=false}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		>$(TMPFILE)
	@cd $(ROOTDIR)/$(DOCDIR) && ( \
		iconv -t utf-8 index.md && echo && \
		echo "# LIBXS Domains" && \
		iconv -t utf-8 libxs_mm.md && echo && \
		iconv -t utf-8 libxs_aux.md && echo && \
		iconv -t utf-8 libxs_prof.md && echo && \
		iconv -t utf-8 libxs_tune.md && echo && \
		iconv -t utf-8 libxs_be.md && echo && \
		echo "# Appendix" && \
		$(SED) "s/^\(##*\) /#\1 /" libxs_compat.md | iconv -t utf-8 && \
		$(SED) "s/^\(##*\) /#\1 /" libxs_valid.md | iconv -t utf-8 && \
		$(SED) "s/^\(##*\) /#\1 /" libxs_scripts.md | iconv -t utf-8 && \
		$(SED) "s/^\(##*\) /#\1 /" libxs_qna.md | iconv -t utf-8; ) \
	| $(SED) \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--template=$(call qndir,$(TMPFILE)) --listings \
		-f gfm+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="LIBXS Documentation" \
		-V author-meta="Hans Pabst, Alexander Heinecke" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $(call qndir,$@)
	@rm $(TMPFILE)

$(DOCDIR)/libxs_samples.md: $(ROOTDIR)/Makefile $(ROOTDIR)/$(SPLDIR)/*/README.md $(ROOTDIR)/$(SPLDIR)/deeplearning/*/README.md $(ROOTDIR)/$(UTLDIR)/*/README.md
	@cd $(ROOTDIR)
	@if [ "$$(command -v git)" ] && [ "$$(git ls-files version.txt)" ]; then \
		git ls-files $(SPLDIR)/*/README.md $(SPLDIR)/deeplearning/*/README.md $(UTLDIR)/*/README.md | xargs -I {} cat {}; \
	else \
		cat $(SPLDIR)/*/README.md $(SPLDIR)/deeplearning/*/README.md $(UTLDIR)/*/README.md; \
	fi \
	| $(SED) \
		-e 's/^#/##/' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
		-e '1s/^/# [LIBXS Samples](https:\/\/github.com\/libxs\/libxs\/raw\/main\/documentation\/libxs_samples.pdf)\n\n/' \
		>$@

$(DOCDIR)/libxs_samples.$(DOCEXT): $(ROOTDIR)/$(DOCDIR)/libxs_samples.md
	$(eval TMPFILE = $(shell $(MKTEMP) .libxs_XXXXXX.tex))
	@pandoc -D latex \
	| $(SED) \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily,showstringspaces=false}/' \
		-e 's/\(\\usepackage.*{hyperref}\)/\\usepackage[hyphens]{url}\n\1/' \
		>$(TMPFILE)
	@iconv -t utf-8 $(ROOTDIR)/$(DOCDIR)/libxs_samples.md \
	| pandoc \
		--template=$(TMPFILE) --listings \
		-f gfm+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="LIBXS Sample Code Summary" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TMPFILE)

.PHONY: documentation
documentation: \
$(DOCDIR)/libxs.$(DOCEXT) \
$(DOCDIR)/libxs_samples.$(DOCEXT)

.PHONY: mkdocs
mkdocs: $(ROOTDIR)/$(DOCDIR)/index.md $(ROOTDIR)/$(DOCDIR)/libxs_samples.md
	@mkdocs build --clean
	@mkdocs serve

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(HEREDIR))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJFILES) $(FTNOBJS)
	@-rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(HEREDIR))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(OUTDIR)/libxs*.$(SLIBEXT) $(OUTDIR)/libxs*.$(DLIBEXT)*
	@-rm -f $(OUTDIR)/libxs*.pc
endif
ifneq ($(call qapath,$(BINDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BINDIR)),$(HEREDIR))
	@-rm -rf $(BINDIR)
endif
endif
	@-rm -f $(INCDIR)/libxs_version.h
	@-rm -f $(INCDIR)/libxs.mod
	@-rm -f $(INCDIR)/libxs.f

.PHONY: deepclean
deepclean: realclean
	@find . -type f \( -name .make -or -name .state \) -exec rm {} \;
	@-rm -rf $(ROOTSCR)/__pycache__
	@-rm -f $(HEREDIR)/python3

.PHONY: distclean
distclean: deepclean
	@find $(ROOTDIR)/$(SPLDIR) $(ROOTDIR)/$(TSTDIR) -type f -name Makefile -exec $(FLOCK) {} \
		"$(MAKE) --no-print-directory deepclean" \; 2>/dev/null || true
	@-rm -rf libxs*

# keep original prefix (:)
ALIAS_PREFIX := $(PREFIX)

# DESTDIR is used as prefix of PREFIX
ifneq (,$(strip $(DESTDIR)))
  override PREFIX := $(call qapath,$(DESTDIR)/$(PREFIX))
endif
# fall-back
ifeq (,$(strip $(PREFIX)))
  override PREFIX := $(HEREDIR)
endif

# setup maintainer-layout
ifeq (,$(strip $(ALIAS_PREFIX)))
  override ALIAS_PREFIX := $(PREFIX)
endif
ifneq ($(ALIAS_PREFIX),$(PREFIX))
  PPKGDIR := libdata/pkgconfig
  PMODDIR := $(PSHRDIR)
endif

# remove existing PREFIX
CLEAN ?= 0

.PHONY: install-minimal
install-minimal: libxs
ifneq ($(PREFIX),$(ABSDIR))
	@echo
ifneq (0,$(CLEAN))
#ifneq (,$(findstring ?$(HOMEDIR),?$(call qapath,$(PREFIX))))
	@if [ -d $(PREFIX) ]; then echo "LIBXS removing $(PREFIX)..." && rm -rf $(PREFIX) || true; fi
#endif
endif
	@echo "LIBXS installing libraries..."
	@$(MKDIR) -p $(PREFIX)/$(POUTDIR)
	@$(CP) -va $(OUTDIR)/libxsf*.$(DLIBEXT)* $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v  $(OUTDIR)/libxsf.$(SLIBEXT)  $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -va $(OUTDIR)/libxs*.$(DLIBEXT)* $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@$(CP) -v  $(OUTDIR)/libxs.$(SLIBEXT)  $(PREFIX)/$(POUTDIR) 2>/dev/null || true
	@echo
	@echo "LIBXS installing pkg-config and module files..."
	@$(MKDIR) -p $(PREFIX)/$(PPKGDIR)
	@$(CP) -va $(OUTDIR)/*.pc $(PREFIX)/$(PPKGDIR) 2>/dev/null || true
	@if [ ! -e $(PREFIX)/$(PMODDIR)/libxs.env ]; then \
		$(MKDIR) -p $(PREFIX)/$(PMODDIR); \
		$(CP) -v $(OUTDIR)/libxs.env $(PREFIX)/$(PMODDIR) 2>/dev/null || true; \
	fi
	@echo
	@echo "LIBXS installing interface..."
	@$(CP) -v  $(HEADERS_MAIN) $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -v  $(INCDIR)/libxs_version.h $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -v  $(INCDIR)/libxs.h $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -v  $(INCDIR)/libxs.f $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@$(CP) -va $(INCDIR)/*.mod* $(PREFIX)/$(PINCDIR) 2>/dev/null || true
	@echo
	@echo "LIBXS installing header-only..."
	@$(MKDIR) -p $(PREFIX)/$(PINCDIR)/$(PSRCDIR)
	@$(CP) -r $(ROOTSRC)/* $(PREFIX)/$(PINCDIR)/$(PSRCDIR) >/dev/null 2>/dev/null || true
# regenerate libxs_source.h
	@$(ROOTSCR)/libxs_source.sh $(PSRCDIR) >$(PREFIX)/$(PINCDIR)/libxs_source.h
endif

.PHONY: install
install: install-minimal
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "LIBXS installing documentation..."
	@$(MKDIR) -p $(PREFIX)/$(PDOCDIR)
	@$(CP) -va $(ROOTDIR)/$(DOCDIR)/*.pdf $(PREFIX)/$(PDOCDIR)
	@$(CP) -va $(ROOTDIR)/$(DOCDIR)/*.md $(PREFIX)/$(PDOCDIR)
	@$(CP) -v  $(ROOTDIR)/SECURITY.md $(PREFIX)/$(PDOCDIR)
	@$(CP) -v  $(ROOTDIR)/version.txt $(PREFIX)/$(PDOCDIR)
	@$(SED) "s/^\"//;s/\\\n\"$$//;/STATIC=/d" $(DIRSTATE)/.state >$(PREFIX)/$(PDOCDIR)/build.txt 2>/dev/null || true
	@$(MKDIR) -p $(PREFIX)/$(LICFDIR)
ifneq ($(call qapath,$(PREFIX)/$(PDOCDIR)/LICENSE.md),$(call qapath,$(PREFIX)/$(LICFDIR)/$(LICFILE)))
	@$(MV) $(PREFIX)/$(PDOCDIR)/LICENSE.md $(PREFIX)/$(LICFDIR)/$(LICFILE)
endif
endif

.PHONY: install-all
install-all: install build-tests
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "LIBXS installing tests..."
	@$(MKDIR) -p $(PREFIX)/$(PSHRDIR)/$(PTSTDIR)
	@$(CP) -v $(basename $(wildcard $(ROOTDIR)/$(TSTDIR)/*.c)) $(PREFIX)/$(PSHRDIR)/$(PTSTDIR) 2>/dev/null || true
endif

.PHONY: install-dev
install-dev: install
ifneq ($(PREFIX),$(ABSDIR))
	@if test -t 0; then \
		echo; \
		echo "================================================================================"; \
		echo "Installing development tools does not respect a common PREFIX, e.g., /usr/local."; \
		echo "For development, consider checking out https://github.com/hfp/libxs,"; \
		echo "or perform plain \"install\" (or \"install-all\")."; \
		echo "Hit CTRL-C to abort, or wait $(WAIT) seconds to continue."; \
		echo "--------------------------------------------------------------------------------"; \
		sleep $(WAIT); \
	fi
	@echo
	@echo "LIBXS installing utilities..."
	@$(MKDIR) -p $(PREFIX)
	@$(CP) -v $(ROOTDIR)/Makefile.inc $(PREFIX) 2>/dev/null || true
	@$(CP) -v $(ROOTDIR)/.mktmp.sh $(PREFIX) 2>/dev/null || true
	@$(CP) -v $(ROOTDIR)/.flock.sh $(PREFIX) 2>/dev/null || true
	@$(CP) -v $(ROOTDIR)/.state.sh $(PREFIX) 2>/dev/null || true
	@echo
	@echo "LIBXS tool scripts..."
	@$(MKDIR) -p $(PREFIX)/$(SCRDIR)
	@$(CP) -v $(ROOTSCR)/tool_getenvars.sh $(PREFIX)/$(SCRDIR) 2>/dev/null || true
	@$(CP) -v $(ROOTSCR)/tool_cpuinfo.sh $(PREFIX)/$(SCRDIR) 2>/dev/null || true
	@$(CP) -v $(ROOTSCR)/tool_pexec.sh $(PREFIX)/$(SCRDIR) 2>/dev/null || true
endif

.PHONY: install-realall
install-realall: install-all install-dev samples
ifneq ($(PREFIX),$(ABSDIR))
	@echo
	@echo "LIBXS installing samples..."
	@$(MKDIR) -p $(PREFIX)/$(PSHRDIR)/$(SPLDIR)
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/cp2k/,cp2k cp2k-perf* cp2k-plot.sh) $(PREFIX)/$(PSHRDIR)/$(SPLDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/hello/,hello helloc hellof) $(PREFIX)/$(PSHRDIR)/$(SPLDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/magazine/,magazine_batch magazine_blas magazine_xsmm benchmark.plt benchmark.set *.sh) \
						$(PREFIX)/$(PSHRDIR)/$(SPLDIR) 2>/dev/null || true
	@$(CP) -v $(addprefix $(ROOTDIR)/$(SPLDIR)/transpose/,transpose transposef) $(PREFIX)/$(PSHRDIR)/$(SPLDIR) 2>/dev/null || true
endif

ifeq (Windows_NT,$(UNAME))
  ALIAS_PRIVLIBS := $(call ldlib,$(LD),$(SLDFLAGS),dbghelp)
else ifneq (Darwin,$(UNAME))
  ifneq (FreeBSD,$(UNAME))
    ALIAS_PRIVLIBS := $(LIBPTHREAD) $(LIBRT) $(LIBDL) $(LIBM) $(LIBC)
  else
    ALIAS_PRIVLIBS := $(LIBDL) $(LIBM) $(LIBC)
  endif
endif
ifneq (,$(OMPFLAG_FORCE))
  ALIAS_PRIVLIBS_EXT := -fopenmp
endif

ALIAS_INCDIR := $(subst $$$$,$(if $(findstring $$$$/,$$$$$(PINCDIR)),,\$${prefix}/),$(subst $$$$$(ALIAS_PREFIX),\$${prefix},$$$$$(PINCDIR)))
ALIAS_LIBDIR := $(subst $$$$,$(if $(findstring $$$$/,$$$$$(POUTDIR)),,\$${prefix}/),$(subst $$$$$(ALIAS_PREFIX),\$${prefix},$$$$$(POUTDIR)))

ifeq (,$(filter-out 0 2,$(BUILD)))
$(OUTDIR)/libxs-static.pc: $(OUTDIR)/libxs.$(SLIBEXT)
	@echo "Name: libxs" >$@
	@echo "Description: Specialized tensor operations" >>$@
	@echo "URL: https://github.com/hfp/libxs/" >>$@
	@echo "Version: $(VERSION_STRING)" >>$@
	@echo >>$@
	@echo "prefix=$(ALIAS_PREFIX)" >>$@
	@echo "includedir=$(ALIAS_INCDIR)" >>$@
	@echo "libdir=$(ALIAS_LIBDIR)" >>$@
	@echo >>$@
	@echo "Cflags: -I\$${includedir}" >>$@
  ifneq (,$(ALIAS_PRIVLIBS))
  ifneq (Windows_NT,$(UNAME))
	@echo "Libs: -L\$${libdir} -l:libxs.$(SLIBEXT) $(ALIAS_PRIVLIBS)" >>$@
  else
	@echo "Libs: -L\$${libdir} -lxsmm $(ALIAS_PRIVLIBS)" >>$@
  endif
  else # no private libraries
	@echo "Libs: -L\$${libdir} -lxsmm" >>$@
  endif
  ifeq (,$(filter-out 0 2,$(BUILD)))
	@ln -fs $(notdir $@) $(OUTDIR)/libxs.pc
  endif
else
.PHONY: $(OUTDIR)/libxs-static.pc
endif

ifeq (,$(filter-out 0 2,$(BUILD)))
$(OUTDIR)/libxsf-static.pc: $(OUTDIR)/libxsf.$(SLIBEXT)
	@echo "Name: libxs/f" >$@
	@echo "Description: LIBXS for Fortran" >>$@
	@echo "URL: https://github.com/hfp/libxs/" >>$@
	@echo "Version: $(VERSION_STRING)" >>$@
	@echo >>$@
	@echo "prefix=$(ALIAS_PREFIX)" >>$@
	@echo "includedir=$(ALIAS_INCDIR)" >>$@
	@echo "libdir=$(ALIAS_LIBDIR)" >>$@
	@echo >>$@
	@echo "Requires: libxsext-static" >>$@
	@echo "Cflags: -I\$${includedir}" >>$@
  ifneq (Windows_NT,$(UNAME))
	@echo "Libs: -L\$${libdir} -l:libxsf.$(SLIBEXT)" >>$@
  else
	@echo "Libs: -L\$${libdir} -lxsmmf" >>$@
  endif
  ifeq (,$(filter-out 0 2,$(BUILD)))
	@ln -fs $(notdir $@) $(OUTDIR)/libxsf.pc
  endif
else
.PHONY: $(OUTDIR)/libxsf-static.pc
endif

ifeq (,$(filter-out 1 2,$(BUILD)))
$(OUTDIR)/libxs-shared.pc: $(OUTDIR)/libxs.$(DLIBEXT)
	@echo "Name: libxs" >$@
	@echo "Description: Specialized tensor operations" >>$@
	@echo "URL: https://github.com/hfp/libxs/" >>$@
	@echo "Version: $(VERSION_STRING)" >>$@
	@echo >>$@
	@echo "prefix=$(ALIAS_PREFIX)" >>$@
	@echo "includedir=$(ALIAS_INCDIR)" >>$@
	@echo "libdir=$(ALIAS_LIBDIR)" >>$@
	@echo >>$@
	@echo "Cflags: -I\$${includedir}" >>$@
  ifneq (,$(ALIAS_PRIVLIBS))
	@echo "Libs: -L\$${libdir} -lxsmm" >>$@
	@echo "Libs.private: $(ALIAS_PRIVLIBS)" >>$@
  else # no private libraries
	@echo "Libs: -L\$${libdir} -lxsmm" >>$@
  endif
  ifeq (,$(filter-out 1,$(BUILD)))
	@ln -fs $(notdir $@) $(OUTDIR)/libxs.pc
  endif
else
.PHONY: $(OUTDIR)/libxs-shared.pc
endif

ifeq (,$(filter-out 1 2,$(BUILD)))
$(OUTDIR)/libxsf-shared.pc: $(OUTDIR)/libxsf.$(DLIBEXT)
	@echo "Name: libxs/f" >$@
	@echo "Description: LIBXS for Fortran" >>$@
	@echo "URL: https://github.com/hfp/libxs/" >>$@
	@echo "Version: $(VERSION_STRING)" >>$@
	@echo >>$@
	@echo "prefix=$(ALIAS_PREFIX)" >>$@
	@echo "includedir=$(ALIAS_INCDIR)" >>$@
	@echo "libdir=$(ALIAS_LIBDIR)" >>$@
	@echo >>$@
	@echo "Requires: libxsext" >>$@
	@echo "Cflags: -I\$${includedir}" >>$@
	@echo "Libs: -L\$${libdir} -lxsmmf" >>$@
  ifeq (,$(filter-out 1,$(BUILD)))
	@ln -fs $(notdir $@) $(OUTDIR)/libxsf.pc
  endif
else
.PHONY: $(OUTDIR)/libxsf-shared.pc
endif

$(OUTDIR)/libxs.env: $(OUTDIR)/.make $(INCDIR)/libxs.h
	@echo "#%Module1.0" >$@
	@echo >>$@
	@echo "module-whatis \"LIBXS $(VERSION_STRING)\"" >>$@
	@echo >>$@
	@echo "set PREFIX \"$(ALIAS_PREFIX)\"" >>$@
	@echo "prepend-path PATH \"\$$PREFIX/bin\"" >>$@
	@echo "prepend-path LD_LIBRARY_PATH \"\$$PREFIX/lib\"" >>$@
	@echo >>$@
	@echo "prepend-path PKG_CONFIG_PATH \"\$$PREFIX/lib\"" >>$@
	@echo "prepend-path LIBRARY_PATH \"\$$PREFIX/lib\"" >>$@
	@echo "prepend-path CPATH \"\$$PREFIX/include\"" >>$@

.PHONY: deb
deb:
	@if [ "$$(command -v git)" ]; then \
		VERSION_ARCHIVE_SONAME=$$($(ROOTSCR)/libxs_version.sh 1); \
		VERSION_ARCHIVE=$$($(ROOTSCR)/libxs_version.sh 5); \
	fi; \
	if [ "$${VERSION_ARCHIVE}" ] && [ "$${VERSION_ARCHIVE_SONAME}" ]; then \
		ARCHIVE_AUTHOR_NAME="$$(git config user.name)"; \
		ARCHIVE_AUTHOR_MAIL="$$(git config user.email)"; \
		ARCHIVE_NAME=libxs$${VERSION_ARCHIVE_SONAME}; \
		ARCHIVE_DATE="$$(LANG=C date -R)"; \
		if [ "$${ARCHIVE_AUTHOR_NAME}" ] && [ "$${ARCHIVE_AUTHOR_MAIL}" ]; then \
			ARCHIVE_AUTHOR="$${ARCHIVE_AUTHOR_NAME} <$${ARCHIVE_AUTHOR_MAIL}>"; \
		else \
			echo "Warning: Please git-config user.name and user.email!"; \
			if [ "$${ARCHIVE_AUTHOR_NAME}" ] || [ "$${ARCHIVE_AUTHOR_MAIL}" ]; then \
				ARCHIVE_AUTHOR="$${ARCHIVE_AUTHOR_NAME}$${ARCHIVE_AUTHOR_MAIL}"; \
			fi \
		fi; \
		if ! [ -e $${ARCHIVE_NAME}_$${VERSION_ARCHIVE}.orig.tar.gz ]; then \
			git archive --prefix $${ARCHIVE_NAME}-$${VERSION_ARCHIVE}/ \
				-o $${ARCHIVE_NAME}_$${VERSION_ARCHIVE}.orig.tar.gz $(VERSION_RELEASE); \
		fi; \
		tar xf $${ARCHIVE_NAME}_$${VERSION_ARCHIVE}.orig.tar.gz; \
		cd $${ARCHIVE_NAME}-$${VERSION_ARCHIVE}; \
		$(MKDIR) -p debian/source; cd debian/source; \
		echo "3.0 (quilt)" >format; \
		cd ..; \
		echo "Source: $${ARCHIVE_NAME}" >control; \
		echo "Section: libs" >>control; \
		echo "Homepage: https://github.com/hfp/libxs/" >>control; \
		echo "Vcs-Git: https://github.com/hfp/libxs/libxs.git" >>control; \
		echo "Maintainer: $${ARCHIVE_AUTHOR}" >>control; \
		echo "Priority: optional" >>control; \
		echo "Build-Depends: debhelper (>= 13)" >>control; \
		echo "Standards-Version: 3.9.8" >>control; \
		echo >>control; \
		echo "Package: $${ARCHIVE_NAME}" >>control; \
		echo "Section: libs" >>control; \
		echo "Architecture: amd64" >>control; \
		echo "Depends: \$${shlibs:Depends}, \$${misc:Depends}" >>control; \
		echo "Description: Specialized tensor operations" >>control; \
		wget -T $(TIMEOUT) -qO- "https://api.github.com/repos/libxs/libxs" \
		| $(SED) -n 's/ *\"description\": \"\(..*\)\".*/\1/p' \
		| fold -s -w 79 | $(SED) -e 's/^/ /' -e 's/[[:space:]][[:space:]]*$$//' >>control; \
		echo "$${ARCHIVE_NAME} ($${VERSION_ARCHIVE}-$(VERSION_PACKAGE)) UNRELEASED; urgency=low" >changelog; \
		echo >>changelog; \
		wget -T $(TIMEOUT) -qO- "https://api.github.com/repos/libxs/libxs/releases/tags/$${VERSION_ARCHIVE}" \
		| $(SED) -n 's/ *\"body\": \"\(..*\)\".*/\1/p' \
		| $(SED) -e 's/\\r\\n/\n/g' -e 's/\\"/"/g' -e 's/\[\([^]]*\)\]([^)]*)/\1/g' \
		| $(SED) -n 's/^\* \(..*\)/\* \1/p' \
		| fold -s -w 78 | $(SED) -e 's/^/  /g' -e 's/^  \* /\* /' -e 's/^/  /' -e 's/[[:space:]][[:space:]]*$$//' >>changelog; \
		echo >>changelog; \
		echo " -- $${ARCHIVE_AUTHOR}  $${ARCHIVE_DATE}" >>changelog; \
		echo "#!/usr/bin/make -f" >rules; \
		echo "export DH_VERBOSE = 1" >>rules; \
		echo >>rules; \
		echo "%:" >>rules; \
		$$(which echo) -e "\tdh \$$@" >>rules; \
		echo >>rules; \
		echo "override_dh_auto_install:" >>rules; \
		$$(which echo) -e "\tdh_auto_install -- prefix=/usr" >>rules; \
		echo >>rules; \
		echo "13" >compat; \
		$(CP) ../LICENSE.md copyright; \
		rm -f ../$(TSTDIR)/mhd_test.mhd; \
		chmod +x rules; \
		debuild \
			-e PREFIX=debian/$${ARCHIVE_NAME}/usr \
			-e PDOCDIR=share/doc/$${ARCHIVE_NAME} \
			-e LICFILE=copyright \
			-e LICFDIR=../.. \
			-e SONAMELNK=1 \
			-e SYM=1 \
			-us -uc; \
	else \
		echo "Error: Git is unavailable or make-deb runs outside of cloned repository!"; \
	fi
