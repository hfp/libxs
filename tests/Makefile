ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
PROJECT := libxs
DEPDIR = $(ROOTDIR)/..
SRCDIR = $(ROOTDIR)
INCDIR = $(ROOTDIR)
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -DLIBXS_BLAS_CONST

# PEDANTIC=2: OpenBLAS headers can cause warnings
override PEDANTIC = 1
BLAS ?= 1
OMP ?= 1
SYM ?= 1

EXCLUDE_STATE := DESTDIR PREFIX CURDIR OUTDIR TEST VERSION_STRING ALIAS_% LIBXS_BUILD_STATE %ROOT

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
#IFLAGS += -I$(call quote,$(DEPDIR)/src)

ifneq (0,$(shell echo "$$((100000>$(GCC_VERSION_NUM)))"))
  MIX ?= 1
else
  MIX ?= 0
endif

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
		   $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh) \
		   $(wildcard $(DEPDIR)/libxs/*.h) $(wildcard $(DEPDIR)/libxs/*.hpp) \
		   $(wildcard $(DEPDIR)/libxs/*.hxx) $(wildcard $(DEPDIR)/libxs/*.hh) \
		   $(DEPDIR)/libxs/libxs_source.h
CPPSRCS := $(shell grep -L '$(CMAIN)' $(SRCDIR)/*.cpp 2>/dev/null | tr -s "\n" " ")
CPPSRCX := $(shell grep -l '$(CMAIN)' $(SRCDIR)/*.cpp 2>/dev/null | tr -s "\n" " ")
CXXSRCS := $(shell grep -L '$(CMAIN)' $(SRCDIR)/*.cxx 2>/dev/null | tr -s "\n" " ")
CXXSRCX := $(shell grep -l '$(CMAIN)' $(SRCDIR)/*.cxx 2>/dev/null | tr -s "\n" " ")
CCXSRCS := $(shell grep -L '$(CMAIN)' $(SRCDIR)/*.cc  2>/dev/null | tr -s "\n" " ")
CCXSRCX := $(shell grep -l '$(CMAIN)' $(SRCDIR)/*.cc  2>/dev/null | tr -s "\n" " ")
CSOURCS := $(shell grep -L '$(CMAIN)' $(SRCDIR)/*.c   2>/dev/null | tr -s "\n" " ")
CSOURCX := $(shell grep -l '$(CMAIN)' $(SRCDIR)/*.c   2>/dev/null | tr -s "\n" " ")
ifneq (1,$(HEADERONLY))
FXXSRCS := $(shell grep -L '$(FMAIN)' $(SRCDIR)/*.f   2>/dev/null | tr -s "\n" " ")
FXXSRCX := $(shell grep -l '$(FMAIN)' $(SRCDIR)/*.f   2>/dev/null | tr -s "\n" " ")
F77SRCS := $(shell grep -L '$(FMAIN)' $(SRCDIR)/*.F   2>/dev/null | tr -s "\n" " ")
F77SRCX := $(shell grep -l '$(FMAIN)' $(SRCDIR)/*.F   2>/dev/null | tr -s "\n" " ")
F90SRCS := $(shell grep -L '$(FMAIN)' $(SRCDIR)/*.f90 2>/dev/null | tr -s "\n" " ") \
           $(shell grep -L '$(FMAIN)' $(SRCDIR)/*.F90 2>/dev/null | tr -s "\n" " ")
F90SRCX := $(shell grep -l '$(FMAIN)' $(SRCDIR)/*.f90 2>/dev/null | tr -s "\n" " ") \
           $(shell grep -l '$(FMAIN)' $(SRCDIR)/*.F90 2>/dev/null | tr -s "\n" " ")
endif
MODULES := $(addsuffix    .mod,$(basename $(FXXSRCS) $(F77SRCS) $(F90SRCS))) \
           $(addsuffix .modmic,$(basename $(FXXSRCS) $(F77SRCS) $(F90SRCS)))
OBJECTS := $(call objname,$(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS))
OBJECTX := $(call objname,$(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX))
FTNOBJS := $(call objname,$(FXXSRCS) $(F77SRCS) $(F90SRCS))
FTNOBJX := $(call objname,$(FXXSRCX) $(F77SRCX) $(F90SRCX))
XFILES  := $(addsuffix .x,$(addprefix $(OUTDIR)/,$(basename $(notdir \
           $(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX) \
           $(FXXSRCX) $(F77SRCX) $(F90SRCX)))))
ifeq (1,$(HEADERONLY))
  XFILES := $(filter-out $(OUTDIR)/headeronly.x,$(XFILES))
endif
TESTNAMES := $(basename $(notdir $(TEST)))
TESTFILES := $(if $(strip $(TEST)),$(foreach NAME,$(TESTNAMES),$(if $(wildcard $(SRCDIR)/$(NAME).sh),$(NULL),$(OUTDIR)/$(NAME).x)),all)

REALTYPE_INDEX := $(shell bash -c 'echo "$$((RANDOM%2+1))"' 2>/dev/null)
REALTYPE := $(word $(if $(REALTYPE_INDEX),$(REALTYPE_INDEX),1),double float)

SAMPLES := $(dir $(shell $(if $(GIT),$(GIT) ls-files,ls -1) \
  $(DEPDIR)/samples/*/Makefile 2>/dev/null))
ifeq (Darwin,$(UNAME))
  SAMPLES := $(filter-out %/ozaki/,$(SAMPLES))
endif

.PHONY: all
all: $(XFILES)
ifneq (Darwin,$(UNAME))
	@$(FLOCK) $(DEPDIR)/samples/ozaki "$(MAKE) --no-print-directory BLAS_STATIC=0"
	@$(FLOCK) $(DEPDIR)/samples/ozaki "$(MAKE) --no-print-directory"
endif
	@for SAMPLE in $(SAMPLES); do \
		$(FLOCK) $${SAMPLE} "$(MAKE) --no-print-directory"; \
	done

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

.PHONY: tests
tests: test

.PHONY: test
test: $(OUTDIR)/.make $(SRCDIR)/test.sh $(TESTFILES)
	@BLDDIR=$(abspath $(OUTDIR)) $(SRCDIR)/test.sh $(TEST)

# determine header-only tests (to avoid linking against LIBXS library; see below)
ifeq (1,$(HEADERONLY))
  HEADER_ONLY = $(basename $(notdir $(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX)))
else
  HEADER_ONLY = $(basename $(notdir $(shell grep -H libxs_source $(SRCDIR)/*.c | cut -d: -f1)))
endif

define DEFINE_LINK_LD_RULE
$(1).x: $(2) $(dir $(1))/.make Makefile $(DEPDIR)/Makefile.inc $(if $(filter $(1),$(HEADER_ONLY)),$(NULL),$(XLIB))
	$(LD) $(SLDFLAGS) -o $(1).x $(2) \
		$(if $(filter $(1),$(HEADER_ONLY)),$(NULL),$(XLIB)) \
		$(call cleanld,$(LDFLAGS) $(CLDFLAGS))
endef

define DEFINE_LINK_FC_RULE
ifneq (,$(strip $(FC)))
$(1).x: $(2) $(XLIB) $(dir $(1))/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(SLDFLAGS) -o $(1).x $(2) $(XLIB) $(FCMTFLAGS) \
		$(call cleanld,$(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS))
else
.PHONY: $(1).x
endif
endef

$(foreach SRC, $(filter-out $(SRCDIR)/headeronly.c,$(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX)), \
  $(eval $(call DEFINE_LINK_LD_RULE, $(basename $(notdir $(SRC))), $(call objname,$(SRC)))))

$(foreach SRC, $(FXXSRCX) $(F77SRCX) $(F90SRCX), \
  $(eval $(call DEFINE_LINK_FC_RULE, $(basename $(notdir $(SRC))), $(call objname,$(SRC)))))

ifeq (1,$(HEADERONLY))
# headeronly.x is redundant (multi-TU explicit test; all tests are header-only)
else ifeq (0,$(MIX))
$(OUTDIR)/headeronly.x: $(OUTDIR)/.make $(BLDDIR)/headeronly-c.o $(BLDDIR)/headeronly_aux-c.o Makefile $(DEPDIR)/Makefile.inc
	$(LD) -o $@ $(BLDDIR)/headeronly-c.o $(BLDDIR)/headeronly_aux-c.o \
		$(call cleanld,$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(NOBLASLIB))
else # compile headeronly_aux.c as C++ translation unit
$(OUTDIR)/headeronly.x: $(OUTDIR)/.make $(BLDDIR)/headeronly-c.o Makefile $(DEPDIR)/Makefile.inc
	@$(CP) $(SRCDIR)/headeronly_aux.c $(SRCDIR)/headeronly_aux.cpp
	$(CXX) $(DFLAGS) $(IFLAGS) $(CXXFLAGS) $(CTARGET) -c $(SRCDIR)/headeronly_aux.cpp -o $(BLDDIR)/headeronly_aux-cpp.o
	$(XLD) -o $@ $(BLDDIR)/headeronly_aux-cpp.o $(BLDDIR)/headeronly-c.o \
		$(call cleanld,$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(NOBLASLIB))
	@-rm -f headeronly_aux.cpp
endif

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc #$(DEPDIR)/include/libxs_source.h
	$(CXX) $(DFLAGS) $(IFLAGS) $(CXXFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/libxs/libxs_source.h
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

#$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxs_source.h
#$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

#$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxs_source.h
#$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

#$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxs_source.h
#$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

#$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxs_source.h
#$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

.PHONY: clean-private
clean-private:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJECTS) $(OBJECTX) $(FTNOBJS) $(FTNOBJX)
endif

.PHONY: realclean-private
realclean-private: clean-private
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(XFILES) $(MODULES)
endif

.PHONY: clean
clean: clean-private
	@for SAMPLE in $(SAMPLES) ozaki; do \
		$(FLOCK) $${SAMPLE} "$(MAKE) --no-print-directory clean"; \
	done

.PHONY: realclean
realclean: realclean-private
	@for SAMPLE in $(SAMPLES) ozaki; do \
		$(FLOCK) $${SAMPLE} "$(MAKE) --no-print-directory realclean"; \
	done

.PHONY: deepclean
deepclean: realclean-private
	@for SAMPLE in $(SAMPLES) ozaki; do \
		$(FLOCK) $${SAMPLE} "$(MAKE) --no-print-directory deepclean"; \
	done
	@-rm -f .make .state
