###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
# This file is part of the LIBXS library.                                     #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxs/                          #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Included after Makefile.inc by a sample that compares against XGBoost; the
# sample sets XGB and links $(XGBLDFLAGS).
# XGBoost for head-to-head comparison
ifneq (0,$(XGB))
  PKGCFG ?= $(call which,pkg-config)
  XGBOOST_ROOT ?= $(shell echo "$$XGBOOST_ROOT")
  # A source checkout carries the headers but no built library, so the library
  # is looked for under XGBOOST_ROOT rather than assumed, and XGBOOST_LIBDIR
  # names one held elsewhere (a directory of links to a library from a Python
  # wheel, say). Nothing is enabled without a library to link: a build that
  # links and then cannot load is worse than one without the comparison.
  XGBOOST_LIBDIR ?= $(shell echo "$$XGBOOST_LIBDIR")
  ifeq (,$(filter Darwin Windows_NT,$(UNAME)))
    XGB_OLDTAGS := $(call linkopt,--disable-new-dtags)
  endif
  ifneq (,$(wildcard $(XGBOOST_ROOT)/include/xgboost/c_api.h))
    ifeq (,$(XGBOOST_LIBDIR))
      XGBOOST_LIBDIR := $(patsubst %/,%,$(dir $(firstword $(wildcard \
        $(XGBOOST_ROOT)/lib64/libxgboost.* $(XGBOOST_ROOT)/lib/libxgboost.*))))
    endif
    # Nothing built under the root: ask the Python package where its own library
    # is. That library is not linkable as it stands, so what is named here is a
    # directory of links under the build tree rather than the package: the
    # library carries a SONAME that no file provides, and the dependencies it
    # vendored sit in a sibling directory. Both are supplied by the link farm
    # below, so nothing outside the build tree is written to.
    ifeq (,$(filter Darwin Windows_NT,$(UNAME)))
    ifeq (,$(XGBOOST_LIBDIR))
      XGBPY_LIB := $(shell $(PYTHON) -c \
        "import xgboost.core as c; print(c._LIB._name)" 2>/dev/null)
      ifneq (,$(wildcard $(XGBPY_LIB)))
        XGBPY_SOV := $(shell $(PYTHON) -c \
          "import xgboost; print(xgboost.__version__.split('.')[0])" 2>/dev/null)
        XGBPY_DEPS := $(shell $(PYTHON) -c "import os,glob,xgboost; \
          print(' '.join(glob.glob(os.path.dirname(xgboost.__file__)+'.libs/*')))" \
          2>/dev/null)
        # The wheel vendors its own OpenMP under a mangled name. Loading it
        # beside the one this library links gives two runtimes, each with a
        # thread pool spinning against the other: measured at 9x the build time
        # (65 s against 7 s over 80k rows) on a run that never calls XGBoost.
        # Linking that name to the runtime already in the process leaves one.
        XGBSYSOMP := $(shell $(CC) -print-file-name=libgomp.$(ILIBEXT) 2>/dev/null)
        ifeq (,$(wildcard $(XGBSYSOMP)))
          XGBSYSOMP := $(NULL)
        endif
        # The links are made while this file is read, not by a rule: the link
        # flags are built here too, and the helper that builds them skips a
        # directory that does not exist yet. A rule runs too late, which left
        # the first build on a clean tree with a bare -lxgboost: it failed
        # outright where no system copy was installed, and quietly resolved
        # against one of a different version where there was.
        XGBFARM := $(BLDDIR)/xgboost
        XGBOK := $(shell $(MKDIR) -p $(XGBFARM) \
          && ln -sf $(XGBPY_LIB) $(XGBFARM)/libxgboost.$(ILIBEXT) \
          && ln -sf $(XGBPY_LIB) $(XGBFARM)/libxgboost.$(ILIBEXT).$(XGBPY_SOV) \
          && for DEP in $(XGBPY_DEPS); do \
            B=`basename $$DEP`; \
            if test -z "$${B%%libgomp*}"; then \
              ln -sf $(if $(XGBSYSOMP),$(XGBSYSOMP),$$DEP) $(XGBFARM)/$$B; \
            else ln -sf $$DEP $(XGBFARM)/; fi; done \
          && echo ok)
        ifeq (ok,$(XGBOK))
          XGBOOST_LIBDIR := $(XGBFARM)
        endif
      endif
    endif
    endif
  endif
  ifneq (,$(and $(wildcard $(XGBOOST_ROOT)/include/xgboost/c_api.h),$(XGBOOST_LIBDIR)))
    IFLAGS += -I$(call quote,$(XGBOOST_ROOT)/include)
    DFLAGS += -D__XGBOOST
    # A library taken from a Python wheel is not self-contained: it names its
    # own dependencies (a vendored OpenMP among them) and finds them through its
    # own rpath, which the loader does not consult for a dependency of a
    # dependency. The old tag is inherited down the chain where RUNPATH is not,
    # so one directory resolves the library and whatever it drags in.
    XGBLDFLAGS := $(XGB_OLDTAGS) $(call abslibpath,$(XGBOOST_LIBDIR)) -lxgboost
  else ifneq (,$(and $(PKGCFG),$(shell $(PKGCFG) --exists xgboost 2>/dev/null && echo yes)))
    IFLAGS += $(shell $(PKGCFG) --cflags xgboost)
    DFLAGS += -D__XGBOOST
    XGBLDFLAGS := $(shell $(PKGCFG) --libs xgboost)
  endif
  # headers of a foreign project are not held to the warning policy of this one
  ifneq (,$(XGBLDFLAGS))
    WERROR_CFLAG := $(NULL)
  endif
endif
