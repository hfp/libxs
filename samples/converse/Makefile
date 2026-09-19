ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
PROJECT := libxs
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

BLAS = 0
OMP = 0
SYM = 1

include $(DEPDIR)/Makefile.inc

ifeq (,$(XLIB))
ifneq (,$(wildcard $(DEPDIR)/libxs/libxs_source.h))
  DFLAGS += -DLIBXS_SOURCE
  HEADERONLY = 1
endif
endif

IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/libxs)

HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(wildcard $(SRCDIR)/*.h) \
           $(wildcard $(DEPDIR)/libxs/*.h) \
           $(DEPDIR)/libxs/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
SOURCES := $(CSOURCS)
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/summarize.x $(OUTDIR)/converse.x \
          $(OUTDIR)/converse-qa.x $(OUTDIR)/converse-lm.x \
          $(OUTDIR)/converse-norm.x

# The corpus, the tokenizer, the word n-gram and the successor embedding: what
# both halves are built from, and what -L persists.
CONVERSE_CORE := $(BLDDIR)/converse-c.o
# The grounded half. converse_recomb.c belongs to it and names no model.
CONVERSE_QA := $(BLDDIR)/converse_qa-c.o $(BLDDIR)/converse_recomb-c.o
# The prediction half. It is the only half that may name the byte model, which
# reaches the other one as an installed judge or not at all.
CONVERSE_LM := $(BLDDIR)/converse_lm-c.o $(BLDDIR)/converse_hier-c.o
# The normalizer. It reads the lexicon as a codebook and names no half.
CONVERSE_NORM := $(BLDDIR)/converse_norm-c.o

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

$(OUTDIR)/summarize.x: $(OUTDIR)/.make $(BLDDIR)/summarize-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/summarize-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/converse.x: $(OUTDIR)/.make $(BLDDIR)/main_all-c.o \
  $(CONVERSE_CORE) $(CONVERSE_QA) $(CONVERSE_LM) $(XLIB)
	$(LD) -o $@ $(BLDDIR)/main_all-c.o $(CONVERSE_CORE) $(CONVERSE_QA) \
	  $(CONVERSE_LM) $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/converse-qa.x: $(OUTDIR)/.make $(BLDDIR)/main_qa-c.o \
  $(CONVERSE_CORE) $(CONVERSE_QA) $(XLIB)
	$(LD) -o $@ $(BLDDIR)/main_qa-c.o $(CONVERSE_CORE) $(CONVERSE_QA) \
	  $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/converse-lm.x: $(OUTDIR)/.make $(BLDDIR)/main_lm-c.o \
  $(CONVERSE_CORE) $(CONVERSE_LM) $(XLIB)
	$(LD) -o $@ $(BLDDIR)/main_lm-c.o $(CONVERSE_CORE) $(CONVERSE_LM) \
	  $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/converse-norm.x: $(OUTDIR)/.make $(BLDDIR)/main_norm-c.o \
  $(CONVERSE_CORE) $(CONVERSE_NORM) $(XLIB)
	$(LD) -o $@ $(BLDDIR)/main_norm-c.o $(CONVERSE_CORE) \
	  $(CONVERSE_NORM) $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR)))
	@-rm -f $(OBJECTS)
endif

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR)))
	@-rm -f $(XFILES) $(MODULES)
endif

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state *.bin

# Relink when only the link flags changed: those are tracked apart from .state so
# that a link-only change no longer recompiles every object, which leaves the link
# as the one step that still has to see them.
$(XFILES): $(LNKSTATE)
