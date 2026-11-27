ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = $(ROOTDIR)/../..
BLDDIR = obj
OUTDIR = .

# CP2K source tree, which provides the grid library and its miniapp
CP2K_ROOT ?= $(wildcard $(abspath $(DEPDIR)/../cp2k))
# LIBXSMM as kernel provider of LIBXS (0: none, 1: required), used if found
LIBXSMMROOT ?= $(wildcard $(abspath $(DEPDIR)/../libxsmm))
LIBXSMM ?= $(if $(wildcard $(LIBXSMMROOT)/lib/libxsmm.a),1,0)
# BLAS for the DGEMM backend and for the comparison of the test
BLAS_LIBS ?= -lblas
# arguments of the bench target: task lists (GLOBAL%GRID%DUMP_TASK_LISTS) and cycles
BENCH ?= $(NULL)
BENCHARG ?= 3

CFLAGS ?= -O3 -march=native
OMPFLAG ?= -fopenmp
# CP2K's sources, compiled with CP2K's flags rather than those of LIBXS
GRIDFLAGS := -std=c11 $(OMPFLAG) -Wall -Wextra -Wno-vla-parameter -D__LIBXS -I$(DEPDIR)
LIBS := $(BLAS_LIBS) -lm -ldl -lpthread

# LIBXS as library if built, header-only otherwise (which needs POSIX beyond C11)
XLIB := $(wildcard $(DEPDIR)/lib/libxs.a)
ifeq (,$(XLIB))
  GRIDFLAGS := $(patsubst -std=c11,-std=gnu11,$(GRIDFLAGS)) -DLIBXS_SOURCE
endif
ifneq (0,$(LIBXSMM))
  ifeq (,$(wildcard $(LIBXSMMROOT)/lib/libxsmm.a))
    $(error LIBXSMM=$(LIBXSMM) but $(LIBXSMMROOT)/lib/libxsmm.a does not exist)
  endif
  # CP2K's DGEMM backend includes libxsmm.h first, which lets LIBXS take its kernels
  GRIDFLAGS += -D__LIBXSMM -I$(LIBXSMMROOT)/include
  LIBS := $(LIBXSMMROOT)/lib/libxsmm.a $(LIBS)
endif

GRIDDIR := $(CP2K_ROOT)/src/grid
SRCDIR := $(CP2K_ROOT)/src
SOURCES := $(filter-out %/grid_miniapp.c %/grid_unittest.c, \
  $(wildcard $(GRIDDIR)/*.c $(GRIDDIR)/common/*.c $(GRIDDIR)/ref/*.c \
             $(GRIDDIR)/cpu/*.c $(GRIDDIR)/dgemm/*.c)) \
  $(SRCDIR)/offload/offload_buffer.c $(SRCDIR)/offload/offload_library.c \
  $(SRCDIR)/offload/offload_mempool.c $(SRCDIR)/mpiwrap/cp_mpi.c
OBJECTS := $(patsubst $(SRCDIR)/%.c,$(BLDDIR)/%.o,$(SOURCES))
HEADERS := $(wildcard $(GRIDDIR)/*.h $(GRIDDIR)/*/*.h $(SRCDIR)/offload/*.h \
  $(SRCDIR)/mpiwrap/*.h $(DEPDIR)/libxs/*.h)
MINIAPP := $(OUTDIR)/grid_miniapp.x
UNITTEST := $(OUTDIR)/grid_unittest.x

# rebuild when the flags change (e.g., LIBXSMM or XLIB given differently)
FLAGS := $(BLDDIR)/.flags
ifneq (,$(CP2K_ROOT))
  $(shell mkdir -p $(BLDDIR) && echo '$(CFLAGS) $(GRIDFLAGS) $(LIBS)' | cmp -s - $(FLAGS) \
    || echo '$(CFLAGS) $(GRIDFLAGS) $(LIBS)' >$(FLAGS))
endif

.PHONY: all
ifneq (,$(CP2K_ROOT))
all: miniapp
else
all:
	@echo "INFO: CP2K not found (CP2K_ROOT), the grid miniapp is not built"
endif

# CP2K's grid miniapp and unit test, built here rather than in CP2K's tree
.PHONY: miniapp
miniapp: $(MINIAPP) $(UNITTEST)

$(OUTDIR)/grid_%.x: $(BLDDIR)/grid/grid_%.o $(OBJECTS) $(XLIB) $(FLAGS)
	$(if $(CP2K_ROOT),,$(error CP2K_ROOT is not set))
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OMPFLAG) -o $@ $(filter-out $(FLAGS),$^) $(LIBS)

$(BLDDIR)/%.o: $(SRCDIR)/%.c $(HEADERS) $(FLAGS) $(ROOTDIR)/Makefile
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(GRIDFLAGS) -c $< -o $@

# CP2K's unit test, then the DGEMM backend's GEMMs by LIBXS against BLAS
.PHONY: test
test: miniapp
	@$(abspath $(UNITTEST)) $(CP2K_ROOT)/
ifneq (0,$(LIBXSMM))
	@$(ROOTDIR)/grid_test.sh $(abspath $(MINIAPP)) $(GRIDDIR)/sample_tasks $(abspath $(BLDDIR))/test
else
	@echo "INFO: without LIBXSMM, LIBXS hands every GEMM to BLAS (LIBXSMM=1 compares them)"
endif

# every given task list with each CPU backend, and the DGEMM backend with BLAS and LIBXS
.PHONY: bench
bench: miniapp
ifneq (,$(BENCH))
	@for FILE in $(BENCH); do \
		for SETUP in "cpu 0" "ref 0" "dgemm 3" $(if $(filter-out 0,$(LIBXSMM)),"dgemm 2"); do \
			set -- $${SETUP}; \
			echo "$${FILE} $${1}$$([ 0 = "$${2}" ] || echo " (LIBXS_GEMM_BACKEND=$${2})"): $$( \
				LIBXS_GEMM_BACKEND=$${2} $(abspath $(MINIAPP)) --backend $${1} $(BENCHARG) $${FILE} 2>&1 \
				| grep -o 'Time: [^ ]*')"; \
		done; \
	done
else
	@echo "INFO: BENCH names the task lists to replay (GLOBAL%GRID%DUMP_TASK_LISTS)"
endif

.PHONY: clean
clean:
	@rm -rf $(BLDDIR)

.PHONY: realclean
realclean: clean
	@rm -f $(MINIAPP) $(UNITTEST)
