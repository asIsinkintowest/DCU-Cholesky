ROCM_PATH ?= /opt/rocm
HIPCC ?= $(ROCM_PATH)/bin/hipcc
MPICC ?= mpicc
CXX ?= g++

CFLAGS ?= -O3
HIPFLAGS ?= -O3
CXXFLAGS ?= -O3 -std=c++17

INCLUDES ?= -I$(ROCM_PATH)/include
ROCM_LIBDIR ?= -L$(ROCM_PATH)/lib

HIP_LIBS ?= -lhipsolver
ROC_LIBS ?= -lrocsolver -lrocblas
SCALAPACK_LIBS ?= -lscalapack -lblacs

BIN_DIR ?= build

HIP_SRC = src/hip_cholesky.cpp
ROC_SRC = src/roc_cholesky.cpp
SCALAPACK_SRC = src/scalapack_cholesky.c
RUN_BENCH_SRC = scripts/run_bench.cpp

HIP_BIN = $(BIN_DIR)/hip_cholesky
ROC_BIN = $(BIN_DIR)/roc_cholesky
SCALAPACK_BIN = $(BIN_DIR)/scalapack_cholesky
RUN_BENCH_BIN = $(BIN_DIR)/run_bench

all: $(HIP_BIN) $(ROC_BIN) $(SCALAPACK_BIN) $(RUN_BENCH_BIN)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(HIP_BIN): $(HIP_SRC) | $(BIN_DIR)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES) $< -o $@ $(ROCM_LIBDIR) $(HIP_LIBS)

$(ROC_BIN): $(ROC_SRC) | $(BIN_DIR)
	$(HIPCC) $(HIPFLAGS) $(INCLUDES) $< -o $@ $(ROCM_LIBDIR) $(ROC_LIBS)

$(SCALAPACK_BIN): $(SCALAPACK_SRC) | $(BIN_DIR)
	$(MPICC) $(CFLAGS) $< -o $@ $(SCALAPACK_LIBS)

$(RUN_BENCH_BIN): $(RUN_BENCH_SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	@rm -rf $(BIN_DIR)

.PHONY: all clean
