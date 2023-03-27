.PHONY: setup clean main

## Compiler
CPP = g++
NVCC = nvcc
CFLAGS = -std=c++11 -O3
CUDA_ARCH = -arch=sm_80

ifeq ($(mode), debug)
	CFLAGS += -g
endif

## Directories
BUILD_DIR = build
ARTIFACTS_DIR = build/artifacts
BIN_DIR = build/bin
INCLUDE_DIR = build/include
LIB_DIR = build/lib
CUDAPATH = /opt/asn/apps/cuda_11.7.0

## Includes
FRAMEWORK_INCLUDE = -Iframework/include -I$(CUDAPATH)/samples/common/inc
GAME_INCLUDE = -Igames/$(game)/include
SIMULATION_INCLUDES = -Isimulation/include -Ibuild/include

## Files
FRAMEWORK_FILES = Logger PlayerManager RandomPlayer MonteCarloPlayer MonteCarloPlayerMT MonteCarloHybridPlayer PureMonteCarloPlayer NaivePureMonteCarloPlayer Timer
FRAMEWORK_CUDA_FILES = MonteCarloUtility
GAME_FILES = GameBoard
SIMULATION_FILES = main

## Links
LINKS = -lpthread
CUDA_LINKS = -L$(CUDAPATH)/lib64 -lcudart

## Libs
GAME_LIB = $(LIB_DIR)/lib$(game).a

## Targets
main: simulation test

# Builds simulation executable
simulation: $(GAME_LIB)
	@$(NVCC) $(CFLAGS) $(SIMULATION_INCLUDES) $(CUDA_LINKS) $(LINKS) simulation/src/*.cpp $(GAME_LIB) -o $(BUILD_DIR)/bin/$@ 

# Builds unit test executable
test: $(GAME_LIB)
	@$(NVCC) $(CFLAGS) $(SIMULATION_INCLUDES) $(CUDA_LINKS) $(LINKS) framework/test/mcts/*.cpp $(GAME_LIB) -o $(BUILD_DIR)/bin/$@

# Builds game library
lib: $(GAME_LIB)
$(GAME_LIB): setup $(FRAMEWORK_FILES) $(FRAMEWORK_CUDA_FILES) $(GAME_FILES)
	@ar rcs $(GAME_LIB) $(ARTIFACTS_DIR)/*.o
	@cp framework/include/* $(INCLUDE_DIR)/
	@cp games/$(game)/include/* $(INCLUDE_DIR)/
	@rm -rf $(ARTIFACTS_DIR)

# Builds objects associated with framework
$(FRAMEWORK_FILES):
	@$(CPP) $(CFLAGS) -c -o $(ARTIFACTS_DIR)/$@.o $(FRAMEWORK_INCLUDE) $(GAME_INCLUDE) framework/src/$@.cpp

# Builds cuda objects associated with framework
$(FRAMEWORK_CUDA_FILES):
	@$(NVCC) $(CUDA_ARCH) $(CFLAGS) -x cu -dc -o $(ARTIFACTS_DIR)/$@.o $(FRAMEWORK_INCLUDE) $(GAME_INCLUDE) $(CUDA_LINKS) framework/src/$@.cu

# Builds objects associated with the game definition
$(GAME_FILES):
	@$(NVCC) $(CUDA_ARCH) $(CFLAGS) -x cu -dc -o $(ARTIFACTS_DIR)/$@.o $(FRAMEWORK_INCLUDE) $(GAME_INCLUDE) $(CUDA_LINKS) games/$(game)/src/$@.cpp

# Setups build enviroment
setup: clean
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(ARTIFACTS_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(INCLUDE_DIR)
	@mkdir -p $(LIB_DIR)

# Cleans build directory
clean:
	@rm -rf $(BUILD_DIR)

# Launches run script
run: build/bin/simulation
	@rm -f *.nsys-rep *.i* *.o* core.*
	@echo -ne "gpu\n4\n\n1gb\n1\nampere\ngame_simulation\n" | \
		run_gpu .runSimulation.sh > /dev/null
	@sleep 5
	@tail -f *.o*
