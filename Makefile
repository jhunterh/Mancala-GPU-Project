.PHONY: setup clean

## Compiler
CPP = g++
NVCC = nvcc
CFLAGS = -std=c++11

## Directories
BUILD_DIR = build
ARTIFACTS_DIR = build/artifacts
BIN_DIR = build/bin
INCLUDE_DIR = build/include
LIB_DIR = build/lib

## Includes
FRAMEWORK_INCLUDE = -Iframework/include
GAME_INCLUDE = -Igames/$(game)/include
SIMULATION_INCLUDES = -Isimulation/include -Ibuild/include

## Files
FRAMEWORK_FILES = PlayerManager RandomPlayer MonteCarloPlayer MonteCarloPlayerMT
GAME_FILES = GameBoard
SIMULATION_FILES = main

## Libs
GAME_LIB = $(LIB_DIR)/lib$(game).a

## Targets
# Builds simulation executable
simulation: $(GAME_LIB)
	@$(CPP) $(CFLAGS) $(SIMULATION_INCLUDES) simulation/src/*.cpp $(GAME_LIB) -o $(BUILD_DIR)/bin/$@ -lpthread

# Builds game library
lib: $(GAME_LIB)
$(GAME_LIB): setup $(FRAMEWORK_FILES) $(GAME_FILES)
	@ar rcs $(GAME_LIB) $(ARTIFACTS_DIR)/*.o
	@cp framework/include/* $(INCLUDE_DIR)/
	@cp games/$(game)/include/* $(INCLUDE_DIR)/
	@rm -rf $(ARTIFACTS_DIR)

# Builds objects associated with framework
$(FRAMEWORK_FILES):
	@$(CPP) $(CFLAGS) -c -o $(ARTIFACTS_DIR)/$@.o $(FRAMEWORK_INCLUDE) $(GAME_INCLUDE) framework/src/$@.cpp

# Builds objects associated with the game definition
$(GAME_FILES):
	@$(CPP) $(CFLAGS) -c -o $(ARTIFACTS_DIR)/$@.o $(FRAMEWORK_INCLUDE) $(GAME_INCLUDE) games/$(game)/src/$@.cpp

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