.PHONY: setup clean

CPP = g++
NVCC = nvcc
BUILD_DIR = build
ARTIFACTS_DIR = build/artifacts
BIN_DIR = build/bin
INCLUDE_DIR = build/include
LIB_DIR = build/lib

FRAMEWORK_INCLUDE = -Iframework/include
GAME_INCLUDE = -Igames/$(game)/include
SIMULATION_INCLUDES = -Isimulation/include -Ibuild/include

FRAMEWORK_FILES = PlayerManager RandomPlayer
GAME_FILES = GameBoard
SIMULATION_FILES = main

GAME_LIB = $(LIB_DIR)/lib$(game).a

simulation: $(GAME_LIB)
	$(CPP) -std=c++11 $(SIMULATION_INCLUDES) simulation/src/*.cpp $(GAME_LIB) -o $(BUILD_DIR)/bin/$@

lib: $(GAME_LIB)
$(GAME_LIB): setup $(FRAMEWORK_FILES) $(GAME_FILES)
	ar rcs $(GAME_LIB) $(ARTIFACTS_DIR)/*.o
	cp framework/include/* $(INCLUDE_DIR)/
	cp games/$(game)/include/* $(INCLUDE_DIR)/
	rm -rf $(ARTIFACTS_DIR)

$(FRAMEWORK_FILES):
	$(CPP) -std=c++11 -c -o $(ARTIFACTS_DIR)/$@.o $(FRAMEWORK_INCLUDE) $(GAME_INCLUDE) framework/src/$@.cpp

$(GAME_FILES):
	$(CPP) -std=c++11 -c -o $(ARTIFACTS_DIR)/$@.o $(FRAMEWORK_INCLUDE) $(GAME_INCLUDE) games/$(game)/src/$@.cpp

setup: clean
	mkdir -p $(BUILD_DIR)
	mkdir -p $(ARTIFACTS_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(INCLUDE_DIR)
	mkdir -p $(LIB_DIR)

clean:
	rm -rf $(BUILD_DIR)