.PHONY: all
.PHONY: clean

CPP = g++
NVCC = nvcc
BUILD_DIR = build
OUTPUT_DIR = bin

FRAMEWORK_INCLUDE = -Iframework/include
SIMULATION_INCLUDE = -Isimulation/include

FRAMEWORK_FILES = PlayerManager RandomPlayer
GAME_FILES = GameBoard
SIMULATION_FILES = main

mancala: GAME = mancala
mancala: setup $(FRAMEWORK_FILES) $(GAME_FILES) $(SIMULATION_FILES)
	$(CPP) -std=c++11 $(FRAMEWORK_INCLUDE) -Igames/$(GAME)/include $(SIMULATION_INCLUDE) $(BUILD_DIR)/*.o -o $(OUTPUT_DIR)/$@

$(FRAMEWORK_FILES):
	$(CPP) -std=c++11 -c -o $(BUILD_DIR)/$@.o $(FRAMEWORK_INCLUDE) -Igames/$(GAME)/include framework/src/$@.cpp

$(GAME_FILES):
	$(CPP) -std=c++11 -c -o $(BUILD_DIR)/$@.o $(FRAMEWORK_INCLUDE) -Igames/$(GAME)/include games/$(GAME)/src/$@.cpp

$(SIMULATION_FILES):
	$(CPP) -std=c++11 -c -o $(BUILD_DIR)/$@.o $(FRAMEWORK_INCLUDE) -Igames/$(GAME)/include $(SIMULATION_INCLUDE) simulation/src/$@.cpp

setup: clean
	mkdir -p $(OUTPUT_DIR)
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(OUTPUT_DIR)