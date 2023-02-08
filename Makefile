.PHONY: all
.PHONY: clean

CPP = g++
NVCC = nvcc
INCLUDE = -I$(CURDIR)/include
OBJECT = $(CURDIR)/build
OUTPUT = -o $(CURDIR)/bin/$@
SRC = $(CURDIR)/src

main: GameManager RandomPlayer
	$(CPP) $(OUTPUT) $(INCLUDE) $(OBJECT)/*.o $(SRC)/$@.cpp

GameManager:
	$(CPP) -c -o $(OBJECT)/$@.o $(INCLUDE) $(SRC)/$@.cpp

RandomPlayer:
	$(CPP) -c -o $(OBJECT)/$@.o $(INCLUDE) $(SRC)/$@.cpp

clean:
	rm -f build/*
	rm -f bin/*