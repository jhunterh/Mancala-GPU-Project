.PHONY: all
.PHONY: clean

CPP = g++
NVCC = nvcc
INCLUDE = -I$(CURDIR)/include
OUTPUT = -o $(CURDIR)/bin/$@
SRC = $(CURDIR)/src

main:
	$(CPP) $(OUTPUT) $(INCLUDE) $(SRC)/$@.cpp

clean:
	rm -f build/*
	rm -f bin/*