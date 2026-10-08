# sm_89 = RTX 4090. Change ARCH for other GPUs (e.g. sm_86 for 3090, sm_90 for H100).
ARCH ?= sm_89

all: anneal wordcheck

anneal: anneal.cu hunt_core.h
	nvcc -O3 -std=c++17 -arch=$(ARCH) -o $@ anneal.cu

wordcheck: wordcheck.cpp
	g++ -O3 -std=c++17 -o $@ wordcheck.cpp

clean:
	rm -f anneal wordcheck

.PHONY: all clean
