CC=g++
NV=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_32
OPENMPFLAGS=-Xcompiler -openmp

UTILITIES=utilities

all: sssp open benchmark datasets

datasets:
	make -C datasets

sssp: sssp.cu 
	$(NV) -o sssp sssp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS)

open: openmp.cu
	$(NV) -o openmp openmp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

benchmark: benchmark.cu 
	$(NV) -o benchmark benchmark.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)


	
clean:
	rm *.obj