CC=g++
NV=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_32
OPENMPFLAGS=-Xcompiler -openmp

UTILITIES=utilities

all: sssp

sssp: sssp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp
	$(NV) -o sssp sssp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(NFLAGS)

# sssp8: sssp8.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp
# 	$(NV) -o dija sssp8.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp

open: openmp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp
	$(NV) -o openmp openmp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(NFLAGS) $(OPENMPFLAGS)



	
clean:
	rm *.obj