CC=g++
NV=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_32
OPENMPFLAGS=-Xcompiler -openmp

UTILITIES=utilities

all: sssp open openmpV2 benchmark

datasets:
	make -C datasets

sssp: sssp.cu 
	$(NV) -o sssp sssp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS)

open: openmp.cu
	$(NV) -o openmp openmp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

benchmark: benchmark.cu 
	$(NV) -o benchmark benchmark.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

temp: temp.cu 
	$(NV) -o temp temp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

temp1: temp1.cu 
	$(NV) -o temp1 temp1.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

temp3: temp3.cu 
	$(NV) -o temp3 temp3.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

temp4: temp4.cu 
	$(NV) -o temp4 temp4.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

temp5: temp5.cu 
	$(NV) -o temp5 temp5.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

openmpV2: openmpV2.cu 
	$(NV) -o openmpV2 openmpV2.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

openmpV3: openmpV3.cu 
	$(NV) -o openmpV3 openmpV3.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp $(UTILITIES)/global.cpp $(UTILITIES)/argument_parser.cpp $(CFLAGS) $(NFLAGS) $(OPENMPFLAGS)

clean:
	rm *.obj