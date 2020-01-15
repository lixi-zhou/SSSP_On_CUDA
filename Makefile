CC=g++
NV=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_32

UTILITIES=utilities

all: sssp7

sssp1: sssp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp
	$(NV) -o a.exe sssp.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp

sssp2: sssp2.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp
	$(NV) -o a.exe $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp

# sssp3: sssp3.cu newGraph.cpp timer.cpp
# 	$(NV) -o a.exe sssp3.cu newGraph.cpp timer.cpp

# sssp4: sssp4.cu newGraph.cpp timer.cpp
# 	$(NV) -o a.exe sssp4.cu newGraph.cpp timer.cpp

sssp6: sssp6.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp
	$(NV) -o a.exe sssp6.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp

sssp7: sssp7.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp
	$(NV) -o a.exe sssp7.cu $(UTILITIES)/graph.cpp $(UTILITIES)/timer.cpp
	
clean:
	rm *.obj