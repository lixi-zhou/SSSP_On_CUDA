CC=g++
NV=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_32

# all: graph sssp

# graph: graph.cpp
# 	$(CC) -c graph.cpp 

# sssp: sssp.cu
# 	$(NV) -c sssp.cu graph.obj
# 	$(NV) sssp.obj graph.obj

all: sssp6

sssp1: sssp.cu graph.cpp timer.cpp
	$(NV) -o a.exe sssp.cu graph.cpp timer.cpp

sssp2: sssp2.cu graph.cpp timer.cpp
	$(NV) -o a.exe sssp2.cu graph.cpp timer.cpp

sssp3: sssp3.cu newGraph.cpp timer.cpp
	$(NV) -o a.exe sssp3.cu newGraph.cpp timer.cpp

sssp4: sssp4.cu newGraph.cpp timer.cpp
	$(NV) -o a.exe sssp4.cu newGraph.cpp timer.cpp

sssp6: sssp6.cu newGraph.cpp timer.cpp
	$(NV) -o a.exe sssp6.cu newGraph.cpp timer.cpp

clean:
	rm *.obj