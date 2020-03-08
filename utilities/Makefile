CC=g++
NV=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_32
OPENMPFLAGS=-Xcompiler -openmp


all: global.o argument_parser.o graph.o timer.o

global.o : global.cpp
	$(CC) -c global.cpp -o global.o $(CFLAGS)

argument_parser.o : argument_parser.cpp
	$(CC) -c argument_parser.cpp -o argument_parser.o $(CFLAGS)

graph.o: graph.cpp
	$(CC) -c graph.cpp -o graph.o $(CFLAGS)

timer.o: timer.cpp
	$(CC) -c timer.cpp -o timer.o $(CFLAGS)

clean:
	rm *.o