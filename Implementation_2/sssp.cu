#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "./utilities/timer.hpp"
#include "./utilities/graph.hpp"
#include "./utilities/gpu_error_check.cuh"


/*

Dijkstra Algorithm with Adjacency list

*/


int main(){
    Graph graph("datasets/simpleGraph.txt");

    graph.readGraph();

    graph.printGraph();
}