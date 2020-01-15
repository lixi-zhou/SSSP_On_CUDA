#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "./utilities/timer.hpp"
#include "./utilities/graph.hpp"
#include "./utilities/gpu_error_check.cuh"
// #include "graph.hpp"

/*

Version 3. Optimize the graph structure

*/


/* int main(){
    printf("hello");
    // NewGraph graph1("simpleGragh.txt");
    // Graph graph1("Wiki-Vote.txt");
    // graph1.readGraph();
    // NewGraph g1;
    NewGraph g1("Wiki-Vote.txt");
    return 0;
} */