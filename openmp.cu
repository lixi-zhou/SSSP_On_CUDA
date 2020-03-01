#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "./utilities/timer.hpp"
#include "./utilities/graph.hpp"
#include "./utilities/gpu_error_check.cuh"
#include "./utilities/global.hpp"
#include "./utilities/argument_parser.hpp"
#include <omp.h>


uint* array_sum(uint *A, uint *B, uint size) {
    uint *result = new uint[size];
    for (int i = 0; i < size; i++) {
        result[i] = A[i] + B[i];
    }

    return result;
}

uint* array_sum_open(uint *A, uint *B, uint size) {
    uint *result = new uint[size];

    #pragma omp parallel num_threads(8)
    {
        int threadId = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int numPerThread = size / numThreads;
        int start = threadId * numPerThread;
        int end = (threadId + 1) * numPerThread;
        if (end > size) {
            end = size;
        }

        for (int i = start; i < end; i++) {
            result[i] = A[i] + B[i];
        }
    }

    return result;
}

void openmpCompare() {
    printf("Start\n");

    // #pragma omp parallel num_threads(16)
    // {
    //     int threadId = omp_get_thread_num();
    //     printf("Thread: %d\n", threadId);
    //     // for (int i = 0; i < 5; i++) {
    //     //     printf("%d\n", i);
    //     // }
    // }

    uint size = 1000000000;
    uint *A = new uint[size];
    uint *B = new uint[size];
    srand((unsigned)time(NULL)); 
    Timer timer;

    //init 
    for (int i = 0; i < size; i++) {
        A[i] = rand() / 5000;
        B[i] = rand() / 5000;
    }

    timer.start();
    uint *result_ser = array_sum(A, B, size);
    timer.stop();
    printf("The execution time of on CPU: %f ms\n", timer.elapsedTime());

    timer.start();
    uint *result_par = array_sum_open(A, B, size);
    timer.stop();
    printf("The execution time of on CPU parallel: %f ms\n", timer.elapsedTime());


    compareResult(result_ser, result_par, size);
    // for (int i = 0; i < 10; i++) {
    //     printf("%d\n", result_ser[i]);
    // }


    printf("End\n");
}

uint* sssp_CPU(Graph* graph, int source){
    int numNodes = graph->numNodes;
    int numEdges = graph->numEdges;
    uint *dist = new uint[numNodes];
    uint *preNode = new uint[numNodes];
    bool *processed = new bool[numNodes];

    for (int i = 0; i < numNodes; i++) {
        dist[i] = MAX_DIST;
        preNode[i] = uint(-1);
        processed[i] = false;
    }


    for (int i = 0; i < numEdges; i++) {
        Edge edge = graph->edges.at(i);
        if (edge.source == source){
            if (edge.weight < dist[edge.end]){
                dist[edge.end] = edge.weight;
                preNode[edge.end] = source;
            }
        } else {
            // Case: edge.source != source
            continue;
        }
    }

    Timer timer;
    bool finished = false;
    uint numIteration = 0;

    dist[source] = 0;
    preNode[source] = 0;
    processed[source] = true;

    timer.start();
    while (!finished) {
        // uint minDist = MAX_DIST;
        finished = true;
        numIteration++;

        for (int i = 0; i < numEdges; i++){
            Edge edge = graph->edges.at(i);
            // Update its neighbor
            uint source = edge.source;
            uint end = edge.end;
            uint weight = edge.weight;

            if (dist[source] + weight < dist[end]) {
                dist[end] = dist[source] + weight;
                preNode[end] = source;
                finished = false;
            }
        }
        
    }
    timer.stop();
    

    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    printf("The execution time of SSSP on CPU: %f ms\n", timer.elapsedTime());

    return dist;
}



uint* sssp_CPU_parallel(Graph *graph, int source) {
    int numNodes = graph->numNodes;
    int numEdges = graph->numEdges;
    uint *dist = new uint[numNodes];
    uint *preNode = new uint[numNodes];
    bool *processed = new bool[numNodes];

    for (int i = 0; i < numNodes; i++) {
        dist[i] = MAX_DIST;
        preNode[i] = uint(-1);
        processed[i] = false;
    }

    for (int i = 0; i < numEdges;i ++) {
        Edge edge = graph->edges.at(i);
        if (edge.source == source) {
            if (edge.weight < dist[edge.end]) {
                dist[edge.end] = edge.weight;
                preNode[edge.end] = source;
            }
        } else {
            // Case: edge.source != source
            continue;
        }
    }
    
    Timer timer;
    bool finished = false;
    uint numIteration = 0;

    dist[source] = 0;
    preNode[source] = 0;
    processed[source] = true;

    timer.start();
    while(!finished) {
        finished = true;
        numIteration++;
        
        #pragma omp parallel 
        {   
            int threadId = omp_get_thread_num();
            int numThreads = omp_get_num_threads();
            int numEdgesPerThread = numEdges / numThreads + 1;
            int start = threadId * numEdgesPerThread;
            int end = (threadId + 1) * numEdgesPerThread;
            // cout << "Thread: " << threadId << " processing edges from: " << start << " to: " << end << endl;
            if (start > numEdges) {
                start = numEdges;
            }
            
            if (end > numEdges) {
                end = numEdges;
            }

            for (int i = start; i < end; i++) {
                Edge edge = graph->edges.at(i);
                uint source =edge.source;
                uint end = edge.end;
                uint weight = edge.weight;

                if (dist[source] + weight < dist[end]) {
                    // #pragma omp atomic
                    dist[end] = dist[source] + weight;
                    // #pragma omp atomic
                    preNode[end] = source;
                    finished = false;
                }
            }
        }
    }
    timer.stop();

    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    printf("The execution time of SSSP on CPU(OpenMP): %f ms\n", timer.elapsedTime());

    return dist;

}



int main(int argc, char **argv) {
    ArgumentParser args(argc, argv);
    Graph graph(args.inputFilePath);
    //  Graph graph("datasets/simpleGraph.txt");

    graph.readGraph();
    
    int sourceNode;

    if (args.hasSourceNode) {
        sourceNode = args.sourceNode;
    } else {
        // Use graph default source 
        sourceNode = graph.defaultSource;
    }

    uint *dist_cpu_parallel = sssp_CPU_parallel(&graph, sourceNode);

    if (args.runOnCPU) {
        uint *dist_cpu = sssp_CPU(&graph, sourceNode);
        compareResult(dist_cpu, dist_cpu_parallel, graph.numNodes);
    }

    return 0;
}