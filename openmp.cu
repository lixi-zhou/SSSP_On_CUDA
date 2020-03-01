#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "./utilities/timer.hpp"
#include "./utilities/graph.hpp"
#include "./utilities/gpu_error_check.cuh"
#include "./utilities/global.hpp"
#include "./utilities/argument_parser.hpp"
#include <omp.h>


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
    uint *edgesSource = new uint[numEdges];
    uint *edgesEnd = new uint[numEdges];
    uint *edgesWeight = new uint[numEdges];

    for (int i = 0; i < numNodes; i++) {
        dist[i] = MAX_DIST;
        preNode[i] = uint(-1);
        processed[i] = false;
    }

    for (int i = 0; i < numEdges;i ++) {
        Edge edge = graph->edges.at(i);
        edgesSource[i] = edge.source;
        edgesEnd[i] = edge.end;
        edgesWeight[i] = edge.weight;

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
            // #pragma omp master 
            //     printf("master\n");
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
                // Edge edge = graph->edges.at(i);
                // uint source = edge.source;
                // uint end = edge.end;
                // uint weight = edge.weight;
                uint source = edgesSource[i];
                uint end = edgesEnd[i];
                uint weight = edgesWeight[i];

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

__global__ void sssp_GPU_Kernel(int splitIndex,
                                int numEdges,
                                int numEdgesPerThread,
                                uint *dist,
                                uint *preNode,
                                uint *edgesSource,
                                uint *edgesEnd,
                                uint *edgesWeight,
                                bool *finished) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = splitIndex + threadId * numEdgesPerThread;
    if (startId >= numEdges) {
        return;
    }

    int endId = splitIndex + (threadId + 1) * numEdgesPerThread;
    if (endId >= numEdges) {
        endId = numEdges;
    }

    // printf("GPU: process edged from: %d to %d \n", startId, endId);
    for (int nodeId = startId; nodeId < endId; nodeId++) {
        uint source = edgesSource[nodeId];
        uint end = edgesEnd[nodeId];
        uint weight = edgesWeight[nodeId];
        
        if (dist[source] + weight < dist[end]) {
            atomicMin(&dist[end], dist[source] + weight);
            preNode[end] = source;
            *finished = false;
        }
    }
}

uint* sssp_Hybrid(Graph *graph, int source) {
    int numNodes = graph->numNodes;
    int numEdges = graph->numEdges;
    uint *dist = new uint[numNodes];
    uint *preNode = new uint[numNodes];
    uint *edgesSource = new uint[numEdges];
    uint *edgesEnd = new uint[numEdges];
    uint *edgesWeight = new uint[numEdges];
    uint *dist_copy = new uint[numNodes];

    for (int i = 0; i < numNodes; i++) {
        dist[i] = MAX_DIST;
        preNode[i] = uint(-1);
    }


    for (int i = 0; i < numEdges; i++) {
        Edge edge = graph->edges.at(i);
        
        // Transfer the vector to the following three arrays
        edgesSource[i] = edge.source;
        edgesEnd[i] = edge.end;
        edgesWeight[i] = edge.weight;

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

    dist[source] = 0;
    preNode[source] = 0;

    uint *d_dist;
    uint *d_preNode;
    bool *d_finished;
    uint *d_edgesSource;
    uint *d_edgesEnd;
    uint *d_edgesWeight;

    gpuErrorcheck(cudaMalloc(&d_dist, numNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_preNode, numNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_edgesSource, numEdges * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_edgesEnd, numEdges * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_edgesWeight, numEdges * sizeof(uint)));

    gpuErrorcheck(cudaMemcpy(d_dist, dist, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_preNode, preNode, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgesSource, edgesSource, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgesEnd, edgesEnd, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgesWeight, edgesWeight, numEdges * sizeof(uint), cudaMemcpyHostToDevice));

    // Copy from gpu memory
    memcpy(dist_copy, dist, numNodes * sizeof(uint));

    Timer timer;
    int numIteration = 0;
    int numEdgesPerThread = 8;
    bool finished = false;
    bool h_finished = false;
    
    
    float splitRatio = 0.1; // cpu_data_size / whole_data_size
    /*
    CPU process edges from 0 to splitIndex   
        number of edges: splitIndex
    GPU process edges from splitIndex to numEdges 
        number of edges: numEdges - splitIndex + 1
    */
    int splitIndex = numEdges * splitRatio;
    int d_numEdgesPerThread = 8;
    int d_numThreadsPerBlock = 512;
    int d_numBlock = (numEdges - splitIndex + 1) / (d_numThreadsPerBlock * d_numEdgesPerThread) + 1;

    timer.start();
    while (!finished) {
        numIteration++;
        finished = true;
        h_finished = true;
        
        #pragma omp parallel num_threads(8)
        {
            int threadId = omp_get_thread_num();
            int h_numThreads = omp_get_num_threads();
            if (threadId == h_numThreads - 1) {
                // Last thread will be used to launch gpu kernel 
                // if thread 0 is used to launch gpu kernel, the first block of 
                // data whose index begining from 0 will not be processed.
                gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
                gpuErrorcheck(cudaMemcpy(d_dist, dist, sizeof(uint) * numNodes, cudaMemcpyHostToDevice));
                sssp_GPU_Kernel<<< d_numBlock, d_numThreadsPerBlock>>> (splitIndex,
                                                                        numEdges,
                                                                        d_numEdgesPerThread,
                                                                        d_dist,
                                                                        d_preNode,
                                                                        d_edgesSource,
                                                                        d_edgesEnd,
                                                                        d_edgesWeight,
                                                                        d_finished);
                gpuErrorcheck(cudaPeekAtLastError());
                gpuErrorcheck(cudaDeviceSynchronize()); 
                gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
                gpuErrorcheck(cudaMemcpy(dist_copy, d_dist, sizeof(uint) * numNodes, cudaMemcpyDeviceToHost));
            } else {
                // printf("Sub threads\n");
                int h_numEdgesPerThread = (splitIndex) / (h_numThreads - 1) + 1;
                int start = threadId * h_numEdgesPerThread;
                int end = (threadId + 1) * h_numEdgesPerThread;
                if (start > splitIndex) {
                    start = splitIndex;
                }
                if (end > splitIndex) {
                    end = splitIndex;
                }

                // cout << "Processs node: from " << start << " to: " << end << endl;
                // printf("Process node from: %d to : %d\n", start, end);
                for (int i = start; i < end; i++) {
                    uint source = edgesSource[i];
                    uint end = edgesEnd[i];
                    uint weight = edgesWeight[i];
                    
                    if (dist[source] + weight < dist[end]) {
                        dist[end] = dist[source] + weight;
                        preNode[end] = source;
                        h_finished = false;
                    }
                }
            }
        }

        finished = finished && h_finished;
        // printDist(dist, numNodes);
        // printDist(dist_copy, numNodes);
        if (!finished) {
            // Need to merge
            for (int i = 0; i < numNodes; i++) {
                if (dist[i] > dist_copy[i]) {
                    // Merge
                    dist[i] = dist_copy[i];
                }
            }
        }
        
    };
    timer.stop();

    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    printf("The execution time of SSSP on Hybrid(CPU-GPU): %f ms\n", timer.elapsedTime());

    gpuErrorcheck(cudaFree(d_dist));
    gpuErrorcheck(cudaFree(d_preNode));
    gpuErrorcheck(cudaFree(d_finished));
    gpuErrorcheck(cudaFree(d_edgesSource));
    gpuErrorcheck(cudaFree(d_edgesEnd));
    gpuErrorcheck(cudaFree(d_edgesWeight));


    return dist;
}



int main(int argc, char **argv) {
    ArgumentParser args(argc, argv);
    Graph graph(args.inputFilePath);
    //Graph graph("datasets/simpleGraph.txt");

    graph.readGraph();
    
    int sourceNode;

    if (args.hasSourceNode) {
        sourceNode = args.sourceNode;
    } else {
        // Use graph default source 
        sourceNode = graph.defaultSource;
    }

    // uint *dist_cpu_parallel = sssp_CPU_parallel(&graph, sourceNode);

    uint *dist_cpu_parallel = sssp_Hybrid(&graph, sourceNode);

    if (args.runOnCPU) {
        uint *dist_cpu = sssp_CPU(&graph, sourceNode);
        compareResult(dist_cpu, dist_cpu_parallel, graph.numNodes);
    }

    return 0;
}