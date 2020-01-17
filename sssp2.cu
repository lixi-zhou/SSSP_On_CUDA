#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "./utilities/timer.hpp"
#include "./utilities/graph.hpp"
#include "./utilities/gpu_error_check.cuh"

/*

Basic implementation

*/

int numNodes;
int numEdges;

int* dist;
int* previousNode;
int ** graph;
bool* finished;
int* graph_static;

void init(Graph* graphData, int source) {
    numNodes = graphData->numNodes;
    graph = graphData->graph;

    int size = numNodes;

    dist = new int[size];
    previousNode = new int[size];
    finished = new bool[size];

    for (int i = 0; i < size; i++) {
        // set the distance to the source node
        dist[i] = graph[source][i];
        finished[i] = false;
        if (dist[i] == MAX_DIST) {
            previousNode[i] = -1;
        }
        else {
            previousNode[i] = source;
        }
    }
    // set dist[source] = 0
    dist[source] = 0;

}

void printShortestDistance(int source) {
    int diameter = 0;
    for (int i = 0; i < numNodes; i++) {
        if(dist[i] != MAX_DIST){
            if(dist[i] > diameter){
                diameter = dist[i];
            }
            // printf("Shortest distance from node: %d to source: %d: is: %d\n", i, source, dist[i]);
        }else{
            // printf("Shortest distance from node: %d to source: %d: is: INF\n", i, source);
        } 
    }
    printf("Diameter: %d\n", diameter);
}

void printGraph(){
    printf("\n\nGraph");
    for (int i = 0; i < numNodes; i++){
        for (int j = 0; j < numNodes; j++){
            printf("%d ", graph[i][j]);
        }
        printf("\n");
    }
}

void printFinished(){
    printf("Finished array\n");
    for(int i = 0; i < numNodes; i++){
        printf("Node: %d, status: %d\n", i, finished[i]);
    }
    printf("\n");
}

void imcompletedAndConnectedNode(){
    int count = 0;
    for(int i = 0; i < numNodes; i++){
        if((!finished[i] && (dist[i] != MAX_DIST))){
            count++;
        }
    }
    printf("Imcompleted Nodes Number: %d\n", count);
}

void dijkstraOnCPU(int source) {
    Timer timer;
    int size = numNodes;
    int numIteration = 0;
    
    // Find the connected nodes to the source point
    // Set the source point
    dist[source] = 0;
    finished[source] = true;

    timer.start();
    for (int i = 0; i < size; i++) {
        int mindist = MAX_DIST;
        // U is the closet point to source, u is not finished yet
        int u = source;

        numIteration++;

        for (int j = 0; j < size; j++) {
            if ((j != u) && (!finished[j]) && dist[j] < mindist) {
                u = j;
                mindist = dist[j];
            }
        }
        if (u == source){
            // Completed
            break;
        }

        finished[u] = true;

        for (int j = 0; j < size; j++) {
            if ((j != u) && (!finished[j]) && graph[u][j] < MAX_DIST) {
                // Find the shorter path
                if (dist[u] + graph[u][j] < dist[j]) {
                    // Update dist
                    dist[j] = dist[u] + graph[u][j];
                    // Update its previous point
                    previousNode[j] = u;
                }
            }
        }
    }
    printf("Number of Iteration Executed: %d\n", numIteration);
    printf("The execution time of SSSP on CPU: %d ms\n", timer.stop());
    // printShortestDistance(0);
}

__global__ void dijkstraOnGPU_kernel1(int numNodes, 
                                int sourceId,
                                int partSize,
                                int* graphData,
                                bool* finished,
                                int* dist,
                                int* prev,
                                int* closestNodeId,
                                int* minimumDist,
                                bool* completed) {
    // kernel to compute the closest node 
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startNodeId = threadId * partSize;
    int endNodeId = (threadId + 1) * partSize;
    if(endNodeId > numNodes){
        endNodeId = numNodes;
    } 

    if(startNodeId > numNodes) return; 

    // printf("Thread: %d process data from: %d to %d \n", threadId, startNodeId, endNodeId);


    for(int nodeId = startNodeId; nodeId < endNodeId; nodeId++){
        if (!finished[nodeId] && dist[nodeId] < *minimumDist){
            *closestNodeId = nodeId;
            *minimumDist = dist[nodeId];
            *completed = false;
        }
    }

}

__global__ void dijkstraOnGPU_kernel2(int numNodes, 
                                        int sourceId,
                                        int partSize,
                                        int* graphData,
                                        bool* finished,
                                        int* dist,
                                        int* prev,
                                        int* closestNodeId,
                                        int GRAPH_MAX_SIZE) {

    // Based on closest node then update its connected node
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startNodeId = threadId * partSize;
    int endNodeId = (threadId + 1) * partSize;
    if(endNodeId > numNodes){
        endNodeId = numNodes;
    } 
    // int nodeId = threadId;

    if(startNodeId > numNodes) return;

    for(int nodeId = startNodeId; nodeId < endNodeId; nodeId++){
        // Convert to 1-D index
        int index = *closestNodeId * GRAPH_MAX_SIZE + nodeId;
        finished[*closestNodeId] = true;
        // printf("graphData[%d][%d]: is %d \n", *closestNodeId, nodeId, graphData[*closestNodeId][nodeId]);
        if (!finished[nodeId] && graphData[index] < MAX_DIST){
            // Find the shorter path
            if(dist[*closestNodeId] + graphData[index] < dist[nodeId]){
                // Update dist
                dist[nodeId] = dist[*closestNodeId] + graphData[index];
                // Update its previous point
                prev[nodeId] = *closestNodeId;
            }
        }
    }
}

void dijkstraOnGPU(int source){
    Timer timer;
    cudaFree(0);
    // Define CPU vars
    int closestNodeId = 6;
    // Define GPU vars
    int* d_graph;   // 2D array is converted to 1-D, row = i / cols, col = i % cols;
    int* d_dist;
    int* d_prev;
    bool* d_finished;
    int* d_closestNodeId;
    int* d_minimumDist;
    bool* d_completed;


    
    gpuErrorcheck(cudaMalloc((void **)&d_graph, GRAPH_MAX_SIZE * GRAPH_MAX_SIZE * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_dist, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_prev, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_finished, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_closestNodeId, sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_completed, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_minimumDist, sizeof(int)));

    gpuErrorcheck(cudaMemcpy(d_graph, graph[0], GRAPH_MAX_SIZE * GRAPH_MAX_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_dist, dist, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_prev, previousNode, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_finished, finished, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_closestNodeId, &closestNodeId, sizeof(int), cudaMemcpyHostToDevice));

    bool completed = true;
    int minimumDist = MAX_DIST;
    int numIteration = 0;

    
    // Each block has 128 threads
   
    int numNodesPerPart = 32;
    int numThreadPerBlock = 64;
    int numBlock = (numNodes) / (numNodesPerPart * numThreadPerBlock) + 1;
    
    timer.start();
    do{
        numIteration++;
        completed = true;
        minimumDist = MAX_DIST;

        gpuErrorcheck(cudaMemcpy(d_completed, &completed, sizeof(bool), cudaMemcpyHostToDevice));
        gpuErrorcheck(cudaMemcpy(d_minimumDist, &minimumDist, sizeof(int), cudaMemcpyHostToDevice));

        if(numIteration % 2 == 1){
            dijkstraOnGPU_kernel1<<< numBlock, numThreadPerBlock >>>(numNodes,
                source,
                numNodesPerPart,
                d_graph,
                d_finished,
                d_dist,
                d_prev,
                d_closestNodeId,
                d_minimumDist,
                d_completed);
            gpuErrorcheck(cudaMemcpy(&completed, d_completed, sizeof(bool), cudaMemcpyDeviceToHost));
        }else{
            dijkstraOnGPU_kernel2<<<numBlock, numThreadPerBlock>>>(numNodes,
                source,
                numNodesPerPart,
                d_graph,
                d_finished,
                d_dist,
                d_prev,
                d_closestNodeId,
                GRAPH_MAX_SIZE);
            completed = false;
        }

        
        gpuErrorcheck(cudaPeekAtLastError());
        gpuErrorcheck(cudaDeviceSynchronize());  
    }while(!completed);

    printf("Number of Iteration Executed: %d\n", numIteration);
    printf("The execution time of SSSP on GPU: %d ms\n", timer.stop());
    cudaMemcpy(&closestNodeId, d_closestNodeId, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dist, d_dist, numNodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_graph);
    cudaFree(d_dist);
    cudaFree(d_prev);
    cudaFree(d_finished);
    cudaFree(d_closestNodeId);
    cudaFree(d_minimumDist);
    cudaFree(d_completed);
}


int main() {

    // Graph graph1("datasets/simpleGragh1.txt");
    // Graph graph1("datasets/email-Eu-core-SIMPLE.txt");
    Graph graph1("datasets/email-Eu-core.txt");
    // Graph graph1("datasets/Wiki-Vote.txt");
    // Graph graph1("datasets/simpleGragh2.txt");
    // Graph graph1("datasets/CA-GrQc.txt");
     //Graph graph("datasets/testGraph.txt");
    graph1.readGraph();
    int sourceId = 0;

    init(&graph1, sourceId);   // source 0
        
    // Run SSSP on CPU
    dijkstraOnCPU(sourceId);
    printShortestDistance(sourceId);

    init(&graph1, sourceId);   // source 0

    // Run SSSP on GPU
    dijkstraOnGPU(sourceId);
    printShortestDistance(sourceId);

    return 0;
}