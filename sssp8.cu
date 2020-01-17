#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "./utilities/timer.hpp"
#include "./utilities/graph.hpp"
#include "./utilities/gpu_error_check.cuh"

/*

Applying Unified Memory

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
    printf("Maximum shortest distance : %d\n", diameter);
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
}

void initOnUnifiedMemory(int sourceId,
                                        int size,
                                        int** graph,
                                        int* dist,
                                        int* previousNode,
                                        int* d_m_graph,
                                        int* d_m_dist,
                                        int* d_m_prev,
                                        bool* d_m_finished,
                                        int* d_m_closestNodeId,
                                        int* d_m_minimumDist,
                                        int* d_m_completed){

    memcpy(d_m_graph, graph[0], GRAPH_MAX_SIZE * GRAPH_MAX_SIZE * sizeof(int));
    memcpy(d_m_dist, dist, size * sizeof(int));
    memcpy(d_m_prev, previousNode, size * sizeof(int));
    memcpy(d_m_finished, finished, size * sizeof(bool));
    *d_m_closestNodeId = 0;
    *d_m_minimumDist = MAX_DIST;
    *d_m_completed = 1;

    d_m_dist[sourceId] = 0;   

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
                                int* completed) {
    // kernel to compute the closest node 
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startNodeId = threadId * partSize;
    int endNodeId = (threadId + 1) * partSize;
    if(endNodeId > numNodes){
        endNodeId = numNodes;
    } 

    if(startNodeId > numNodes) return; 

    for(int nodeId = startNodeId; nodeId < endNodeId; nodeId++){
        if (!finished[nodeId] && dist[nodeId] < *minimumDist){
            *closestNodeId = nodeId;
            *minimumDist = dist[nodeId];
            *completed = 0;
            
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
                                        int* minimumDist,
                                        int* completed,
                                        int GRAPH_MAX_SIZE) {

    // Based on closest node then update its connected node
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startNodeId = threadId * partSize;
    int endNodeId = (threadId + 1) * partSize;
    if(endNodeId > numNodes){
        endNodeId = numNodes;
    } 

    *completed = 0;

    if(startNodeId > numNodes) return;

    for (int nodeId = startNodeId; nodeId < endNodeId; nodeId++){
        // Version 2
        // Process the nodes, whose dist = minimumDist
         if ((!finished[nodeId]) && dist[nodeId] == *minimumDist){
            finished[nodeId] = true;
            for (int connectedNodeId = 0; connectedNodeId < numNodes; connectedNodeId++){
                int index = nodeId * GRAPH_MAX_SIZE + connectedNodeId;
                if ((nodeId != connectedNodeId) && (graphData[index] < MAX_DIST)){
                    if (dist[nodeId] + graphData[index] < dist[connectedNodeId]){
                        finished[connectedNodeId] = false;
                        dist[connectedNodeId] = dist[nodeId] + graphData[index]; 
                    }
                }
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

    int* d_m_graph;
    int* d_m_dist;
    int* d_m_prev;
    bool* d_m_finished;
    int* d_m_closestNodeId;
    int* d_m_minimumDist;
    int* d_m_completed;

    gpuErrorcheck(cudaMallocManaged((void**) &d_m_graph, GRAPH_MAX_SIZE * GRAPH_MAX_SIZE * sizeof(int)));
    gpuErrorcheck(cudaMallocManaged(&d_m_dist, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMallocManaged(&d_m_prev, numNodes *sizeof(int)));
    gpuErrorcheck(cudaMallocManaged(&d_m_finished, numNodes * sizeof(bool)));
    gpuErrorcheck(cudaMallocManaged(&d_m_closestNodeId, sizeof(int)));
    gpuErrorcheck(cudaMallocManaged(&d_m_minimumDist, sizeof(int)));
    gpuErrorcheck(cudaMallocManaged(&d_m_completed, sizeof(int)));

    cudaDeviceSynchronize();

    initOnUnifiedMemory(source,
                        numNodes,
                        graph,
                        dist,
                        previousNode,
                        d_m_graph,
                        d_m_dist,
                        d_m_prev,
                        d_m_finished,
                        d_m_closestNodeId,
                        d_m_minimumDist,
                        d_m_completed);

    *d_m_completed = 1;
    *d_m_minimumDist = MAX_DIST;

    int numIteration = 0;
   
    int numNodesPerPart = 2;
    int numThreadPerBlock = 64;
    int numBlock = (numNodes) / (numNodesPerPart * numThreadPerBlock) + 1;
    
    timer.start();
    do{
        numIteration++;
        *d_m_completed = 1;

        if(numIteration % 2 == 1){
            *d_m_minimumDist = MAX_DIST;
            
            numNodesPerPart = 128;
            numThreadPerBlock = 64;
            numBlock = (numNodes) / (numNodesPerPart * numThreadPerBlock) + 1;
            
            // First: find closest node
            dijkstraOnGPU_kernel1<<< numBlock, numThreadPerBlock >>>(numNodes,
                source,
                numNodesPerPart,
                d_m_graph,
                d_m_finished,
                d_m_dist,
                d_m_prev,
                d_m_closestNodeId,
                d_m_minimumDist,
                d_m_completed);
        }else{
            numNodesPerPart = 1;
            numThreadPerBlock = 64;
            numBlock = (numNodes) / (numNodesPerPart * numThreadPerBlock) + 1;

            // Second: update its connected node
                dijkstraOnGPU_kernel2<<<numBlock, numThreadPerBlock>>>(numNodes,
                    source,
                    numNodesPerPart,
                    d_m_graph,
                    d_m_finished,
                    d_m_dist,
                    d_m_prev,
                    d_m_closestNodeId,
                    d_m_minimumDist,
                    d_m_completed,
                    GRAPH_MAX_SIZE
                );
        }
        gpuErrorcheck(cudaPeekAtLastError());
        gpuErrorcheck(cudaDeviceSynchronize());  
    }while(!(*d_m_completed));

    printf("Number of Iteration Executed: %d\n", numIteration);
    printf("The execution time of SSSP on GPU: %d ms\n", timer.stop());

    memcpy(dist, d_m_dist, numNodes * sizeof(int));


    cudaFree(d_m_graph);
    cudaFree(d_m_dist);
    cudaFree(d_m_prev);
    cudaFree(d_m_finished);
    cudaFree(d_m_closestNodeId);
    cudaFree(d_m_minimumDist);
    cudaFree(d_m_completed);
}

int main() {

    Graph graph1("datasets/simpleGragh.txt");
    // Graph graph1("datasets/email-Eu-core-SIMPLE.txt");
    // Graph graph1("datasets/email-Eu-core.txt");
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