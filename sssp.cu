#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "graph.hpp"
#include "gpu_error_check.cuh"

// const int MAX_DIST = 65535; //  initial value of distance
int numNodes;
int numEdges;

int* dist;
int* previousNode;
// int dist[NUMBER];   //  array to store the distance from source to each nodes
// int previousNode[NUMBER];   //  
// int graph[NUMBER][NUMBER];  //  a matrix to represent the graph
int ** graph;
bool* finished;

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
}


void dijkstraOnCPU(int source) {
    int size = numNodes;
    


    
    

    // Find the connected nodes to the source point
    // Set the source point
    dist[source] = 0;
    finished[source] = true;

    for (int i = 0; i < size; i++) {
        int mindist = MAX_DIST;
        // U is the closet point to source, u is not finished yet
        int u = source;

        for (int j = 0; j < size; j++) {
            if ((j != u) && (!finished[j]) && dist[j] < mindist) {
                u = j;
                mindist = dist[j];
            }
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
}


void printShortestDistance(int source) {
    for (int i = 0; i < numNodes; i++) {
        if(dist[i] != MAX_DIST){
            printf("Shortest distance from node: %d to source: %d: is: %d\n", i, source, dist[i]);
        }else{
            printf("Shortest distance from node: %d to source: %d: is: INF\n", i, source);
        } 
    }
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

__global__ void dijkstraOnGPU_kernel1(int numNodes, 
                                int sourceId,
                                int** graphData,
                                bool* finished,
                                int* dist,
                                int* prev,
                                int* closestNodeId) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int nodeId = threadId;
    //int nodeBeginId = partSize * partId;
    //int nodeEndId = partSize * (partId + 1);
    //if (nodeEndId > numNodes) nodeEndId = numNodes;

    //printf("Thread %d: processes the nodes from %d to %d\n", partId, nodeBeginId, nodeEndId);
    
    if (nodeId < numNodes){
        printf("This thread id is: %d\n", threadId);
        printf("dist[%d] is %d, and the closest distance is %d\n\n", nodeId, dist[nodeId], dist[*closestNodeId]);
        if (!finished[nodeId] && dist[nodeId] < dist[*closestNodeId]){
            // printf("Finished?");
            *closestNodeId = nodeId;
            printf("updated closetNodeId: %d\n", *closestNodeId);
        }
    }
    // printf("Graph[0][2]: %d\n", graphData[0][2]);
    printf("Print graph from GPU. Num nodes: %d \n", numNodes);
    for(int i = 0; i < numNodes; i++){
        for(int j = 0; j < numNodes; j++){
            printf("[%d][%d]: %d ", i, j, graphData[i][j]);
        }
        printf("\n");
    }
    printf("\n");


}

__global__ void dijkstraOnGPU_kernel2(int numNodes, 
                                        int sourceId,
                                        int** graphData,
                                        bool* finished,
                                        int* dist,
                                        int* prev,
                                        int* closestNodeId) {

    // Based on closest node then update its connected node
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int nodeId = threadId;
    if(nodeId > numNodes) return;

    // printf("graphData[%d][%d]: is %d \n", *closestNodeId, nodeId, graphData[*closestNodeId][nodeId]);
    if (!finished[nodeId] && graphData[*closestNodeId][nodeId] < MAX_DIST){
        // Find the shorter path
        if(dist[*closestNodeId] + graphData[*closestNodeId][nodeId] < dist[nodeId]){
            // Update dist
            dist[nodeId] = dist[*closestNodeId] + graphData[*closestNodeId][nodeId];
            // Update its previous point
            prev[nodeId] = *closestNodeId;
        }
    }
}

int main() {

    time_t start, finish;

    Graph graph1("simpleGragh.txt");
    // Graph graph("email-Eu-core.txt");
     //Graph graph("testGraph.txt");
    graph1.readGraph();

    init(&graph1, 0);   // source 0
    
    // printGraph();
        
    /**************

    CPU Part


    */


    // start = clock();
    
    // dijkstraOnCPU(0);

    // finish = clock();

    printShortestDistance(0);

    // cout << "Execution time of SSSP: " << (finish - start) << " ms" << endl;
    //dijkstraOnCPU<<<1,10>>>();
    // dim3 block(1);
    // dim3 grid(1);


        
    /**************

    GPU Part


    */


    cudaFree(0);
    // Define CPU vars
    // int* closestNodeId = new int(6);
    int closestNodeId = 6;
    // Define GPU vars
    int** d_graph;
    int* d_dist;
    int* d_prev;
    bool* d_finished;
    int* d_closestNodeId;

    gpuErrorcheck(cudaMalloc((void **)&d_graph, numNodes * numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_dist, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_prev, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_finished, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_closestNodeId, sizeof(int)));

    gpuErrorcheck(cudaMemcpy(d_graph, graph, numNodes * numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_dist, dist, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_prev, previousNode, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_finished, finished, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_closestNodeId, &closestNodeId, sizeof(int), cudaMemcpyHostToDevice));




    // Each block has 128 threads
    int numThreadPerBlock = 128;
    int numBlock = (numNodes / numThreadPerBlock) + 1;


    dijkstraOnGPU_kernel1<<<numBlock, numThreadPerBlock >>>(numNodes,
                                                    0,
                                                    d_graph,
                                                    d_finished,
                                                    d_dist,
                                                    d_prev,
                                                    d_closestNodeId);

    gpuErrorcheck(cudaPeekAtLastError());
    gpuErrorcheck(cudaDeviceSynchronize());
    
    // dijkstraOnGPU_kernel2<<<numBlock, numThreadPerBlock>>>(numNodes,
    //                                                 0,
    //                                                 d_graph,
    //                                                 d_finished,
    //                                                 d_dist,
    //                                                 d_prev,
    //                                                 d_closestNodeId);


    // cudaDeviceSynchronize();    
    // print("%d", d_closestNodeId);
    cudaMemcpy(&closestNodeId, d_closestNodeId, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dist, d_dist, numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d", (int)(*closestNodeId));
    printf("%d", closestNodeId);

    // printGraph();

    
    // graph.printGraph();

    cudaFree(d_graph);
    cudaFree(d_dist);
    cudaFree(d_prev);
    cudaFree(d_finished);
    cudaFree(d_closestNodeId);

    printShortestDistance(0);

    return 0;
}