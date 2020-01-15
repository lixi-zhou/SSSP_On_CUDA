#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.hpp"
#include "graph.hpp"
#include "gpu_error_check.cuh"

/*

Version 2. 128 Nodes Per Thread



*/

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
    // finished[source] = true;

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
        // imcompletedAndConnectedNode();
        // break;
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

    printf("Thread: %d process data from: %d to %d \n", threadId, startNodeId, endNodeId);


    for(int nodeId = startNodeId; nodeId < endNodeId; nodeId++){
        // if (nodeId < numNodes){
            // printf("This thread id is: %d\n", threadId);
            // printf("dist[%d] is %d, and the closest distance is %d\n\n", nodeId, dist[nodeId], dist[*closestNodeId]);
            if (!finished[nodeId] && dist[nodeId] < *minimumDist){
                // printf("Finished?");
                *closestNodeId = nodeId;
                *minimumDist = dist[nodeId];
                // printf("thread: %d, updated closetNodeId: %d, dist: %d\n", threadId, nodeId, dist[nodeId]);
                *completed = false;
            }
        // }      
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
        
        
        
         // rowIndex = closestNodeId;
        // colIndex = nodeId;

        // Process the nodes with same 
        // Convert to 1-D index
      /*   int index = *closestNodeId * GRAPH_MAX_SIZE + nodeId;
        finished[*closestNodeId] = true;
        // printf("graphData[%d][%d]: is %d \n", *closestNodeId, nodeId, graphData[*closestNodeId][nodeId]);
        if (!finished[nodeId] && graphData[index] < MAX_DIST){
            // Find the shorter path
            if(dist[*closestNodeId] + graphData[index] < dist[nodeId]){
                // Update dist
                dist[nodeId] = dist[*closestNodeId] + graphData[index];
                // Update its previous point
                prev[nodeId] = *closestNodeId;
                // printf("update prev[%d] = %d\n", nodeId, *closestNodeId);
                // printf("update dist[%d] = %d\n", nodeId, dist[*closestNodeId] + graphData[index]);
            }
        } */
    }
}

void dijkstraOnGPU(int source){
    Timer timer;
    cudaFree(0);
    // Define CPU vars
    // int* closestNodeId = new int(6);
    int closestNodeId = 6;
    // Define GPU vars
    int* d_graph;   // 2D array is converted to 1-D, row = i / cols, col = i % cols;
    int* d_dist;
    int* d_prev;
    bool* d_finished;
    int* d_closestNodeId;
    int* d_minimumDist;
    bool* d_completed;


    /* int width = GRAPH_MAX_SIZE, height = GRAPH_MAX_SIZE;
    size_t pitch;
    size_t size = sizeof(int) * width;    */

    // gpuErrorcheck(cudaMallocPitch((void **)&d_graph, &pitch, size, height));
    // gpuErrorcheck(cudaMemset2D(d_graph, pitch, 0, size, height));
    gpuErrorcheck(cudaMalloc((void **)&d_graph, GRAPH_MAX_SIZE * GRAPH_MAX_SIZE * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_dist, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_prev, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_finished, numNodes * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_closestNodeId, sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_completed, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_minimumDist, sizeof(int)));

    // gpuErrorcheck(cudaMemcpy2D(d_graph, pitch, graph1.graph, size, size, height, cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_graph, graph[0], GRAPH_MAX_SIZE * GRAPH_MAX_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_dist, dist, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_prev, previousNode, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_finished, finished, numNodes * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_closestNodeId, &closestNodeId, sizeof(int), cudaMemcpyHostToDevice));

    bool completed = true;
    int minimumDist = MAX_DIST;
    int numIteration = 0;

    
    // Each block has 128 threads
   
    int numNodesPerPart = 2;
    int numThreadPerBlock = 64;
    int numBlock = (numNodes) / (numNodesPerPart * numThreadPerBlock) + 1;
    
    timer.start();
    do{
        numIteration++;
        completed = true;
        
        gpuErrorcheck(cudaMemcpy(d_completed, &completed, sizeof(bool), cudaMemcpyHostToDevice));
        

        if(numIteration % 2 == 1){
            minimumDist = MAX_DIST;
            gpuErrorcheck(cudaMemcpy(d_minimumDist, &minimumDist, sizeof(int), cudaMemcpyHostToDevice));
            // First: find closest node
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
            for(int i = 0; i < numNodes; i++){
                printf("CPU processing\n");
            }
            
                gpuErrorcheck(cudaMemcpy(&completed, d_completed, sizeof(bool), cudaMemcpyDeviceToHost));
        }else{
            // Second: update its connected node
            dijkstraOnGPU_kernel2<<<numBlock, numThreadPerBlock>>>(numNodes,
                source,
                numNodesPerPart,
                d_graph,
                d_finished,
                d_dist,
                d_prev,
                d_closestNodeId,
                d_minimumDist,
                GRAPH_MAX_SIZE);
            completed = false;
        }

        
        // gpuErrorcheck(cudaPeekAtLastError());
        // gpuErrorcheck(cudaDeviceSynchronize());
        
        


        gpuErrorcheck(cudaDeviceSynchronize());  
        
        
        // printFinished();
        // printf("finished: %d\n", completed);
    }while(!completed);

    // gpuErrorcheck(cudaMemcpy(finished, d_finished, numNodes * sizeof(bool), cudaMemcpyDeviceToHost));
    printf("Number of Iteration Executed: %d\n", numIteration);
    printf("The execution time of SSSP on GPU: %d ms\n", timer.stop());
    // print("%d", d_closestNodeId);
    cudaMemcpy(&closestNodeId, d_closestNodeId, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dist, d_dist, numNodes * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("%d", (int)(*closestNodeId));
    // printf("%d", closestNodeId);

    // printGraph();

    
    // graph.printGraph();

    cudaFree(d_graph);
    cudaFree(d_dist);
    cudaFree(d_prev);
    cudaFree(d_finished);
    cudaFree(d_closestNodeId);
    cudaFree(d_minimumDist);
    cudaFree(d_completed);

    // printShortestDistance(0);
}

int main() {

    // Graph graph1("simpleGragh2.txt");
    // Graph graph1("email-Eu-core-SIMPLE.txt");
    // Graph graph1("email-Eu-core.txt");
    Graph graph1("Wiki-Vote.txt");
    // Graph graph1("simpleGragh2.txt");
    // Graph graph1("CA-GrQc.txt");
     //Graph graph("testGraph.txt");
    graph1.readGraph();
    int sourceId = 30;

    init(&graph1, sourceId);   // source 0
        
    // Run SSSP on CPU
    dijkstraOnCPU(sourceId);
    printShortestDistance(sourceId);
    // printGraph();

    init(&graph1, sourceId);   // source 0
    // Run SSSP on GPU
    dijkstraOnGPU(sourceId);
    printShortestDistance(sourceId);
    // printGraph();


   

    return 0;
}