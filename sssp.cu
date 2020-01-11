#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "graph.hpp"

// const int MAX_DIST = 65535; //  initial value of distance
int numNodes;
int numEdges;

int* dist;
int* previousNode;
// int dist[NUMBER];   //  array to store the distance from source to each nodes
// int previousNode[NUMBER];   //  
// int graph[NUMBER][NUMBER];  //  a matrix to represent the graph
int ** graph;

void init(Graph* graphData) {
    // int test[NUMBER][NUMBER] = {1};
    numNodes = graphData->numNodes;
    graph = graphData->graph;
    
    // printf("size of array: %d\n", length);
    /* for (int i = 0; i < NUMBER; i++) {
        for (int j = 0; j < NUMBER; j++) {
            graph[i][j] = 1;
            printf("%d ", graph[i][j]);
        }
        printf("\n");
    } */
}


void dijkstra(int source) {
    int size = numNodes;
    dist = new int[size];
    previousNode = new int[size];
    bool* finished = new bool[size];


    // Find the connected nodes to the source point
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


int main() {

    time_t start, finish;

    //Graph graph("simpleGragh.txt");
    Graph graph("email-Eu-core.txt");
     //Graph graph("testGraph.txt");
    graph.readGraph();

    init(&graph);

    
    start = clock();
    
    dijkstra(0);

    finish = clock();

    //printShortestDistance(0);

    cout << "Execution time of SSSP: " << (finish - start) << " ms" << endl;

    // printGraph();

    
    // graph.printGraph();

    return 0;
}