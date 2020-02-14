#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "./utilities/timer.hpp"
#include "./utilities/graph.hpp"
#include "./utilities/gpu_error_check.cuh"
#include "./utilities/global.hpp"
#include "./utilities/argument_parser.hpp"


/*

Dijkstra Algorithm with Adjacency list

*/


void dijkstraCPU(Graph* graph, int source){
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
        uint minDist = MAX_DIST;
        finished = true;
        numIteration++;

        

        for (int i = 0; i < numNodes; i++){
            if (i != source && (!processed[i]) && dist[i] < minDist){
                // Find the minimum distance in un-processed sets
                minDist = dist[i];
                finished = false;
            }    
        }

        vector<uint> sets;

        for (int i = 0; i < numEdges; i++){
            Edge edge = graph->edges.at(i);
            // Update its neighbor
            uint source = edge.source;
            uint end = edge.end;
            uint weight = edge.weight;

            if ((!processed[end]) && dist[end] == minDist){
                // To handle the node which does not have other neighbors
                sets.push_back(end);
            }

            if ((!processed[source]) && (dist[source] == minDist)) {
                sets.push_back(source);
                if (dist[source] + weight < dist[end]){
                    // Update dist
                    dist[end] = dist[source] + weight;
                    preNode[end] = source;
                    processed[end] = false;
                }   
            }
        }
        
        // Mark the processed node
        for (int i = 0; i < sets.size(); i++){
            processed[sets.at(i)] = true;
        }
    }
    timer.stop();
    
    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    printf("The execution time of SSSP on CPU: %d ms\n", timer.elapsedTime());
}

void dijkstraCPU1(Graph* graph, int source){
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
    

    // printDist(dist, numNodes);
    // printPreNode(dist, numNodes);
    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    printf("The execution time of SSSP on CPU: %d ms\n", timer.elapsedTime());
}

int main(int argc, char **argv){
    ArgumentParser args(argc, argv);
    // cout << "Input file : " << argumentParser.inputFilePath << endl;
    Graph graph(args.inputFilePath);

    graph.readGraph();

    dijkstraCPU1(&graph, graph.defaultSource);
}