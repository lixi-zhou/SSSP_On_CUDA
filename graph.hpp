#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>



const int GRAPH_MAX_SIZE = 9000;
const int MAX_DIST = 65535;

using namespace std;

class Graph {
private:

public:
    string graphFilePath;
    int numNodes;
    int numEdges;
    int** graph = new int* [GRAPH_MAX_SIZE];


    Graph(string graphFilePath);
    void readGraph();
    void printGraph();
};

#endif