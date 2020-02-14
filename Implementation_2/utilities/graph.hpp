#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include "global.hpp"



// const int GRAPH_MAX_SIZE = 12000;
// const int MAX_DIST = 65535;
// typedef unsigned int uint;
using namespace std;

struct Edge{
    uint source;
    uint end;
    uint weight;
};

class Graph {
private:

public:
    string graphFilePath;
    uint numNodes;
    uint numEdges;
    uint defaultSource;
    bool hasZeroId;
    // int** graph = new int* [GRAPH_MAX_SIZE];
    vector<Edge> edges;
    Graph(string graphFilePath);
    void readGraph();
    void printGraph();
};

#endif