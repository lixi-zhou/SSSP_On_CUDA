#ifndef NEWGRAPH_HPP
#define NEWGRAPH_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>


int const MAX_SIZE = 120000;


using namespace std;

typedef struct{
    int source;
    int target;
} Edge;

typedef struct{
    int id;
    bool visited;
} Vertex;




class NewGraph{
    private:


    public:
        string graphFilePath;
        int numNodes;
        int numEdges;
        Edge* edges;
        int* weights;

        // NewGraph();
        NewGraph(string graphFilePath);



};



#endif