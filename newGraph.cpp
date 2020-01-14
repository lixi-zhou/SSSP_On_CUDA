#include "newGraph.hpp"


/* NewGraph::NewGraph(){
    printf("created");
} */

NewGraph::NewGraph(string graphFilePath){
    std::cout << "dadsa" << endl;
    std::cout << graphFilePath << endl;

    Edge* ed = new Edge[MAX_SIZE];
    int* w = new int[MAX_SIZE];

    ifstream infile;
    infile.open(graphFilePath);

    string line;
    int edgeCounter = 0;
    int maxNodeNumber = 0;
    int index = 0;

    
    while (getline(infile, line)){

        if (line[0] < '0' || line[0] > '9'){
            continue;
        }
        stringstream ss(line);
        edgeCounter++;
        // read start and target vertex
        int start;
        int target;
        int weight;
        ss >> start;
        ss >> target;
        
        if (ss >> weight){
            // load weight
        }else{
            // load default weight
            weight = 1;
        }
        ed[index] = Edge{start, target};
         w[index] = weight;
        index++;
        if (maxNodeNumber < start){
            maxNodeNumber = start;
        }
        if (maxNodeNumber < target){
            maxNodeNumber = target;
        }

        
    }
    this->numNodes = maxNodeNumber + 1;
    this->numEdges = edgeCounter;

    this->edges = ed;
    this->weights = w;

    cout << "Read graph from " << graphFilePath << ". This graph contains " << this->numNodes \
		<< " nodes, and " << edgeCounter << " edges" << endl;
}