#include "graph.hpp"

// Default Graph MAX SIZE: 5000 x 5000


Graph::Graph(string graphFilePath) {
	this->graphFilePath = graphFilePath;
	this->hasZeroId = false;
	/* this->graph[0] = new int[GRAPH_MAX_SIZE * GRAPH_MAX_SIZE];
	for (int i = 1; i < GRAPH_MAX_SIZE; i++){
		// Make its memoery contiguous
		this->graph[i] = graph[i-1] + GRAPH_MAX_SIZE;
	} */


	/* for (int i = 0; i < GRAPH_MAX_SIZE; i++) {
		// this->graph[i] = new int[GRAPH_MAX_SIZE]();
		
		// memset(this->graph[i], 1, sizeof(int) * GRAPH_MAX_SIZE);
		for(int j = 0; j < GRAPH_MAX_SIZE; j++){
			this->graph[i][j] = MAX_DIST;
		}
	} */
	//this->graph = new int *[MAX_INT][MAX_INT];
}

void Graph::readGraph() {
	ifstream infile;
	infile.open(graphFilePath);

	string line;
	uint edgeCounter = 0;
	uint maxNodeNumber = 0;

	Edge newEdge;

	while (getline(infile, line)) {
		// ignore non graph data
		if (line[0] < '0' || line[0] >'9') {
			continue;
		}

		stringstream ss(line);
		edgeCounter++;
		

		ss >> newEdge.source;
		ss >> newEdge.end;

		if (ss >> newEdge.weight) {
			// load weight 
		}
		else {
			// load default weight
			newEdge.weight = 1;
		}

		// this->graph[start][end] = weight;
		if (newEdge.source == 0){
			this->hasZeroId = true;
		}
		if (newEdge.end == 0){
			this->hasZeroId = true;
		}

		if (maxNodeNumber < newEdge.source) {
			maxNodeNumber = newEdge.source;
		}
		if (maxNodeNumber < newEdge.end) {
			maxNodeNumber = newEdge.end;
		}

		this->edges.push_back(newEdge);

	}

	this->numEdges = edgeCounter;
	if (this->hasZeroId){
		this->numNodes++;
	}
	this->numNodes = maxNodeNumber;

	std::cout << "Read graph from " << this->graphFilePath << ". This graph contains " << this->numNodes \
		<< " nodes, and " << edgeCounter << " edges" << endl;
}

void Graph::printGraph() {
	// print the graph
	std::cout << "This graph has " << this->numNodes << " nodes and " << this->numEdges << " edges." << endl;
	int size = this->numNodes;

	for (int i = 0; i < this->numEdges; i++){
		Edge edge = edges.at(i);
		std::cout << "Node: " << edge.source << " -> Node: " << edge.end << " Weight: " << edge.weight << endl;
	}
}