#include "graph.hpp"

// Default Graph MAX SIZE: 5000 x 5000


Graph::Graph(string graphFilePath) {
	this->graphFilePath = graphFilePath;
	this->hasZeroId = false;
}

void Graph::readGraph() {
	ifstream infile;
	infile.open(graphFilePath);

	string line;
	stringstream ss;
	uint edgeCounter = 0;
	uint maxNodeNumber = 0;
	uint minNodeNumber = MAX_DIST;

	Edge newEdge;

	while (getline(infile, line)) {
		// ignore non graph data
		if (line[0] < '0' || line[0] >'9') {
			continue;
		}

		// stringstream ss(line);
		ss.clear();
		ss << line;
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
		if (minNodeNumber > newEdge.source) {
			minNodeNumber = newEdge.source;
		}
		if (minNodeNumber > newEdge.end) {
			minNodeNumber = newEdge.source;
		}
		

		this->edges.push_back(newEdge);

	}

	
	if (this->hasZeroId){
		maxNodeNumber++;
	}
	this->numNodes = maxNodeNumber;
	this->numEdges = edgeCounter;
	this->defaultSource = minNodeNumber;

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