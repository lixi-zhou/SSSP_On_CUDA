#include "graph.hpp"

// Default Graph MAX SIZE: 5000 x 5000


Graph::Graph(string graphFilePath) {
	this->graphFilePath = graphFilePath;
	this->graph[0] = new int[GRAPH_MAX_SIZE * GRAPH_MAX_SIZE];
	for (int i = 1; i < GRAPH_MAX_SIZE; i++){
		// Make its memoery contiguous
		this->graph[i] = graph[i-1] + GRAPH_MAX_SIZE;
	}


	for (int i = 0; i < GRAPH_MAX_SIZE; i++) {
		// this->graph[i] = new int[GRAPH_MAX_SIZE]();
		
		// memset(this->graph[i], 1, sizeof(int) * GRAPH_MAX_SIZE);
		for(int j = 0; j < GRAPH_MAX_SIZE; j++){
			this->graph[i][j] = MAX_DIST;
		}
	}
	//this->graph = new int *[MAX_INT][MAX_INT];
}

void Graph::readGraph() {
	ifstream infile;
	infile.open(graphFilePath);

	string line;
	int edgeCounter = 0;
	int maxNodeNumber = 0;

	while (getline(infile, line)) {
		// ignore non graph data
		if (line[0] < '0' || line[0] >'9') {
			continue;
		}


		stringstream ss(line);
		edgeCounter++;
		// read start and end node
		int start;
		int end;
		int weight;

		ss >> start;
		ss >> end;

		if (ss >> weight) {
			// load weight 
		}
		else {
			// load default weight
			weight = 1;
		}

		this->graph[start][end] = weight;


		if (maxNodeNumber < start) {
			maxNodeNumber = start;
		}
		if (maxNodeNumber < end) {
			maxNodeNumber = end;
		}

		//cout << "Read edge. Start node: " << start << " - End node: " << end << " Weight: " << weight << endl;

	}

	this->numEdges = edgeCounter;
	this->numNodes = maxNodeNumber + 1;

	cout << "Read graph from " << this->graphFilePath << ". This graph contains " << this->numNodes \
		<< " nodes, and " << edgeCounter << " edges" << endl;
}

void Graph::printGraph() {
	// print the graph
	cout << "This graph has " << this->numNodes << " nodes and " << this->numEdges << " edges." << endl;
	int size = this->numNodes;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			cout << this->graph[i][j] << " ";
		}
		cout << endl;
	}
}