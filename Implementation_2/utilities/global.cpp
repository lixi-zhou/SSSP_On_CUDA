#include "global.hpp"


void printDist(uint* dist, uint size) {
    for (int i = 0; i < size; i++) {
        std::cout << "dist[" << i << "]: " << dist[i] << std::endl;
    }
}

void printProcessed(bool* processed, uint size) {
    for (int i = 0; i < size; i++) {
        std::cout << "processed[" << i << "]: " << processed[i] << std::endl;
    }
}

void printPreNode(uint* preNode, uint size){
    for (int i = 0; i < size; i++) {
        std::cout << "prevNode[" << i << "]: " << preNode[i] << std::endl;
    }
}

