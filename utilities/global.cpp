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

void printPreNode(uint* preNode, uint size) {
    for (int i = 0; i < size; i++) {
        std::cout << "prevNode[" << i << "]: " << preNode[i] << std::endl;
    }
}

void compareResult(uint* dist1, uint* dist2, uint numNodes) {
    uint diffCount = 0;
    for (int i = 0; i < numNodes; i++) {
        if (dist1[i] != dist2[i]) {
            diffCount++;
        }
    }

    if (diffCount == 0) {
        std::cout << "Good! These two result are identical!" <<  std::endl;
    } else {
        std::cout << diffCount << " of " << numNodes << " does not match!" << std::endl;
    }
}

void printLoopInfo(vector<LoopInfo> info) {

    for (int i = 0; i < info.size(); i++) {
        LoopInfo loopInfo = info[i];
        printf("No. itr: %d , updated splitRatio: %f\n", loopInfo.numIteration, loopInfo.splitRatio);
        printf("CPU PART TIME: %f\n", loopInfo.time_cpu);
        printf("GPU PART TIME: %f\n", loopInfo.time_gpu);
        // printf("Copy dist from host to device : %f ms \n", timer_host_to_device.elapsedTime());
        // printf("Copy dist from device to host : %f ms \n", timer_device_to_host.elapsedTime());
            
    }


    

}