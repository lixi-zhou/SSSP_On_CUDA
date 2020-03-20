#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "./utilities/timer.hpp"
#include "./utilities/graph.hpp"
#include "./utilities/gpu_error_check.cuh"
#include "./utilities/global.hpp"
#include "./utilities/argument_parser.hpp"
#include <omp.h>


uint* sssp_CPU_parallel(Graph *graph, int source) {
    int numNodes = graph->numNodes;
    int numEdges = graph->numEdges;
    uint *dist = new uint[numNodes];
    uint *preNode = new uint[numNodes];
    bool *processed = new bool[numNodes];
    uint *edgesSource = new uint[numEdges];
    uint *edgesEnd = new uint[numEdges];
    uint *edgesWeight = new uint[numEdges];

    for (int i = 0; i < numNodes; i++) {
        dist[i] = MAX_DIST;
        preNode[i] = uint(-1);
        processed[i] = false;
    }

    for (int i = 0; i < numEdges;i ++) {
        Edge edge = graph->edges.at(i);
        edgesSource[i] = edge.source;
        edgesEnd[i] = edge.end;
        edgesWeight[i] = edge.weight;

        if (edge.source == source) {
            if (edge.weight < dist[edge.end]) {
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
    while(!finished) {
        finished = true;
        numIteration++;
        
        #pragma omp parallel 
        {   
            // #pragma omp master 
            int threadId = omp_get_thread_num();
            int numThreads = omp_get_num_threads();
            int numEdgesPerThread = numEdges / numThreads + 1;
            int start = threadId * numEdgesPerThread;
            int end = (threadId + 1) * numEdgesPerThread;
            if (start > numEdges) {
                start = numEdges;
            }
            
            if (end > numEdges) {
                end = numEdges;
            }

            for (int i = start; i < end; i++) {
                uint source = edgesSource[i];
                uint end = edgesEnd[i];
                uint weight = edgesWeight[i];

                if (dist[source] + weight < dist[end]) {
                    // #pragma omp atomic
                    dist[end] = dist[source] + weight;
                    // #pragma omp atomic
                    preNode[end] = source;
                    finished = false;
                }
            }
        }
    }
    timer.stop();

    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    printf("The execution time of SSSP on CPU(OpenMP): %f ms\n", timer.elapsedTime());

    return dist;

}

__global__ void sssp_GPU_Kernel(int numEdges,
                                int numEdgesPerThread,
                                uint *dist,
                                uint *preNode,
                                uint *edgesSource,
                                uint *edgesEnd,
                                uint *edgesWeight,
                                bool *finished) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * numEdgesPerThread;

    if (startId >= numEdges) {
        return;
    }

    int endId = (threadId + 1) * numEdgesPerThread;
    if (endId >= numEdges) {
        endId = numEdges;
    }

    for (int nodeId = startId; nodeId < endId; nodeId++) {
        uint source = edgesSource[nodeId];
        uint end = edgesEnd[nodeId];
        uint weight = edgesWeight[nodeId];

        if (dist[source] + weight < dist[end]) {
        atomicMin(&dist[end], dist[source] + weight);
        // dist[end] = dist[source] + weight;
        preNode[end] = source;
        *finished = false;
        }
    }

}

uint* sssp_GPU(Graph *graph, int source) {
    int numNodes = graph->numNodes;
    int numEdges = graph->numEdges;
    uint *dist = new uint[numNodes];
    uint *preNode = new uint[numNodes];
    uint *edgesSource = new uint[numEdges];
    uint *edgesEnd = new uint[numEdges];
    uint *edgesWeight = new uint[numEdges];

    for (int i = 0; i < numNodes; i++) {
        dist[i] = MAX_DIST;
        preNode[i] = uint(-1);
    }


    for (int i = 0; i < numEdges; i++) {
        Edge edge = graph->edges.at(i);

        // Transfer the vector to the following three arrays
        edgesSource[i] = edge.source;
        edgesEnd[i] = edge.end;
        edgesWeight[i] = edge.weight;

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

    dist[source] = 0;
    preNode[source] = 0;


    uint *d_dist;
    uint *d_preNode;
    bool *d_finished;
    uint *d_edgesSource;
    uint *d_edgesEnd;
    uint *d_edgesWeight;

    gpuErrorcheck(cudaMalloc(&d_dist, numNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_preNode, numNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_edgesSource, numEdges * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_edgesEnd, numEdges * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_edgesWeight, numEdges * sizeof(uint)));

    gpuErrorcheck(cudaMemcpy(d_dist, dist, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_preNode, preNode, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgesSource, edgesSource, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgesEnd, edgesEnd, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgesWeight, edgesWeight, numEdges * sizeof(uint), cudaMemcpyHostToDevice));

    Timer timer;
    int numIteration = 0;
    int numEdgesPerThread = 8;
    int numThreadsPerBlock = 512;
    int numBlock = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
    bool finished = true;

    timer.start();
    do {
        numIteration++;
        finished = true;

        gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));

        // TO-DO PARALLEL
        sssp_GPU_Kernel<<< numBlock, numThreadsPerBlock >>> (numEdges,
                                        numEdgesPerThread,
                                        d_dist,
                                        d_preNode,
                                        d_edgesSource,
                                        d_edgesEnd,
                                        d_edgesWeight,
                                        d_finished);

        gpuErrorcheck(cudaPeekAtLastError());
        gpuErrorcheck(cudaDeviceSynchronize()); 
        gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
    } while(!finished);
    timer.stop();


    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    printf("The execution time of SSSP on GPU: %f ms\n", timer.elapsedTime());

    gpuErrorcheck(cudaMemcpy(dist, d_dist, numNodes * sizeof(uint), cudaMemcpyDeviceToHost));

    gpuErrorcheck(cudaFree(d_dist));
    gpuErrorcheck(cudaFree(d_preNode));
    gpuErrorcheck(cudaFree(d_finished));
    gpuErrorcheck(cudaFree(d_edgesSource));
    gpuErrorcheck(cudaFree(d_edgesEnd));
    gpuErrorcheck(cudaFree(d_edgesWeight));

    return dist;
}

__global__ void sssp_GPU_Hybrid_Message(int numNodesPerThread,
                                    uint *dist,
                                    uint *preNode,
                                    int *d_messageToDeviceSize,
                                    int *d_messageToDeviceNodeIndex,
                                    int *d_messageToDeviceDist) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = threadId * numNodesPerThread;
    if (startId >= *d_messageToDeviceSize) {
        return;
    }

    int endId = (threadId + 1) * numNodesPerThread;
    if (endId >= *d_messageToDeviceSize) {
        endId = *d_messageToDeviceSize;
    }

    for (int i = startId; i < endId; i++) {
        int nodeId = d_messageToDeviceNodeIndex[i];
        int updateDist = d_messageToDeviceDist[i];
        if (dist[nodeId] > updateDist) {
            atomicMin(&dist[nodeId], updateDist);
        }
    }
}



__global__ void sssp_GPU_Hybrid_Kernel(int splitIndex,
                                int numEdges,
                                int numEdgesPerThread,
                                uint *dist,
                                uint *preNode,
                                uint *edgesSource,
                                uint *edgesEnd,
                                uint *edgesWeight,
                                bool *finished,
                                int *d_messageToHostSize,
                                int *d_messageToHostNodeIndex,
                                int *d_messageToHostDist) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int startId = splitIndex + threadId * numEdgesPerThread;
    if (startId >= numEdges) {
        return;
    }

    int endId = splitIndex + (threadId + 1) * numEdgesPerThread;
    if (endId >= numEdges) {
        endId = numEdges;
    }

    // printf("GPU: process edged from: %d to %d \n", startId, endId);
    for (int nodeId = startId; nodeId < endId; nodeId++) {
        uint source = edgesSource[nodeId];
        uint end = edgesEnd[nodeId];
        uint weight = edgesWeight[nodeId];
        
        uint tempDist = dist[source] + weight;
        if (tempDist < dist[end]) {
            atomicMin(&dist[end], tempDist);
            preNode[end] = source;
            *finished = false;

            // Message
            // int index = *d_messageToHostSize++;
            int index = atomicAdd(d_messageToHostSize, 1);
            d_messageToHostNodeIndex[index] = end;
            d_messageToHostDist[index] = tempDist;
            // printf("GPUUUUUUUU Message index: %d\n", index);
        }
    }
}

uint* sssp_Hybrid(Graph *graph, int source) {
    int numNodes = graph->numNodes;
    int numEdges = graph->numEdges;
    uint *dist = new uint[numNodes];
    uint *preNode = new uint[numNodes];
    uint *edgesSource = new uint[numEdges];
    uint *edgesEnd = new uint[numEdges];
    uint *edgesWeight = new uint[numEdges];

    // Message from host to device
    int *messageToDeviceNodeIndex = new int[numEdges];
    int *messageToDeviceDist = new int[numEdges];
    int messageToDeviceSize = 0;
    // Message from device to host
    int *messageToHostNodeIndex  = new int[numEdges];
    int *messageToHostDist = new int[numEdges];
    int messageToHostSize = 0;

    for (int i = 0; i < numNodes; i++) {
        dist[i] = MAX_DIST;
        preNode[i] = uint(-1);

        messageToDeviceNodeIndex[i] = -1;
        messageToDeviceDist[i] = -1;
        messageToHostNodeIndex[i] = -1;
        messageToHostDist[i] = -1;
    }


    for (int i = 0; i < numEdges; i++) {
        Edge edge = graph->edges.at(i);
        
        // Transfer the vector to the following three arrays
        edgesSource[i] = edge.source;
        edgesEnd[i] = edge.end;
        edgesWeight[i] = edge.weight;

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

    dist[source] = 0;
    preNode[source] = 0;

    uint *d_dist;
    uint *d_preNode;
    bool *d_finished;
    uint *d_edgesSource;
    uint *d_edgesEnd;
    uint *d_edgesWeight;
    int *d_messageToHostNodeIndex;
    int *d_messageToHostDist;
    int *d_messageToHostSize;
    int *d_messageToDeviceNodeIndex;
    int *d_messageToDeviceDist;
    int *d_messageToDeviceSize;

    gpuErrorcheck(cudaMalloc(&d_dist, numNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_preNode, numNodes * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_edgesSource, numEdges * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_edgesEnd, numEdges * sizeof(uint)));
    gpuErrorcheck(cudaMalloc(&d_edgesWeight, numEdges * sizeof(uint)));
    
    gpuErrorcheck(cudaMalloc(&d_messageToHostNodeIndex, numEdges * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_messageToHostDist, numEdges * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_messageToHostSize, sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_messageToDeviceNodeIndex, numEdges * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_messageToDeviceDist, numEdges * sizeof(int)));
    gpuErrorcheck(cudaMalloc(&d_messageToDeviceSize, sizeof(int)));


    gpuErrorcheck(cudaMemcpy(d_dist, dist, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_preNode, preNode, numNodes * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgesSource, edgesSource, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgesEnd, edgesEnd, numEdges * sizeof(uint), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgesWeight, edgesWeight, numEdges * sizeof(uint), cudaMemcpyHostToDevice));

    gpuErrorcheck(cudaMemcpy(d_messageToHostNodeIndex, messageToHostNodeIndex, numEdges * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_messageToHostDist, messageToHostDist, numEdges * sizeof(int), cudaMemcpyHostToDevice));



    Timer timer;
    int numIteration = 0;
    bool finished = false;
    bool h_finished = false;
    
    
    float splitRatio; // cpu_data_size / whole_data_size

    // Automatic select a prior value of spritRatio based on experience
    if (numEdges < 300000) {
        splitRatio = 0.95;
    } else if (numEdges < 800000) {
        splitRatio = 0.7;
    } else {
        splitRatio = 0.1;
    }

    /*
    CPU process edges from 0 to splitIndex   
        number of edges: splitIndex
    GPU process edges from splitIndex to numEdges 
        number of edges: numEdges - splitIndex + 1
    */
    int splitIndex = numEdges * splitRatio;
    int d_numEdgesPerThread = 8;
    // int d_numNodesPerThread = 8;
    int d_numThreadsPerBlock = 512;
    int d_numBlock = (numEdges - splitIndex + 1) / (d_numThreadsPerBlock * d_numEdgesPerThread) + 1;
    
    Timer timer_cpu, timer_gpu;
    Timer timer_host_to_device;
    Timer timer_device_to_host;

    // Default: enable cpu and gpu 
    // Once splitRatio equals to 0 only enable gpu
    // Once splitRatio equals to 1 only enable cpu
    
    bool cpu_enable = true;
    bool gpu_enable = true;

    vector<LoopInfo> infos;
    LoopInfo loopInfo;

    





    timer.start();
    do {
        // splitRatio = 0.4;

        numIteration++;
        finished = true;
        h_finished = true;
        splitIndex = numEdges * splitRatio;
        d_numBlock = (numEdges - splitIndex + 1) / (d_numThreadsPerBlock * d_numEdgesPerThread) + 1;
        
        messageToDeviceSize = 0;
        messageToHostSize = 0;


        timer_gpu.start();
        timer_cpu.start();
        #pragma omp parallel //num_threads(8)
        {   
            int threadId = omp_get_thread_num();
            int h_numThreads = omp_get_num_threads();
            if (threadId == h_numThreads - 1 && splitIndex < numEdges  && gpu_enable) {
                // Last thread will be used to launch gpu kernel 
                // if thread 0 is used to launch gpu kernel, the first block of 
                // data whose index begining from 0 will not be processed.
                gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
                // timer_host_to_device.start();
                // gpuErrorcheck(cudaMemcpy(d_dist, dist, sizeof(uint) * numNodes, cudaMemcpyHostToDevice));
                // timer_host_to_device.stop();
                gpuErrorcheck(cudaMemcpy(d_messageToHostSize, &messageToHostSize, sizeof(int), cudaMemcpyHostToDevice));

                sssp_GPU_Hybrid_Kernel<<< d_numBlock, d_numThreadsPerBlock>>> (splitIndex,
                                                                        numEdges,
                                                                        d_numEdgesPerThread,
                                                                        d_dist,
                                                                        d_preNode,
                                                                        d_edgesSource,
                                                                        d_edgesEnd,
                                                                        d_edgesWeight,
                                                                        d_finished,
                                                                        d_messageToHostSize,
                                                                        d_messageToHostNodeIndex,
                                                                        d_messageToHostDist);
                gpuErrorcheck(cudaPeekAtLastError());
                gpuErrorcheck(cudaDeviceSynchronize()); 
                gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
                // timer_device_to_host.start();
                // gpuErrorcheck(cudaMemcpy(dist_copy, d_dist, sizeof(uint) * numNodes, cudaMemcpyDeviceToHost));
                // timer_device_to_host.stop();

                gpuErrorcheck(cudaMemcpy(&messageToHostSize, d_messageToHostSize, sizeof(int), cudaMemcpyDeviceToHost));
                int messageSize = messageToHostSize;
                gpuErrorcheck(cudaMemcpy(messageToHostNodeIndex, d_messageToHostNodeIndex, sizeof(int) * messageSize, cudaMemcpyDeviceToHost));
                gpuErrorcheck(cudaMemcpy(messageToHostDist, d_messageToHostDist, sizeof(int) * messageSize, cudaMemcpyDeviceToHost));

                /* printf("messageToHost size: %d\n", messageSize);
                for (int i = 0; i < messageSize; i++) {
                    printf("index: %d dist: %d\n", messageToHostNodeIndex[i], messageToHostDist[i]);
                } */

                timer_gpu.stop();
            } else if (cpu_enable && threadId != (h_numThreads - 1)) {
                // printf("Sub threads\n");
                int h_numEdgesPerThread = (splitIndex) / (h_numThreads - 1) + 1;
                int start = threadId * h_numEdgesPerThread;
                int end = (threadId + 1) * h_numEdgesPerThread;
                if (start > splitIndex) {
                    start = splitIndex;
                }
                if (end > splitIndex) {
                    end = splitIndex;
                }

                // cout << "Processs node: from " << start << " to: " << end << endl;
                // printf("Process node from: %d to : %d\n", start, end);
                for (int i = start; i < end; i++) {
                    uint source = edgesSource[i];
                    uint end = edgesEnd[i];
                    uint weight = edgesWeight[i];
                    
                    uint tempDist = dist[source] + weight;
                    if (tempDist < dist[end]) {
                        dist[end] = tempDist;
                        preNode[end] = source;
                    // New message  
                    #pragma omp critical
                    {
                        messageToDeviceNodeIndex[messageToDeviceSize] = end;
                        messageToDeviceDist[messageToDeviceSize] = tempDist;
                        messageToDeviceSize++;
                    }            
                    h_finished = false;
                    }
                }
                timer_cpu.stop();
            }
        }
       
        finished = finished && h_finished;

        #pragma omp parallel //num_threads(8)
        {   
            int threadId = omp_get_thread_num();
            int h_numThreads = omp_get_num_threads();
            if (threadId == h_numThreads - 1) {
                // Process message from host in GPU
                int d_numNodesPerThread = 8;
                /* if (messageToDeviceSize > numNodes) {
                    messageToDeviceSize = numNodes;
                } */
                d_numBlock = (messageToDeviceSize) / (d_numThreadsPerBlock * d_numNodesPerThread) + 1;
                gpuErrorcheck(cudaMemcpy(d_messageToDeviceSize, &messageToDeviceSize, sizeof(int), cudaMemcpyHostToDevice));
                gpuErrorcheck(cudaMemcpy(d_messageToDeviceNodeIndex, messageToDeviceNodeIndex, sizeof(int) * messageToDeviceSize, cudaMemcpyHostToDevice));
                gpuErrorcheck(cudaMemcpy(d_messageToDeviceDist, messageToDeviceDist, sizeof(int) * messageToDeviceSize, cudaMemcpyHostToDevice));

                sssp_GPU_Hybrid_Message<<< d_numBlock, d_numThreadsPerBlock>>> (d_numNodesPerThread,
                                                                            d_dist,
                                                                            d_preNode,
                                                                            d_messageToDeviceSize,
                                                                            d_messageToDeviceNodeIndex,
                                                                            d_messageToDeviceDist);

                gpuErrorcheck(cudaPeekAtLastError());
                gpuErrorcheck(cudaDeviceSynchronize());    
            } else if (threadId != h_numThreads - 1) {
                int h_numNodesPerThread = (messageToHostSize) / (h_numThreads - 1) + 1;
                int start = threadId * h_numNodesPerThread;
                int end = (threadId + 1) * h_numNodesPerThread;
                if (start > messageToHostSize) {
                    start = messageToHostSize;
                }
                if (end > messageToHostSize) {
                    end = messageToHostSize;
                }
                for (int i = start; i < end; i++) {
                    int nodeId = messageToHostNodeIndex[i];
                    int updateDist = messageToHostDist[i];
                    if (dist[nodeId] > updateDist) {
                        dist[nodeId] = updateDist;
                    }
                }
            }
        }

        // Load Balancing

        if (cpu_enable && gpu_enable) {
            float factor = (timer_cpu.elapsedTime() / timer_gpu.elapsedTime());
            if (factor > 1.1) {
                splitRatio = splitRatio - 0.05;
                if (splitRatio < 0) {
                    splitRatio = 0;
                    cpu_enable = false;
                }
    
            } else if (factor < 0.9) {
                splitRatio = splitRatio + 0.05;
                if (splitRatio > 1) {
                    splitRatio = 1;
                    gpu_enable = false;
                }
            }

            // printf("No. itr: %d , updated splitRatio: %f, factor: %f\n", numIteration, splitRatio, factor);

            // printf("CPU PART TIME: %f\n", timer_cpu.elapsedTime());
            // printf("GPU PART TIME: %f\n", timer_gpu.elapsedTime());
            // printf("Copy dist from host to device : %f ms \n", timer_host_to_device.elapsedTime());
            // printf("Copy dist from device to host : %f ms \n", timer_device_to_host.elapsedTime());
            loopInfo.numIteration = numIteration;
            loopInfo.time_cpu = timer_cpu.elapsedTime() > 0 ? timer_cpu.elapsedTime() : 0;
            loopInfo.time_gpu = timer_gpu.elapsedTime() > 0 ? timer_gpu.elapsedTime() : 0;
            loopInfo.splitRatio = splitRatio;
            infos.push_back(loopInfo);
        } 
    } while(!finished);
    timer.stop();

    printLoopInfo(infos);
    printf("Process Done!\n");
    printf("Number of Iteration: %d\n", numIteration);
    printf("The execution time of SSSP on Hybrid(CPU-GPU): %f ms\n", timer.elapsedTime());

    gpuErrorcheck(cudaFree(d_dist));
    gpuErrorcheck(cudaFree(d_preNode));
    gpuErrorcheck(cudaFree(d_finished));
    gpuErrorcheck(cudaFree(d_edgesSource));
    gpuErrorcheck(cudaFree(d_edgesEnd));
    gpuErrorcheck(cudaFree(d_edgesWeight));
    gpuErrorcheck(cudaFree(d_messageToHostNodeIndex));
    gpuErrorcheck(cudaFree(d_messageToHostDist));
    gpuErrorcheck(cudaFree(d_messageToHostSize));
    gpuErrorcheck(cudaFree(d_messageToDeviceNodeIndex));
    gpuErrorcheck(cudaFree(d_messageToDeviceDist));
    gpuErrorcheck(cudaFree(d_messageToDeviceSize));

    return dist;
}



int main(int argc, char **argv) {
    Timer timer_total, timer_load;
    timer_total.start();
    
    ArgumentParser args(argc, argv);

    timer_load.start();
    Graph graph(args.inputFilePath);
    //Graph graph("datasets/simpleGraph.txt");
    graph.readGraph();
    timer_load.stop();
    

    int sourceNode;

    if (args.hasSourceNode) {
        sourceNode = args.sourceNode;
    } else {
        // Use graph default source 
        sourceNode = graph.defaultSource;
    }


    uint *dist_hybrid = sssp_Hybrid(&graph, sourceNode);
    uint *dist_gpu = sssp_GPU(&graph, sourceNode);

    compareResult(dist_hybrid, dist_gpu, graph.numNodes);

    if (args.runOnCPU) {
        uint *dist_cpu = sssp_CPU_parallel(&graph, sourceNode);
        compareResult(dist_cpu, dist_hybrid, graph.numNodes);
    }

    timer_total.stop();
    printf("Total execution time: %f ms\n", timer_total.elapsedTime());
    printf("Graph loading execution time: %f ms\n", timer_load.elapsedTime());

    return 0;
}