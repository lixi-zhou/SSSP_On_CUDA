# SSSP_On_CUDA

---

Implement Dijkstra's SSSP on both CPU and GPU (CUDA)

---

<!-- TOC -->

- [SSSP_On_CUDA](#sssponcuda)
  - [Instruction](#instruction)
  - [Description](#description)
      - [Implementation of dijkstra on CPU](#implementation-of-dijkstra-on-cpu)
    - [Implementation of dijkstra on GPU (sssp2)](#implementation-of-dijkstra-on-gpu-sssp2)
    - [Implementation of dijkstra on GPU (sssp6)](#implementation-of-dijkstra-on-gpu-sssp6)
    - [Implementation of dijkstra on GPU (sssp7)](#implementation-of-dijkstra-on-gpu-sssp7)
    - [Implementation of dijkstra on GPU (sssp8)](#implementation-of-dijkstra-on-gpu-sssp8)
    - [Known issues](#known-issues)

<!-- /TOC -->

## Instruction

Run `make [OPTION]` in the root folder to generate the executable file. There are mainly **4** different implementation of the SSSP.

*The option can be sssp2, sssp6, sssp7, sssp8*

There are some limitation now:

1. Argument parsing is not support yet. The path of graph data has to be hard code in the source file.
2. For now, it only supports the graph, whose number of node is no bigger than 12,000.

## Description

#### Implementation of dijkstra on CPU

- Set *finished*: the nodes have been processed.
- Set *unprocessed*: the nodes have not been processed. `All nodes are placed in unprocessed set after initialization.`

1. Use for loop to find the closest node to the source node from **unprocessed** set.
2. Use for loop to update the connected nodes' distance of the closest node.
3. Repeat *Step 1* and *Step 2* until all nodes are in **finished** set.

### Implementation of dijkstra on GPU (sssp2)

Basic implementation of dijkstra algorithm on GPU.

1. Launch multiple thread to find the closest node parallel from **unprocessed** set.
2. Launch multiple thread to update the connected nodes' distance of the closest node.
3. Repeat  *Step 1* and *Step 2* untill all nodes are in **finished** set. 

### Implementation of dijkstra on GPU (sssp6)

1. Launch multiple thread the find the minimum distance of **unprocessed** nodes.
2. Launch multiple thread to update the connected node's distance of the nodes, whose distance to source node equals to the minimum distance of *Step 1*.
3. Repeat  *Step 1* and *Step 2* untill all nodes are in **finished** set. 

### Implementation of dijkstra on GPU (sssp7)

1. Use CPU to find the minimum distance of **unprocessed** nodes.
2. Launch multiple thread to update the connected node's distance of the nodes, whose distance to source node equals to the minimum distance of *Step 1*.
3. Repeat  *Step 1* and *Step 2* untill all nodes are in **finished** set. 

### Implementation of dijkstra on GPU (sssp8)

Allocate the memory by using the unified memory. The other part is the same as the sssp6.


---

### Known issues
- [x] error when copying graph data from host to device
- [x] SSSP on GPU/CPU implemented, but performance of GPU's is worse than CPU'S
- [ ] SSSP version1 may have a accuracy problem when processing big graph
- [ ] In some graph, its maximum node id is not its number of nodes