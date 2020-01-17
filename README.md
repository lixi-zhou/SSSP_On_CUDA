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

Run `make [OPTION]` in the root folder to generate the executable file. There are mainly **3** different implementation of the SSSP.

## Description

#### Implementation of dijkstra on CPU

- Set *finished*: the nodes have been processed.
- Set *unprocessed*: the nodes have not been processed. `All nodes are placed in unprocessed set after initialization.`

1. Find the closest node to the source node.
2. Update the connected nodes' distance of the closest node.
3. Loop *Step 1* and *Step 2* until all nodes are in **finished** set.

### Implementation of dijkstra on GPU (sssp2)

### Implementation of dijkstra on GPU (sssp6)

### Implementation of dijkstra on GPU (sssp7)

### Implementation of dijkstra on GPU (sssp8)



- SSSP2: Basic 


---

### Known issues
- [x] error when copying graph data from host to device
- [x] SSSP on GPU/CPU implemented, but performance of GPU's is worse than CPU'S
- [ ] SSSP version1 may have a accuracy problem when processing big graph
- [ ] In some graph, its maximum node id is not its number of nodes