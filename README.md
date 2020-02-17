# SSSP_On_CUDA

---

Implement Single-Source Shortest Paths (SSSP) on both CPU and GPU (CUDA)

---

<!-- TOC -->

- [SSSP_On_CUDA](#sssponcuda)
  - [Instruction](#instruction)
  - [Description](#description)
      - [Implementation of Dijkstra on CPU](#implementation-of-dijkstra-on-cpu)
    - [Implementation of Dijkstra on GPU (sssp2)](#implementation-of-dijkstra-on-gpu-sssp2)
    - [Implementation of Dijkstra on GPU (sssp6)](#implementation-of-dijkstra-on-gpu-sssp6)
    - [Implementation of Dijkstra on GPU (sssp7)](#implementation-of-dijkstra-on-gpu-sssp7)
    - [Implementation of Dijkstra on GPU (sssp8)](#implementation-of-dijkstra-on-gpu-sssp8)
    - [Known issues](#known-issues)

<!-- /TOC -->

## Instruction

Run `make` in the root folder to generate the executable file.

The core algorithm of this project is **Bellman-Ford Algorithm**.

## Description

#### Implementation on CPU

1. Loop all edges to update vertexs' distance to source node.
2. Repeate *Step 1* until there is no vertex needs to update its distance to source.

### Implementation of GPU

1. Divide all edges into multiple parts.
2. Launch multiple threads to process the edges assigned from *Step 1*.
3. Repeate *Step 1* and *Step 2* until there is no vertex needs to update its distance to source.
Basic implementation of dijkstra algorithm on GPU.

## Running Application

```shell
$./sssp --input path_of_graph
```

#### Application Argument

```
Optional arguments:
  [--oncpu]: Run this graph on CPU. Its value must be true/false (default: false). E.g., --oncpu true
  [--source]: Set the source node (default: minimum node number). E.g., --source 0  
```

## Input Graph Format

Input graphs should be in form of plain text files. The format of each edge is as following:

```
source end weight
```

if the weight is not specified, it will be assigned to a default value: **1**.


---

### Known issues
