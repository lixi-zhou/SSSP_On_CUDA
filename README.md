# SSSP_On_CUDA

---

Implement Dijkstra's SSSP on both CPU and GPU (CUDA)

---

## Instruction

Run `make` in the root folder.


---

### Known issues
- [x] error when copying graph data from host to device
- [x] SSSP on GPU/CPU implemented, but performance of GPU's is worse than CPU'S
- [ ] SSSP version1 may have a accuracy problem when processing big graph
- [ ] In some graph, its maximum node id is not its number of nodes