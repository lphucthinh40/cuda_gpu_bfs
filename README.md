# cuda_gpu_bfs

Compile
```
# in root directory
make
```

Run
```
./run_bfs <dataset_name> <source_vertice>
    - <dataset_name>: ["roadNet-CA", "wiki-Talk", "wiki-Vote"]
    - <source_vertice>: default to 0
```

![Alt text](/output/sample_output.png "output")

Note: Our dataset graphs are not fully connected. Thus, visited_nodes & visited_edges are normally less than total_nodes & total_edges.
