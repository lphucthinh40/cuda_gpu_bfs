#ifndef GRAPH_H
#define GRAPH_H

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <iostream>

typedef unsigned long long graph_data_t;
typedef unsigned long long graph_size_t;

// Graph Model
class Graph {
public:
	Graph(const char *idx_file, const char *csr_file);
    ~Graph();
    graph_data_t* adjacencyList; // neighbours of consecutive vertexes
    graph_data_t* startIdx; // starting index of the first edge for any node
    graph_size_t verticesCount;
    graph_size_t edgesCount;
    bool loaded;

private:
    bool loadIndexFile(const char *idx_file);
    bool loadCSRFile(const char *csr_file);
};

#endif // GRAPH_H
