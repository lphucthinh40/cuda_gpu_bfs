#ifndef GRAPH_H
#define GRAPH_H

#include <climits>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <iostream>

// Use ULL type for vertices to handle very-large graph
typedef unsigned long long vertex_t;
typedef unsigned long long index_t;
typedef unsigned int depth_t;
const vertex_t INFTY = ULLONG_MAX;

// Graph Model
class Graph {
public:
	Graph(const char *idx_file, const char *csr_file);
    ~Graph();
    vertex_t* adjacencyList; // neighbours of consecutive vertexes
    vertex_t* startIdx; // starting index of the first edge for any node
    index_t verticesCount;
    index_t edgesCount;
    bool loaded;

private:
    bool loadIndexFile(const char *idx_file);
    bool loadCSRFile(const char *csr_file);
};


inline off_t getFileSize_inBytes(const char *filename) {
	struct stat st; 
	if (stat(filename, &st) == 0)
		return st.st_size;
	return -1; 
}

Graph::Graph(const char *idx_file, const char *csr_file) {
	loaded = true;
	loaded &= loadIndexFile(idx_file);
	loaded &= loadCSRFile(csr_file);
	if (loaded) { 
		std::cout<<"Graph successfully loaded\n";
		std::cout<<"-- Num of Vertices: " << verticesCount << std::endl;
		std::cout<<"-- Num of Edges: " << edgesCount << std::endl;
	}
}

Graph::~Graph() {
    delete [] startIdx;
    delete [] adjacencyList;
}

bool Graph::loadIndexFile(const char *idx_file) {
	FILE *file;
	index_t ret;
	verticesCount=getFileSize_inBytes(idx_file)/sizeof(vertex_t) - 1;
	file=fopen(idx_file, "rb");
	if (file!=NULL) {
		startIdx = new vertex_t[verticesCount+1];
		ret=fread(startIdx, sizeof(vertex_t), verticesCount+1, file);
		assert(ret==verticesCount+1);
		fclose(file);
		return true;
	} else {
		fclose(file);
		return false;
	}
}

bool Graph::loadCSRFile(const char *csr_file) {
	FILE *file;
	index_t ret;
	edgesCount=getFileSize_inBytes(csr_file)/sizeof(vertex_t);
	file=fopen(csr_file, "rb");
	if (file!=NULL) {
		if(posix_memalign((void **)&adjacencyList,32,sizeof(index_t)*edgesCount)) perror("posix_memalign");
		ret=fread(adjacencyList, sizeof(vertex_t), edgesCount, file);
		assert(ret==edgesCount);
		fclose(file);
		return true;
	} else {
		fclose(file);
		return false;
	}
}

#endif // GRAPH_H
