#include "graph.h"

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

bool Graph::loadIndexFile(const char *idx_file) {
	FILE *file;
	graph_size_t ret;
	verticesCount=getFileSize_inBytes(idx_file)/sizeof(graph_data_t) - 1;
	file=fopen(idx_file, "rb");
	if (file!=NULL) {
		startIdx = new graph_data_t[verticesCount+1];
		ret=fread(startIdx, sizeof(graph_data_t), verticesCount+1, file);
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
	graph_size_t ret;
	edgesCount=getFileSize_inBytes(csr_file)/sizeof(graph_data_t);
	file=fopen(csr_file, "rb");
	if (file!=NULL) {
		if(posix_memalign((void **)&adjacencyList,32,sizeof(graph_size_t)*edgesCount)) perror("posix_memalign");
		ret=fread(adjacencyList, sizeof(graph_data_t), edgesCount, file);
		assert(ret==edgesCount);
		fclose(file);
		return true;
	} else {
		fclose(file);
		return false;
	}
}