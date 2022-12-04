#include <iostream>

#include "graph/graph.h"

using namespace std;

int main(int argc, char **argv){
	
	if(argc < 3){
        std::cout << "Missing required arguments beg_pos and adj_list of input graph\n";
        exit(-1);
    }

    std::string file_start_idx = std::string(argv[1]);
    std::string file_adj_list = std::string(argv[2]);

	Graph* graph = new Graph(file_start_idx.c_str(), file_adj_list.c_str());

}