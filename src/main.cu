#include <iostream>
#include <vector>
#include <algorithm>

#include "graph/graph.hpp"
#include "helper/warmup.cuh"

#include "bfs/cpu/bfs_cpu.hpp"
#include "bfs/gpu_simple/bfs_gpu_simple.cuh"
#include "bfs/gpu_surf/bfs_gpu_surf.cuh"

bool in_strlist(const std::string &value, const std::vector<std::string> &array)
{
    return std::find(array.begin(), array.end(), value) != array.end();
}

const std::string FILE_PREFIX = "./data/";
const std::string FILE_START_IDX_POSTFIX = ".mtx_beg_pos.bin";
const std::string FILE_ADJ_LIST_POSTFIX = ".mtx_adj_list.bin";
const std::vector<std::string> DATASETS {"roadNet-CA", "wiki-Talk", "wiki-Vote"};


int main(int argc, char **argv){
	
	if(argc < 2){
        std::cout << "Missing required argument: dataset_name\n";
        exit(-1);
    }

    std::string dataset_name = std::string(argv[1]);
    if (!in_strlist(dataset_name, DATASETS)) {
        std::cout << "Unknown dataset_name. Please try again!\n";
        exit(-1);
    }

    std::string file_start_idx = FILE_PREFIX + dataset_name + "/" + dataset_name + FILE_START_IDX_POSTFIX;
    std::string file_adj_list = FILE_PREFIX + dataset_name + "/" + dataset_name + FILE_ADJ_LIST_POSTFIX;

    index_t src = 0;
    if (argc >= 3) {
        std::string src_arg = argv[2];
        std::size_t pos;
        index_t temp_src = std::stoull(src_arg, &pos);
        if (pos > 0) {
            src = temp_src;
        }
    }
    
	Graph* graph = new Graph(file_start_idx.c_str(), file_adj_list.c_str());
    
    warm_up_gpu();

    bfs_cpu(graph, src);
    bfs_gpu_simple(graph, src);
    bfs_gpu_surf(graph, src);

    delete graph;

    return 0;
}