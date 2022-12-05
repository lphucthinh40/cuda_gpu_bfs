#ifndef BFS_GPU_SIMPLE_CUH
#define BFS_GPU_SIMPLE_CUH

#include "../../graph/graph.hpp"
#include "../../helper/wtime.h"

#define MAX_THREADS_PER_BLOCK 256

__global__ void
bfs_gpu_kernel( vertex_t* g_graph_nodes, vertex_t* g_graph_edges, bool* g_graph_mask, bool* g_graph_visited, bool* g_edges_visited, int* g_cost, bool *g_over, index_t n_vertices, index_t n_edges) 
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if (tid<n_vertices && g_graph_mask[tid]) {
		g_graph_mask[tid]=false;
		g_graph_visited[tid]=true;
        index_t no_of_edges = (tid<n_vertices-1) ? g_graph_nodes[tid+1] - g_graph_nodes[tid] : n_edges - g_graph_nodes[tid];
		for (int i=g_graph_nodes[tid]; i<(g_graph_nodes[tid]+no_of_edges); i++) {
            int id = g_graph_edges[i];
            g_edges_visited[i]=true;
			if (!g_graph_visited[id]) {
				g_cost[id]=g_cost[tid]+1;
				g_graph_mask[id]=true;
				//Change the loop stop value such that loop continues
				*g_over=true;
            }
        }
	}
}

void bfs_gpu_simple(Graph *G, index_t start) {
    double t_start;
    double t_elapsed;
    int num_of_blocks = 1;
	int num_of_threads_per_block = G->verticesCount;
	
	if (G->verticesCount>MAX_THREADS_PER_BLOCK) {
        num_of_blocks = (int)ceil(G->verticesCount/(double)MAX_THREADS_PER_BLOCK); 
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}
		
	// allocate host memory
    bool *h_graph_mask = (bool*) malloc(sizeof(bool)*G->verticesCount);
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*G->verticesCount);
    bool *h_edges_visited = (bool*) malloc(sizeof(bool)*G->edgesCount);

    // initalize the memory
    for( index_t i = 0; i < G->verticesCount; i++) 
    {
        h_graph_mask[i]=false;
        h_graph_visited[i]=false;
    }
    for( index_t i = 0; i < G->edgesCount; i++) 
    {
        h_edges_visited[i]=false;
    }

    //set the source node as true in the mask
    h_graph_mask[start]=true;

    //Copy the Node list to device memory
    vertex_t* d_graph_nodes;
    cudaMalloc( (void**) &d_graph_nodes, sizeof(vertex_t)*G->verticesCount);
    cudaMemcpy( d_graph_nodes, G->startIdx, sizeof(vertex_t)*G->verticesCount, cudaMemcpyHostToDevice);

	//Copy the Edge List to device Memory
	vertex_t* d_graph_edges;
    cudaMalloc( (void**) &d_graph_edges, sizeof(vertex_t)*G->edgesCount);
    cudaMemcpy( d_graph_edges, G->adjacencyList, sizeof(vertex_t)*G->edgesCount, cudaMemcpyHostToDevice);
    
    //Copy the Mask to device memory
    bool* d_graph_mask;
    cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*G->verticesCount);
    cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*G->verticesCount, cudaMemcpyHostToDevice);
    
    //Copy the Visited nodes array to device memory
    bool* d_graph_visited;
    cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*G->verticesCount);
    cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*G->verticesCount, cudaMemcpyHostToDevice);

    //Copy the Visited edges array to device memory
    bool* d_edges_visited;
    cudaMalloc( (void**) &d_edges_visited, sizeof(bool)*G->edgesCount);
    cudaMemcpy( d_edges_visited, h_edges_visited, sizeof(bool)*G->edgesCount, cudaMemcpyHostToDevice);
    
    // allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*G->verticesCount);
	for(index_t i=0;i<G->verticesCount;i++)
	    h_cost[i]=-1;
	h_cost[start]=0;

	// allocate device memory for result
    int* d_cost;
    cudaMalloc( (void**) &d_cost, sizeof(int)*G->verticesCount);
    cudaMemcpy( d_cost, h_cost, sizeof(int)*G->verticesCount, cudaMemcpyHostToDevice);

    //make a bool to check if the execution is over
    bool *d_over;
    cudaMalloc( (void**) &d_over, sizeof(bool));

	// printf("Copied Everything to GPU memory\n");

    // setup execution parameters
    dim3  grid( num_of_blocks, 1, 1);
    dim3  threads( num_of_threads_per_block, 1, 1);

    int k=0;
	
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
    t_start = wtime();

    do
    {
	//if no thread changes this value then the loop stops
	stop=false;
	cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);
    bfs_gpu_kernel<<< grid, threads, 0 >>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_graph_visited, d_edges_visited, d_cost, d_over, G->verticesCount, G->edgesCount);
	cudaDeviceSynchronize();
	// check if kernel execution generated and error
	cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
    k++;
	}
    while(stop);

	cudaDeviceSynchronize();

    t_elapsed = wtime() - t_start;
    
    // copy result from device to host
    cudaMemcpy( h_cost, d_cost, sizeof(index_t)*G->verticesCount, cudaMemcpyDeviceToHost);    
    cudaMemcpy( h_graph_visited, d_graph_visited, sizeof(bool)*G->verticesCount, cudaMemcpyDeviceToHost);
    cudaMemcpy( h_edges_visited, d_edges_visited, sizeof(bool)*G->edgesCount, cudaMemcpyDeviceToHost);

    index_t nodes_visited_count = 0;
    for( index_t i = 0; i < G->verticesCount; i++) 
    {
        nodes_visited_count += (h_graph_visited[i])? 1:0;
    }

    index_t edges_visited_count = 0;
    for( index_t i = 0; i < G->edgesCount; i++) 
    {
        edges_visited_count += (h_edges_visited[i])? 1:0;
    }

    printf( "===================================\n"
            "SIMPLE_GPU_BFS: started from %llu\n"
            "- traversed %llu/%llu vertices\n"
            "- traversed %llu/%llu edges\n"
            "- number of iterations (level): %d\n"
            "- elapsed time: %f\n"
    , start, nodes_visited_count, G->verticesCount, edges_visited_count, G->edgesCount, k, t_elapsed);
	
    // cleanup memory
    free( h_graph_mask);
    free( h_graph_visited);
    free( h_cost);
    cudaFree(d_graph_nodes);
    cudaFree(d_graph_edges);
    cudaFree(d_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);

    printf( "SIMPLE_GPU BFS finished...\n" );
}

#endif // BFS_GPU_SIMPLE_CUH