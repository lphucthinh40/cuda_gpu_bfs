#ifndef BFS_GPU_SURF_CUH
#define BFS_GPU_SURF_CUH

#include "../../graph/graph.hpp"
#include "../../helper/wtime.h"

#include "./utils/alloc.cuh"
#include "./utils/comm.cuh"
#include "./utils/fqg.cuh"
#include "./utils/mcpy.cuh"
#include "./model/model.h"

// BFS Top-Dowm Single Step (PUSH Phase)
void bfs_td(
	depth_t *sa_d,
	const vertex_t * __restrict__ adj_list_d,
	const index_t * __restrict__ offset_d,
	const index_t * __restrict__ adj_deg_d,
	const index_t vert_count,
	depth_t &level,
	vertex_t *fq_td_in_d,
	vertex_t *fq_td_in_curr_sz,
	vertex_t *fq_sz_h,
	vertex_t *fq_td_out_d,
	vertex_t *fq_td_out_curr_sz
){
    if(*fq_sz_h < (vertex_t) (par_beta * vert_count)) {
		fqg_td_wccao<vertex_t, index_t, depth_t> // warp-cooperative chained atomic operations
			<<<BLKS_NUM_TD_WCCAO, THDS_NUM_TD_WCCAO>>>(
			sa_d,
			adj_list_d,
			offset_d,
			adj_deg_d,
			level,
			fq_td_in_d,
			fq_td_in_curr_sz,
			fq_td_out_d,
			fq_td_out_curr_sz
		);
        cudaDeviceSynchronize();
    } else {     
        fqg_td_wcsac<vertex_t, index_t, depth_t> // warp-cooperative status array check
        	<<<BLKS_NUM_TD_WCSAC, THDS_NUM_TD_WCSAC>>>(
			sa_d,
			adj_list_d,
			offset_d,
			adj_deg_d,
			level,
			fq_td_in_d,
			fq_td_in_curr_sz
        );
        cudaDeviceSynchronize();
        fqg_td_tcfe<vertex_t, index_t, depth_t> // thread-centric frontier enqueue
        	<<<BLKS_NUM_TD_TCFE, THDS_NUM_TD_TCFE>>>(
			sa_d,
			vert_count,
			level,
			fq_td_out_d,
			fq_td_out_curr_sz
        );
        cudaDeviceSynchronize();
    }
    H_ERR(cudaMemcpy(fq_sz_h, fq_td_out_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

// BFS Bottom-Up Single Step (PULL Phase)
void bfs_bu(

        depth_t *sa_d,
        const vertex_t * __restrict__ adj_list_d,
        const index_t * __restrict__ offset_d,
        const index_t * __restrict__ adj_deg_d,
        const index_t vert_count,
        depth_t &level,
        vertex_t *fq_sz_h,
        vertex_t *fq_bu_curr_sz
){
    fqg_bu_wcsac<vertex_t, index_t, depth_t>
    <<<BLKS_NUM_BU_WCSA, THDS_NUM_BU_WCSA>>>(

            sa_d,
            adj_list_d,
            offset_d,
            adj_deg_d,
            vert_count,
            level,
            fq_bu_curr_sz
    );
    cudaDeviceSynchronize();
    H_ERR(cudaMemcpy(fq_sz_h, fq_bu_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

void bfs_rev(
	depth_t *sa_d,
	const index_t vert_count,
	depth_t &level,
	vertex_t *fq_sz_h,
	vertex_t *fq_td_in_d,
	vertex_t *fq_td_in_curr_sz
){
    fqg_rev_tcfe<vertex_t, index_t, depth_t> // thread-centric frontier enqueue
    <<<BLKS_NUM_REV_TCFE, THDS_NUM_REV_TCFE>>>(
		sa_d,
		vert_count,
		level,
		fq_td_in_d,
		fq_td_in_curr_sz
    );
    cudaDeviceSynchronize();
    H_ERR(cudaMemcpy(fq_sz_h, fq_td_in_curr_sz, sizeof(vertex_t), cudaMemcpyDeviceToHost));
}

// iterate BFS step until the graph is fully explored (PUSH or PULL in each iteration)
void bfs_tdbu(
	depth_t *sa_d,
	const vertex_t * __restrict__ adj_list_d,
	const index_t * __restrict__ offset_d,
	const index_t * __restrict__ adj_deg_d,
	const index_t vert_count,
	depth_t &level,
	vertex_t *fq_td_1_d,
	vertex_t *temp_fq_td_d,
	vertex_t *fq_td_1_curr_sz,
	vertex_t *temp_fq_curr_sz,
	vertex_t *fq_sz_h,
	vertex_t *fq_td_2_d,
	vertex_t *fq_td_2_curr_sz,
	vertex_t *fq_bu_curr_sz,
	vertex_t INFTY
){
    vertex_t prev_fq_sz = 0;
    vertex_t curr_fq_sz = 0;
    vertex_t unvisited = (vertex_t) vert_count;
    double prev_slope = 0.0;
    double curr_slope = 0.0; // slope (variation)
    double curr_conv = 0.0; // convexity (tendency)
    double curr_proc = 0.0; // processed (progress)
    double remn_proc = 0.0;

    double neuron_input[6];
    neuron_input[0] = avg_deg;
    neuron_input[1] = prob_high;
    bool label = false;

    double t_st_pred;
    t_pred = 0.0;

    bool fq_swap = true;
    bool reversed = false;
    bool TD_BU = false; // true: bottom-up, false: top-down

    *fq_sz_h = 1;

    for(level = 0; ; level++){

        if(level != 0){

            t_st_pred = wtime();

            prev_fq_sz = curr_fq_sz;
            prev_slope = curr_slope;

            curr_fq_sz = *fq_sz_h;
            curr_slope = ((double) curr_fq_sz - prev_fq_sz) / vert_count;
            curr_conv = curr_slope - prev_slope;

            unvisited -= curr_fq_sz;
            curr_proc = (double) curr_fq_sz / vert_count;
            remn_proc = (double) unvisited / vert_count;

            neuron_input[2] = curr_slope;
            neuron_input[3] = curr_conv;
            neuron_input[4] = curr_proc;
            neuron_input[5] = remn_proc;

            label = predict(neuron_input);

            t_pred += (wtime() - t_st_pred);
        }

        if(!label){

            if(TD_BU)
                reversed = true;

            TD_BU = false;
        }

        else
            TD_BU = true;

        if(!TD_BU){

            if(!fq_swap)
                fq_swap = true;
            else
                fq_swap = false;

            if(level != 0){

                if(!reversed){

                    if(!fq_swap){

                        mcpy_init_fq_td<vertex_t, index_t, depth_t>
                        <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                vert_count,
                                temp_fq_td_d,
                                temp_fq_curr_sz,
                                fq_td_2_d,
                                fq_td_2_curr_sz,
                                INFTY
                        );
                    }

                    else{

                        if(level == 1){

                            init_fqg_2<vertex_t, index_t, depth_t>
                            <<<1, 1>>>(

                                    fq_td_1_d,
                                    fq_td_1_curr_sz,
                                    INFTY
                            );
                        }

                        else{

                            mcpy_init_fq_td<vertex_t, index_t, depth_t>
                            <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                                    vert_count,
                                    temp_fq_td_d,
                                    temp_fq_curr_sz,
                                    fq_td_1_d,
                                    fq_td_1_curr_sz,
                                    INFTY
                            );
                        }
                    }
                }

                else{

                    reversed = false;
                    fq_swap = false;

                    mcpy_init_fq_td<vertex_t, index_t, depth_t>
                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                            vert_count,
                            temp_fq_td_d,
                            temp_fq_curr_sz,
                            fq_td_2_d,
                            fq_td_2_curr_sz,
                            INFTY
                    );

                    mcpy_init_fq_td<vertex_t, index_t, depth_t>
                    <<<BLKS_NUM_INIT_RT, THDS_NUM_INIT_RT>>>(

                            vert_count,
                            temp_fq_td_d,
                            temp_fq_curr_sz,
                            fq_td_1_d,
                            fq_td_1_curr_sz,
                            INFTY
                    );
                    cudaDeviceSynchronize();

                    bfs_rev(
						sa_d,
						vert_count,
						level,
						fq_sz_h,
						fq_td_1_d,
						fq_td_1_curr_sz
                    );
                }
            }

            cudaDeviceSynchronize();

            if(!fq_swap){
                bfs_td(
					sa_d,
					adj_list_d,
					offset_d,
					adj_deg_d,
					vert_count,
					level,
					fq_td_1_d,
					fq_td_1_curr_sz,
					fq_sz_h,
					fq_td_2_d,
					fq_td_2_curr_sz
                );
            }

            else{
                bfs_td(
					sa_d,
					adj_list_d,
					offset_d,
					adj_deg_d,
					vert_count,
					level,
					fq_td_2_d,
					fq_td_2_curr_sz,
					fq_sz_h,
					fq_td_1_d,
					fq_td_1_curr_sz
                );
            }

            cudaDeviceSynchronize();
        }
        else{

            flush_fq<vertex_t, index_t, depth_t>
            <<<1, 1>>>(

                    fq_bu_curr_sz
            );
            cudaDeviceSynchronize();

            bfs_bu(
				sa_d,
				adj_list_d,
				offset_d,
				adj_deg_d,
				vert_count,
				level,
				fq_sz_h,
				fq_bu_curr_sz
            );
            cudaDeviceSynchronize();
        }

        if(*fq_sz_h == 0)
            break;
    }
}

void bfs_gpu_surf(Graph *G, index_t start) {
    // Init MLD model with pre-trained weights
    init_model();

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    depth_t *sa_d;              // status array on GPU
    depth_t *sa_h;              // status array on CPU
    depth_t *temp_sa;           // initial state of status array (used for iterative test)
    index_t *adj_deg_d;         // the number of neighbors for each vertex
    index_t *adj_deg_h;
    vertex_t *adj_list_d;       // adjacent lists
    index_t *offset_d;          // offset
    vertex_t *fq_td_1_d;        // frontier queue for top-down traversal
    vertex_t *fq_td_1_curr_sz;  // used for the top-down queue size
                                // synchronized index of frontier queue for top-down traversal, the size must be 1
    vertex_t *fq_td_2_d;
    vertex_t *fq_td_2_curr_sz;
    vertex_t *temp_fq_td_d;
    vertex_t *temp_fq_curr_sz;
    vertex_t *fq_sz_h;
    vertex_t *fq_bu_curr_sz;    // used for the number of vertices examined at each level, the size must be 1
	
	depth_t level = 0;
    
    double t_start;
    double t_elapsed;
    
    // Allocate memory on GPU (check alloc.cuh)
    alloc<vertex_t, index_t, depth_t>::
    alloc_mem(
		sa_d,
		sa_h,
		temp_sa,
		adj_list_d,
		adj_deg_d,
		adj_deg_h,
		offset_d,
		G->startIdx,
		G->adjacencyList,
		G->verticesCount,
		G->edgesCount,
		fq_td_1_d,
		temp_fq_td_d,
		fq_td_1_curr_sz,
		temp_fq_curr_sz,
		fq_sz_h,
		fq_td_2_d,
		fq_td_2_curr_sz,
		fq_bu_curr_sz
    );

    // Initialize temp frontier_queue & frontier_queue_size
    mcpy_init_temp<vertex_t, index_t, depth_t>
    <<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(
		G->verticesCount,
		temp_fq_td_d,
		temp_fq_curr_sz,
		INFTY
    );
    cudaDeviceSynchronize();

    // Warm up GPU
    // warm_up_gpu<<<BLKS_NUM_INIT, THDS_NUM_INIT>>>();
    // cudaDeviceSynchronize();

    // Setup temp status array (useful for iterative test)
    H_ERR(cudaMemcpy(sa_d, temp_sa, sizeof(depth_t) * G->verticesCount, cudaMemcpyHostToDevice));
    H_ERR(cudaMemcpy(sa_h, temp_sa, sizeof(depth_t) * G->verticesCount, cudaMemcpyHostToHost));

	// Initialize all queues
	mcpy_init_fq_td<vertex_t, index_t, depth_t>
	<<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(
		G->verticesCount,
		temp_fq_td_d,
		temp_fq_curr_sz,
		fq_td_1_d,
		fq_td_1_curr_sz,
		INFTY
	);
	cudaDeviceSynchronize();

	mcpy_init_fq_td<vertex_t, index_t, depth_t>
	<<<BLKS_NUM_INIT, THDS_NUM_INIT>>>(
		G->verticesCount,
		temp_fq_td_d,
		temp_fq_curr_sz,
		fq_td_2_d,
		fq_td_2_curr_sz,
		INFTY
	);
	cudaDeviceSynchronize();

	init_fqg<vertex_t, index_t, depth_t>
	<<<1, 1>>>(
		start,
		sa_d,
		fq_td_1_d,
		fq_td_1_curr_sz
	);
	cudaDeviceSynchronize();

	// calculate parallel options
	calc_par_opt<vertex_t, index_t>(
			adj_deg_h,
			G->verticesCount,
			G->edgesCount
	);
    cudaDeviceSynchronize();

    t_start = wtime();
        
	bfs_tdbu(
		sa_d,
		adj_list_d,
		offset_d,
		adj_deg_d,
		G->verticesCount,
		level,
		fq_td_1_d,
		temp_fq_td_d,
		fq_td_1_curr_sz,
		temp_fq_curr_sz,
		fq_sz_h,
		fq_td_2_d,
		fq_td_2_curr_sz,
		fq_bu_curr_sz,
		INFTY
	);
    cudaDeviceSynchronize();

    t_elapsed = wtime() - t_start;

	// for validation
	index_t tr_vert = 0;
	index_t tr_edge = 0;

	H_ERR(cudaMemcpy(sa_h, sa_d, sizeof(depth_t) * G->verticesCount, cudaMemcpyDeviceToHost));

	for(index_t j = 0; j < G->verticesCount; j++){
		if(sa_h[j] != UNVISITED){
			tr_vert++;
			tr_edge += adj_deg_h[j];
		}
	}

    printf( "===================================\n"
            "SURF_GPU_BFS: started from %llu\n"
            "- traversed %llu/%llu vertices\n"
            "- traversed %llu/%llu edges\n"
            "- number of iterations (level): %d\n"
            "- elapsed time: %f\n"
    , start, tr_vert, G->verticesCount, tr_edge, G->edgesCount, level+1, t_elapsed);

    cudaDeviceSynchronize();

	alloc<vertex_t, index_t, depth_t>::
    dealloc_mem(
		sa_d,
		sa_h,
		temp_sa,
		adj_list_d,
		adj_deg_d,
		adj_deg_h,
		offset_d,
		fq_td_1_d,
		temp_fq_td_d,
		fq_td_1_curr_sz,
		temp_fq_curr_sz,
		fq_sz_h,
		fq_td_2_d,
		fq_td_2_curr_sz,
		fq_bu_curr_sz
    );

    std::cout << "SURF_GPU BFS finished..." << std::endl;
}

#endif // BFS_GPU_SURF_CUH
