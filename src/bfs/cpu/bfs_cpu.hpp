#ifndef BFS_CPU_HPP
#define BFS_CPU_HPP

#include <queue>
#include <vector>
#include <bits/stdc++.h>

#include "../../graph/graph.hpp"
#include "../../helper/wtime.h"

void bfs_cpu(Graph *G, index_t start) {
    double t_start;
    double t_elapsed;
    std::vector<unsigned int> distance = std::vector<unsigned int>(G->verticesCount);
	std::vector<bool> visitedVertices = std::vector<bool>(G->verticesCount);
	std::vector<bool> visitedEdges = std::vector<bool>(G->edgesCount);

    fill(distance.begin(), distance.end(), UINT_MAX);
	fill(visitedVertices.begin(), visitedVertices.end(), false);
    fill(visitedEdges.begin(), visitedEdges.end(), false);

	distance[start] = 0;
	
	std::queue<vertex_t> to_visit;
	to_visit.push(start);

	t_start = wtime();
	while (!to_visit.empty()) {
		vertex_t current = to_visit.front();
		visitedVertices[current] = true;
		to_visit.pop();
		index_t no_of_edges = (current<G->verticesCount-1) ? G->startIdx[current+1] - G->startIdx[current] : G->edgesCount - G->startIdx[current];
		for (index_t i = G->startIdx[current]; i < G->startIdx[current] + no_of_edges; ++i) {
			vertex_t v = G->adjacencyList[i];
			visitedEdges[i] = true;
			if (distance[v] == UINT_MAX) {
				distance[v] = distance[current] + 1;
				to_visit.push(v);
			}
		}
	}
	t_elapsed = wtime() - t_start;

	index_t vertices_visited_count = 0;
    for( index_t i = 0; i < G->verticesCount; i++) 
    {
        vertices_visited_count += (visitedVertices[i])? 1:0;
    }

    index_t edges_visited_count = 0;
    for( index_t i = 0; i < G->edgesCount; i++) 
    {
        edges_visited_count += (visitedEdges[i])? 1:0;
    }

    printf( "===================================\n"
            "CPU_BFS: started from %llu\n"
            "- traversed %llu/%llu vertices\n"
            "- traversed %llu/%llu edges\n"
            "- elapsed time: %f\n"
			"CPU BFS finished...\n"
    , start, vertices_visited_count, G->verticesCount, edges_visited_count, G->edgesCount, t_elapsed);

}

#endif // BFS_CPU_HPP
