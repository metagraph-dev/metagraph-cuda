from metagraph import concrete_algorithm, NodeID
from ..registry import has_cugraph
from typing import Tuple, Any

if has_cugraph:
    import cugraph
    from ..types import CuGraph, CuDFVector

    @concrete_algorithm("traversal.bfs_iter")
    def breadth_first_search(
        graph: CuGraph, source_node: NodeID, depth_limit: int
    ) -> CuDFVector:
        bfs_df = cugraph.bfs(graph.edges.value, source_node)
        bfs_df = bfs_df[bfs_df.predecessor.isin(bfs_df.vertex) | (bfs_df.distance == 0)]
        bfs_ordered_vertices = bfs_df.sort_values("distance").vertex.reset_index(
            drop=True
        )
        return CuDFVector(bfs_ordered_vertices)
