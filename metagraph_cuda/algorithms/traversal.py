from metagraph import concrete_algorithm
from ..registry import has_cugraph
from typing import Tuple, Any

if has_cugraph:
    import cugraph
    from ..types import CuGraph, CuDFNodes
    from metagraph.plugins.numpy.types import NumpyVector

    @concrete_algorithm("traversal.breadth_first_search")
    def breadth_first_search(graph: CuGraph, source_node: Any) -> NumpyVector:
        bfs_df = cugraph.bfs(graph.value, 0)
        bfs_df = bfs_df[bfs_df.predecessor.isin(bfs_df.vertex) | (bfs_df.distance == 0)]
        bfs_ordered_vertices = bfs_df.sort_values("distance").vertex.to_array()
        return NumpyVector(bfs_ordered_vertices)
