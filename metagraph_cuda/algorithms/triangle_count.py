from metagraph import concrete_algorithm
from ..registry import has_cugraph


if has_cugraph:
    import cugraph
    from ..types import CuGraph

    @concrete_algorithm("cluster.triangle_count")
    def cugraph_triangle_count(graph: CuGraph) -> int:
        return cugraph.triangles(graph.value) // 3
