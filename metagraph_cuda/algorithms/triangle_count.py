from metagraph import concrete_algorithm
from ..registry import has_cugraph


if has_cugraph:
    import cugraph

    @concrete_algorithm("cluster.triangle_count")
    def cugraph_triangle_count(graph: cugraph.DiGraph) -> int:
        return cugraph.triangles(graph) // 3
