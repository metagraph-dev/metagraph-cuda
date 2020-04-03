from metagraph import concrete_algorithm
from ..registry import has_cugraph


if has_cugraph:
    import cugraph

    @concrete_algorithm("cluster.triangle_count")
    def auto_cugraph_triangle_count(graph: cugraph.Graph) -> int:
        return cugraph.triangles(graph) // 3
