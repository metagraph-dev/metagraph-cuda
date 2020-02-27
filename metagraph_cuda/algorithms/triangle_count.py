from metagraph import abstract_algorithm, concrete_algorithm
from ..registry import cugraph


if cugraph:

    @concrete_algorithm("cluster.triangle_count")
    def cugraph_triangle_count(graph: cugraph.DiGraph) -> int:
        return cugraph.triangles(graph) // 3
