from metagraph import concrete_algorithm
from ..registry import has_cugraph


if has_cugraph:
    import cugraph
    from ..types import CuGraph, CuDFNodeMap

    @concrete_algorithm("clustering.connected_components")
    def connected_components(graph: CuGraph) -> CuDFNodeMap:
        df = cugraph.weakly_connected_components(graph.edges.value)
        df = df.set_index("vertices")
        return CuDFNodeMap(df, "labels")

    @concrete_algorithm("clustering.strongly_connected_components")
    def strongly_connected_components(graph: CuGraph) -> CuDFNodeMap:
        df = cugraph.strongly_connected_components(graph.edges.value)
        df = df.set_index("vertices")
        return CuDFNodeMap(df, "labels")
