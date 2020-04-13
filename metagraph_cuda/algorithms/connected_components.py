from metagraph import concrete_algorithm
from ..registry import has_cugraph


if has_cugraph:
    import cugraph
    from ..types import CuGraph, CuDFNodes

    @concrete_algorithm("clustering.connected_components")
    def connected_components(graph: CuGraph) -> CuDFNodes:
        df = cugraph.weakly_connected_components(graph.value)
        return CuDFNodes(df, "vertices", "labels", weights="non-negative")

    @concrete_algorithm("clustering.strongly_connected_components")
    def strongly_connected_components(graph: CuGraph) -> CuDFNodes:
        df = cugraph.strongly_connected_components(graph.value)
        return CuDFNodes(df, "vertices", "labels", weights="non-negative")
