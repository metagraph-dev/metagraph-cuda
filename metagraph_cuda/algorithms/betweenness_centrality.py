from metagraph import concrete_algorithm
from ..registry import has_cugraph


if has_cugraph:
    import cugraph
    from ..types import CuGraph, CuDFNodes

    @concrete_algorithm("vertex_ranking.betweenness_centrality")
    def cugraph_betweenness_centrality(
        graph: CuGraph, k: int, enable_normalization: bool, include_endpoints: bool,
    ) -> CuDFNodes:
        node_to_score_df = cugraph.betweenness_centrality(
            graph.value,
            k=k,
            normalized=enable_normalization,
            endpoints=include_endpoints,
        )
        return CuDFNodes(
            node_to_score_df,
            "vertex",
            "betweenness_centrality",
            node_index=graph.node_index,
            dtype="float",
            weights="non-negative",
        )
