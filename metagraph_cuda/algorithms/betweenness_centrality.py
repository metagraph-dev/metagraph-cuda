from metagraph import concrete_algorithm
from ..registry import has_cugraph, has_cudf


if has_cugraph and has_cudf:
    import cugraph
    import cudf
    import random
    from ..types import CuGraphEdgeMap, CuDFNodeMap

    @concrete_algorithm("vertex_ranking.betweenness_centrality")
    def cugraph_betweenness_centrality(
        graph: CuGraphEdgeMap,
        k: int,
        enable_normalization: bool,
        include_endpoints: bool,
    ) -> CuDFNodeMap:
        number_of_nodes = graph.value.number_of_nodes()
        if number_of_nodes >= k:
            sampled_graph = graph.value
        else:
            # workaround for cudf.Series not yet supportting sampling
            edge_list = graph.value.view_edge_list()
            node_series = cudf.concat([edge_list.src, edge_list.dst]).unique()
            nodes = list(node_series)
            sampled_nodes = random.sample(nodes, k)
            sampled_graph = cugraph.subgraph(graph.value, sampled_nodes)
        node_to_score_df = cugraph.betweenness_centrality(
            sampled_graph, normalized=enable_normalization, endpoints=include_endpoints,
        )
        node_to_score_df = node_to_score_df.set_index("vertex")
        return CuDFNodeMap(node_to_score_df, "betweenness_centrality",)
