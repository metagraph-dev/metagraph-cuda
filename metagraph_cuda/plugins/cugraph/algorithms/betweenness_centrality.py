import metagraph as mg
from metagraph import concrete_algorithm
from ..registry import has_cugraph, has_cudf


if has_cugraph and has_cudf:
    import cugraph
    import cudf
    from ..types import CuGraph, CuDFNodeSet, CuDFNodeMap

    # TODO only works for unweighted graphs ; make this only work for unweighted graphs
    # @concrete_algorithm("centrality.betweenness")
    # def cugraph_betweenness_centrality(
    #         graph: CuGraph,
    #         nodes: mg.Optional[CuDFNodeSet], # TODO verify this is correct
    #         normalize: bool,
    # ) -> CuDFNodeMap:
    #     node_to_score_df = cugraph.betweenness_centrality(graph.edges.value, k=None if nodes is None else nodes.value.tolist(), normalized=False) # TODO use 'weights' param
    #     node_to_score_df = node_to_score_df.set_index("vertex")
    #     return CuDFNodeMap(node_to_score_df, "betweenness_centrality",)
