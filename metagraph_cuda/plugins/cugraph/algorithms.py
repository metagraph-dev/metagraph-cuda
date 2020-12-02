import metagraph as mg
from metagraph import concrete_algorithm, NodeID
from .. import has_cugraph
from typing import Callable, Tuple, Any

if has_cugraph:
    import cugraph
    import cudf
    import cupy
    from .types import CuGraph, CuGraphBipartiteGraph, CuGraphEdgeSet, CuGraphEdgeMap
    from ..cudf.types import CuDFVector, CuDFNodeSet, CuDFNodeMap

    # TODO only works for unweighted graphs ; make this only work for unweighted graphs
    # @concrete_algorithm("centrality.betweenness")
    # def cugraph_betweenness_centrality(
    #         graph: CuGraph,
    #         nodes: mg.Optional[CuDFNodeSet], # TODO verify this is correct
    #         normalize: bool,
    # ) -> CuDFNodeMap:
    #     node_to_score_df = cugraph.betweenness_centrality(graph.value, k=None if nodes is None else nodes.value.tolist(), normalized=False) # TODO use 'weights' param
    #     node_to_score_df = node_to_score_df.set_index("vertex")
    #     return CuDFNodeMap(node_to_score_df, "betweenness_centrality",)

    @concrete_algorithm("clustering.connected_components")
    def connected_components(graph: CuGraph) -> CuDFNodeMap:
        series = cugraph.weakly_connected_components(graph.value).set_index("vertices")[
            "labels"
        ]
        return CuDFNodeMap(series)

    @concrete_algorithm("clustering.strongly_connected_components")
    def strongly_connected_components(graph: CuGraph) -> CuDFNodeMap:
        series = cugraph.strongly_connected_components(graph.value).set_index(
            "vertices"
        )["labels"]
        return CuDFNodeMap(series)

    @concrete_algorithm("centrality.pagerank")
    def cugraph_pagerank(
        graph: CuGraph, damping: float, maxiter: int, tolerance: float
    ) -> CuDFNodeMap:
        pagerank = cugraph.pagerank(
            graph.value, alpha=damping, max_iter=maxiter, tol=tolerance
        ).set_index("vertex")["pagerank"]
        return CuDFNodeMap(pagerank)

    @concrete_algorithm("traversal.bfs_iter")
    def breadth_first_search(
        graph: CuGraph, source_node: NodeID, depth_limit: int
    ) -> CuDFVector:
        bfs_df = cugraph.bfs(graph.value, source_node)
        bfs_df = bfs_df[bfs_df.predecessor.isin(bfs_df.vertex) | (bfs_df.distance == 0)]
        bfs_ordered_vertices = bfs_df.sort_values("distance").vertex.reset_index(
            drop=True
        )
        return CuDFVector(bfs_ordered_vertices)

    @concrete_algorithm("cluster.triangle_count")
    def cugraph_triangle_count(graph: CuGraph) -> int:
        return cugraph.triangles(graph.value) // 3

    # TODO cupy.ufunc's reduce not supported https://docs.cupy.dev/en/stable/reference/ufunc.html#universal-functions-ufunc
    # @concrete_algorithm("util.graph.aggregate_edges")
    # def cugraph_graph_aggregate_edges(
    #     graph: CuGraph,
    #     func: Callable[[Any, Any], Any],
    #     initial_value: Any,
    #     in_edges: bool,
    #     out_edges: bool,
    # ) -> CuDFNodeMap:
    #     pass

    @concrete_algorithm("util.graph.filter_edges")
    def cugraph_graph_filter_edges(
        graph: CuGraph, func: Callable[[Any], bool]
    ) -> CuGraph:
        new_g = cugraph.DiGraph() if graph.value.is_directed() else cugraph.Graph()
        # TODO do something more optimal when there's an adjlist and no edgelist
        df = graph.value.view_edge_list()
        keep_mask = df["weights"].applymap(func).values
        # TODO reset_index is workaround for https://github.com/rapidsai/cugraph/issues/1080
        keep_df = df.iloc[keep_mask].reset_index(drop=True)
        new_g.from_cudf_edgelist(
            keep_df, source="src", destination="dst", edge_attr="weights"
        )
        # TODO handle node weights
        graph_nodes = cudf.concat([df["src"], df["dst"]]).unique()
        new_g_nodes = cudf.concat([keep_df["src"], keep_df["dst"]]).unique()
        any_nodes_lost = (~graph_nodes.isin(new_g_nodes)).any()
        return CuGraph(new_g, graph_nodes)

    def _assign_uniform_weight(
        input_graph: cugraph.Graph, default_value: Any
    ) -> cugraph.Graph:
        g = cugraph.DiGraph() if input_graph.is_directed() else cugraph.Graph()
        if input_graph.edgelist is not None:
            # TODO avoid copying the 'src' and 'dst' columns twice
            gdf = input_graph.view_edge_list().copy()
            gdf["weight"] = cupy.full(len(gdf), default_value)
            g.from_cudf_edgelist(
                gdf, source="src", destination="dst", edge_attr="weight"
            )
        else:
            offset_col, index_col, _ = g.view_adj_list()
            value_col = cudf.Series(cupy.full(len(offset_col), default_value))
            g.from_cudf_adjlist(offset_col, index_col, value_col)
        return g

    @concrete_algorithm("util.edgemap.from_edgeset")
    def cugraph_edge_map_from_edgeset(
        edgeset: CuGraphEdgeSet, default_value: Any
    ) -> CuGraphEdgeMap:
        g = _assign_uniform_weight(edgeset.value, default_value)
        return CuGraphEdgeMap(g)

    @concrete_algorithm("util.graph.assign_uniform_weight")
    def cugraph_graph_assign_uniform_weight(graph: CuGraph, weight: Any) -> CuGraph:
        new_edges = _assign_uniform_weight(graph.value, weight)
        new_nodes = None if graph.nodes is None else graph.nodes.copy()
        return CuGraph(new_edges, new_nodes, graph.has_node_weights)

    @concrete_algorithm("util.graph.build")
    def cugraph_graph_build(
        edges: mg.Union[CuGraphEdgeSet, CuGraphEdgeMap],
        nodes: mg.Optional[mg.Union[CuDFNodeSet, CuDFNodeMap]],
    ) -> CuGraph:
        return CuGraph(edges.value, nodes.value, isinstance(nodes, CuDFNodeMap))

    @concrete_algorithm("bipartite.graph_projection")
    def graph_projection(bgraph: CuGraphBipartiteGraph, nodes_retained: int) -> CuGraph:
        g = cugraph.DiGraph() if bgraph.value.is_directed() else cugraph.Graph()
        two_hop_neighbors_df = bgraph.value.get_two_hop_neighbors()
        nodes_to_keep = bgraph.value.sets()[nodes_retained]
        keep_mask = two_hop_neighbors_df.first.isin(nodes_to_keep)
        edge_list_df = two_hop_neighbors_df[keep_mask]
        g.from_cudf_edgelist(edge_list_df, source="first", destination="second")
        nodes = nodes_to_keep.set_index(nodes_to_keep)
        return CuGraph(g, nodes=nodes)

    @concrete_algorithm("clustering.louvain_community")
    def cugraph_louvain_community(graph: CuGraph) -> Tuple[CuDFNodeMap, float]:
        label_df, modularity_score = cugraph.louvain(graph.value)
        label_series = label_df.set_index("vertex")["partition"]
        orphan_mask: cupy.ndarray = ~graph.nodes.index.isin(label_series.index)
        orphan_nodes = graph.nodes.index[orphan_mask]
        orphan_count = orphan_mask.astype(int).sum().item()
        max_label = label_df.index.max()
        orphan_labels = cupy.arange(max_label + 1, max_label + 1 + orphan_count)
        orphan_series = cudf.Series(orphan_labels, index=orphan_nodes)
        label_series = cudf.concat([label_series, orphan_series])
        # TODO more orphan should worsen the modularity score but do not in this implementation
        return (
            CuDFNodeMap(label_series),
            modularity_score,
        )
