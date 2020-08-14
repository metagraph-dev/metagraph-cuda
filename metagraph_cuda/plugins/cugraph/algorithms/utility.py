from metagraph import concrete_algorithm
from ..registry import has_cudf, has_cugraph


if has_cudf:
    import cudf
    import cupy
    from ..types import CuDFNodeSet

    # TODO cudf and cupy currently don't support sampling
    # @concrete_algorithm("util.nodeset.choose_random")
    # def cudf_nodeset_choose_random(x: CuDFNodeSet, k: int) -> CuDFNodeSet:
    #     pass

    @concrete_algorithm("util.nodeset.from_vector")
    def cudf_nodeset_from_vector(x: CuDFVector) -> CuDFNodeSet:
        return CuDFNodeSet(x.value.copy())

    @concrete_algorithm("util.nodemap.sort")
    def cudf_nodemap_sort(
        x: CuDFNodeMap, ascending: bool, limit: mg.Optional[int]
    ) -> CuDFVector:
        data = x.value.sort_values(x.value_label, ascending=ascending)[
            x.value_label
        ].reset_index(drop=True)
        if limit is not None:
            data = data.iloc[0 : k + 1]
        return CuDFVector(data)

    @concrete_algorithm("util.nodemap.select")
    def cudf_nodemap_select(x: CuDFNodeMap, nodes: CuDFNodeSet) -> CuDFNodeMap:
        data = x.value.iloc[nodes.value].copy()
        return CuDFNodeMap(data, x.value_label)

    @concrete_algorithm("util.nodemap.filter")
    def cudf_nodemap_filter(x: CuDFNodeMap, func: Callable[[Any], bool]) -> CuDFNodeSet:
        keep_mask = x.value[x.value_label].applymap(func).values
        indices = cupy.flatnonzero(keep_mask)
        nodes = cudf.Series(indices)
        return CuDFNodeSet(nodes)

    @concrete_algorithm("util.nodemap.apply")
    def cudf_nodemap_apply(x: CuDFNodeMap, func: Callable[[Any], Any]) -> CuDFNodeMap:
        new_column_name = "applied"
        data = x.value[x.value_label].applymap(func).to_frame(new_column_name)
        return CuDFNodeMap(data, new_column_name)

    # TODO cupy.ufunc's reduce not supported https://docs.cupy.dev/en/stable/reference/ufunc.html#universal-functions-ufunc
    # @concrete_algorithm("util.nodemap.reduce")
    # def cudf_nodemap_reduce(x: CuDFNodeMap, func: Callable[[Any, Any], Any]) -> Any:
    #     pass

    @concrete_algorithm("util.edge_map.from_edgeset")
    def cudf_edge_map_from_edgeset(
        edgeset: CuDFEdgeSet, default_value: Any,
    ) -> CuDFEdgeMap:
        df = edgeset.value.copy()
        df["weight"] = cupy.full(len(df), default_value)
        return CuDFEdgeMap(
            df, edgeset.src_label, edgeset.dst_label, "weight", edgeset.is_directed
        )


if has_cugraph:
    import cudf
    import cugraph
    import cupy
    from ..types import CuDFNodeSet, CuGraph

    @concrete_algorithm("util.edge_map.from_edgeset")
    def cugraph_edge_map_from_edgeset(
        edgeset: CuGraphEdgeSet, default_value: Any,
    ) -> CuGraphEdgeMap:
        g = cugraph.DiGraph() if edgeset.value.is_directed() else cugraph.Graph()
        if edgeset.value.edgelist is not None:
            # TODO avoid copying the 'src' and 'dst' columns twice
            gdf = edgeset.value.view_edge_list().copy()
            gdf["weight"] = cupy.full(len(df), default_value)
            g.from_cudf_edgelist(
                gdf, source="src", destination="dst", edge_attr="weight"
            )
        else:
            offset_col, index_col, _ = g.view_adj_list()
            value_col = cudf.Series(cupy.full(len(offset_col), default_value))
            g.from_cudf_adjlist(offset_col, index_col, value_col)
        return CuGraphEdgeMap(g)

    # TODO cupy.ufunc's reduce not supported https://docs.cupy.dev/en/stable/reference/ufunc.html#universal-functions-ufunc
    # @concrete_algorithm("util.graph.aggregate_edges")
    # def cugraph_graph_aggregate_edges(
    #     graph: CuGraph(edge_type="map"),
    #     func: Callable[[Any, Any], Any],
    #     initial_value: Any,
    #     in_edges: bool,
    #     out_edges: bool,
    # ) -> CuDFNodeMap:
    #     pass

    @concrete_algorithm("util.graph.filter_edges")
    def cugraph_graph_filter_edges(
        graph: CuGraph(edge_type="map"), func: Callable[[Any], bool]
    ) -> CuGraph:
        g = cugraph.DiGraph() if edgeset.value.is_directed() else cugraph.Graph()
        # TODO do something more optimal when there's an adjlist and no edgelist
        df = graph.edges.view_edge_list()
        keep_mask = df["weights"].applymap(func).values
        indices = cupy.flatnonzero(keep_mask)
        keep_df = df.iloc[indices]
        g.from_cudf_edgelist(
            keep_df, source="src", destination="dst", edge_attr="weight"
        )
        nodes = graph.nodes.copy()
        return CuGraph(g, nodes)

    @concrete_algorithm("util.graph.assign_uniform_weight")
    def cugraph_graph_assign_uniform_weight(
        graph: CuGraph, weight: Any
    ) -> CuGraph(edge_type="map"):
        new_edges = cugraph_edge_map_from_edgeset.func(graph.edges, weight)
        new_nodes = graph.nodes.copy()
        return CuGraph(new_edges, new_nodes)

    @concrete_algorithm("util.graph.build")
    def cugraph_graph_build(
        edges: mg.Union[CuGraphEdgeSet, CuGraphEdgeMap],
        nodes: mg.Optional[mg.Union[CuDFNodeSet, CuDFNodeMap]],
    ) -> CuGraph:
        return CuGraph(edges, nodese)
