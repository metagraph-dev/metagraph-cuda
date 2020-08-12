from metagraph import translator, dtypes
from metagraph.plugins import has_pandas, has_networkx, has_scipy


import numpy as np
import cudf
import cugraph
import cupy
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap, NumpyVector
from metagraph.plugins.python.types import PythonNodeSet, PythonNodeMap, dtype_casting
from .types import (
    CuDFVector,
    CuDFNodeSet,
    CuDFNodeMap,
    CuDFEdgeSet,
    CuDFEdgeMap,
    CuGraphEdgeSet,
    CuGraphEdgeMap,
    CuGraph,
)


@translator
def cudf_nodemap_to_nodeset(x: CuDFNodeMap, **props) -> CuDFNodeSet:
    return CuDFNodeSet(x.value.index.to_series())


@translator
def cudf_edgemap_to_edgeset(x: CuDFEdgeMap, **props) -> CuDFEdgeSet:
    return CuDFEdgeSet(
        x.value.copy(), x.src_label, x.dst_label, is_directed=x.is_directed
    )


@translator
def translate_nodes_cudfnodemap2pythonnodemap(x: CuDFNodeMap, **props) -> PythonNodeMap:
    cast = dtype_casting[dtypes.dtypes_simplified[x.value[x.value_label].dtype]]
    data = {
        i.item(): cast(x.value.loc[i.item()].loc[x.value_label])
        for i in x.value.index.values
    }
    return PythonNodeMap(data)


@translator
def translate_nodes_pythonnodemap2cudfnodemap(x: PythonNodeMap, **props) -> CuDFNodeMap:
    keys, values = zip(*x.value.items())
    # TODO consider special casing the situation when all the keys form a compact range
    data = cudf.DataFrame({"key": keys, "value": values}).set_index("key")
    return CuDFNodeMap(data, "value")


@translator
def translate_nodes_cudfnodeset2pythonnodeset(x: CuDFNodeSet, **props) -> PythonNodeSet:
    return PythonNodeSet(set(x.value.index))


@translator
def translate_nodes_pythonnodeset2cudfnodeset(x: PythonNodeSet, **props) -> CuDFNodeSet:
    return CuDFNodeSet(cudf.Series(x.value))


@translator
def translate_nodes_numpyvector2cudfvector(x: NumpyVector, **props) -> CuDFVector:
    if x.mask is not None:
        data = x.value[x.mask]
        series = cudf.Series(data).set_index(np.flatnonzero(x.mask))
    else:
        data = x.value
        series = cudf.Series(data)
    return CuDFVector(series)


@translator
def translate_vector_cudfvector2numpyvector(x: CuDFVector, **props) -> NumpyVector:
    is_dense = CuDFVector.Type.compute_abstract_properties(x, {"is_dense"})["is_dense"]
    if is_dense:
        np_vector = cupy.asnumpy(x.value.sort_index().values)
        mask = None
    else:
        index_as_np_array = x.value.index.to_array()
        np_vector = np.empty(len(x), dtype=x.value.dtype)
        np_vector[index_as_np_array] = cupy.asnumpy(x.value.values)
        mask = np.zeros(len(x), dtype=bool)
        mask[index_as_np_array] = True
    return NumpyVector(np_vector, mask)


@translator
def translate_nodes_numpynodemap2cudfnodemap(x: NumpyNodeMap, **props) -> CuDFNodeMap:
    idx = np.arange(len(x.value))
    vals = x.value[idx]
    data = cudf.DataFrame({"value": vals})
    return CuDFNodeMap(data, "value")


@translator
def translate_nodes_cudfnodemap2numpynodemap(x: CuDFNodeMap, **props) -> NumpyNodeMap:
    if isinstance(x.value.index, cudf.core.index.RangeIndex):
        x_index_min = x.value.index.start
        x_index_max = x.value.index.stop - 1
    else:
        x_index_min = x.value.index.min()
        x_index_max = x.value.index.max()
    x_density = (x_index_max + 1 - x_index_min) / (x_index_max + 1)
    if x_density == 1.0:
        data = np.empty(len(x.value), dtype=x.value[x.value_label].dtype)
        data[cupy.asnumpy(x.value.index.values)] = cupy.asnumpy(
            x.value[x.value_label].values
        )
        mask = None
        node_ids = None
    elif x_density > 0.5:  # TODO consider moving this threshold out to a global
        data = np.empty(x_index_max + 1, dtype=x.value[x.value_label].dtype)
        position_selector = cupy.asnumpy(x.value.index.values)
        data[position_selector] = cupy.asnumpy(x.value[x.value_label].values)
        mask = np.zeros(x_index_max + 1, dtype=bool)
        mask[position_selector] = True
        node_ids = None
    else:
        df_index_sorted = (
            x.value.sort_index()
        )  # O(n log n), but n is small since not dense
        data = cupy.asnumpy(df_index_sorted[x.value_label].values)
        node_ids = dict(map(reversed, enumerate(df_index_sorted.index)))
        mask = None
    return NumpyNodeMap(data, mask=mask, node_ids=node_ids)


@translator
def translate_nodes_numpynodeset2cudfnodeset(x: NumpyNodeSet, **props) -> CuDFNodeSet:
    data = cudf.Series(x.nodes())
    return CuDFNodeSet(data)


@translator
def translate_nodes_cudfnodeset2numpynodeset(x: CuDFNodeSet, **props) -> NumpyNodeSet:
    if isinstance(x.value.index, cudf.core.index.RangeIndex):
        x_index_min = x.value.index.start
        x_index_max = x.value.index.stop - 1
    else:
        x_index_min = x.value.index.min()
        x_index_max = x.value.index.max()
    x_density = (x_index_max + 1 - x_index_min) / (x_index_max + 1)
    node_positions = cupy.asnumpy(x.value.index.values)
    if x_density > 0.5:  # TODO consider moving this threshold out to a global
        mask = np.zeros(x_index_max + 1, dtype=bool)
        mask[node_positions] = True
        node_ids = None
    else:
        node_ids = node_positions
        mask = None
    return NumpyNodeSet(node_ids=node_ids, mask=mask)


@translator
def translate_edgeset_cudfedgeset2cugraphedgeset(
    x: CuDFEdgeSet, **props
) -> CuGraphEdgeSet:
    cugraph_graph = cugraph.DiGraph() if x.is_directed else cugraph.Graph()
    cugraph_graph.from_cudf_edgelist(x.value, x.src_label, x.dst_label)
    return CuGraphEdgeSet(cugraph_graph)


@translator
def translate_edgemap_cudfedgemap2cugraphedgemap(
    x: CuDFEdgeMap, **props
) -> CuGraphEdgeMap:
    cugraph_graph = cugraph.DiGraph() if x.is_directed else cugraph.Graph()
    cugraph_graph.from_cudf_edgelist(
        x.value, x.src_label, x.dst_label, edge_attr=x.weight_label
    )
    return CuGraphEdgeMap(cugraph_graph)


@translator
def translate_edgeset_cugraphedgeset2cudfedgeset(
    x: CuGraphEdgeSet, **props
) -> CuDFEdgeSet:
    return CuDFEdgeSet(
        x.value.view_edge_list(),
        src_label="src",
        dst_label="dst",
        is_directed=isinstance(x.value, cugraph.DiGraph),
    )


@translator
def translate_edgemap_cugraphedgemap2cudfedgemap(
    x: CuGraphEdgeMap, **props
) -> CuDFEdgeMap:
    return CuDFEdgeMap(
        x.value.view_edge_list(),
        src_label="src",
        dst_label="dst",
        weight_label="weights",
        is_directed=x.value.is_directed(),
    )


if has_networkx:
    import networkx as nx
    from metagraph.plugins.networkx.types import NetworkXGraph

    @translator
    def translate_graph_cugraphgraph2networkxgraph(
        x: CuGraph, **props
    ) -> NetworkXGraph:
        aprops = CuGraph.Type.compute_abstract_properties(
            x, {"is_directed", "edge_type"}
        )
        is_directed = aprops["is_directed"]
        out = nx.DiGraph() if is_directed else nx.Graph()
        column_name_to_series = {
            column_name: series
            for column_name, series in x.edges.value.view_edge_list().iteritems()
        }
        if aprops["edge_type"] == "set":
            ebunch = zip(
                column_name_to_series["src"].tolist(),
                column_name_to_series["dst"].tolist(),
            )
            out.add_edges_from(ebunch)
        else:
            ebunch = zip(
                column_name_to_series["src"].tolist(),
                column_name_to_series["dst"].tolist(),
                column_name_to_series["weights"].tolist(),
            )
            out.add_weighted_edges_from(ebunch)
        # TODO take care of node weights
        return NetworkXGraph(out)

    @translator
    def translate_graph_networkx2cugraph(x: NetworkXGraph, **props) -> CuGraph:
        aprops = NetworkXGraph.Type.compute_abstract_properties(
            x, {"edge_type", "node_type", "is_directed"}
        )
        is_weighted = aprops["edge_type"] == "map"
        has_node_weights = aprops["node_type"] == "map"
        is_directed = aprops["is_directed"]

        g = cugraph.DiGraph() if is_directed else cugraph.Graph()
        if is_weighted:
            edgelist = x.value.edges(data=True)
            source_nodes, target_nodes, node_data_dicts = zip(*edgelist)
            cdf_data = {"source": source_nodes, "destination": target_nodes}
            cdf_data[x.edge_weight_label] = [
                data_dict[x.edge_weight_label] for data_dict in node_data_dicts
            ]
            cdf = cudf.DataFrame(cdf_data)
            g.from_cudf_edgelist(
                cdf,
                source="source",
                destination="destination",
                edge_attr=x.edge_weight_label,
            )
            edges = CuGraphEdgeMap(g)
        else:
            edgelist = x.value.edges()
            source_nodes, target_nodes = zip(*edgelist)
            cdf_data = {"source": source_nodes, "destination": target_nodes}
            cdf = cudf.DataFrame(cdf_data)
            g.from_cudf_edgelist(cdf, source="source", destination="destination")
            edges = CuGraphEdgeSet(g)
        if has_node_weights:
            nodes = None
            # TODO implement this via a CuDFNodeMap
        else:
            # TODO check for orphan nodes
            nodes = None
            # TODO implement this via a CuDFNodeMap
        return CuGraph(edges, nodes)


if has_pandas:
    from metagraph.plugins.pandas.types import PandasEdgeSet, PandasEdgeMap

    @translator
    def translate_edgeset_pdedgeset2cudfedgeset(
        x: PandasEdgeSet, **props
    ) -> CuDFEdgeSet:
        df = cudf.from_pandas(x.value)
        return CuDFEdgeSet(
            df, src_label=x.src_label, dst_label=x.dst_label, is_directed=x.is_directed
        )

    @translator
    def translate_edgemap_pdedgemap2cudfedgemap(
        x: PandasEdgeMap, **props
    ) -> CuDFEdgeMap:
        df = cudf.from_pandas(x.value)
        return CuDFEdgeMap(
            df,
            src_label=x.src_label,
            dst_label=x.dst_label,
            weight_label=x.weight_label,
            is_directed=x.is_directed,
        )

    @translator
    def translate_edgeset_cudfedgeset2pdedgeset(
        x: CuDFEdgeSet, **props
    ) -> PandasEdgeSet:
        pdf = x.value.to_pandas()
        return PandasEdgeSet(
            pdf,
            src_label=x.src_label,
            dst_label=x.dst_label,
            is_directed=x.is_directed,
        )

    @translator
    def translate_edgemap_cudfedgemap2pdedgemap(
        x: CuDFEdgeMap, **props
    ) -> PandasEdgeMap:
        pdf = x.value.to_pandas()
        return PandasEdgeMap(
            pdf,
            src_label=x.src_label,
            dst_label=x.dst_label,
            weight_label=x.weight_label,
            is_directed=x.is_directed,
        )

    @translator
    def translate_edgeset_pdedgeset2cugraphedgeset(
        x: PandasEdgeSet, **props
    ) -> CuGraphEdgeSet:
        df = cudf.from_pandas(x.value)
        g = cugraph.DiGraph() if x.is_directed else cugraph.Graph()
        g.from_cudf_edgelist(
            df, source=x.src_label, destination=x.dst_label, renumber=False
        )
        return CuGraphEdgeSet(g)

    @translator
    def translate_edgemap_pdedgemap2cugraphedgemap(
        x: PandasEdgeMap, **props
    ) -> CuGraphEdgeMap:
        df = cudf.from_pandas(x.value)
        g = cugraph.DiGraph() if x.is_directed else cugraph.Graph()
        g.from_cudf_edgelist(
            df,
            source=x.src_label,
            destination=x.dst_label,
            edge_attr=x.weight_label,
            renumber=False,
        )
        return CuGraphEdgeMap(g)

    @translator
    def translate_edgeset_cugraphedgeset2pdedgeset(
        x: CuGraphEdgeSet, **props
    ) -> PandasEdgeSet:
        is_directed = CuGraphEdgeSet.Type.compute_abstract_properties(
            x, {"is_directed"}
        )["is_directed"]
        pdf = x.value.view_edge_list().to_pandas()
        return PandasEdgeSet(
            pdf, src_label="src", dst_label="dst", is_directed=is_directed,
        )

    @translator
    def translate_edgemap_cugraphedgemap2pdedgemap(
        x: CuGraphEdgeMap, **props
    ) -> PandasEdgeMap:
        is_directed = CuGraphEdgeSet.Type.compute_abstract_properties(
            x, {"is_directed"}
        )["is_directed"]
        pdf = x.value.view_edge_list().to_pandas()
        return PandasEdgeMap(
            pdf,
            src_label="src",
            dst_label="dst",
            weight_label="weights",
            is_directed=is_directed,
        )


if has_scipy:
    import cudf
    import scipy.sparse as ss
    from metagraph.plugins.scipy.types import ScipyEdgeSet, ScipyEdgeMap

    @translator
    def translate_edgeset_scipyedgeset2cugraphedgeset(
        x: ScipyEdgeSet, **props
    ) -> CuGraphEdgeSet:
        is_directed = ScipyEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rc_pairs = zip(row_ids, column_ids)
        if not is_directed:
            rc_pairs = filter(lambda pair: pair[0] < pair[1], rc_pairs)
        rc_pairs = list(rc_pairs)
        cdf = cudf.DataFrame(rc_pairs, columns=["source", "target"])
        graph = cugraph.DiGraph() if is_directed else cugraph.Graph()
        graph.from_cudf_edgelist(
            cdf, source="source", destination="target",
        )
        return CuGraphEdgeSet(graph)

    @translator
    def translate_edgemap_scipyedgemap2cugraphedgemap(
        x: ScipyEdgeMap, **props
    ) -> CuGraphEdgeMap:
        is_directed = ScipyEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rcw_triples = zip(row_ids, column_ids, coo_matrix.data)
        if not is_directed:
            rcw_triples = filter(lambda triple: triple[0] < triple[1], rcw_triples)
        rcw_triples = list(rcw_triples)
        cdf = cudf.DataFrame(rcw_triples, columns=["source", "target", "weight"])
        graph = cugraph.DiGraph() if is_directed else cugraph.Graph()
        graph.from_cudf_edgelist(
            cdf, source="source", destination="target", edge_attr="weight",
        )
        return CuGraphEdgeMap(graph)

    @translator
    def translate_edgeset_cugraphedgeset2scipyedgeset(
        x: CuGraphEdgeSet, **props
    ) -> ScipyEdgeSet:
        is_directed = x.value.is_directed()
        node_list = x.value.nodes().copy().sort_values().tolist()
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        gdf = x.value.view_edge_list()
        source_positions = list(map(get_id_pos, gdf["src"].values.tolist()))
        target_positions = list(map(get_id_pos, gdf["dst"].values.tolist()))
        if not is_directed:
            source_positions, target_positions = (
                source_positions + target_positions,
                target_positions + source_positions,
            )
        source_positions = np.array(source_positions)
        target_positions = np.array(target_positions)
        matrix = ss.coo_matrix(
            (np.ones(len(source_positions)), (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeSet(matrix, node_list)

    @translator
    def translate_edgemap_cugraphedgemap2scipyedgemap(
        x: CuGraphEdgeMap, **props
    ) -> ScipyEdgeMap:
        is_directed = x.value.is_directed()
        node_list = x.value.nodes().copy().sort_values().tolist()
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        gdf = x.value.view_edge_list()
        source_positions = list(map(get_id_pos, gdf["src"].values.tolist()))
        target_positions = list(map(get_id_pos, gdf["dst"].values.tolist()))
        weights = cupy.asnumpy(gdf["weights"].values)
        if not is_directed:
            source_positions, target_positions = (
                source_positions + target_positions,
                target_positions + source_positions,
            )
            weights = np.concatenate([weights, weights])
        source_positions = np.array(source_positions)
        target_positions = np.array(target_positions)
        matrix = ss.coo_matrix(
            (weights, (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeMap(matrix, node_list)

    @translator
    def translate_edgeset_scipyedgeset2cudfedgeset(
        x: ScipyEdgeSet, **props
    ) -> CuDFEdgeSet:
        is_directed = ScipyEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rc_pairs = zip(row_ids, column_ids)
        if not is_directed:
            rc_pairs = filter(lambda pair: pair[0] < pair[1], rc_pairs)
        rc_pairs = list(rc_pairs)
        df = cudf.DataFrame(rc_pairs, columns=["source", "target"])
        return CuDFEdgeSet(df, is_directed=is_directed)

    @translator
    def translate_edgemap_scipyedgemap2cudfedgemap(
        x: ScipyEdgeMap, **props
    ) -> CuDFEdgeMap:
        is_directed = ScipyEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rcw_triples = zip(row_ids, column_ids, coo_matrix.data)
        if not is_directed:
            rcw_triples = filter(lambda triple: triple[0] < triple[1], rcw_triples)
        rcw_triples = list(rcw_triples)
        df = cudf.DataFrame(rcw_triples, columns=["source", "target", "weight"])
        return CuDFEdgeMap(df, is_directed=is_directed)

    @translator
    def translate_edgeset_cudfedgeset2scipyedgeset(
        x: CuDFEdgeSet, **props
    ) -> ScipyEdgeSet:
        is_directed = x.is_directed
        node_list = np.unique(
            cupy.asnumpy(x.value[[x.src_label, x.dst_label]].values).ravel("K")
        )
        node_list.sort()
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        source_positions = list(map(get_id_pos, x.value[x.src_label].values.tolist()))
        target_positions = list(map(get_id_pos, x.value[x.dst_label].values.tolist()))
        if not is_directed:
            source_positions, target_positions = (
                source_positions + target_positions,
                target_positions + source_positions,
            )
        source_positions = np.array(source_positions)
        target_positions = np.array(target_positions)
        matrix = ss.coo_matrix(
            (np.ones(len(source_positions)), (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeSet(matrix, node_list)

    @translator
    def translate_edgemap_cudfedgemap2scipyedgemap(
        x: CuDFEdgeMap, **props
    ) -> ScipyEdgeMap:
        is_directed = x.is_directed
        node_list = np.unique(
            cupy.asnumpy(x.value[[x.src_label, x.dst_label]].values).ravel("K")
        )
        node_list.sort()
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        source_positions = list(map(get_id_pos, x.value[x.src_label].tolist()))
        target_positions = list(map(get_id_pos, x.value[x.dst_label].tolist()))
        weights = cupy.asnumpy(x.value[x.weight_label].values)
        if not is_directed:
            source_positions, target_positions = (
                source_positions + target_positions,
                target_positions + source_positions,
            )
            weights = np.concatenate([weights, weights])
        matrix = ss.coo_matrix(
            (weights, (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeMap(matrix, node_list)

    @translator
    def translate_graph_cugraph2scipygraph(x: CuGraph, **props) -> ScipyGraph:
        # TODO create composite translators since this uses a lot of boilerplate
        if isinstance(x.edges, CuGraphEdgeSet):
            new_edges = translate_edgeset_cugraphedgeset2scipyedgeset.func(x.edges)
        else:
            new_edges = translate_edgemap_cugraphedgemap2scipyedgemap.func(x.edges)
        if isinstance(x.nodes, CuDFNodeSet):
            new_nodes = translate_nodes_cudfnodeset2numpynodeset.func(x.nodes)
        else:
            new_nodes = translate_nodes_cudfnodemap2numpynodemap.func(x.nodes)
        return ScipyGraph(new_edges, new_nodes)

    @translator
    def translate_graph_scipygraph2cugraph(x: ScipyGraph, **props) -> CuGraph:
        # TODO create composite translators since this uses a lot of boilerplate
        if isinstance(x.edges, ScipyEdgeSet):
            new_edges = translate_edgeset_scipyedgeset2cugraphedgeset.func(x.edges)
        else:
            new_edges = translate_edgemap_scipyedgemap2cugraphedgemap.func(x.edges)
        if isinstance(x.nodes, NumpyNodeSet):
            new_nodes = translate_nodes_numpynodeset2cudfnodeset.func(x.nodes)
        else:
            new_nodes = translate_nodes_numpynodemap2cudfnodemap.func(x.nodes)
        return CuGraph(new_edges, new_nodes)
