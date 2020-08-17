from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx, has_scipy
import numpy as np
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap, NumpyVector
from metagraph.plugins.python.types import PythonNodeSet, PythonNodeMap, dtype_casting
from .. import has_cugraph

if has_cugraph:
    import cudf
    import cugraph
    import cupy

    from .types import (
        CuGraphEdgeSet,
        CuGraphEdgeMap,
        CuGraph,
    )
    from ..cudf.types import (
        CuDFVector,
        CuDFNodeSet,
        CuDFNodeMap,
        CuDFEdgeSet,
        CuDFEdgeMap,
    )
    from ..cudf.translators import (
        translate_nodes_cudfnodeset2numpynodeset,
        translate_nodes_cudfnodemap2numpynodemap,
        translate_nodes_numpynodeset2cudfnodeset,
        translate_nodes_numpynodemap2cudfnodemap,
    )

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


if has_cugraph and has_pandas:
    from metagraph.plugins.pandas.types import PandasEdgeSet, PandasEdgeMap

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


if has_cugraph and has_scipy:
    import scipy.sparse as ss
    from metagraph.plugins.scipy.types import ScipyEdgeSet, ScipyEdgeMap, ScipyGraph

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
    def translate_graph_cugraph2scipygraph(x: CuGraph, **props) -> ScipyGraph:
        # TODO create composite translators since this uses a lot of boilerplate
        if isinstance(x.edges, CuGraphEdgeSet):
            new_edges = translate_edgeset_cugraphedgeset2scipyedgeset.func(x.edges)
        else:
            new_edges = translate_edgemap_cugraphedgemap2scipyedgemap.func(x.edges)
        if x.nodes is None:
            new_nodes = None
        elif isinstance(x.nodes, CuDFNodeSet):
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
        if x.nodes is None:
            new_nodes = None
        if isinstance(x.nodes, NumpyNodeSet):
            new_nodes = translate_nodes_numpynodeset2cudfnodeset.func(x.nodes)
        else:
            new_nodes = translate_nodes_numpynodemap2cudfnodemap.func(x.nodes)
        return CuGraph(new_edges, new_nodes)


if has_cugraph and has_networkx:
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
        if isinstance(x.nodes, CuDFNodeSet):
            out.add_nodes_from(x.nodes)
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
