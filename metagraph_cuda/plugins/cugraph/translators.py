from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx, has_scipy
import numpy as np
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap
from metagraph.plugins.python.types import dtype_casting
from metagraph.core.dtypes import dtypes_simplified
from .. import has_cugraph

if has_cugraph:
    import cudf
    import cugraph
    import cupy

    from .types import (
        CuGraphEdgeSet,
        CuGraphEdgeMap,
        CuGraph,
        CuGraphBipartiteGraph,
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
    def cugraph_edgemap_to_edgeset(x: CuGraphEdgeMap, **props) -> CuGraphEdgeSet:
        is_directed = CuGraphEdgeMap.Type.compute_abstract_properties(
            x, {"is_directed"}
        )["is_directed"]
        g = cugraph.DiGraph() if is_directed else cugraph.Graph()
        if x.value.edgelist is not None:
            gdf = x.value.view_edge_list()
            g.from_cudf_edgelist(gdf, source="src", destination="dst")
        else:
            offset_col, index_col, _ = x.value.view_adj_list()
            g.from_cudf_adjlist(offset_col, index_col, value_col)
        return CuGraphEdgeSet(g)

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
        is_directed = CuGraphEdgeMap.Type.compute_abstract_properties(
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
        # TODO can we special case CSR? example at https://docs.rapids.ai/api/cugraph/stable/api.html#cugraph.structure.graph.Graph.from_cudf_adjlist
        coo_matrix = x.value.tocoo()
        row_ids = x.node_list[coo_matrix.row]
        column_ids = x.node_list[coo_matrix.col]
        if not is_directed:
            mask = row_ids <= column_ids
            row_ids = row_ids[mask]
            column_ids = column_ids[mask]
        cdf = cudf.DataFrame({"source": row_ids, "target": column_ids})
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
        # TODO can we special case CSR? example at https://docs.rapids.ai/api/cugraph/stable/api.html#cugraph.structure.graph.Graph.from_cudf_adjlist
        coo_matrix = x.value.tocoo()
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rcw_triples = zip(row_ids, column_ids, coo_matrix.data)
        if not is_directed:
            rcw_triples = filter(lambda triple: triple[0] <= triple[1], rcw_triples)
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
        node_list = x.value.nodes().values_host
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        gdf = x.value.view_edge_list()
        source_positions = list(map(get_id_pos, gdf["src"].values_host))
        target_positions = list(map(get_id_pos, gdf["dst"].values_host))
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
        return ScipyEdgeSet(matrix, node_list, aprops={"is_directed": is_directed})

    @translator
    def translate_edgemap_cugraphedgemap2scipyedgemap(
        x: CuGraphEdgeMap, **props
    ) -> ScipyEdgeMap:
        is_directed = x.value.is_directed()
        node_list = x.value.nodes().values_host
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        gdf = x.value.view_edge_list()
        if not is_directed:
            self_loop_mask = gdf.src == gdf.dst
            self_loop_df = gdf[self_loop_mask]
            no_self_loop_df = gdf[~self_loop_mask]
            repeat_df = no_self_loop_df.rename(columns={"src": "dst", "dst": "src"})
            gdf = cudf.concat([no_self_loop_df, repeat_df, self_loop_df,])
        source_positions = list(map(get_id_pos, gdf["src"].values_host))
        target_positions = list(map(get_id_pos, gdf["dst"].values_host))
        weights = cupy.asnumpy(gdf["weights"].values)
        source_positions = np.array(source_positions)
        target_positions = np.array(target_positions)
        matrix = ss.coo_matrix(
            (weights, (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeMap(matrix, node_list, aprops={"is_directed": is_directed})

    @translator
    def translate_graph_cugraph2scipygraph(x: CuGraph, **props) -> ScipyGraph:
        aprops = props.copy()
        aprops.update(
            CuGraph.Type.compute_abstract_properties(
                x, {"is_directed", "edge_type"} - props.keys()
            )
        )
        is_weighted = "map" == aprops["edge_type"]
        is_directed = aprops["is_directed"]
        node_list = cupy.asnumpy(x.nodes.index.to_array())
        node_values = cupy.asnumpy(x.nodes.values) if x.has_node_weights else None
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        gdf = x.value.view_edge_list()
        if not is_directed:
            self_loop_mask = gdf.src == gdf.dst
            self_loop_df = gdf[self_loop_mask]
            no_self_loop_df = gdf[~self_loop_mask]
            repeat_df = no_self_loop_df.rename(columns={"src": "dst", "dst": "src"})
            gdf = cudf.concat([no_self_loop_df, repeat_df, self_loop_df,])
        source_positions = list(map(get_id_pos, gdf["src"].values_host))
        target_positions = list(map(get_id_pos, gdf["dst"].values_host))
        weights = (
            cupy.asnumpy(gdf["weights"].values)
            if is_weighted
            else np.ones(len(source_positions), dtype=bool)
        )
        source_positions = np.array(source_positions)
        target_positions = np.array(target_positions)
        matrix = ss.coo_matrix(
            (weights, (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyGraph(matrix, node_list, node_values, aprops=aprops)

    @translator
    def translate_graph_scipygraph2cugraph(x: ScipyGraph, **props) -> CuGraph:
        aprops = props.copy()
        aprops.update(
            ScipyGraph.Type.compute_abstract_properties(
                x,
                {"is_directed", "edge_type", "edge_dtype", "node_type"} - props.keys(),
            )
        )
        is_directed = aprops["is_directed"]
        is_weighted = aprops["edge_type"] == "map"
        has_node_weights = aprops["node_type"] == "map"
        expected_edge_dtype = dtype_casting.get(aprops["edge_dtype"])

        coo_matrix = x.value.tocoo()
        if (
            expected_edge_dtype is not None
            and expected_edge_dtype != dtypes_simplified[coo_matrix.dtype]
        ):
            coo_matrix = coo_matrix.astype(expected_edge_dtype)
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        ebunch = (
            zip(row_ids, column_ids, coo_matrix.data)
            if is_weighted
            else zip(row_ids, column_ids)
        )
        if not is_directed:
            ebunch = filter(lambda pair: pair[0] <= pair[1], ebunch)
        ebunch = list(ebunch)
        columns = (
            ["source", "target", "weight"] if is_weighted else ["source", "target"]
        )
        cdf = cudf.DataFrame(ebunch, columns=columns)
        graph = cugraph.DiGraph() if is_directed else cugraph.Graph()
        graph.from_cudf_edgelist(
            cdf,
            source="source",
            destination="target",
            edge_attr="weight" if is_weighted else None,
        )
        if has_node_weights:
            nodes = cudf.Series(x.node_vals).set_index(x.node_list)
        else:
            nodes = cudf.Series(x.node_list).set_index(x.node_list)
        return CuGraph(graph, nodes, has_node_weights)


if has_cugraph and has_networkx:
    import networkx as nx
    from metagraph.plugins.networkx.types import NetworkXGraph, NetworkXBipartiteGraph

    @translator
    def translate_graph_cugraphgraph2networkxgraph(
        x: CuGraph, **props
    ) -> NetworkXGraph:
        aprops = props.copy()
        aprops.update(
            CuGraph.Type.compute_abstract_properties(
                x,
                {"is_directed", "edge_type", "node_type", "node_dtype"} - props.keys(),
            )
        )
        is_directed = aprops["is_directed"]
        is_weighted = aprops["edge_type"] == "map"
        has_node_weights = aprops["node_type"] == "map"
        out = nx.DiGraph() if is_directed else nx.Graph()
        column_name_to_edge_list_values = {
            column_name: series.values_host.tolist()
            for column_name, series in x.value.view_edge_list().iteritems()
        }
        node_weight_label = "weight"
        edge_weight_label = "weight"
        if is_weighted:
            ebunch = zip(
                column_name_to_edge_list_values["src"],
                column_name_to_edge_list_values["dst"],
                column_name_to_edge_list_values["weights"],
            )
            out.add_weighted_edges_from(ebunch)
        else:
            ebunch = zip(
                column_name_to_edge_list_values["src"],
                column_name_to_edge_list_values["dst"],
            )
            out.add_edges_from(ebunch)
        nodes = x.nodes.index.values_host
        if has_node_weights:
            caster = dtype_casting[aprops["node_dtype"]]
            for node, weight in zip(nodes, x.nodes.values_host):
                out.add_node(node, weight=caster(weight))
        else:
            out.add_nodes_from(nodes)
        return NetworkXGraph(
            out,
            node_weight_label=node_weight_label,
            edge_weight_label=edge_weight_label,
        )

    @translator
    def translate_graph_networkx2cugraph(x: NetworkXGraph, **props) -> CuGraph:
        aprops = props.copy()
        aprops.update(
            NetworkXGraph.Type.compute_abstract_properties(
                x, {"edge_type", "node_type", "is_directed"} - props.keys()
            )
        )
        is_weighted = aprops["edge_type"] == "map"
        has_node_weights = aprops["node_type"] == "map"
        is_directed = aprops["is_directed"]

        g = cugraph.DiGraph() if is_directed else cugraph.Graph()
        if is_weighted:
            edgelist = x.value.edges(data=x.edge_weight_label)
            source_nodes, target_nodes, weights = zip(*edgelist)
            cdf_data = {
                "source": source_nodes,
                "destination": target_nodes,
                "weight": weights,
            }
            cdf = cudf.DataFrame(cdf_data)
            g.from_cudf_edgelist(
                cdf, source="source", destination="destination", edge_attr="weight",
            )
        else:
            edgelist = x.value.edges()
            source_nodes, target_nodes = zip(*edgelist)
            cdf_data = {"source": source_nodes, "destination": target_nodes}
            cdf = cudf.DataFrame(cdf_data)
            g.from_cudf_edgelist(cdf, source="source", destination="destination")
        if has_node_weights:
            node_ids, node_weights = list(zip(*x.value.nodes(data=x.node_weight_label)))
            nodes = cudf.Series(node_weights).set_index(node_ids)
        else:
            node_list = list(x.value.nodes())
            nodes = cudf.Series(node_list).set_index(node_list)
        return CuGraph(g, nodes, has_node_weights)

    @translator
    def translate_bipartitegraph_cugraphgraph2networkxgraph(
        x: CuGraphBipartiteGraph, **props
    ) -> NetworkXBipartiteGraph:
        aprops = props.copy()
        aprops.update(
            CuGraphBipartiteGraph.Type.compute_abstract_properties(
                x, {"is_directed", "edge_dtype", "edge_type"} - props.keys()
            )
        )
        is_weighted = aprops["edge_type"] == "map"
        is_directed = aprops["is_directed"]
        nx_graph = nx.DiGraph() if is_directed else nx.Graph()
        edge_list_df = x.value.view_edge_list()
        columns = edge_list_df.columns.values
        src_index = np.where(columns == "src")[0].item()
        dst_index = np.where(columns == "dst")[0].item()
        if is_weighted:
            weight_index = np.where(columns == "weights")[0].item()
            ebunch = cupy.asnumpy(edge_list_df.values)
            ebunch = ebunch[:, [src_index, dst_index, weight_index]]
            ebunch = ebunch.tolist()
            caster = dtype_casting[aprops["edge_dtype"]]
            ebunch = [(src, dst, caster(weight)) for src, dst, weight in ebunch]
        else:
            ebunch = cupy.asnumpy(edge_list_df.values)
            ebunch = ebunch[:, [src_index, dst_index]]
            ebunch = ebunch.tolist()
        if "weights" in edge_list_df.columns:
            nx_graph.add_weighted_edges_from(ebunch, weight="weight")
        else:
            nx_graph.add_edges_from(ebunch)
        nodes0 = x.nodes0.index.to_arrow().to_pylist()
        nodes1 = x.nodes1.index.to_arrow().to_pylist()
        if x.nodes0_have_weights:
            nodes0_weights = cupy.asnumpy(x.nodes0)
            for node, node_weight in zip(nodes0, nodes0_weights):
                nx_graph.add_node(node, weight=nodes0_weight)
        if x.nodes1_have_weights:
            nodes1_weights = cupy.asnumpy(x.nodes1)
            for node, node_weight in zip(nodes1, nodes1_weights):
                nx_graph.add_node(node, weight=nodes1_weight)
        nodes = (nodes0, nodes1)
        return NetworkXBipartiteGraph(
            nx_graph, nodes, node_weight_label="weight", edge_weight_label="weight"
        )

    @translator
    def translate_bipartitegraph_networkx2cugraph(
        x: NetworkXBipartiteGraph, **props
    ) -> CuGraphBipartiteGraph:
        # TODO abstract out common functionality among this and the non-bipartite graph translators
        aprops = NetworkXBipartiteGraph.Type.compute_abstract_properties(
            x, {"edge_type", "is_directed"}
        )
        is_weighted = aprops["edge_type"] == "map"
        is_directed = aprops["is_directed"]
        g = cugraph.DiGraph() if is_directed else cugraph.Graph()
        top_nodes = list(x.nodes[0])
        bottom_nodes = list(x.nodes[1])
        g.add_nodes_from(top_nodes, bipartite="top")
        g.add_nodes_from(bottom_nodes, bipartite="bottom")
        if is_weighted:
            edgelist = x.value.edges(data=x.edge_weight_label)
            source_nodes, target_nodes, weights = zip(*edgelist)
            cdf_data = {
                "source": source_nodes,
                "destination": target_nodes,
                "weight": weights,
            }
            cdf = cudf.DataFrame(cdf_data)
            g.from_cudf_edgelist(
                cdf, source="source", destination="destination", edge_attr="weight",
            )
            edges = CuGraphEdgeMap(g)
        else:
            edgelist = x.value.edges()
            source_nodes, target_nodes = zip(*edgelist)
            cdf_data = {"source": source_nodes, "destination": target_nodes}
            cdf = cudf.DataFrame(cdf_data)
            g.from_cudf_edgelist(cdf, source="source", destination="destination")
            edges = CuGraphEdgeSet(g)
        # TODO handle node weights
        return CuGraphBipartiteGraph(g)
