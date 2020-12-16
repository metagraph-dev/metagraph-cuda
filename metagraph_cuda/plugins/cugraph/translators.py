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
    from cugraph.structure.number_map import NumberMap

    @translator
    def translate_edgeset_scipyedgeset2cugraphedgeset(
        x: ScipyEdgeSet, **props
    ) -> CuGraphEdgeSet:
        is_directed = (
            props["is_directed"]
            if "is_directed" in props
            else ScipyEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})[
                "is_directed"
            ]
        )
        matrix = x.value
        graph = cugraph.DiGraph() if is_directed else cugraph.Graph()
        if isinstance(matrix, ss.csr_matrix):
            offset_col = cudf.Series(matrix.indptr)
            index_col = cudf.Series(matrix.indices)
            value_col = None
            graph.from_cudf_adjlist(offset_col, index_col, value_col)
            if x.node_list[0] != 0 or np.any(np.ediff1d(x.node_list) != 1):
                renumber_df = cudf.DataFrame({"src": x.node_list, "dst": x.node_list})
                _, renumber_map = NumberMap.renumber(renumber_df, "src", "dst")
                # assert np.all(renumber_map.implementation.df["0"].value_host == x.node_list)
                # assert np.all(renumber_map.implementation.df["id"].value_host == np.arange(len(x.node_list)))
                graph.renumbered = True
                graph.renumber_map = renumber_map
        elif isinstance(matrix, ss.coo_matrix):
            row_ids = x.node_list[matrix.row]
            column_ids = x.node_list[matrix.col]
            if not is_directed:
                mask = row_ids <= column_ids
                row_ids = row_ids[mask]
                column_ids = column_ids[mask]
            cdf = cudf.DataFrame({"source": row_ids, "target": column_ids})
            graph.from_cudf_edgelist(
                cdf, source="source", destination="target",
            )
            print(f"2 graph.nodes() {repr(graph.nodes())}")
        else:
            raise TypeError(f"{matrix} must be a in CSR or COO format.")
        return CuGraphEdgeSet(graph)

    @translator
    def translate_edgemap_scipyedgemap2cugraphedgemap(
        x: ScipyEdgeMap, **props
    ) -> CuGraphEdgeMap:
        is_directed = (
            props["is_directed"]
            if "is_directed" in props
            else ScipyEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})[
                "is_directed"
            ]
        )

        matrix = x.value
        graph = cugraph.DiGraph() if is_directed else cugraph.Graph()
        if isinstance(matrix, ss.csr_matrix):
            offset_col = cudf.Series(matrix.indptr)
            index_col = cudf.Series(matrix.indices)
            value_col = cudf.Series(matrix.data)
            graph.from_cudf_adjlist(offset_col, index_col, value_col)
            if x.node_list[0] != 0 or np.any(np.ediff1d(x.node_list) != 1):
                renumber_df = cudf.DataFrame({"src": x.node_list, "dst": x.node_list})
                _, renumber_map = NumberMap.renumber(renumber_df, "src", "dst")
                # assert np.all(renumber_map.implementation.df["0"].value_host == x.node_list)
                # assert np.all(renumber_map.implementation.df["id"].value_host == np.arange(len(x.node_list)))
                graph.renumbered = True
                graph.renumber_map = renumber_map
        elif isinstance(matrix, ss.coo_matrix):
            row_ids = x.node_list[matrix.row]
            column_ids = x.node_list[matrix.col]
            weights = matrix.data
            if not is_directed:
                mask = row_ids <= column_ids
                row_ids = row_ids[mask]
                column_ids = column_ids[mask]
                weights = weights[mask]
            cdf = cudf.DataFrame(
                {"source": row_ids, "target": column_ids, "weight": weights}
            )
            graph.from_cudf_edgelist(
                cdf, source="source", destination="target", edge_attr="weight",
            )
        else:
            raise TypeError(f"{matrix} must be a in CSR or COO format.")
        return CuGraphEdgeMap(graph)

    @translator
    def translate_edgeset_cugraphedgeset2scipyedgeset(
        x: CuGraphEdgeSet, **props
    ) -> ScipyEdgeSet:
        is_directed = x.value.is_directed()

        if x.value.adjlist is not None:
            # TODO test this block
            indptr, indices, _ = x.value.view_adj_list()
            indptr = cupy.asnumpy(indptr.values)
            indices = cupy.asnumpy(indices.values)
            data = np.ones(len(indices), dtype=bool)
            matrix = ss.csr_matrix((data, indices, indptr))
            if x.value.renumber_map is None:
                node_list = None
            else:
                # TODO test this branch
                nrows = matrix.shape[0]
                internal_ids = cudf.Series(cupy.arange(nrows))
                external_id_df = x.value.renumber_map.from_internal_vertex_id(
                    internal_ids
                )
                external_ids = external_id_df["0"]
                node_list = cupy.asnumpy(external_ids.values)
        else:
            node_list = cupy.sort(x.value.nodes().values)
            num_nodes = len(node_list)
            gdf = x.value.view_edge_list()
            source_positions = cupy.searchsorted(node_list, gdf["src"].values)
            target_positions = cupy.searchsorted(node_list, gdf["dst"].values)
            if not is_directed:
                non_self_loop_mask = source_positions != target_positions
                source_positions, target_positions = (
                    cupy.concatenate(
                        [source_positions, target_positions[non_self_loop_mask]]
                    ),
                    cupy.concatenate(
                        [target_positions, source_positions[non_self_loop_mask]]
                    ),
                )

            node_list = cupy.asnumpy(node_list)
            source_positions = cupy.asnumpy(source_positions)
            target_positions = cupy.asnumpy(target_positions)
            matrix = ss.coo_matrix(
                (
                    np.ones(len(source_positions), dtype=bool),
                    (source_positions, target_positions),
                ),
                shape=(num_nodes, num_nodes),
            )
        return ScipyEdgeSet(matrix, node_list, aprops={"is_directed": is_directed})

    @translator
    def translate_edgemap_cugraphedgemap2scipyedgemap(
        x: CuGraphEdgeMap, **props
    ) -> ScipyEdgeMap:
        is_directed = x.value.is_directed()
        if x.value.adjlist is not None:
            # TODO test this block
            indptr, indices, data = x.value.view_adj_list()
            indptr = cupy.asnumpy(indptr.values)
            indices = cupy.asnumpy(indices.values)
            data = cupy.asnumpy(data.values)
            edge_dtype = (
                props["dtype"]
                if "dtype" in props
                else CuGraphEdgeMap.Type.compute_abstract_properties(x, {"dtype"})[
                    "dtype"
                ]
            )
            caster = dtype_casting[edge_dtype]
            data = data.astype(caster)
            matrix = ss.csr_matrix((data, indices, indptr))
            if x.value.renumber_map is None:
                node_list = None
            else:
                nrows = matrix.shape[0]
                internal_ids = cudf.Series(cupy.arange(nrows))
                external_id_df = x.value.renumber_map.from_internal_vertex_id(
                    internal_ids
                )
                external_ids = external_id_df["0"]
                node_list = cupy.asnumpy(external_ids.values)
        else:
            node_list = cupy.sort(x.value.nodes().values)
            num_nodes = len(node_list)
            gdf = x.value.view_edge_list()
            if not is_directed:
                self_loop_mask = gdf.src == gdf.dst
                self_loop_df = gdf[self_loop_mask]
                no_self_loop_df = gdf[~self_loop_mask]
                repeat_df = no_self_loop_df.rename(columns={"src": "dst", "dst": "src"})
                gdf = cudf.concat([no_self_loop_df, repeat_df, self_loop_df,])
            source_positions = cupy.searchsorted(node_list, gdf["src"].values)
            target_positions = cupy.searchsorted(node_list, gdf["dst"].values)

            node_list = cupy.asnumpy(node_list)
            source_positions = cupy.asnumpy(source_positions)
            target_positions = cupy.asnumpy(target_positions)
            weights = cupy.asnumpy(gdf["weights"].values)
            matrix = ss.coo_matrix(
                (weights, (source_positions, target_positions)),
                shape=(num_nodes, num_nodes),
            )
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

        if x.value.adjlist is not None:
            # TODO test this block
            indptr, indices, data = x.value.view_adj_list()

            # Handle indptr
            indptr: cupy.ndarray = indptr.values

            # Calculate intermediate node_list (non-orphan nodes)
            if x.value.renumber_map is not None:
                internal_ids = cudf.Series(cupy.arange(x.value.number_of_vertices()))
                external_id_df = x.value.renumber_map.from_internal_vertex_id(
                    internal_ids
                )
                external_ids = external_id_df["0"]
                node_list: cupy.ndarray = external_ids.values
            else:
                # TODO is x.value.nodes() guaranteed to not contain orphans? Test this.
                node_list: cupy.ndarray = x.value.nodes()

            # Handle orphans
            orphan_node_mask: cupy.ndarray = ~x.nodes.index.isin(node_list)
            orphan_nodes: cupy.ndarray = x.nodes.index[orphan_node_mask].values

            # Calculate final node_list
            node_list: cupy.ndarray = cupy.concatenate([node_list, orphan_nodes])
            node_list: np.ndarray = cupy.asnumpy(node_list)

            indptr: cupy.ndarray = cupy.concatenate(
                [indptr, cupy.full(len(orphan_nodes), indptr[-1])]
            )
            indptr: np.ndarray = cupy.asnumpy(indptr)

            # Handle indices
            indices: np.ndarray = cupy.asnumpy(indices.values)

            # Hande data
            if not is_weighted:
                data: np.ndarray = np.ones(len(indices), dtype=bool)
            else:
                edge_dtype = (
                    props["edge_dtype"]
                    if "edge_dtype" in props
                    else CuGraph.Type.compute_abstract_properties(x, {"edge_dtype"})[
                        "edge_dtype"
                    ]
                )
                caster = dtype_casting[edge_dtype]
                data: np.ndarray = cupy.asnumpy(data.values).astype(caster)

            # Calculate node_values
            if x.has_node_weights:
                node_values = cupy.asnumpy(x.nodes.loc[node_list].values)
            else:
                node_values = None

            node_count = len(node_list)
            matrix = ss.csr_matrix(
                (data, indices, indptr), shape=[node_count, node_count]
            )

        else:
            gdf = x.value.view_edge_list()
            if not is_directed:
                self_loop_mask = gdf.src == gdf.dst
                self_loop_df = gdf[self_loop_mask]
                no_self_loop_df = gdf[~self_loop_mask]
                repeat_df = no_self_loop_df.rename(columns={"src": "dst", "dst": "src"})
                gdf = cudf.concat([no_self_loop_df, repeat_df, self_loop_df,])

            node_list_sorted_indices = cupy.argsort(
                x.nodes.index.values
            )  # TODO consider storing the "nodes" attribute of a CuGraph as a sorted array
            node_list = x.nodes.index.values[node_list_sorted_indices]
            num_nodes = len(node_list)
            source_positions = cupy.searchsorted(node_list, gdf["src"].values)
            target_positions = cupy.searchsorted(node_list, gdf["dst"].values)

            source_positions = cupy.asnumpy(source_positions)
            target_positions = cupy.asnumpy(target_positions)
            weights = (
                cupy.asnumpy(gdf["weights"].values)
                if is_weighted
                else np.ones(len(source_positions), dtype=bool)
            )
            matrix = ss.coo_matrix(
                (weights, (source_positions, target_positions)),
                shape=(num_nodes, num_nodes),
            )
            node_list = cupy.asnumpy(node_list)
            node_values = (
                cupy.asnumpy(x.nodes.values[node_list_sorted_indices])
                if x.has_node_weights
                else None
            )

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

        matrix = x.value
        graph = cugraph.DiGraph() if is_directed else cugraph.Graph()
        edge_dtype_conversion_needed = (
            expected_edge_dtype is not None
            and expected_edge_dtype != dtypes_simplified[matrix.dtype]
        )

        if isinstance(matrix, ss.csr_matrix):
            offset_col = cudf.Series(matrix.indptr)
            index_col = cudf.Series(matrix.indices)
            if is_weighted:
                value_col = cudf.Series(matrix.data)
                if edge_dtype_conversion_needed:
                    value_col = value_col.astype(expected_edge_dtype)
            else:
                value_col = None
            graph.from_cudf_adjlist(offset_col, index_col, value_col)
            if x.node_list[0] != 0 or np.any(np.ediff1d(x.node_list) != 1):
                renumber_df = cudf.DataFrame({"src": x.node_list, "dst": x.node_list})
                _, renumber_map = NumberMap.renumber(renumber_df, "src", "dst")
                # assert np.all(renumber_map.implementation.df["0"].value_host == x.node_list)
                # assert np.all(renumber_map.implementation.df["id"].value_host == np.arange(len(x.node_list)))
                graph.renumbered = True
                graph.renumber_map = renumber_map
        elif isinstance(matrix, ss.coo_matrix):
            row_ids = x.node_list[matrix.row]
            column_ids = x.node_list[matrix.col]
            if is_weighted:
                weights = matrix.data
                if edge_dtype_conversion_needed:
                    weights = weights.astype(expected_edge_dtype)
            if not is_directed:
                # TODO should we move this to the GPU before filtering?
                # We can do this by generating the cudf.DataFrame first and then filtering
                mask = row_ids <= column_ids
                row_ids = row_ids[mask]
                column_ids = column_ids[mask]
                if is_weighted:
                    weights = weights[mask]
            cdf = cudf.DataFrame(
                {"source": row_ids, "target": column_ids, "weight": weights}
                if is_weighted
                else {"source": row_ids, "target": column_ids}
            )
            graph.from_cudf_edgelist(
                cdf,
                source="source",
                destination="target",
                edge_attr="weight" if is_weighted else None,
            )
        else:
            raise TypeError(f"{matrix} must be a in CSR or COO format.")

        if has_node_weights:
            nodes = cudf.Series(x.node_vals, index=x.node_list)
        else:
            nodes = cudf.Series(x.node_list, index=x.node_list)

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
                {"is_directed", "edge_type", "edge_dtype", "node_type", "node_dtype"}
                - props.keys(),
            )
        )
        is_directed = aprops["is_directed"]
        is_weighted = aprops["edge_type"] == "map"
        has_node_weights = aprops["node_type"] == "map"

        out = nx.DiGraph() if is_directed else nx.Graph()
        edge_list_df = x.value.view_edge_list()
        node_weight_label = "weight"
        edge_weight_label = "weight"

        if is_weighted:
            weights = edge_list_df["weights"]
            caster = dtype_casting[aprops["edge_dtype"]]
            if len(weights) > 0 and type(weights[0]) != caster:
                weights = weights.astype(caster)
            weights = weights.values_host.tolist()
            ebunch = zip(
                edge_list_df["src"].values_host.tolist(),
                edge_list_df["dst"].values_host.tolist(),
                weights,
            )
            out.add_weighted_edges_from(ebunch)
        else:
            ebunch = zip(
                edge_list_df["src"].values_host.tolist(),
                edge_list_df["dst"].values_host.tolist(),
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
