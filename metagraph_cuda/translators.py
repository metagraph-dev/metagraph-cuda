from metagraph import translator
from metagraph.plugins import has_pandas, has_networkx


import numpy as np
import cudf
import cugraph
from .types import CuDFEdgeList, CuGraph, CuDFNodes
from metagraph.plugins.python.types import PythonNodes, dtype_casting
from metagraph.plugins.numpy.types import NumpyNodes


@translator
def translate_nodes_cudfnodes2pythonnodes(x: CuDFNodes, **props) -> PythonNodes:
    cast = dtype_casting[x._dtype]
    data = {key: cast(x[key]) for key in x.node_index}
    return PythonNodes(
        data, dtype=x._dtype, weights=x._weights, node_index=x.node_index
    )


@translator
def translate_nodes_pythonnodes2cudfnodes(x: PythonNodes, **props) -> CuDFNodes:
    keys, values = zip(*x.value.items())
    data = cudf.DataFrame({"key": keys, "value": values})
    return CuDFNodes(data, "key", "value", weights=x._weights, node_index=x.node_index)


@translator
def translate_nodes_numpynodes2cudfnodes(x: NumpyNodes, **props) -> CuDFNodes:
    idx = np.arange(len(x.value))
    vals = x.value[idx]
    data = cudf.DataFrame({"key": idx, "value": vals})
    return CuDFNodes(data, "key", "value", weights=x._weights)


@translator
def translate_graph_cudfedge2cugraph(x: CuDFEdgeList, **props) -> CuGraph:
    cugraph_graph = cugraph.DiGraph() if x.is_directed else cugraph.Graph()
    cugraph_graph.from_cudf_edgelist(x.value, x.src_label, x.dst_label, x.weight_label)
    return CuGraph(cugraph_graph)


@translator
def translate_graph_cugraph2cudfedge(x: CuGraph, **props) -> CuDFEdgeList:
    type_info = CuGraph.Type.get_type(x)
    weight_label = None
    if (x.value.edgelist and "weights" in x.value.view_edge_list().columns) or (
        x.value.adjlist and x.value.view_adj_list()[2] is not None
    ):
        weight_label = "weights"
    return CuDFEdgeList(
        x.value.view_edge_list(),
        src_label="src",
        dst_label="dst",
        weight_label=weight_label,
        is_directed=type_info["is_directed"],
        weights=type_info["weights"],
        node_index=x.node_index,
    )


if has_networkx:
    import cudf
    import networkx as nx
    from .types import CuGraph
    from metagraph.plugins.networkx.types import NetworkXGraph

    @translator
    def translate_graph_cugraph2networkx(x: CuGraph, **props) -> NetworkXGraph:
        type_info = CuGraph.Type.get_type(x)
        weight_label = None
        out = nx.DiGraph() if x.is_directed else nx.Graph()
        column_name_to_series_map = {
            column_name: series
            for column_name, series in x.value.view_edge_list().iteritems()
        }
        if "weights" in column_name_to_series_map:
            source_destination_weight_triples = zip(
                column_name_to_series_map["src"],
                column_name_to_series_map["dst"],
                column_name_to_series_map["weights"],
            )
            out.add_weighted_edges_from(source_destination_weight_triples)
            weight_label = "weight"
        else:
            source_destination_weight_pairs = zip(
                column_name_to_series_map["src"], column_name_to_series_map["dst"]
            )
            out.add_edges_from(source_destination_weight_pairs)
        return NetworkXGraph(
            out,
            weight_label=weight_label,
            weights=type_info["weights"],
            dtype=type_info["dtype"],
            node_index=x.node_index,
        )


if has_networkx:
    import cudf
    from .types import CuDFEdgeList
    from metagraph.plugins.networkx.types import NetworkXGraph

    @translator
    def translate_graph_networkx2cudf(x: NetworkXGraph, **props) -> CuDFEdgeList:
        type_info = NetworkXGraph.Type.get_type(x)
        edgelist = x.value.edges(data=True)
        source_nodes, target_nodes, node_data_dicts = zip(*edgelist)
        cdf_data = {"source": source_nodes, "destination": target_nodes}
        if x.weight_label:
            cdf_data.update(
                {
                    x.weight_label: [
                        data_dict.get(x.weight_label, float("nan"))
                        for data_dict in node_data_dicts
                    ]
                }
            )
        cdf = cudf.DataFrame(cdf_data)
        return CuDFEdgeList(
            cdf,
            src_label="source",
            dst_label="destination",
            weight_label=x.weight_label,
            is_directed=type_info["is_directed"],
            weights=type_info["weights"],
            node_index=x.node_index,
        )


if has_pandas:
    import cudf
    from .types import CuDFEdgeList
    from metagraph.plugins.pandas.types import PandasEdgeList

    @translator
    def translate_graph_pdedge2cudf(x: PandasEdgeList, **props) -> CuDFEdgeList:
        df = cudf.from_pandas(x.value)
        return CuDFEdgeList(df, src_label=x.src_label, dst_label=x.dst_label)

    @translator
    def translate_graph_cudf2pdedge(x: CuDFEdgeList, **props) -> PandasEdgeList:
        type_info = CuDFEdgeList.Type.get_type(x)
        pdf = x.value.to_pandas()
        return PandasEdgeList(
            pdf,
            src_label=x.src_label,
            dst_label=x.dst_label,
            weight_label=x.weight_label,
            is_directed=type_info["is_directed"],
            weights=type_info["weights"],
            node_index=x.node_index,
        )
