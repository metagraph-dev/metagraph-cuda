from metagraph import translator, dtypes
from metagraph.plugins import has_pandas, has_networkx


import numpy as np
import cudf
import cugraph
from metagraph.plugins.numpy.types import NumpyNodeMap
from metagraph.plugins.python.types import PythonNodeMap, dtype_casting
from .types import CuDFNodeMap, CuDFEdgeSet, CuDFEdgeMap, CuGraphEdgeSet, CuGraphEdgeMap


@translator
def translate_nodes_cudfnodemap2pythonnodemap(x: CuDFNodeMap, **props) -> PythonNodeMap:
    cast = dtype_casting[dtypes.dtypes_simplified[x.value[x.value_label].dtype]]
    [print(x.value.loc[i.item()].loc[x.value_label]) for i in x.value.index.values]
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
def translate_nodes_numpynodemap2cudfnodemap(x: NumpyNodeMap, **props) -> CuDFNodeMap:
    idx = np.arange(len(x.value))
    vals = x.value[idx]
    data = cudf.DataFrame({"value": vals})
    return CuDFNodeMap(data, "value")


@translator
def translate_graph_cudfedgeset2cugraphedgeset(
    x: CuDFEdgeSet, **props
) -> CuGraphEdgeSet:
    cugraph_graph = cugraph.DiGraph() if x.is_directed else cugraph.Graph()
    cugraph_graph.from_cudf_edgelist(x.value, x.src_label, x.dst_label)
    return CuGraphEdgeSet(cugraph_graph)


@translator
def translate_graph_cudfedgemap2cugraphedgemap(
    x: CuDFEdgeMap, **props
) -> CuGraphEdgeMap:
    cugraph_graph = cugraph.DiGraph() if x.is_directed else cugraph.Graph()
    cugraph_graph.from_cudf_edgelist(
        x.value, x.src_label, x.dst_label, edge_attr=x.weight_label
    )
    return CuGraphEdgeMap(cugraph_graph)


@translator
def translate_graph_cugraphedgeset2cudfedgeset(
    x: CuGraphEdgeSet, **props
) -> CuDFEdgeSet:
    return CuDFEdgeSet(
        x.value.view_edge_list(),
        src_label="src",
        dst_label="dst",
        is_directed=isinstance(x.value, cugraph.DiGraph),
    )


@translator
def translate_graph_cugraphedgemap2cudfedgemap(
    x: CuGraphEdgeMap, **props
) -> CuDFEdgeMap:
    return CuDFEdgeMap(
        x.value.view_edge_list(),
        src_label="src",
        dst_label="dst",
        weight_label="weights",
        is_directed=isinstance(x.value, cugraph.DiGraph),
    )


if has_networkx:
    import cudf
    import networkx as nx
    from metagraph.plugins.networkx.types import NetworkXEdgeMap, NetworkXEdgeSet

    @translator
    def translate_graph_cugraphedgeset2networkxedgeset(
        x: CuGraphEdgeSet, **props
    ) -> NetworkXEdgeSet:
        out = nx.DiGraph() if isinstance(x.value, cugraph.DiGraph) else nx.Graph()
        column_name_to_series_set = {
            column_name: series
            for column_name, series in x.value.view_edge_list().iteritems()
        }
        source_destination_weight_pairs = zip(
            column_name_to_series_set["src"], column_name_to_series_set["dst"]
        )
        out.add_edges_from(source_destination_weight_pairs)
        return NetworkXEdgeSet(out)

    @translator
    def translate_graph_cugraphedgemap2networkxedgemap(
        x: CuGraphEdgeMap, **props
    ) -> NetworkXEdgeMap:
        out = nx.DiGraph() if isinstance(x.value, cugraph.DiGraph) else nx.Graph()
        column_name_to_series_map = {
            column_name: series
            for column_name, series in x.value.view_edge_list().iteritems()
        }
        source_destination_weight_triples = zip(
            column_name_to_series_map["src"],
            column_name_to_series_map["dst"],
            column_name_to_series_map["weights"],
        )
        out.add_weighted_edges_from(source_destination_weight_triples)
        return NetworkXEdgeMap(out)


if has_networkx:
    import cudf
    from metagraph.plugins.networkx.types import NetworkXEdgeMap, NetworkXEdgeSet

    @translator
    def translate_graph_networkxedgeset2cudfedgeset(
        x: NetworkXEdgeSet, **props
    ) -> CuDFEdgeSet:
        edgelist = x.value.edges(data=False)
        source_nodes, target_nodes = zip(*edgelist)
        cdf = cudf.DataFrame({"source": source_nodes, "destination": target_nodes})
        return CuDFEdgeSet(
            cdf,
            src_label="source",
            dst_label="destination",
            is_directed=isinstance(x.value, nx.DiGraph),
        )

    @translator
    def translate_graph_networkxedgemap2cudfedgemap(
        x: NetworkXEdgeMap, **props
    ) -> CuDFEdgeMap:
        edgelist = x.value.edges(data=True)
        source_nodes, target_nodes, node_data_dicts = zip(*edgelist)
        cdf_data = {"source": source_nodes, "destination": target_nodes}
        cdf_data.update(
            {
                x.weight_label: [
                    data_dict.get(x.weight_label, float("nan"))
                    for data_dict in node_data_dicts
                ]
            }
        )
        cdf = cudf.DataFrame(cdf_data)
        return CuDFEdgeMap(
            cdf,
            src_label="source",
            dst_label="destination",
            weight_label=x.weight_label,
            is_directed=isinstance(x.value, nx.DiGraph),
        )


if has_pandas:
    import cudf
    from metagraph.plugins.pandas.types import PandasEdgeSet, PandasEdgeMap

    @translator
    def translate_graph_pdedgeset2cudfedgeset(x: PandasEdgeSet, **props) -> CuDFEdgeSet:
        df = cudf.from_pandas(x.value)
        return CuDFEdgeSet(
            df, src_label=x.src_label, dst_label=x.dst_label, is_directed=x.is_directed
        )

    @translator
    def translate_graph_pdedgemap2cudfedgemap(x: PandasEdgeMap, **props) -> CuDFEdgeMap:
        df = cudf.from_pandas(x.value)
        return CuDFEdgeMap(
            df,
            src_label=x.src_label,
            dst_label=x.dst_label,
            weight_label=x.weight_label,
            is_directed=x.is_directed,
        )

    @translator
    def translate_graph_cudfedgeset2pdedgeset(x: CuDFEdgeSet, **props) -> PandasEdgeSet:
        pdf = x.value.to_pandas()
        return PandasEdgeSet(
            pdf,
            src_label=x.src_label,
            dst_label=x.dst_label,
            is_directed=x.is_directed,
        )

    @translator
    def translate_graph_cudfedgemap2pdedgemap(x: CuDFEdgeMap, **props) -> PandasEdgeMap:
        pdf = x.value.to_pandas()
        return PandasEdgeMap(
            pdf,
            src_label=x.src_label,
            dst_label=x.dst_label,
            weight_label=x.weight_label,
            is_directed=x.is_directed,
        )
