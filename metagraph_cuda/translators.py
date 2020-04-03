from metagraph import translator
from .registry import has_cudf, has_cugraph
from metagraph.plugins import has_pandas

if has_cudf and has_cugraph:
    import cugraph
    from .types import CuDFEdgeList, CuGraph

    @translator
    def translate_graph_cudfedge2cugraph(x: CuDFEdgeList, **props) -> CuGraph:
        cugraph_graph = cugraph.Graph()
        cugraph_graph.from_cudf_edgelist(x.value, x.src_label, x.dest_label)
        return CuGraph(cugraph_graph)


if has_pandas and has_cudf:
    import cudf
    from .types import CuDFEdgeList
    from metagraph.plugins.pandas.types import PandasEdgeList

    @translator
    def translate_graph_pdedge2cudf(x: PandasEdgeList, **props) -> CuDFEdgeList:
        df = cudf.from_pandas(x.value)
        return CuDFEdgeList(df, src_label=x.src_label, dest_label=x.dest_label)
