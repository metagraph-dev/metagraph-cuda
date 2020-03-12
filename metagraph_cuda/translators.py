from metagraph import translator
from .wrappers import CuDFEdgeList, CuGraphType
from .registry import cudf, cugraph, pandas

if cudf and cugraph:
    @translator
    def translate_graph_cudfedge2cugraph(x: CuDFEdgeList, **props) -> CuGraphType:
        g = cugraph.DiGraph()
        g.from_cudf_edgelist(x.value, x.src_label, x.dest_label)
        return g


if pandas and cudf:
    pd = pandas
    from metagraph.default_plugins.wrappers.pandas import PandasEdgeList

    @translator
    def translate_graph_pdedge2cudf(x: PandasEdgeList, **props) -> CuDFEdgeList:
        df = cudf.from_pandas(x.value)
        return CuDFEdgeList(df, src_label=x.src_label, dest_label=x.dest_label)
