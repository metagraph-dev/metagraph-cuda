from metagraph import translator
from .registry import has_cudf, has_cugraph
from metagraph.plugins import has_pandas

if has_cudf and has_cugraph:
    import cugraph
    from .types import CuDFEdgeList, CuGraphType

    @translator
    def translate_graph_cudfedge2cugraph(x: CuDFEdgeList, **props) -> CuGraphType:
        g = cugraph.DiGraph()
        g.from_cudf_edgelist(x.value, x.src_label, x.dest_label)
        return g


if has_pandas and has_cudf:
    import cudf
    from .types import CuDFEdgeList
    from metagraph.plugins.pandas.types import PandasEdgeList

    @translator
    def translate_graph_pdedge2cudf(x: PandasEdgeList, **props) -> CuDFEdgeList:
        df = cudf.from_pandas(x.value)
        return CuDFEdgeList(df, src_label=x.src_label, dest_label=x.dest_label)
