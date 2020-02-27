from metagraph import abstract_algorithm, concrete_algorithm
from ..registry import cugraph, numpy


if cugraph and numpy:
    np = numpy
    from metagraph.default_plugins.wrappers.numpy import NumpySparseVector

    @concrete_algorithm("link_analysis.pagerank")
    def cugraph_pagerank(
        graph: cugraph.DiGraph,
        damping: float = 0.85,
        maxiter: int = 50,
        tolerance: float = 1e-05,
    ) -> NumpySparseVector:
        pagerank = cugraph.pagerank(
            graph, alpha=damping, max_iter=maxiter, tol=tolerance
        )
        out = np.full((graph.number_of_nodes(),), np.nan)
        out[pagerank["vertex"]] = pagerank["pagerank"]
        return NumpySparseVector(out, missing_value=np.nan)
