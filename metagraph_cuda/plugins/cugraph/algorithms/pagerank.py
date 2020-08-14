from metagraph import concrete_algorithm
from ..registry import has_cugraph
import numpy as np


if has_cugraph:
    import cugraph
    from ..types import CuGraph, CuDFNodeMap

    @concrete_algorithm("centrality.pagerank")
    def cugraph_pagerank(
        graph: CuGraph, damping: float, maxiter: int, tolerance: float,
    ) -> CuDFNodeMap:
        pagerank = cugraph.pagerank(
            graph.edges.value, alpha=damping, max_iter=maxiter, tol=tolerance
        ).set_index("vertex")
        return CuDFNodeMap(pagerank, "pagerank")
