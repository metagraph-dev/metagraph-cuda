import metagraph as mg
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as ss
import cudf
import cugraph
from metagraph_cuda.plugins.cugraph.types import CuGraph, CuGraphEdgeSet, CuGraphEdgeMap

# TODO failing due to https://github.com/rapidsai/cugraph/issues/1315
# def test_scipy_edge_set_to_cugraph_edge_set():
#     """
#               +-+
#      ------>  |1|
#      |        +-+
#      |
#      |         |
#      |         v

#     +-+  <--  +-+       +-+
#     |0|       |2|  <--  |3|
#     +-+  -->  +-+       +-+"""
#     dpr = mg.resolver
#     scipy_sparse_matrix = ss.csr_matrix(
#         np.array(
#             [
#                 [0, 1, 1, 0],
#                 [0, 0, 1, 0],
#                 [1, 0, 0, 0],
#                 [0, 0, 1, 0],
#             ]
#         )
#     )
#     x = dpr.wrappers.EdgeSet.ScipyEdgeSet(scipy_sparse_matrix)

#     sources = [0, 0, 1, 2, 3]
#     destinations = [1, 2, 2, 0, 2]
#     cdf = cudf.DataFrame({"Source": sources, "Destination": destinations})
#     g = cugraph.DiGraph()
#     g.from_cudf_edgelist(cdf, source="Source", destination="Destination")
#     intermediate = dpr.wrappers.EdgeSet.CuGraphEdgeSet(g)

#     y = dpr.translate(x, CuGraphEdgeSet)
#     dpr.assert_equal(y, intermediate)
#     assert len(dpr.plan.translate(x, CuGraphEdgeSet)) == 1


def test_scipy_edge_map_to_cugraph_edge_map():
    """
              +-+
     ----9->  |1|
     |        +-+
     |
     |         |
     |         6
     |         |
     |         v

    +-+  <-7-  +-+        +-+
    |0|        |2|  <-5-  |3|
    +-+  -8->  +-+        +-+"""
    dpr = mg.resolver
    scipy_sparse_matrix = ss.csr_matrix(
        np.array([[0, 9, 8, 0], [0, 0, 6, 0], [7, 0, 0, 0], [0, 0, 5, 0],])
    )
    x = dpr.wrappers.EdgeMap.ScipyEdgeMap(scipy_sparse_matrix)

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    weights = [9, 8, 6, 7, 5]
    cdf = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weights": weights}
    )
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(
        cdf, source="source", destination="destination", edge_attr="weights"
    )
    intermediate = dpr.wrappers.EdgeMap.CuGraphEdgeMap(g)

    y = dpr.translate(x, CuGraphEdgeMap)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuGraphEdgeMap)) == 1


def test_unweighted_directed_networkx_to_cugraph():
    """
              +-+
     ------>  |1|
     |        +-+
     |
     |         |
     |         v

    +-+  <--  +-+       +-+
    |0|       |2|  <--  |3|
    +-+  -->  +-+       +-+"""
    dpr = mg.resolver
    networkx_graph_data = [
        (0, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (3, 2),
    ]
    networkx_graph_unwrapped = nx.DiGraph()
    networkx_graph_unwrapped.add_edges_from(networkx_graph_data)
    x = dpr.wrappers.Graph.NetworkXGraph(networkx_graph_unwrapped)

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    cdf = cudf.DataFrame({"source": sources, "destination": destinations})
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(cdf, source="source", destination="destination")
    intermediate = dpr.wrappers.Graph.CuGraph(g, None)
    y = dpr.translate(x, CuGraph)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuGraph)) == 1


def test_weighted_directed_networkx_to_cugraph():
    """
              +-+
     ----9->  |1|
     |        +-+
     |
     |         |
     |         6
     |         |
     |         v

    +-+  <-7-  +-+        +-+
    |0|        |2|  <-5-  |3|
    +-+  -8->  +-+        +-+"""
    dpr = mg.resolver
    networkx_graph_data = [
        (0, 1, 9),
        (0, 2, 8),
        (2, 0, 7),
        (1, 2, 6),
        (3, 2, 5),
    ]
    networkx_graph_unwrapped = nx.DiGraph()
    networkx_graph_unwrapped.add_weighted_edges_from(
        networkx_graph_data, weight="weight"
    )
    x = dpr.wrappers.Graph.NetworkXGraph(networkx_graph_unwrapped)

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    weights = [9, 8, 6, 7, 5]
    cdf = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(
        cdf, source="source", destination="destination", edge_attr="weight"
    )
    intermediate = dpr.wrappers.Graph.CuGraph(g, None)
    y = dpr.translate(x, CuGraph)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuGraph)) == 1


def test_pandas_edge_set_to_cugraph_edge_set():
    """
              +-+
     ------>  |1|
     |        +-+
     |
     |         |
     |         v

    +-+  <--  +-+       +-+
    |0|       |2|  <--  |3|
    +-+  -->  +-+       +-+"""
    dpr = mg.resolver
    pdf = pd.DataFrame({"src": (0, 0, 2, 1, 3), "dst": (1, 2, 0, 2, 2)})
    x = dpr.wrappers.EdgeSet.PandasEdgeSet(
        pdf, src_label="src", dst_label="dst", is_directed=True
    )

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    cdf = cudf.DataFrame({"source": sources, "destination": destinations})
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(cdf, source="source", destination="destination")
    intermediate = dpr.wrappers.EdgeSet.CuGraphEdgeSet(g)
    y = dpr.translate(x, CuGraphEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuGraphEdgeSet)) == 1


def test_pandas_edge_map_to_cugraph_edge_map():
    """
              +-+
     ----9->  |1|
     |        +-+
     |
     |         |
     |         6
     |         |
     |         v

    +-+  <-7-  +-+        +-+
    |0|        |2|  <-5-  |3|
    +-+  -8->  +-+        +-+"""
    dpr = mg.resolver
    pdf = pd.DataFrame(
        {"src": (0, 0, 2, 1, 3), "dst": (1, 2, 0, 2, 2), "w": (9, 8, 7, 6, 5)}
    )
    x = dpr.wrappers.EdgeMap.PandasEdgeMap(
        pdf, src_label="src", dst_label="dst", weight_label="w", is_directed=True
    )

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    weights = [9, 8, 7, 6, 5]
    cdf = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weights": weights}
    )
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(
        cdf, source="source", destination="destination", edge_attr="weights"
    )
    intermediate = dpr.wrappers.EdgeMap.CuGraphEdgeMap(g)

    y = dpr.translate(x, CuGraphEdgeMap)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuGraphEdgeMap)) == 1


def test_scipy_graph_to_cugraph_graph():
    """
              +-+       +-+
     ------>  |1|       |4|
     |        +-+       +-+
     |
     |         |
     |         v

    +-+  <--  +-+       +-+
    |0|       |2|  <--  |3|
    +-+  -->  +-+       +-+
    """
    dpr = mg.resolver
    scipy_sparse_matrix = ss.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
    )
    x = dpr.wrappers.Graph.ScipyGraph(scipy_sparse_matrix, np.arange(5))

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    cdf = cudf.DataFrame({"Source": sources, "Destination": destinations})
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(cdf, source="Source", destination="Destination")
    intermediate = dpr.wrappers.Graph.CuGraph(g, cudf.Series(range(5)))

    y = dpr.translate(x, CuGraph)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuGraph)) == 1
