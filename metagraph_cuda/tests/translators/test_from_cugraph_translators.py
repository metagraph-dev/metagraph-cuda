import metagraph as mg
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as ss
import io
import cudf
import cugraph
from metagraph.plugins.networkx.types import NetworkXGraph
from metagraph.plugins.pandas.types import PandasEdgeSet, PandasEdgeMap
from metagraph.plugins.scipy.types import ScipyEdgeSet, ScipyEdgeMap, ScipyGraph


def test_cugraph_edge_set_to_scipy_edge_set():
    """
          +-+
 ------>  |1|
 |        +-+
 | 
 |         |
 |         v

+-+  <--  +-+       +-+
|0|       |2|  <--  |3|
+-+  -->  +-+       +-+
"""
    dpr = mg.resolver
    csv_data = """
Source,Destination
0,1
0,2
1,2
2,0
3,2
"""
    csv_file = io.StringIO(csv_data)
    cdf = cudf.read_csv(csv_file)
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(cdf, source="Source", destination="Destination")
    x = dpr.wrappers.EdgeSet.CuGraphEdgeSet(g)

    scipy_sparse_matrix = ss.csr_matrix(
        np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0],])
    )
    intermediate = ScipyEdgeSet(scipy_sparse_matrix)
    y = dpr.translate(x, ScipyEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, ScipyEdgeSet)) == 1


def test_cugraph_edge_map_to_scipy_edge_map():
    """
           +-+
 ------>   |1|
 |         +-+
 | 
 |          |
 9          6
 |          |
 |          v

+-+  <-8-  +-+        +-+
|0|        |2|  <-5-  |3|
+-+  -7->  +-+        +-+
"""
    dpr = mg.resolver
    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    weights = [9, 7, 6, 8, 5]
    cdf = cudf.DataFrame(
        {"Source": sources, "Destination": destinations, "Weight": weights}
    )
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(
        cdf, source="Source", destination="Destination", edge_attr="Weight"
    )
    x = dpr.wrappers.EdgeMap.CuGraphEdgeMap(g)

    scipy_sparse_matrix = ss.csr_matrix(
        np.array([[0, 9, 7, 0], [0, 0, 6, 0], [8, 0, 0, 0], [0, 0, 5, 0],])
    )
    intermediate = ScipyEdgeMap(scipy_sparse_matrix)
    y = dpr.translate(x, ScipyEdgeMap)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, ScipyEdgeMap)) == 1


def test_unweighted_directed_edge_list_cugraph_to_nextworkx():
    """
0 < -   1       5   - > 6
      ^       ^ ^       
|   /   |   /   |   /    
v       v /       v      
3   - > 4 < -   2   - > 7
    """
    dpr = mg.resolver
    sources = [0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    destinations = [3, 0, 4, 4, 5, 7, 1, 4, 5, 6, 2]
    gdf = cudf.DataFrame({"source": sources, "dst": destinations})
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_edgelist(gdf, source="source", destination="dst")
    x = dpr.wrappers.Graph.CuGraph(cugraph_graph_unwrapped)

    networkx_graph_data = [
        (0, 3),
        (1, 0),
        (1, 4),
        (2, 4),
        (2, 5),
        (2, 7),
        (3, 1),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 2),
    ]
    networkx_graph_unwrapped = nx.DiGraph()
    networkx_graph_unwrapped.add_edges_from(networkx_graph_data)
    intermediate = NetworkXGraph(networkx_graph_unwrapped)
    y = dpr.translate(x, NetworkXGraph)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, NetworkXGraph)) == 1


def test_weighted_directed_edge_list_cugraph_to_nextworkx():
    """
0 <--2-- 1        5 --10-> 6
|      ^ |      ^ ^      / 
|     /  |     /  |     /   
1    7   3    9   5   11   
|   /    |  /     |   /    
v        v /        v      
3 --8--> 4 <--4-- 2 --6--> 7
    """
    dpr = mg.resolver
    sources = [0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    destinations = [3, 0, 4, 4, 5, 7, 1, 4, 5, 6, 2]
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    gdf = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    cugraph_graph = cugraph.DiGraph()
    cugraph_graph.from_cudf_edgelist(
        gdf, source="source", destination="destination", edge_attr="weight"
    )
    x = dpr.wrappers.Graph.CuGraph(cugraph_graph)

    networkx_graph_data = [
        (0, 3, 1),
        (1, 0, 2),
        (1, 4, 3),
        (2, 4, 4),
        (2, 5, 5),
        (2, 7, 6),
        (3, 1, 7),
        (3, 4, 8),
        (4, 5, 9),
        (5, 6, 10),
        (6, 2, 11),
    ]
    networkx_graph_unwrapped = nx.DiGraph()
    networkx_graph_unwrapped.add_weighted_edges_from(networkx_graph_data)
    intermediate = NetworkXGraph(networkx_graph_unwrapped)
    y = dpr.translate(x, NetworkXGraph)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, NetworkXGraph)) == 1


def test_unweighted_directed_adjacency_list_cugraph_to_networkx():
    """
0 -----> 1 
^^       | 
| \_     | 
|   \_   |
|     \  | 
|      \ v 
2 <----- 3 
    """
    dpr = mg.resolver
    sparse_matrix = ss.csr_matrix(
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]]),
        dtype=np.int8,
    )
    offsets = cudf.Series(sparse_matrix.indptr)
    indices = cudf.Series(sparse_matrix.indices)
    cugraph_graph = cugraph.DiGraph()
    cugraph_graph.from_cudf_adjlist(offsets, indices, None)
    x = dpr.wrappers.Graph.CuGraph(cugraph_graph)

    networkx_graph_data = [
        (0, 1),
        (1, 3),
        (2, 0),
        (3, 0),
        (3, 2),
    ]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(networkx_graph_data)
    intermediate = dpr.wrappers.Graph.NetworkXGraph(networkx_graph)
    y = dpr.translate(x, NetworkXGraph)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, NetworkXGraph)) == 1


def test_weighted_directed_adjacency_list_cugraph_to_networkx():
    """
0 -1.1-> 1 
^^       | 
| \     2.2 
|  4.4   |
3.3   \  | 
|      \ v 
2 <-5.5- 3 
    """
    dpr = mg.resolver
    sparse_matrix = ss.csr_matrix(
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]]),
        dtype=np.int8,
    )
    offsets = cudf.Series(sparse_matrix.indptr)
    indices = cudf.Series(sparse_matrix.indices)
    weights = cudf.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    cugraph_graph = cugraph.DiGraph()
    cugraph_graph.from_cudf_adjlist(offsets, indices, weights)
    x = dpr.wrappers.Graph.CuGraph(cugraph_graph)

    networkx_graph_data = [
        (0, 1, 1.1),
        (1, 3, 2.2),
        (2, 0, 3.3),
        (3, 0, 4.4),
        (3, 2, 5.5),
    ]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_weighted_edges_from(networkx_graph_data)
    intermediate = dpr.wrappers.Graph.NetworkXGraph(networkx_graph)
    y = dpr.translate(x, dpr.wrappers.Graph.NetworkXGraph)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, NetworkXGraph)) == 1


def test_cugraph_edge_set_to_pandas_edge_set():
    """
0 < -   1       5   - > 6
      ^       ^ ^       
|   /   |   /   |   /    
v       v /       v      
3   - > 4 < -   2   - > 7
    """
    dpr = mg.resolver
    sources = [0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    destinations = [3, 0, 4, 4, 5, 7, 1, 4, 5, 6, 2]
    gdf = cudf.DataFrame({"source": sources, "dst": destinations})
    cugraph_graph = cugraph.DiGraph()
    cugraph_graph.from_cudf_edgelist(gdf, source="source", destination="dst")
    x = dpr.wrappers.EdgeSet.CuGraphEdgeSet(cugraph_graph)

    pdf = pd.DataFrame({"source": sources, "dst": destinations})
    intermediate = PandasEdgeSet(
        pdf, src_label="source", dst_label="dst", is_directed=True
    )
    y = dpr.translate(x, PandasEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, PandasEdgeSet)) == 1


def test_cugraph_edge_map_to_pandas_edge_map():
    """
0 <--2-- 1        5 --10-> 6
|      ^ |      ^ ^      / 
|     /  |     /  |     /   
1    7   3    9   5   11   
|   /    |  /     |   /    
v        v /        v      
3 --8--> 4 <--4-- 2 --6--> 7
    """
    dpr = mg.resolver
    sources = [0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    destinations = [3, 0, 4, 4, 5, 7, 1, 4, 5, 6, 2]
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    gdf = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    cugraph_graph = cugraph.DiGraph()
    cugraph_graph.from_cudf_edgelist(
        gdf, source="source", destination="destination", edge_attr="weight"
    )
    x = dpr.wrappers.EdgeMap.CuGraphEdgeMap(cugraph_graph)

    pdf = pd.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    intermediate = PandasEdgeMap(
        pdf,
        src_label="source",
        dst_label="destination",
        weight_label="weight",
        is_directed=True,
    )
    y = dpr.translate(x, PandasEdgeMap)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, PandasEdgeMap)) == 1


def test_cugraph_graph_to_scipy_graph():
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

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    cdf = cudf.DataFrame({"Source": sources, "Destination": destinations})
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(
        cdf, source="Source", destination="Destination",
    )
    cudf_nodes = dpr.wrappers.NodeSet.CuDFNodeSet(cudf.Series(range(5)))
    x = dpr.wrappers.Graph.CuGraph(g)

    scipy_sparse_matrix = ss.csr_matrix(
        np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0],])
    )
    ss_edge_set = dpr.wrappers.EdgeSet.ScipyEdgeSet(scipy_sparse_matrix)
    np_nodes = dpr.wrappers.NodeSet.NumpyNodeSet(np.arange(5))
    intermediate = dpr.wrappers.Graph.ScipyGraph(ss_edge_set, np_nodes)

    y = dpr.translate(x, ScipyGraph)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, ScipyGraph)) == 1
