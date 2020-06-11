import metagraph as mg
import pandas as pd
import networkx as nx
import numpy as np
import scipy
import cugraph
import cudf
import io
from metagraph_cuda.types import CuDFEdgeSet, CuDFNodeMap


def test_pandas_edge_set_to_cudf_edge_set():
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
    pdf_unwrapped = pd.read_csv(csv_file)
    x = dpr.wrappers.EdgeSet.PandasEdgeSet(pdf_unwrapped, "Source", "Destination")

    csv_file = io.StringIO(csv_data)
    cdf_unwrapped = cudf.read_csv(csv_file)
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "Source", "Destination"
    )
    y = dpr.translate(x, CuDFEdgeSet)
    CuDFEdgeSet.Type.assert_equal(y, intermediate, {}, {})


def test_unweighted_directed_networkx_to_cudf_edge_set():
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
    networkx_graph_data = [
        (0, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (3, 2),
    ]
    networkx_graph_unwrapped = nx.DiGraph()
    networkx_graph_unwrapped.add_edges_from(networkx_graph_data)
    x = dpr.wrappers.EdgeSet.NetworkXEdgeSet(networkx_graph_unwrapped)

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    cdf_unwrapped = cudf.DataFrame({"source": sources, "destination": destinations})
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "source", "destination"
    )
    y = dpr.translate(x, CuDFEdgeSet)
    CuDFEdgeSet.Type.assert_equal(y, intermediate, {}, {})


def test_weighted_directed_networkx_to_cudf_edge_set():
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
+-+  -8->  +-+        +-+
"""
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
    x = dpr.wrappers.EdgeSet.NetworkXEdgeSet(networkx_graph_unwrapped)

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    weights = [9, 8, 6, 7, 5]
    cdf_unwrapped = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "source", "destination"
    )
    y = dpr.translate(x, CuDFEdgeSet)
    CuDFEdgeSet.Type.assert_equal(y, intermediate, {}, {})


def test_unweighted_directed_edge_set_cugraph_to_cudf_edge_set():
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
    x = dpr.wrappers.EdgeSet.CuGraphEdgeSet(cugraph_graph_unwrapped)

    cdf_unwrapped = cudf.DataFrame({"source": sources, "destination": destinations})
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "source", "destination"
    )
    y = dpr.translate(x, CuDFEdgeSet)
    CuDFEdgeSet.Type.assert_equal(y, intermediate, {}, {})


def test_weighted_directed_edge_set_cugraph_to_cudf_edge_set():
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
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_edgelist(
        gdf, source="source", destination="destination", edge_attr="weight"
    )
    cugraph_graph = dpr.wrappers.EdgeSet.CuGraphEdgeSet(cugraph_graph_unwrapped)
    x = dpr.translate(cugraph_graph, dpr.types.EdgeSet.CuDFEdgeSetType)

    cdf_unwrapped = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "source", "destination"
    )
    y = dpr.translate(x, CuDFEdgeSet)
    CuDFEdgeSet.Type.assert_equal(y, intermediate, {}, {})


def test_unweighted_directed_adjacency_set_cugraph_to_cudf_edge_set():
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
    sparse_matrix = scipy.sparse.csr_matrix(
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]]),
        dtype=np.int8,
    )
    offsets = cudf.Series(sparse_matrix.indptr)
    indices = cudf.Series(sparse_matrix.indices)
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_adjlist(offsets, indices, None)
    cugraph_graph = dpr.wrappers.EdgeSet.CuGraphEdgeSet(cugraph_graph_unwrapped)
    x = dpr.translate(cugraph_graph, dpr.types.EdgeSet.CuDFEdgeSetType)

    sources = [0, 1, 2, 3, 3]
    destinations = [1, 3, 0, 2, 0]
    cdf_unwrapped = cudf.DataFrame({"source": sources, "destination": destinations})
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "source", "destination"
    )
    y = dpr.translate(x, CuDFEdgeSet)
    CuDFEdgeSet.Type.assert_equal(y, intermediate, {}, {})


def test_weighted_directed_adjacency_set_cugraph_to_cudf_edge_set():
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
    sparse_matrix = scipy.sparse.csr_matrix(
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]]),
        dtype=np.int8,
    )
    offsets = cudf.Series(sparse_matrix.indptr)
    indices = cudf.Series(sparse_matrix.indices)
    weights = cudf.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_adjlist(offsets, indices, weights)
    x = dpr.wrappers.EdgeSet.CuGraphEdgeSet(cugraph_graph_unwrapped)

    sources = [0, 1, 2, 3, 3]
    destinations = [1, 3, 0, 0, 2]
    weights = [1.1, 2.2, 3.3, 4.4, 5.5]
    cdf_unwrapped = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "source", "destination"
    )
    y = dpr.translate(x, CuDFEdgeSet)
    CuDFEdgeSet.Type.assert_equal(y, intermediate, {}, {})


def test_numpy_nodes_to_cudf_nodes():
    dpr = mg.resolver
    numpy_data = np.array([33, 22, 11])
    numpy_nodes = dpr.wrappers.NodeMap.NumpyNodeMap(numpy_data)
    x = dpr.translate(numpy_nodes, dpr.types.NodeMap.CuDFNodeMapType)

    keys = [0, 1, 2]
    labels = [33, 22, 11]
    cdf_unwrapped = cudf.DataFrame({"key": keys, "label": labels}).set_index("key")
    intermediate = dpr.wrappers.NodeMap.CuDFNodeMap(cdf_unwrapped, "label")
    y = dpr.translate(x, CuDFNodeMap)
    CuDFNodeMap.Type.assert_equal(y, intermediate, {}, {})


def test_python_nodes_to_cudf_nodes():
    dpr = mg.resolver
    python_data = {1: 11, 2: 22, 3: 33}
    python_nodes = dpr.wrappers.NodeMap.PythonNodeMap(python_data)
    x = dpr.translate(python_nodes, dpr.types.NodeMap.CuDFNodeMapType)

    keys = [1, 2, 3]
    labels = [11, 22, 33]
    cdf_unwrapped = cudf.DataFrame({"key": keys, "label": labels}).set_index("key")
    intermediate = dpr.wrappers.NodeMap.CuDFNodeMap(cdf_unwrapped, "label")
    y = dpr.translate(x, CuDFNodeMap)
    CuDFNodeMap.Type.assert_equal(y, intermediate, {}, {})
