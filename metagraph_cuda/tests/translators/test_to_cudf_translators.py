import metagraph as mg
import pandas as pd
import networkx as nx
import numpy as np
import scipy
import cugraph
import cudf
import io


def test_pandas_edge_list_to_cudf_edge_list():
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
    r = mg.resolver
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
    pdf = r.wrapper.Graph.PandasEdgeList(pdf_unwrapped, "Source", "Destination")
    cdf = r.translate(pdf, r.types.Graph.CuDFEdgeListType)
    assert len(pdf.value) == len(cdf.value)
    assert len(pdf.value) == 5
    assert set(cdf.value[cdf.value["Source"] == 0]["Destination"]) == set(
        pdf.value[pdf.value["Source"] == 0]["Destination"]
    )
    assert set(cdf.value[cdf.value["Source"] == 0]["Destination"]) == {1, 2}
    assert set(cdf.value[cdf.value["Source"] == 1]["Destination"]) == set(
        pdf.value[pdf.value["Source"] == 1]["Destination"]
    )
    assert set(cdf.value[cdf.value["Source"] == 1]["Destination"]) == {2}
    assert set(cdf.value[cdf.value["Source"] == 2]["Destination"]) == set(
        pdf.value[pdf.value["Source"] == 2]["Destination"]
    )
    assert set(cdf.value[cdf.value["Source"] == 2]["Destination"]) == {0}
    assert set(cdf.value[cdf.value["Source"] == 3]["Destination"]) == set(
        pdf.value[pdf.value["Source"] == 3]["Destination"]
    )
    assert set(cdf.value[cdf.value["Source"] == 3]["Destination"]) == {2}


def test_unweighted_directed_networkx_to_cudf_edge_list():
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
    r = mg.resolver
    networkx_graph_data = [
        (0, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (3, 2),
    ]
    networkx_graph_unwrapped = nx.DiGraph()
    networkx_graph_unwrapped.add_edges_from(networkx_graph_data)
    networkx_graph = r.wrapper.Graph.NetworkXGraph(networkx_graph_unwrapped)
    cdf = r.translate(networkx_graph, r.types.Graph.CuDFEdgeListType)
    assert len(cdf.value) == 5
    assert set(cdf.value[cdf.value["source"] == 0]["destination"]) == {1, 2}
    assert set(cdf.value[cdf.value["source"] == 1]["destination"]) == {2}
    assert set(cdf.value[cdf.value["source"] == 2]["destination"]) == {0}
    assert set(cdf.value[cdf.value["source"] == 3]["destination"]) == {2}


def test_weighted_directed_networkx_to_cudf_edge_list():
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
    r = mg.resolver
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
    networkx_graph = r.wrapper.Graph.NetworkXGraph(
        networkx_graph_unwrapped, weight_label="weight"
    )
    cdf = r.translate(networkx_graph, r.types.Graph.CuDFEdgeListType)
    assert len(cdf.value) == 5
    assert set(cdf.value[cdf.value["source"] == 0]["destination"]) == {1, 2}
    assert set(cdf.value[cdf.value["source"] == 0]["weight"]) == {9, 8}
    assert set(cdf.value[cdf.value["source"] == 1]["destination"]) == {2}
    assert set(cdf.value[cdf.value["source"] == 1]["weight"]) == {6}
    assert set(cdf.value[cdf.value["source"] == 2]["destination"]) == {0}
    assert set(cdf.value[cdf.value["source"] == 2]["weight"]) == {7}
    assert set(cdf.value[cdf.value["source"] == 3]["destination"]) == {2}
    assert set(cdf.value[cdf.value["source"] == 3]["weight"]) == {5}


def test_unweighted_directed_edge_list_cugraph_to_cudf_edge_list():
    """
0 < -   1       5   - > 6
      ^       ^ ^       
|   /   |   /   |   /    
v       v /       v      
3   - > 4 < -   2   - > 7
    """
    r = mg.resolver
    sources = [0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    destinations = [3, 0, 4, 4, 5, 7, 1, 4, 5, 6, 2]
    nodes = set(sources + destinations)
    gdf = cudf.DataFrame({"source": sources, "dst": destinations})
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_edgelist(gdf, source="source", destination="dst")
    cugraph_graph = r.wrapper.Graph.CuGraph(cugraph_graph_unwrapped)
    cdf = r.translate(cugraph_graph, r.types.Graph.CuDFEdgeListType)
    assert len(cdf.value) == 11
    assert set(cdf.value[cdf.value["src"] == 0]["dst"]) == {3}
    assert set(cdf.value[cdf.value["src"] == 1]["dst"]) == {0, 4}
    assert set(cdf.value[cdf.value["src"] == 2]["dst"]) == {4, 5, 7}
    assert set(cdf.value[cdf.value["src"] == 3]["dst"]) == {1, 4}
    assert set(cdf.value[cdf.value["src"] == 4]["dst"]) == {5}
    assert set(cdf.value[cdf.value["src"] == 5]["dst"]) == {6}
    assert set(cdf.value[cdf.value["src"] == 6]["dst"]) == {2}
    assert set(cdf.value[cdf.value["src"] == 7]["dst"]) == set()


def test_weighted_directed_edge_list_cugraph_to_cudf_edge_list():
    """
0 <--2-- 1        5 --10-> 6
|      ^ |      ^ ^      / 
|     /  |     /  |     /   
1    7   3    9   5   11   
|   /    |  /     |   /    
v        v /        v      
3 --8--> 4 <--4-- 2 --6--> 7
    """
    r = mg.resolver
    sources = [0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    destinations = [3, 0, 4, 4, 5, 7, 1, 4, 5, 6, 2]
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    nodes = set(sources + destinations)
    gdf = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_edgelist(
        gdf, source="source", destination="destination", edge_attr="weight"
    )
    cugraph_graph = r.wrapper.Graph.CuGraph(cugraph_graph_unwrapped)
    cdf = r.translate(cugraph_graph, r.types.Graph.CuDFEdgeListType)
    assert len(cdf.value) == 11
    assert set(cdf.value[cdf.value["src"] == 0]["dst"]) == {3}
    assert set(cdf.value[cdf.value["src"] == 0]["weights"]) == {1}
    assert set(cdf.value[cdf.value["src"] == 1]["dst"]) == {0, 4}
    assert set(cdf.value[cdf.value["src"] == 1]["weights"]) == {2, 3}
    assert set(cdf.value[cdf.value["src"] == 2]["dst"]) == {4, 5, 7}
    assert set(cdf.value[cdf.value["src"] == 2]["weights"]) == {4, 5, 6}
    assert set(cdf.value[cdf.value["src"] == 3]["dst"]) == {1, 4}
    assert set(cdf.value[cdf.value["src"] == 3]["weights"]) == {7, 8}
    assert set(cdf.value[cdf.value["src"] == 4]["dst"]) == {5}
    assert set(cdf.value[cdf.value["src"] == 4]["weights"]) == {9}
    assert set(cdf.value[cdf.value["src"] == 5]["dst"]) == {6}
    assert set(cdf.value[cdf.value["src"] == 5]["weights"]) == {10}
    assert set(cdf.value[cdf.value["src"] == 6]["dst"]) == {2}
    assert set(cdf.value[cdf.value["src"] == 6]["weights"]) == {11}
    assert set(cdf.value[cdf.value["src"] == 7]["dst"]) == set()


def test_unweighted_directed_adjacency_list_cugraph_to_cudf_edge_list():
    """
0 -----> 1 
^^       | 
| \_     | 
|   \_   |
|     \  | 
|      \ v 
2 <----- 3 
    """
    r = mg.resolver
    sparse_matrix = scipy.sparse.csr_matrix(
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]]),
        dtype=np.int8,
    )
    offsets = cudf.Series(sparse_matrix.indptr)
    indices = cudf.Series(sparse_matrix.indices)
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_adjlist(offsets, indices, None)
    cugraph_graph = r.wrapper.Graph.CuGraph(cugraph_graph_unwrapped)
    cdf = r.translate(cugraph_graph, r.types.Graph.CuDFEdgeListType)
    assert len(cdf.value) == 5
    assert set(cdf.value[cdf.value["src"] == 0]["dst"]) == {1}
    assert set(cdf.value[cdf.value["src"] == 1]["dst"]) == {3}
    assert set(cdf.value[cdf.value["src"] == 2]["dst"]) == {0}
    assert set(cdf.value[cdf.value["src"] == 3]["dst"]) == {0, 2}


def test_weighted_directed_adjacency_list_cugraph_to_cudf_edge_list():
    """
0 -1.1-> 1 
^^       | 
| \     2.2 
|  4.4   |
3.3   \  | 
|      \ v 
2 <-5.5- 3 
    """
    r = mg.resolver
    sparse_matrix = scipy.sparse.csr_matrix(
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]]),
        dtype=np.int8,
    )
    offsets = cudf.Series(sparse_matrix.indptr)
    indices = cudf.Series(sparse_matrix.indices)
    weights = cudf.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_adjlist(offsets, indices, weights)
    cugraph_graph = r.wrapper.Graph.CuGraph(cugraph_graph_unwrapped)
    cdf = r.translate(cugraph_graph, r.types.Graph.CuDFEdgeListType)
    assert len(cdf.value) == 5
    assert set(cdf.value[cdf.value["src"] == 0]["dst"]) == {1}
    assert set(cdf.value[cdf.value["src"] == 0]["weights"]) == {1.1}
    assert set(cdf.value[cdf.value["src"] == 1]["dst"]) == {3}
    assert set(cdf.value[cdf.value["src"] == 1]["weights"]) == {2.2}
    assert set(cdf.value[cdf.value["src"] == 2]["dst"]) == {0}
    assert set(cdf.value[cdf.value["src"] == 2]["weights"]) == {3.3}
    assert set(cdf.value[cdf.value["src"] == 3]["dst"]) == {0, 2}
    assert set(cdf.value[cdf.value["src"] == 3]["weights"]) == {4.4, 5.5}


def test_numpy_nodes_to_cudf_nodes():
    r = mg.resolver
    numpy_data = np.array([33, 22, 11])
    numpy_nodes = r.wrapper.Nodes.NumpyNodes(numpy_data)
    cudf_nodes = r.translate(numpy_nodes, r.types.Nodes.CuDFNodesType)
    assert len(cudf_nodes.value) == 3
    for k, v in enumerate(numpy_data):
        assert cudf_nodes[k] == v


def test_python_nodes_to_cudf_nodes():
    r = mg.resolver
    python_data = {1: 11, 2: 22, 3: 33}
    python_nodes = r.wrapper.Nodes.PythonNodes(python_data)
    cudf_nodes = r.translate(python_nodes, r.types.Nodes.CuDFNodesType)
    assert len(cudf_nodes.value) == 3
    for k, v in python_data.items():
        assert cudf_nodes[k] == v
