import metagraph as mg
import networkx as nx
import scipy
import numpy as np
import cudf
import cugraph


def test_unweighted_directed_edge_list_cugraph_to_nextworkx():
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
    networkx_graph = r.translate(cugraph_graph, r.types.Graph.NetworkXGraphType)
    assert networkx_graph.value.number_of_edges() == 11
    assert networkx_graph.value.number_of_nodes() == 8
    assert networkx_graph.value.is_directed()
    assert set(networkx_graph.value.neighbors(0)) == {3}
    assert set(networkx_graph.value.neighbors(1)) == {0, 4}
    assert set(networkx_graph.value.neighbors(2)) == {4, 5, 7}
    assert set(networkx_graph.value.neighbors(3)) == {1, 4}
    assert set(networkx_graph.value.neighbors(4)) == {5}
    assert set(networkx_graph.value.neighbors(5)) == {6}
    assert set(networkx_graph.value.neighbors(6)) == {2}
    assert set(networkx_graph.value.neighbors(7)) == set()


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
    networkx_graph = r.translate(cugraph_graph, r.types.Graph.NetworkXGraphType)
    assert networkx_graph.value.number_of_edges() == 11
    assert networkx_graph.value.number_of_nodes() == 8
    assert networkx_graph.value.is_directed()
    assert set(networkx_graph.value.neighbors(0)) == {3}
    assert networkx_graph.value[0][3]["weight"] == 1
    assert set(networkx_graph.value.neighbors(1)) == {0, 4}
    assert networkx_graph.value[1][0]["weight"] == 2
    assert networkx_graph.value[1][4]["weight"] == 3
    assert set(networkx_graph.value.neighbors(2)) == {4, 5, 7}
    assert networkx_graph.value[2][4]["weight"] == 4
    assert networkx_graph.value[2][5]["weight"] == 5
    assert networkx_graph.value[2][7]["weight"] == 6
    assert set(networkx_graph.value.neighbors(3)) == {1, 4}
    assert networkx_graph.value[3][1]["weight"] == 7
    assert networkx_graph.value[3][4]["weight"] == 8
    assert set(networkx_graph.value.neighbors(4)) == {5}
    assert networkx_graph.value[4][5]["weight"] == 9
    assert set(networkx_graph.value.neighbors(5)) == {6}
    assert networkx_graph.value[5][6]["weight"] == 10
    assert set(networkx_graph.value.neighbors(6)) == {2}
    assert networkx_graph.value[6][2]["weight"] == 11
    assert set(networkx_graph.value.neighbors(7)) == set()


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
    networkx_graph = r.translate(cugraph_graph, r.types.Graph.NetworkXGraphType)
    assert networkx_graph.value.number_of_edges() == 5
    assert networkx_graph.value.number_of_nodes() == 4
    assert set(networkx_graph.value.neighbors(0)) == {1}
    assert set(networkx_graph.value.neighbors(1)) == {3}
    assert set(networkx_graph.value.neighbors(2)) == {0}
    assert set(networkx_graph.value.neighbors(3)) == {0, 2}


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
    networkx_graph = r.translate(cugraph_graph, r.types.Graph.NetworkXGraphType)
    assert networkx_graph.value.number_of_edges() == 5
    assert networkx_graph.value.number_of_nodes() == 4
    assert set(networkx_graph.value.neighbors(0)) == {1}
    assert networkx_graph.value[0][1]["weight"] == 1.1
    assert set(networkx_graph.value.neighbors(1)) == {3}
    assert networkx_graph.value[1][3]["weight"] == 2.2
    assert set(networkx_graph.value.neighbors(2)) == {0}
    assert networkx_graph.value[2][0]["weight"] == 3.3
    assert set(networkx_graph.value.neighbors(3)) == {0, 2}
    assert networkx_graph.value[3][0]["weight"] == 4.4
    assert networkx_graph.value[3][2]["weight"] == 5.5
