import metagraph as mg
import networkx as nx
import scipy
import numpy as np
import cudf
import cugraph
from metagraph_cuda.types import CuGraphEdgeSet, CuGraphEdgeMap
from metagraph.plugins.networkx.types import NetworkXEdgeSet, NetworkXEdgeMap


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
    nodes = set(sources + destinations)
    gdf = cudf.DataFrame({"source": sources, "dst": destinations})
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_edgelist(gdf, source="source", destination="dst")
    x = dpr.wrappers.EdgeSet.CuGraphEdgeSet(cugraph_graph_unwrapped)

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
    intermediate = NetworkXEdgeSet(networkx_graph_unwrapped)
    y = dpr.translate(x, NetworkXEdgeSet)
    NetworkXEdgeSet.Type.assert_equal(y, intermediate, {}, {})


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
    nodes = set(sources + destinations)
    gdf = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_edgelist(
        gdf, source="source", destination="destination", edge_attr="weight"
    )
    x = dpr.wrappers.EdgeMap.CuGraphEdgeMap(cugraph_graph_unwrapped)

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
    intermediate = NetworkXEdgeMap(networkx_graph_unwrapped, weight_label="weight")
    y = dpr.translate(x, NetworkXEdgeMap)
    NetworkXEdgeMap.Type.assert_equal(y, intermediate, {}, {})


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
    sparse_matrix = scipy.sparse.csr_matrix(
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]]),
        dtype=np.int8,
    )
    offsets = cudf.Series(sparse_matrix.indptr)
    indices = cudf.Series(sparse_matrix.indices)
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_adjlist(offsets, indices, None)
    x = dpr.wrappers.EdgeSet.CuGraphEdgeSet(cugraph_graph_unwrapped)

    networkx_graph_data = [
        (0, 1),
        (1, 3),
        (2, 0),
        (3, 0),
        (3, 2),
    ]
    networkx_graph_unwrapped = nx.DiGraph()
    networkx_graph_unwrapped.add_edges_from(networkx_graph_data)
    intermediate = NetworkXEdgeSet(networkx_graph_unwrapped)
    y = dpr.translate(x, NetworkXEdgeSet)
    NetworkXEdgeSet.Type.assert_equal(y, intermediate, {}, {})


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
    sparse_matrix = scipy.sparse.csr_matrix(
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]]),
        dtype=np.int8,
    )
    offsets = cudf.Series(sparse_matrix.indptr)
    indices = cudf.Series(sparse_matrix.indices)
    weights = cudf.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_adjlist(offsets, indices, weights)
    x = dpr.wrappers.EdgeMap.CuGraphEdgeMap(cugraph_graph_unwrapped)

    networkx_graph_data = [
        (0, 1, 1.1),
        (1, 3, 2.2),
        (2, 0, 3.3),
        (3, 0, 4.4),
        (3, 2, 5.5),
    ]
    networkx_graph_unwrapped = nx.DiGraph()
    networkx_graph_unwrapped.add_weighted_edges_from(networkx_graph_data)
    intermediate = NetworkXEdgeMap(networkx_graph_unwrapped, weight_label="weight")
    y = dpr.translate(x, NetworkXEdgeMap)
    NetworkXEdgeMap.Type.assert_equal(y, intermediate, {}, {})
