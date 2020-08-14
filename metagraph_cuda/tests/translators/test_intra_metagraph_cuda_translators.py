import metagraph as mg
import scipy.sparse as ss
import numpy as np
import cudf
import cugraph
from metagraph_cuda.types import CuGraphEdgeMap, CuDFEdgeSet, CuDFEdgeMap


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
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuDFEdgeSet)) == 1


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
    x = dpr.wrappers.EdgeSet.CuGraphEdgeSet(cugraph_graph_unwrapped)

    cdf_unwrapped = cudf.DataFrame(
        {"source": sources, "destination": destinations, "weight": weights}
    )
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "source", "destination"
    )
    y = dpr.translate(x, CuDFEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuDFEdgeSet)) == 1


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
    sparse_matrix = ss.csr_matrix(
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 1, 0]]),
        dtype=np.int8,
    )
    offsets = cudf.Series(sparse_matrix.indptr)
    indices = cudf.Series(sparse_matrix.indices)
    cugraph_graph_unwrapped = cugraph.DiGraph()
    cugraph_graph_unwrapped.from_cudf_adjlist(offsets, indices, None)
    x = dpr.wrappers.EdgeSet.CuGraphEdgeSet(cugraph_graph_unwrapped)

    sources = [0, 1, 2, 3, 3]
    destinations = [1, 3, 0, 2, 0]
    cdf_unwrapped = cudf.DataFrame({"source": sources, "destination": destinations})
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "source", "destination"
    )
    y = dpr.translate(x, CuDFEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuDFEdgeSet)) == 1


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
    sparse_matrix = ss.csr_matrix(
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
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuDFEdgeSet)) == 1


def test_cudf_edge_map_to_cugraph_edge_map():
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
    cdf_unwrapped = cudf.DataFrame(
        {"Source": sources, "Destination": destinations, "Weight": weights}
    )
    x = dpr.wrappers.EdgeMap.CuDFEdgeMap(
        cdf_unwrapped, "Source", "Destination", "Weight"
    )

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    weights = [9, 7, 6, 8, 5]
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


def test_cugraph_edge_map_to_cudf_edge_map():
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
        {"source": sources, "destination": destinations, "weights": weights}
    )
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(
        cdf, source="source", destination="destination", edge_attr="weights"
    )
    x = dpr.wrappers.EdgeMap.CuGraphEdgeMap(g)

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    weights = [9, 7, 6, 8, 5]
    cdf_unwrapped = cudf.DataFrame(
        {"Source": sources, "Destination": destinations, "Weight": weights}
    )
    intermediate = dpr.wrappers.EdgeMap.CuDFEdgeMap(
        cdf_unwrapped, "Source", "Destination", "Weight"
    )

    y = dpr.translate(x, CuDFEdgeMap)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, CuDFEdgeMap)) == 1
