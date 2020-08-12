import metagraph as mg
import scipy.sparse as ss
import pandas as pd
import networkx as nx
import numpy as np
import scipy
import cudf
import cugraph
import io
from metagraph_cuda.types import CuDFNodeMap, CuDFNodeSet, CuDFEdgeSet, CuDFEdgeMap


def test_cudf_node_map_to_cudf_node_set():
    dpr = mg.resolver
    keys = [3, 2, 1]
    values = [33, 22, 11]
    map_data = cudf.DataFrame({"key": keys, "val": values}).set_index("key")
    x = dpr.wrappers.NodeMap.CuDFNodeMap(map_data, "val")

    nodes = cudf.Series([3, 1, 2])
    intermediate = dpr.wrappers.NodeSet.CuDFNodeSet(nodes)

    y = dpr.translate(x, CuDFNodeSet)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, CuDFNodeSet) == 1


def test_cudf_edge_map_to_cudf_edge_set():
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
    cdf_weighted = cudf.DataFrame(
        {"Source": sources, "Destination": destinations, "Weight": weights}
    )
    x = dpr.wrappers.EdgeMap.CuDFEdgeMap(
        cdf_weighted, "Source", "Destination", "Weight"
    )

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    cdf_unweighted = cudf.DataFrame({"Source": sources, "Destination": destinations})
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unweighted, "Source", "Destination"
    )

    y = dpr.translate(x, CuDFEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, CuDFEdgeSet) == 1


def test_scipy_edge_set_to_cudf_edge_set():
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
    scipy_sparse_matrix = ss.csr_matrix(
        np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0],])
    )
    x = dpr.wrappers.EdgeSet.ScipyEdgeSet(scipy_sparse_matrix)

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    cdf_unwrapped = cudf.DataFrame({"Source": sources, "Destination": destinations})
    intermediate = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "Source", "Destination"
    )

    y = dpr.translate(x, CuDFEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, CuDFEdgeSet) == 1


def test_scipy_edge_map_to_cudf_edge_map():
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
    scipy_sparse_matrix = ss.csr_matrix(
        np.array([[0, 9, 7, 0], [0, 0, 6, 0], [8, 0, 0, 0], [0, 0, 5, 0],])
    )
    x = dpr.wrappers.EdgeMap.ScipyEdgeMap(scipy_sparse_matrix)

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
    assert dpr.plan.num_translations(x, CuDFEdgeMap) == 1


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
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, CuDFEdgeSet) == 1


def test_numpy_node_map_to_cudf_node_map():
    dpr = mg.resolver
    numpy_data = np.array([33, 22, 11])
    x = dpr.wrappers.NodeMap.NumpyNodeMap(numpy_data)

    keys = [0, 1, 2]
    labels = [33, 22, 11]
    cdf_unwrapped = cudf.DataFrame({"key": keys, "label": labels}).set_index("key")
    intermediate = dpr.wrappers.NodeMap.CuDFNodeMap(cdf_unwrapped, "label")
    y = dpr.translate(x, CuDFNodeMap)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, CuDFNodeMap) == 1


def test_python_node_map_to_cudf_node_map():
    dpr = mg.resolver
    python_data = {1: 11, 2: 22, 3: 33}
    x = dpr.wrappers.NodeMap.PythonNodeMap(python_data)

    keys = [1, 2, 3]
    labels = [11, 22, 33]
    cdf_unwrapped = cudf.DataFrame({"key": keys, "label": labels}).set_index("key")
    intermediate = dpr.wrappers.NodeMap.CuDFNodeMap(cdf_unwrapped, "label")
    y = dpr.translate(x, CuDFNodeMap)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, CuDFNodeMap) == 1


def test_python_node_set_to_cudf_node_set():
    dpr = mg.resolver
    python_data = {3, 4, 2, 1}
    x = dpr.wrappers.NodeSet.PythonNodeSet(python_data)

    intermediate = dpr.wrappers.NodeSet.CuDFNodeSet(cudf.Series([2, 3, 4, 1]))
    y = dpr.translate(x, CuDFNodeSet)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, CuDFNodeSet) == 1
