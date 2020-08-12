import metagraph as mg
import scipy.sparse as ss
import pandas as pd
import numpy as np
import cudf
import io
from metagraph.plugins.pandas.types import PandasEdgeSet
from metagraph.plugins.python.types import PythonNodeSet, PythonNodeMap
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap
from metagraph.plugins.scipy.types import ScipyEdgeSet, ScipyEdgeMap


def test_cudf_edge_set_to_scipy_edge_set():
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
    cdf_unwrapped = cudf.read_csv(csv_file)
    x = dpr.wrappers.EdgeSet.CuDFEdgeSet(cdf_unwrapped, "Source", "Destination")

    scipy_sparse_matrix = ss.csr_matrix(
        np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0],])
    )
    intermediate = ScipyEdgeSet(scipy_sparse_matrix)
    y = dpr.translate(x, ScipyEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, ScipyEdgeSet) == 1


def test_cudf_edge_map_to_scipy_edge_map():
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

    scipy_sparse_matrix = ss.csr_matrix(
        np.array([[0, 9, 7, 0], [0, 0, 6, 0], [8, 0, 0, 0], [0, 0, 5, 0],])
    )
    intermediate = ScipyEdgeMap(scipy_sparse_matrix)
    y = dpr.translate(x, ScipyEdgeMap)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, ScipyEdgeMap) == 1


def test_cudf_edge_set_to_pandas_edge_set():
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
    cdf_unwrapped = cudf.read_csv(csv_file)
    x = dpr.wrappers.EdgeSet.CuDFEdgeSet(
        cdf_unwrapped, "Source", "Destination", is_directed=True
    )

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    pdf = pd.DataFrame(
        {"source": sources, "destination": destinations},
        columns=["source", "destination"],
    )
    intermediate = PandasEdgeSet(pdf, "source", "destination")
    y = dpr.translate(x, PandasEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, PandasEdgeSet) == 1


def test_cudf_node_map_to_python_node_map():
    dpr = mg.resolver
    keys = [3, 2, 1]
    values = [33, 22, 11]
    cudf_data = cudf.DataFrame({"key": keys, "val": values}).set_index("key")
    x = dpr.wrappers.NodeMap.CuDFNodeMap(cudf_data, "val")

    python_dict = {k: v for k, v in zip(keys, values)}
    intermediate = PythonNodeMap(python_dict)
    y = dpr.translate(x, PythonNodeMap)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, PythonNodeMap) == 1


def test_fully_dense_cudf_node_map_to_numpy_node_map():
    dpr = mg.resolver
    keys = [3, 2, 1, 0]
    values = [33, 22, 11, 00]
    cudf_data = cudf.DataFrame({"key": keys, "val": values}).set_index("key")
    x = dpr.wrappers.NodeMap.CuDFNodeMap(cudf_data, "val")

    intermediate = NumpyNodeMap(np.array([00, 11, 22, 33], dtype=int))
    y = dpr.translate(x, NumpyNodeMap)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, NumpyNodeMap) == 1


def test_mostly_dense_cudf_node_map_to_numpy_node_map():
    dpr = mg.resolver
    keys = [3, 2, 1]
    values = [33, 22, 11]
    cudf_data = cudf.DataFrame({"key": keys, "val": values}).set_index("key")
    x = dpr.wrappers.NodeMap.CuDFNodeMap(cudf_data, "val")

    intermediate = NumpyNodeMap(
        np.array([987654321, 11, 22, 33], dtype=int),
        mask=np.array([0, 1, 1, 1], dtype=bool),
    )
    y = dpr.translate(x, NumpyNodeMap)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, NumpyNodeMap) == 1


def test_sparse_cudf_node_map_to_numpy_node_map():
    dpr = mg.resolver
    keys = [400, 300]
    values = [4, 3]
    cudf_data = cudf.DataFrame({"key": keys, "val": values}).set_index("key")
    x = dpr.wrappers.NodeMap.CuDFNodeMap(cudf_data, "val")

    intermediate = NumpyNodeMap(np.array([3, 4], dtype=int), node_ids={300: 0, 400: 1})
    y = dpr.translate(x, NumpyNodeMap)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, NumpyNodeMap) == 1


def test_cudf_node_set_to_python_node_set():
    dpr = mg.resolver
    cudf_node_set = dpr.wrappers.NodeSet.CuDFNodeSet(cudf.Series([2, 3, 4, 1]))
    x = dpr.translate(cudf_node_set, dpr.types.NodeSet.CuDFNodeSetType)

    intermediate = dpr.wrappers.NodeSet.PythonNodeSet({3, 4, 2, 1})
    y = dpr.translate(x, PythonNodeSet)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, PythonNodeSet) == 1


def test_dense_cudf_node_set_to_numpy_node_set():
    dpr = mg.resolver
    numpy_nodes = dpr.wrappers.NodeSet.CuDFNodeSet(cudf.Series([2, 3, 4, 1]))
    x = dpr.translate(numpy_nodes, dpr.types.NodeSet.CuDFNodeSetType)

    intermediate = dpr.wrappers.NodeSet.NumpyNodeSet(
        mask=np.array([0, 1, 1, 1, 1], dtype=bool)
    )
    y = dpr.translate(x, NumpyNodeSet)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, NumpyNodeSet) == 1


def test_sparse_cudf_node_set_to_numpy_node_set():
    dpr = mg.resolver
    numpy_nodes = dpr.wrappers.NodeSet.CuDFNodeSet(cudf.Series([200, 300, 400, 100]))
    x = dpr.translate(numpy_nodes, dpr.types.NodeSet.CuDFNodeSetType)

    intermediate = dpr.wrappers.NodeSet.NumpyNodeSet(node_ids={200, 300, 400, 100})
    y = dpr.translate(x, NumpyNodeSet)
    dpr.assert_equal(y, intermediate)
    assert dpr.plan.num_translations(x, NumpyNodeSet) == 1
