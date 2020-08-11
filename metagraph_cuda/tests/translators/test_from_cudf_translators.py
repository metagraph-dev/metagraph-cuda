import metagraph as mg
import pandas as pd
import numpy as np
import cugraph
import cudf
import io
from metagraph.plugins.pandas.types import PandasEdgeSet
from metagraph.plugins.python.types import PythonNodeSet, PythonNodeMap
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap


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
    x = dpr.wrappers.EdgeSet.CuDFEdgeSet(cdf_unwrapped, "Source", "Destination")

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    pdf = pd.DataFrame(
        {"source": sources, "destination": destinations},
        columns=["source", "destination"],
    )
    intermediate = PandasEdgeSet(pdf, "source", "destination")
    y = dpr.translate(x, PandasEdgeSet)
    dpr.assert_equal(y, intermediate)


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


def test_fully_dense_cudf_node_map_to_numpy_node_map():
    dpr = mg.resolver
    keys = [3, 2, 1, 0]
    values = [33, 22, 11, 00]
    cudf_data = cudf.DataFrame({"key": keys, "val": values}).set_index("key")
    x = dpr.wrappers.NodeMap.CuDFNodeMap(cudf_data, "val")

    intermediate = NumpyNodeMap(np.array([00, 11, 22, 33], dtype=int))
    y = dpr.translate(x, NumpyNodeMap)
    dpr.assert_equal(y, intermediate)


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


def test_sparse_cudf_node_map_to_numpy_node_map():
    dpr = mg.resolver
    keys = [400, 300]
    values = [4, 3]
    cudf_data = cudf.DataFrame({"key": keys, "val": values}).set_index("key")
    x = dpr.wrappers.NodeMap.CuDFNodeMap(cudf_data, "val")

    intermediate = NumpyNodeMap(np.array([3, 4], dtype=int), node_ids={300: 0, 400: 1})
    y = dpr.translate(x, NumpyNodeMap)
    dpr.assert_equal(y, intermediate)


def test_cudf_node_set_to_python_node_set():
    dpr = mg.resolver
    cudf_node_set = dpr.wrappers.NodeSet.CuDFNodeSet(cudf.Series([2, 3, 4, 1]))
    x = dpr.translate(cudf_node_set, dpr.types.NodeSet.CuDFNodeSetType)

    intermediate = dpr.wrappers.NodeSet.PythonNodeSet({3, 4, 2, 1})
    y = dpr.translate(x, PythonNodeSet)
    dpr.assert_equal(y, intermediate)


def test_dense_cudf_node_set_to_numpy_node_set():
    dpr = mg.resolver
    numpy_nodes = dpr.wrappers.NodeSet.CuDFNodeSet(cudf.Series([2, 3, 4, 1]))
    x = dpr.translate(numpy_nodes, dpr.types.NodeSet.CuDFNodeSetType)

    intermediate = dpr.wrappers.NodeSet.NumpyNodeSet(
        mask=np.array([0, 1, 1, 1, 1], dtype=bool)
    )
    y = dpr.translate(x, NumpyNodeSet)
    dpr.assert_equal(y, intermediate)


def test_sparse_cudf_node_set_to_numpy_node_set():
    dpr = mg.resolver
    numpy_nodes = dpr.wrappers.NodeSet.CuDFNodeSet(cudf.Series([200, 300, 400, 100]))
    x = dpr.translate(numpy_nodes, dpr.types.NodeSet.CuDFNodeSetType)

    intermediate = dpr.wrappers.NodeSet.NumpyNodeSet(node_ids={200, 300, 400, 100})
    y = dpr.translate(x, NumpyNodeSet)
    dpr.assert_equal(y, intermediate)
