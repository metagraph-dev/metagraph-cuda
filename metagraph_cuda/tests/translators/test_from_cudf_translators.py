import metagraph as mg
import scipy.sparse as ss
import pandas as pd
import numpy as np
import cudf
import io
from metagraph.plugins.pandas.types import PandasEdgeSet
from metagraph.plugins.python.types import PythonNodeSetType, PythonNodeMapType
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
    +-+  -->  +-+       +-+"""
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
        np.array(
            [
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
            ]
        )
    )
    intermediate = ScipyEdgeSet(scipy_sparse_matrix)
    y = dpr.translate(x, ScipyEdgeSet)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, ScipyEdgeSet)) == 1


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
    +-+  -7->  +-+        +-+"""
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
        np.array(
            [
                [0, 9, 7, 0],
                [0, 0, 6, 0],
                [8, 0, 0, 0],
                [0, 0, 5, 0],
            ]
        )
    )
    intermediate = ScipyEdgeMap(scipy_sparse_matrix)
    y = dpr.translate(x, ScipyEdgeMap)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, ScipyEdgeMap)) == 1


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
    +-+  -->  +-+       +-+"""
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
    assert len(dpr.plan.translate(x, PandasEdgeSet)) == 1


def test_cudf_node_map_to_python_node_map():
    dpr = mg.resolver
    keys = [3, 2, 1]
    values = [33, 22, 11]
    cudf_data = cudf.Series(values).set_index(keys)
    x = dpr.wrappers.NodeMap.CuDFNodeMap(cudf_data)

    intermediate = {k: v for k, v in zip(keys, values)}
    y = dpr.translate(x, PythonNodeMapType)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, PythonNodeMapType)) == 1


def test_cudf_node_map_to_numpy_node_map():
    dpr = mg.resolver
    keys = [3, 2, 1, 0]
    values = [33, 22, 11, 00]
    cudf_data = cudf.Series(values).set_index(keys)
    x = dpr.wrappers.NodeMap.CuDFNodeMap(cudf_data)

    intermediate = NumpyNodeMap(np.array([00, 11, 22, 33], dtype=int))
    y = dpr.translate(x, NumpyNodeMap)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, NumpyNodeMap)) == 1


def test_cudf_node_set_to_python_node_set():
    dpr = mg.resolver
    cudf_node_set = dpr.wrappers.NodeSet.CuDFNodeSet(cudf.Series([2, 3, 4, 1]))
    x = dpr.translate(cudf_node_set, dpr.types.NodeSet.CuDFNodeSetType)

    intermediate = {3, 4, 2, 1}
    y = dpr.translate(x, PythonNodeSetType)
    dpr.assert_equal(y, intermediate)
    assert len(dpr.plan.translate(x, PythonNodeSetType)) == 1
