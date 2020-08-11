import metagraph as mg
import pandas as pd
import cugraph
import cudf
import io
from metagraph.plugins.pandas.types import PandasEdgeSet
from metagraph.plugins.python.types import PythonNodeMap


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
