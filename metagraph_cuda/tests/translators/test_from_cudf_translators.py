import metagraph as mg
import pandas as pd
import cugraph
import cudf
import io
from metagraph.plugins.pandas.types import PandasEdgeList
from metagraph.plugins.python.types import PythonNodes


def test_cudf_edge_list_to_pandas_edge_list():
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
    x = dpr.wrappers.Graph.CuDFEdgeList(cdf_unwrapped, "Source", "Destination")

    sources = [0, 0, 1, 2, 3]
    destinations = [1, 2, 2, 0, 2]
    pdf = pd.DataFrame(
        {"source": sources, "destination": destinations},
        columns=["source", "destination"],
    )
    intermediate = PandasEdgeList(pdf, "source", "destination")
    y = dpr.translate(x, PandasEdgeList)
    PandasEdgeList.Type.assert_equal(y, intermediate)


def test_cudf_nodes_to_python_nodes():
    dpr = mg.resolver
    keys = [3, 2, 1]
    values = [33, 22, 11]
    cudf_data = cudf.DataFrame({"key": keys, "val": values})
    x = dpr.wrappers.Nodes.CuDFNodes(cudf_data, "key", "val")

    python_dict = {k: v for k, v in zip(keys, values)}
    intermediate = PythonNodes(python_dict)
    y = dpr.translate(x, PythonNodes)
    PythonNodes.Type.assert_equal(y, intermediate)
