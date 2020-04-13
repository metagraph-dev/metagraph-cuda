import metagraph as mg
import pandas as pd
import cugraph
import cudf
import io


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
    cdf_unwrapped = cudf.read_csv(csv_file)
    cdf = r.wrapper.Graph.CuDFEdgeList(cdf_unwrapped, "Source", "Destination")
    pdf = r.translate(cdf, r.types.Graph.PandasEdgeListType)
    assert len(pdf.value) == len(pdf.value)
    assert len(pdf.value) == 5
    assert set(pdf.value[pdf.value["Source"] == 0]["Destination"]) == set(
        pdf.value[pdf.value["Source"] == 0]["Destination"]
    )
    assert set(pdf.value[pdf.value["Source"] == 0]["Destination"]) == {1, 2}
    assert set(pdf.value[pdf.value["Source"] == 1]["Destination"]) == set(
        pdf.value[pdf.value["Source"] == 1]["Destination"]
    )
    assert set(pdf.value[pdf.value["Source"] == 1]["Destination"]) == {2}
    assert set(pdf.value[pdf.value["Source"] == 2]["Destination"]) == set(
        pdf.value[pdf.value["Source"] == 2]["Destination"]
    )
    assert set(pdf.value[pdf.value["Source"] == 2]["Destination"]) == {0}
    assert set(pdf.value[pdf.value["Source"] == 3]["Destination"]) == set(
        pdf.value[pdf.value["Source"] == 3]["Destination"]
    )
    assert set(pdf.value[pdf.value["Source"] == 3]["Destination"]) == {2}
