import metagraph as mg
import cugraph
import cudf
import io

# Simple graph with 5 triangles
# 0 - 1    5 - 6
# | X |    | /
# 3 - 4 -- 2 - 7

graph_csv_data = """
0,1
1,0
0,3
3,0
0,4
4,0
1,3
3,1
1,4
4,1
2,4
4,2
2,5
5,2
2,6
6,2
2,7
7,2
3,4
4,3
5,6
6,5
"""


def test_triangle_count_on_cugraph_digraph():
    r = mg.resolver
    # Load Graph Data
    csv_file = io.StringIO(graph_csv_data)
    gdf = cudf.read_csv(
        csv_file, names=["Source", "Destination"], dtype=["int32", "int32"]
    )
    cugraph_graph = cugraph.Graph()
    cugraph_graph.from_cudf_edgelist(gdf, source="Source", destination="Destination")
    # Verify Algorithm Presence
    g = r.wrapper.Graph.CuGraph(cugraph_graph)
    assert r.find_algorithm_exact("cluster.triangle_count", g)
    # Verify Triangle Count Result Type
    number_of_triangles = r.algo.cluster.triangle_count(g)
    assert number_of_triangles == 5
    assert isinstance(number_of_triangles, int)
