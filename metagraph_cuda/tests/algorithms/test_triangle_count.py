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
    csv_file = io.StringIO(data)
    gdf = cudf.read_csv(
        csv_file, names=["Source", "Destination"], dtype=["int32", "int32"]
    )
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(gdf, source="Source", destination="Destination")
    # Verify Graph Data
    assert g.number_of_vertices() == 8
    assert g.number_of_edges() == 22
    # Verify Resolver Support For Graph Type
    g_type = type(g)
    assert g_type == cugraph.DiGraph
    assert g_type in r.class_to_concrete
    g_concrete_type = r.class_to_concrete[g_type]
    assert g_concrete_type in r.concrete_types
    # Verify Algorithm Presence
    assert r.find_algorithm_exact("cluster.triangle_count", g)
    # Verify Triangle Count Result Type
    number_of_triangles = r.algo.cluster.triangle_count(g)
    assert isinstance(number_of_triangles, int)
