import metagraph as mg
import cugraph
import cudf
import io
import numpy as np
import scipy


def test_triangle_count_on_cugraph_digraph_trivial():
    r = mg.resolver
    # Load Graph Data
    data = """
0,1
1,0

1,2
2,1

2,0
0,2
"""
    csv_file = io.StringIO(data)
    gdf = cudf.read_csv(
        csv_file, names=["Source", "Destination"], dtype=["int32", "int32"]
    )
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(gdf, source="Source", destination="Destination")
    # Validate Graph Data
    assert g.number_of_vertices() == 3
    assert g.number_of_edges() == 6
    # Test Triangle Count
    assert r.algo.cluster.triangle_count(g) == 1


def test_triangle_count_on_cugraph_digraph_fully_connected_graph():
    r = mg.resolver
    # Generate & Load Graph Data
    number_of_nodes = 30
    sources = []
    destinations = []
    for node_a in range(number_of_nodes):
        for node_b in range(node_a):
            sources.append(node_a)
            destinations.append(node_b)
            sources.append(node_b)
            destinations.append(node_a)
    g = cugraph.DiGraph()
    gdf = cudf.DataFrame({"Source": sources, "Destination": destinations})
    g.from_cudf_edgelist(gdf, source="Source", destination="Destination")
    # Validate Graph Data
    assert g.number_of_vertices() == number_of_nodes
    assert g.number_of_edges() == number_of_nodes * (
        number_of_nodes - 1
    )  # n * (n-1) / 2 * two_edges_per_direction
    # Test Triangle Count
    assert r.algo.cluster.triangle_count(g) == 4060  # 4060 == 30 choose 3


def test_triangle_count_on__scipy_adjacency_matrix_trivial():
    r = mg.resolver
    # Load Graph Data
    sparse_matrix = scipy.sparse.csr_matrix(
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), dtype=np.int8
    )
    g = r.wrapper.Graph.ScipyAdjacencyMatrix(sparse_matrix)
    # Test Triangle Count
    assert r.algo.cluster.triangle_count(g) == 1
