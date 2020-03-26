import metagraph as mg
import cugraph
import cudf
import io
import numpy as np


def test_triangle_count_on_cugraph_digraph_via_trivial_triangle_graph():
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


def test_triangle_count_on_cugraph_digraph_via_fully_connected_graph():
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
    # n * (n-1) / 2 * two_edges_per_direction
    assert g.number_of_edges() == number_of_nodes * (number_of_nodes - 1)
    # Test Triangle Count
    assert r.algo.cluster.triangle_count(g) == 4060  # 4060 == 30 choose 3


def test_triangle_count_on_cugraph_digraph_via_wheel_graph():
    r = mg.resolver
    # Generate & Load Graph Data
    for number_of_nodes in [10, 100, 1_000, 10_000, 100_000]:
        hub_node = 0
        sources = []
        destinations = []
        all_nodes = range(number_of_nodes)
        outer_nodes = filter(lambda node: node != hub_node, all_nodes)
        for outer_node in outer_nodes:
            neighbor_node = outer_node - 1
            if neighbor_node == hub_node:
                neighbor_node = (hub_node - 1) % number_of_nodes
            sources.append(outer_node)
            destinations.append(hub_node)
            sources.append(hub_node)
            destinations.append(outer_node)
            sources.append(outer_node)
            destinations.append(neighbor_node)
            sources.append(neighbor_node)
            destinations.append(outer_node)
        g = cugraph.DiGraph()
        gdf = cudf.DataFrame({"Source": sources, "Destination": destinations})
        g.from_cudf_edgelist(gdf, source="Source", destination="Destination")
        # Validate Graph Data
        assert g.number_of_vertices() == number_of_nodes
        assert g.number_of_edges() == (number_of_nodes - 1) * 4
        # Test Triangle Count
        assert r.algo.cluster.triangle_count(g) == (number_of_nodes - 1)
