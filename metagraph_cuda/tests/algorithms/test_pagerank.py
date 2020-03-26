import metagraph as mg
import cugraph
import cudf
import io
import numpy as np


def test_pagerank_on_cugraph_digraph_via_tiny_fully_connected_graph():
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
    # Test PageRank
    rankings = r.algo.link_analysis.pagerank(g)
    assert np.std(rankings.value) < 1e-8


def test_pagerank_on_cugraph_digraph_via_fully_connected_graphs():
    r = mg.resolver
    # Generate & Load Graph Data
    for number_of_nodes in [10, 100, 1_000]:
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
        assert g.number_of_edges() == number_of_nodes * (number_of_nodes - 1)
        # Test PageRank
        rankings = r.algo.link_analysis.pagerank(g)
        assert np.std(rankings.value) < 1e-8


def test_pagerank_on_cugraph_digraph_via_star_like_graphs():
    """
    Generated graphs have a hub that all other nodes point to. 
    A fraction of the outer nodes point to a neighbor that is not the hub.
    """
    r = mg.resolver
    hub_node = 0
    fraction_of_nodes_to_point_to_non_hub_node = 0.5
    for number_of_nodes in [10, 100, 1_000]:
        # Generate & Load Graph Data
        sources = []
        destinations = []
        all_nodes = range(number_of_nodes)
        outer_nodes = filter(lambda node: node != hub_node, all_nodes)
        expected_number_of_edges = 0
        for outer_node in outer_nodes:
            sources.append(outer_node)
            destinations.append(hub_node)
            expected_number_of_edges += 1
            if (
                outer_node / number_of_nodes
                > fraction_of_nodes_to_point_to_non_hub_node
            ):
                neighbor_node = outer_node - 1 if outer_node + 1 else outer_node + 1
                sources.append(outer_node)
                destinations.append(neighbor_node)
                expected_number_of_edges += 1
        g = cugraph.DiGraph()
        gdf = cudf.DataFrame({"Source": sources, "Destination": destinations})
        g.from_cudf_edgelist(gdf, source="Source", destination="Destination")
        # Validate Graph Data
        assert g.number_of_vertices() == number_of_nodes
        assert g.number_of_edges() == expected_number_of_edges
        # Test PageRank
        rankings = r.algo.link_analysis.pagerank(g)
        assert hub_node == 0
        assert (rankings.value[hub_node] > rankings.value[hub_node + 1 :]).all()
