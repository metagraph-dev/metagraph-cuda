import metagraph as mg
import networkx as nx
import numpy as np
import cugraph
import cudf
import io


def test_pagerank_on_cugraph_digraph():
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
    # Get Expected networkx Results
    networkx_graph_data = [
        (0, 1),
        (0, 2),
        (2, 0),
        (1, 2),
        (3, 2),
    ]
    networkx_graph = nx.DiGraph()
    networkx_graph.add_edges_from(networkx_graph_data)
    networkx_pagerank = nx.pagerank(networkx_graph)
    expected_pagerank_results = np.empty(4)
    for node, rank in networkx_pagerank.items():
        expected_pagerank_results[node] = rank
    # Load Graph Data
    csv_data = "\n".join(
        [f"{source},{destination}" for source, destination in networkx_graph_data]
    )
    csv_file = io.StringIO(csv_data)
    gdf = cudf.read_csv(
        csv_file, names=["Source", "Destination"], dtype=["int32", "int32"]
    )
    g = cugraph.DiGraph()
    g.from_cudf_edgelist(gdf, source="Source", destination="Destination")
    # Verify Graph Data
    assert g.number_of_vertices() == 4
    assert g.number_of_edges() == 5
    # Verify Resolver Support For Graph Type
    g_type = type(g)
    assert g_type == cugraph.DiGraph
    assert g_type in r.class_to_concrete
    g_concrete_type = r.class_to_concrete[g_type]
    assert g_concrete_type in r.concrete_types
    # Verify Algorithm Presence
    assert r.find_algorithm_exact("link_analysis.pagerank", g)
    # Verify PageRank Result Type & Result
    rankings = r.algo.link_analysis.pagerank(g)
    assert isinstance(rankings.value, np.ndarray)
    assert np.std(rankings.value - expected_pagerank_results) < 1e-4
