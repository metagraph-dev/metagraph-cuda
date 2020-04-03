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
    cugraph_digraph = cugraph.DiGraph()
    cugraph_digraph.from_cudf_edgelist(gdf, source="Source", destination="Destination")
    # Verify Graph Data
    assert cugraph_digraph.number_of_vertices() == 4
    assert cugraph_digraph.number_of_edges() == 5
    # Verify Algorithm Presence
    g = r.wrapper.Graph.CuGraph(cugraph_digraph)
    assert r.find_algorithm("link_analysis.pagerank", g)
    # Verify PageRank Result Type & Result
    rankings = r.algo.link_analysis.pagerank(g)
    assert isinstance(rankings.value, np.ndarray)
    assert np.std(rankings.value - expected_pagerank_results) < 1e-4
