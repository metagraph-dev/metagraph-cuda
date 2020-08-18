import numpy as np
from metagraph.wrappers import (
    EdgeSetWrapper,
    EdgeMapWrapper,
    CompositeGraphWrapper,
    BipartiteGraphWrapper,
)
from metagraph import dtypes
from metagraph.types import (
    Graph,
    BipartiteGraph,
    EdgeSet,
    EdgeMap,
)
from .. import has_cugraph
from typing import List, Set, Dict, Any

if has_cugraph:
    import cugraph
    import cudf

    from ..cudf.types import CuDFNodeSet, CuDFNodeMap

    class CuGraphEdgeSet(EdgeSetWrapper, abstract=EdgeSet):
        def __init__(self, graph):
            self.value = graph

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: List[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"is_directed"} - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = obj.value.is_directed()

                return ret

            @classmethod
            def assert_equal(
                cls,
                obj1,
                obj2,
                aprops1,
                aprops2,
                cprops1,
                cprops2,
                *,
                rel_tol=None,
                abs_tol=None,
            ):
                assert (
                    aprops1 == aprops2
                ), f"abstract property mismatch: {aprops1} != {aprops2}"
                g1 = obj1.value
                g2 = obj2.value
                # Compare
                assert all(
                    g1.nodes() == g2.nodes()
                ), f"node mismatch: {g1.nodes()} != {g2.nodes()}"
                assert all(
                    g1.edges() == g2.edges()
                ), f"edge mismatch: {g1.edges()} != {g2.edges()}"

    class CuGraphEdgeMap(EdgeMapWrapper, abstract=EdgeMap):
        def __init__(self, graph):
            self.value = graph
            self._assert_instance(graph, cugraph.Graph)

        def _determine_dtype(self, all_values):
            all_types = {type(v) for v in all_values}
            if not all_types or (all_types - {float, int, bool}):
                return "str"
            for type_ in (float, int, bool):
                if type_ in all_types:
                    return str(type_.__name__)

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: List[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"is_directed", "dtype"} - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = obj.value.is_directed()
                    if prop == "dtype":
                        if obj.value.edgelist:
                            obj_dtype = obj.value.view_edge_list().weights.dtype
                        else:
                            obj_dtype = obj.value.view_adj_list()[2].dtype
                        ret[prop] = dtypes.dtypes_simplified[obj_dtype]

                # slow properties, only compute if asked
                slow_props = props - ret.keys()
                if "has_negative_weights" in slow_props:
                    if obj.value.edgelist:
                        weights = obj.value.view_edge_list().weights
                    else:
                        weights = obj.value.view_adj_list()[2]
                    ret["has_negative_weights"] = any(weights < 0)

                return ret

            @classmethod
            def assert_equal(
                cls,
                obj1,
                obj2,
                aprops1,
                aprops2,
                cprops1,
                cprops2,
                *,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                assert (
                    aprops1 == aprops2
                ), f"abstract property mismatch: {aprops1} != {aprops2}"
                g1 = obj1.value
                g2 = obj2.value
                # Compare
                assert (
                    g1.number_of_nodes() == g2.number_of_nodes()
                ), f"{g1.number_of_nodes()} != {g2.number_of_nodes()}"
                assert (
                    g1.number_of_edges() == g2.number_of_edges()
                ), f"{g1.number_of_edges()} != {g2.number_of_edges()}"

                if g1.edgelist:
                    g1_edge_list = g1.view_edge_list()
                    g1_nodes = cudf.concat(
                        [g1_edge_list["src"], g1_edge_list["dst"]]
                    ).unique()
                    g2_edge_list = g2.view_edge_list()
                    g2_nodes = cudf.concat(
                        [g2_edge_list["src"], g2_edge_list["dst"]]
                    ).unique()
                    assert (
                        g1_nodes.isin(g2_nodes).all() and g2_nodes.isin(g1_nodes).all()
                    ), "g1 and g2 have different nodes"
                    assert len(g1_edge_list) == len(
                        g2_edge_list
                    ), f"g1 and g2 have a different number of edges"
                    # TODO the below takes an additional possibly unneeded O(n) memory
                    assert g1_edge_list.set_index(
                        ["src", "dst", "weights"]
                    ) == g2_edge_list.set_index(
                        ["src", "dst", "weights"]
                    ), "g1 and g2 have different edges"
                else:
                    assert (
                        g1.number_of_nodes() == g2.number_of_nodes()
                    ), "g1 and g2 have different nodes"
                    for i, g1_series in enumerate(g1.view_adj_list()):
                        g2_series = g2.view_adj_list()[i]
                        assert (g1_series == None) == (
                            g2_series == None
                        ), "one of g1 or g2 is weighted while the other is not"
                        if g1_series != None:
                            if np.issubdtype(g1_series.dtype.type, np.float):
                                assert cupy.isclose(g1_series == g2_series)
                            else:
                                assert all(
                                    g1_series == g2_series
                                ), "g1 and g2 have different edges"

    class CuGraph(CompositeGraphWrapper, abstract=Graph):
        def __init__(self, edges, nodes=None):
            if isinstance(edges, cugraph.Graph):
                if edges.edgelist:
                    if edges.edgelist.weights:
                        edges = CuGraphEdgeMap(edges)
                    else:
                        edges = CuGraphEdgeSet(edges)
                elif edges.adjlist:
                    if edges.view_adj_list()[-1] is not None:
                        edges = CuGraphEdgeMap(edges)
                    else:
                        edges = CuGraphEdgeSet(edges)
            self._assert_instance(edges, (CuGraphEdgeSet, CuGraphEdgeMap))
            if nodes is not None:
                self._assert_instance(nodes, (CuDFNodeSet, CuDFNodeMap))
            super().__init__(edges, nodes)

    class CuGraphBipartiteGraph(BipartiteGraphWrapper, abstract=BipartiteGraph):
        def __init__(self, graph, nodes):
            """
            :param graph: cugraph.Graph instance
            :param nodes: A 2-tuple of cudf.Series specifying the node partitions
            """
            self._assert_instance(graph, cugraph.Graph)
            self._assert_instance(nodes, tuple)
            self._assert(nodes, tuple)
            self._assert(len(nodes) == 2, "nodes must have length of 2")
            self._assert_instance(nodes[0], CuDF.Series)
            self._assert_instance(nodes[1], CuDF.Series)
            # TODO handle node weights
            self.value = graph
            self.nodes = (nodes[0].unique(), nodes[1].unique())
            self.nodes[0] = self.nodes[0].set_index(self.nodes[0])
            self.nodes[1] = self.nodes[1].set_index(self.nodes[1])
            common_nodes = self.nodes[0][
                self.nodes[0].isin(self.nodes[1])
            ]  # O(n^2), but cheaper than converting to Python sets
            if len(common_nodes) != 0:
                raise ValueError(
                    f"Node IDs found in both parts of the graph: {common_nodes.values.tolist()}"
                )
            partition_nodes = cudf.concat([self.nodes[0], self.nodes[1]])
            unclaimed_nodes_mask = ~graph.nodes().isin(partition_nodes)
            if unclaimed_nodes.any():
                unclaimed_nodes = graph.nodes()[unclaimed_nodes_mask].values.tolist()
                raise ValueError(
                    f"Node IDs found in graph, but not listed in either partition: {unclaimed_nodes}"
                )

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                if {"edge_type", "edge_dtype", "edge_has_negative_weights"} & (
                    props - ret.keys()
                ):
                    if obj.value.edgelist:
                        edgelist = obj.value.view_edge_list()
                        weights = (
                            edgelist.weights if "weights" in edgelist.columns else None
                        )
                    else:
                        weights = obj.value.view_adj_list()[2]

                # fast properties
                for prop in {
                    "is_directed",
                    "node0_type",
                    "node0_dtype",
                    "node1_type",
                    "node1_dtype",
                } - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = obj.value.is_directed()
                    elif prop == "node0_type":
                        # TODO properly handle when node weights are supported
                        ret[prop] = "set"
                    elif prop == "node1_type":
                        # TODO properly handle when node weights are supported
                        ret[prop] = "set"
                    elif prop == "node0_dtype":
                        ret[prop] = dtypes.dtypes_simplified[obj.nodes[0].dtype]
                    elif prop == "node1_dtype":
                        ret[prop] = dtypes.dtypes_simplified[obj.nodes[1].dtype]
                    elif prop == "edge_type":
                        ret[prop] = "set" if weights is None else "map"
                    elif prop == "edge_dtype":
                        ret[prop] = dtypes.dtypes_simplified[weights.dtype]

                # slow properties, only compute if asked
                slow_props = props - ret.keys()
                if {"edge_has_negative_weights"} & slow_props:
                    for prop in slow_props:
                        if prop == "edge_has_negative_weights":
                            ret[prop] = weights.lt(0).any()

                return ret

            @classmethod
            def assert_equal(
                cls,
                obj1,
                obj2,
                aprops1,
                aprops2,
                cprops1,
                cprops2,
                *,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                assert aprops1 == aprops2, f"property mismatch: {aprops1} != {aprops2}"
                g1 = obj1.value
                g2 = obj2.value
                # Compare
                assert len(obj1.nodes[0]) == len(
                    obj2.nodes[0]
                ), f"{len(obj1.nodes[0])} == {len(obj2.nodes[0])}"
                assert len(obj1.nodes[1]) == len(
                    obj2.nodes[1]
                ), f"{len(obj1.nodes[1])} == {len(obj2.nodes[1])}"
                assert all(
                    obj1.nodes[0] == obj2.nodes[0]
                ), f"{obj1.nodes[0]} != {obj2.nodes[0]}"
                assert all(
                    obj1.nodes[1] == obj2.nodes[1]
                ), f"{obj1.nodes[1]} != {obj2.nodes[1]}"
                assert (
                    g1.number_of_edges() == g2.number_of_edges()
                ), f"{g1.number_of_edges()} != {g2.number_of_edges()}"
                if g1.edgelist:
                    g1_edge_list = g1.view_edge_list()
                    g2_edge_list = g2.view_edge_list()
                    assert len(g1_edge_list) == len(
                        g2_edge_list
                    ), f"g1 and g2 have a different number of edges"
                    assert len(g1_edge_list.columns) == len(
                        g2_edge_list.columns
                    ), "one of g1 or g2 is weighted while the other is not"
                    columns = g1_edge_list.columns
                    # TODO the below takes an additional possibly unneeded O(n) memory
                    assert g1_edge_list.set_index(columns) == g2_edge_list.set_index(
                        columns
                    ), "g1 and g2 have different edges"

                else:
                    for i, g1_series in enumerate(g1.view_adj_list()):
                        g2_series = g1.view_adj_list()[i]
                        assert (g1_series is None) == (
                            g2_series is None
                        ), "one of g1 or g2 is weighted while the other is not"
                        if g1_series is not None:
                            if np.issubdtype(g1_series.dtype.type, np.float):
                                assert cupy.isclose(g1_series == g2_series)
                            else:
                                assert all(
                                    g1_series == g2_series
                                ), "g1 and g2 have different edges"

                if aprops1.get("node0_type") == "map":
                    pass  # TODO handle this when node weights are supported

                if aprops1.get("node1_type") == "map":
                    pass  # TODO handle this when node weights are supported
