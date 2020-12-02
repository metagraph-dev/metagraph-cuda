import numpy as np
from metagraph.wrappers import (
    EdgeSetWrapper,
    EdgeMapWrapper,
    GraphWrapper,
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
from typing import List, Set, Tuple, Dict, Any, Optional

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
                g1_type = type(g1.nodes())
                g2_type = type(g2.nodes())
                assert g1_type == g2_type, f"node type mismatch: {g1_type} != {g2_type}"
                nodes_equal = (g1.nodes() == g2.nodes()).all()
                if isinstance(nodes_equal, cudf.DataFrame):
                    nodes_equal = nodes_equal.all()
                assert nodes_equal, f"node mismatch: {g1.nodes()} != {g2.nodes()}"
                assert len(g1.edges()) == len(
                    g2.edges()
                ), f"edge mismatch: {g1.edges()} != {g2.edges()}"
                g1_edges_reindexed = g1.edges().set_index(["src", "dst"])
                g2_edges_reindexed = g2.edges().set_index(["src", "dst"])
                assert (
                    g2_edges_reindexed.index.isin(g2_edges_reindexed.index).all().item()
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
                    if ret["dtype"] == "bool":
                        ret["has_negative_weights"] = None
                    else:
                        if obj.value.edgelist:
                            weights = obj.value.view_edge_list().weights
                        else:
                            weights = obj.value.view_adj_list()[2]
                        ret["has_negative_weights"] = (weights < 0).any()

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
                    assert len(g1.edges()) == len(
                        g2.edges()
                    ), f"edge mismatch: {g1.edges()} != {g2.edges()}"
                    g1_edges_reindexed = g1_edge_list.set_index(
                        ["src", "dst", "weights"]
                    )
                    g2_edges_reindexed = g2_edge_list.set_index(
                        ["src", "dst", "weights"]
                    )
                    assert (
                        g2_edges_reindexed.index.isin(g2_edges_reindexed.index)
                        .all()
                        .item()
                    ), f"edge mismatch: {g1.edges()} != {g2.edges()}"
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

    class CuGraph(GraphWrapper, abstract=Graph):
        def __init__(
            self,
            graph: cugraph.Graph,
            nodes: Optional[cudf.Series] = None,
            has_node_weights: bool = False,
        ):
            """
            The index of nodes specify the nodes. If has_node_weights is true, then the values of nodes specify the node weights.
            nodes can contain orphan nodes not maintained in graph.
            """
            self._assert_instance(graph, cugraph.Graph)
            self._assert(
                not has_node_weights or nodes is not None,
                f"Node weights not specified.",
            )
            if nodes is None:
                nodes = graph.nodes()
                nodes = nodes.set_index(nodes)
            self._assert_instance(nodes, cudf.Series)
            self._assert(
                graph.nodes().isin(nodes.index).all(),
                f"{graph} contains nodes ({graph.nodes()[~graph.nodes().isin(nodes.index)]}) not specified in {nodes.index.to_arrow().to_pylist()}.",
            )
            self.value = graph
            self.nodes = nodes
            self.has_node_weights = has_node_weights

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
                    "node_type",
                    "node_dtype",
                    "edge_type",
                    "edge_dtype",
                } - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = obj.value.is_directed()
                    elif prop == "node_type":
                        ret[prop] = "map" if obj.has_node_weights else "set"
                    elif prop == "node_dtype":
                        ret[prop] = (
                            dtypes.dtypes_simplified[obj.nodes.dtype]
                            if obj.has_node_weights
                            else None
                        )
                    elif prop == "edge_type":
                        ret[prop] = "map" if weights is not None else "set"
                    elif prop == "edge_dtype":
                        ret[prop] = (
                            dtypes.dtypes_simplified[weights.dtype]
                            if weights is not None
                            else None
                        )

                # slow properties, only compute if asked
                slow_props = props - ret.keys()
                for prop in slow_props:
                    if prop == "edge_has_negative_weights":
                        if ret["edge_dtype"] == "bool" or ret["edge_type"] == "set":
                            ret[prop] = None
                        else:
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
                canonicalize_nodes = lambda series: series.set_index(series)
                # Compare
                assert len(obj1.nodes) == len(
                    obj2.nodes
                ), f"{len(obj1.nodes)} == {len(obj2.nodes)}"
                assert all(
                    obj1.nodes.index.isin(obj2.nodes.index)
                ), f"{obj1} contains nodes missing from {obj2}"
                assert all(
                    obj2.nodes.index.isin(obj1.nodes.index)
                ), f"{obj2} contains nodes missing from {obj1}"
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
                    columns = list(g1_edge_list.columns)
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
                if aprops1["node_type"] == "map":
                    assert obj1.nodes.equal(obj2.nodes)

    class CuGraphBipartiteGraph(BipartiteGraphWrapper, abstract=BipartiteGraph):
        def __init__(
            self,
            graph,
            nodes: Optional[Tuple[cudf.Series, cudf.Series]] = None,
            nodes0_have_weights: bool = False,
            nodes1_have_weights: bool = False,
        ):
            """
            :param graph: cugraph.Graph instance s.t. cugraph.Graph.is_bipartite() returns True
            :param nodes: Optional tuple of cudf.Series nodes0 and nodes1; indices are node ids and values are node weights
            """
            self._assert_instance(graph, cugraph.Graph)
            self._assert(graph.is_bipartite(), f"{graph} is not bipartite")

            graph_node_sets = graph.sets()
            self._assert(
                len(graph_node_sets) == 2, "{graph} must have exactly 2 partitions"
            )
            self._assert_instance(graph_node_sets[0], cudf.Series)
            self._assert_instance(graph_node_sets[1], cudf.Series)
            # O(n^2), but cheaper than converting to Python sets
            common_nodes = graph_node_sets[0][
                graph_node_sets[0].isin(graph_node_sets[1])
            ]
            if len(common_nodes) != 0:
                raise ValueError(
                    f"Node IDs found in both partitions of the graph: {common_nodes.values.tolist()}"
                )
            partition_nodes = cudf.concat([graph_node_sets[0], graph_node_sets[1]])
            unclaimed_nodes_mask = ~graph.nodes().isin(partition_nodes)
            if unclaimed_nodes_mask.any():
                unclaimed_nodes = graph.nodes()[unclaimed_nodes_mask].values.tolist()
                raise ValueError(
                    f"Node IDs found in graph, but not listed in either partition: {unclaimed_nodes}"
                )

            nodes0, nodes1 = (None, None) if nodes is None else nodes
            if nodes0 is None:
                self._assert(
                    not nodes0_have_weights,
                    f"{self.__class__} initialized with node weights for partition 0, but no node weights specified.",
                )
                nodes0 = graph_node_sets[0].set_index(graph_node_sets[0])
            if nodes1 is None:
                self._assert(
                    not nodes1_have_weights,
                    f"{self.__class__} initialized with node weights for partition 1, but no node weights specified.",
                )
                nodes1 = graph_node_sets[1].set_index(graph_node_sets[1])

            self._assert_instance(nodes0, cudf.Series)
            self._assert_instance(nodes1, cudf.Series)
            self._assert(
                graph_node_sets[0].isin(nodes0.index).all(),
                f"Partition 0 of {graph} contains nodes ({graph_node_sets[0][~graph_node_sets[0].isin(nodes0.index)]}) not specified in {nodes0.index.to_arrow().to_pylist()}.",
            )
            self._assert(
                graph_node_sets[1].isin(nodes1.index).all(),
                f"Partition 1 of {graph} contains nodes ({graph_node_sets[1][~graph_node_sets[1].isin(nodes1.index)]}) not specified in {nodes1.index.to_arrow().to_pylist()}.",
            )

            self.value = graph
            self.nodes0 = nodes0
            self.nodes1 = nodes1
            self.nodes0_have_weights = nodes0_have_weights
            self.nodes1_have_weights = nodes1_have_weights

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
                    "edge_type",
                    "edge_dtype",
                    "node0_type",
                    "node1_type",
                    "node0_dtype",
                    "node1_dtype",
                } - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = obj.value.is_directed()
                    elif prop == "edge_type":
                        ret[prop] = "set" if weights is None else "map"
                    elif prop == "edge_dtype":
                        ret[prop] = (
                            None
                            if weights is None
                            else dtypes.dtypes_simplified[weights.dtype]
                        )
                    elif prop == "node0_type":
                        ret[prop] = "map" if obj.nodes0_have_weights else "set"
                    elif prop == "node1_type":
                        ret[prop] = "map" if obj.nodes1_have_weights else "set"
                    elif prop == "node0_dtype":
                        ret[prop] = (
                            dtypes.dtypes_simplified[obj.value.sets()[0].dtype]
                            if obj.nodes0_have_weights
                            else None
                        )
                    elif prop == "node1_dtype":
                        ret[prop] = (
                            dtypes.dtypes_simplified[obj.value.sets()[1].dtype]
                            if obj.nodes1_have_weights
                            else None
                        )

                # slow properties, only compute if asked
                slow_props = props - ret.keys()
                if {"edge_has_negative_weights"} & slow_props:
                    for prop in slow_props:
                        if prop == "edge_has_negative_weights":
                            if ret["edge_dtype"] == "bool" or ret["edge_type"] == "set":
                                ret[prop] = None
                            else:
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
                # Compare
                assert (
                    obj1.nodes0_have_weights == obj2.nodes0_have_weights
                ), f"{obj1.nodes0_have_weights} != {obj2.nodes0_have_weights}"
                assert (
                    obj1.nodes1_have_weights == obj2.nodes1_have_weights
                ), f"{obj1.nodes1_have_weights} != {obj2.nodes1_have_weights}"

                assert len(obj1.nodes0) == len(
                    obj2.nodes0
                ), f"{len(obj1.nodes0)} == {len(obj2.nodes0)}"
                assert len(obj1.nodes1) == len(
                    obj2.nodes1
                ), f"{len(obj1.nodes1)} == {len(obj2.nodes1)}"

                assert all(
                    obj1.nodes0.index.isin(obj2.nodes0.index)
                ), f"{obj1.nodes0} != {obj2.nodes0}"
                assert all(
                    obj1.nodes1.index.isin(obj2.nodes1.index)
                ), f"{obj1.nodes1} != {obj2.nodes1}"

                if aprops1.get("node0_type") == "map":
                    assert all(
                        obj1.nodes0 == obj2.nodes0.loc[obj1.nodes0.index]
                    ), f"{obj1.nodes0} != {obj2.nodes0}"
                if aprops1.get("node1_type") == "map":
                    assert all(
                        obj1.nodes1 == obj2.nodes1.loc[obj1.nodes1.index]
                    ), f"{obj1.nodes1} != {obj2.nodes1}"

                g1 = obj1.value
                g2 = obj2.value
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
