import numpy as np
from metagraph.wrappers import (
    EdgeSetWrapper,
    EdgeMapWrapper,
    CompositeGraphWrapper,
)
from metagraph import dtypes
from metagraph.types import (
    Graph,
    EdgeSet,
    EdgeMap,
)
from .. import has_cugraph
from typing import List, Dict, Any

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
                    # TODO the below takes O(n) memory
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
                        g2_series = g1.view_adj_list()[i]
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
