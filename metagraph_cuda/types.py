import numpy as np
from metagraph.wrappers import (
    NodeSetWrapper,
    NodeMapWrapper,
    EdgeSetWrapper,
    EdgeMapWrapper,
    CompositeGraphWrapper,
)
from metagraph import ConcreteType, Wrapper, dtypes
from metagraph.types import (
    Graph,
    DataFrame,
    Vector,
    Matrix,
    NodeSet,
    NodeMap,
    EdgeSet,
    EdgeMap,
)
from .registry import has_cudf, has_cugraph
from typing import Set, List, Dict, Any

if has_cudf:
    import cudf
    import cupy

    class CuDFType(ConcreteType, abstract=DataFrame):
        value_type = cudf.DataFrame

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
            raise NotImplementedError

    class CuDFVector(Wrapper, abstract=Vector):
        """
        CuDFVector stores data in format where the index is the vector position and the values are the values.
        """

        def __init__(self, data):
            self._assert_instance(data, cudf.Series)
            self._assert(
                data.index.dtype == np.dtype("int64"),
                f"{data} does not have an integer index.",
            )
            self.value = data

        def __contains__(self, node_id):
            return node_id in self.value

        def __getitem__(self, position):
            return self.value.iloc[position]

        def __len__(self):
            return self.value.index.max() + 1

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: Set[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"is_dense", "dtype"} - ret.keys():
                    if prop == "is_dense":
                        if isinstance(obj.value.index, cudf.core.index.RangeIndex):
                            ret[prop] = (
                                obj.value.index.start == 0
                                and obj.value.index.stop == len(obj.value)
                            )
                        else:
                            ret[prop] = (
                                obj.value.index.min() == 0
                                and obj.value.index.max() == len(obj.value) - 1
                            )
                    if prop == "dtype":
                        ret[prop] = dtypes.dtypes_simplified[obj.value.dtype]

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
                assert len(obj1.value) == len(
                    obj2.value
                ), f"{len(obj1.value)} != {len(obj2.value)}"
                assert (
                    aprops1 == aprops2
                ), f"abstract property mismatch: {aprops1} != {aprops2}"
                assert (obj1.value == obj2.value).all()

    class CuDFNodeMap(NodeMapWrapper, abstract=NodeMap):
        """
        CuDFNodeMap stores data in format where the node values are used as the index and the entries
        in the column with the name specified by value_label correspond to the mapped values.
        """

        def __init__(self, data, value_label):
            self._assert_instance(data, cudf.DataFrame)
            self.value = data
            self.value_label = value_label

        def __contains__(self, node_id):
            return node_id in self.value.index

        def __getitem__(self, node_id):
            return self.value.loc[node_id].loc[self.value]

        @property
        def num_nodes(self):
            return len(self.value)

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: List[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in props - ret.keys():
                    if prop == "dtype":
                        ret[prop] = dtypes.dtypes_simplified[
                            obj.value[obj.value_label].dtype
                        ]

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
                d1, d2 = obj1.value, obj2.value
                if aprops1.get("dtype") == "float":
                    assert all(cupy.isclose(d1[obj1.value_label], d2[obj2.value_label]))
                else:
                    assert all(d1[obj1.value_label] == d2[obj2.value_label])

    class CuDFEdgeSet(EdgeSetWrapper, abstract=EdgeSet):
        def __init__(
            self, df, src_label="source", dst_label="target", *, is_directed=True
        ):
            self._assert_instance(df, cudf.DataFrame)
            self.value = df
            self.is_directed = is_directed
            self.src_label = src_label
            self.dst_label = dst_label
            self._assert(src_label in df, f"Indicated src_label not found: {src_label}")
            self._assert(dst_label in df, f"Indicated dst_label not found: {dst_label}")
            # Build the MultiIndex representing the edges
            self.index = df.set_index([src_label, dst_label]).index

        @property
        def num_nodes(self):
            src_nodes, dst_nodes = self.index.levels
            return len(src_nodes | dst_nodes)

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: List[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"is_directed"} - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = obj.is_directed

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
                assert len(g1) == len(g2), f"{len(g1)} != {len(g2)}"
                assert g1.index.equals(g2.index), f"{g1.index} != {g2.index}"

    class CuDFEdgeMap(EdgeMapWrapper, abstract=EdgeMap):
        def __init__(
            self,
            df,
            src_label="source",
            dst_label="target",
            weight_label="weight",
            *,
            is_directed=True,
        ):
            self._assert_instance(df, cudf.DataFrame)
            self.value = df
            self.is_directed = is_directed
            self.src_label = src_label
            self.dst_label = dst_label
            self.weight_label = weight_label
            self._assert(src_label in df, f"Indicated src_label not found: {src_label}")
            self._assert(dst_label in df, f"Indicated dst_label not found: {dst_label}")
            self._assert(
                weight_label in df, f"Indicated weight_label not found: {weight_label}"
            )
            # Build the MultiIndex representing the edges
            self.index = df.set_index([src_label, dst_label]).index

        @property
        def num_nodes(self):
            src_nodes, dst_nodes = self.index.levels
            return len(src_nodes | dst_nodes)

        class TypeMixin:
            @classmethod
            def _compute_abstract_properties(
                cls, obj, props: List[str], known_props: Dict[str, Any]
            ) -> Dict[str, Any]:
                ret = known_props.copy()

                # fast properties
                for prop in {"is_directed", "dtype"} - ret.keys():
                    if prop == "is_directed":
                        ret[prop] = obj.is_directed
                    if prop == "dtype":
                        ret[prop] = dtypes.dtypes_simplified[
                            obj.value[obj.weight_label].dtype
                        ]

                # slow properties, only compute if asked
                for prop in props - ret.keys():
                    if prop == "weights":
                        if ret["dtype"] == "str":
                            weights = "any"
                        elif ret["dtype"] == "bool":
                            weights = "non-negative"
                        else:
                            min_val = obj.value[obj.weight_label].min()
                            if min_val < 0:
                                weights = "any"
                            elif min_val == 0:
                                weights = "non-negative"
                            else:
                                weights = "positive"
                        ret[prop] = weights

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
                assert len(g1) == len(g2), f"{len(g1)} != {len(g2)}"
                assert len(obj1.index & obj2.index) == len(
                    obj1.index
                ), f"{len(obj1.index & obj2.index)} != {len(obj1.index)}"
                # Ensure dataframes are indexed the same
                if not (obj1.index == obj2.index).all():
                    g2 = (
                        g2.set_index(obj2.index)
                        .reindex(obj1.index)
                        .reset_index(drop=True)
                    )
                # Compare
                v1 = g1[obj1.weight_label]
                v2 = g2[obj2.weight_label]
                if issubclass(v1.dtype.type, np.floating):
                    assert np.isclose(v1, v2, rtol=rel_tol, atol=abs_tol).all()
                else:
                    assert (v1 == v2).all()

    class CuDFNodeSet(NodeSetWrapper, abstract=NodeSet):
        def __init__(self, data):
            self._assert_instance(data, cudf.Series)
            unique_values = data.unique()
            self.value = cudf.Series(unique_values).set_index(unique_values)

        @property
        def num_nodes(self):
            return len(self.value)

        def __contains__(self, item):
            return item in self.value.index

        class TypeMixin:
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
                v1, v2 = obj1.value, obj2.value
                assert len(v1) == len(v2), f"size mismatch: {len(v1)} != {len(v2)}"
                assert all(v1 == v2), f"node sets do not match"


if has_cugraph:
    import cugraph
    import cudf

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
                    # TODO the below takes O(n) memory
                    assert len(g1_edge_list) == len(g2_edge_list) and len(
                        g1_edge_list.merge(g2_edge_list, how="outer")
                    ) == len(g1_edge_list), "g1 and g2 have different edges"
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
