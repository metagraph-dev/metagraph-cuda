import numpy as np
from metagraph.wrappers import (
    NodeSetWrapper,
    NodeMapWrapper,
    EdgeSetWrapper,
    EdgeMapWrapper,
)
from metagraph import ConcreteType, Wrapper, dtypes
from metagraph.types import (
    DataFrame,
    Vector,
    NodeSet,
    NodeMap,
    EdgeSet,
    EdgeMap,
)
from .. import has_cudf
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
        CuDFNodeMap stores data in a cudf.DataFrame where the index corresponds go the node ids
        and the entries in the column with the name specified by value_label correspond to
        the mapped values.
        """

        def __init__(self, data, value_label):
            # TODO store this as a series instead
            self._assert_instance(data, cudf.DataFrame)
            self.value = data
            self.value_label = value_label

        def __contains__(self, node_id):
            return node_id in self.value.index

        def __getitem__(self, node_id):
            return self.value.loc[node_id].loc[self.value]

        def copy(self):
            return CuDFNodeMap(self.value.copy(), self.value_label)

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
                    assert (
                        cupy.isclose(d1[obj1.value_label], d2[obj2.value_label])
                    ).all()
                else:
                    assert (d1[obj1.value_label] == d2[obj2.value_label]).all()

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

        def copy(self):
            return CuDFEdgeSet(
                self.value.copy(),
                self.src_label,
                self.dst_label,
                bool(self.is_directed),
            )

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
                    if prop == "has_negative_weights":
                        ret[prop] = obj.value[obj.weight_label].lt(0).any()

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
                assert (
                    g1.index.isin(g2.index).all() and g2.index.isin(g1.index).all()
                ), f"obj1 and obj2 are indexed differently."
                # Ensure dataframes are indexed the same
                if not (g1.index == g2.index).values.all():
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

        def copy(self):
            return CuDFNodeSet(self.value.copy())

        @property
        def num_nodes(self):
            return len(self.value)

        def __iter__(self):
            return iter(self.value.values_host)

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
                assert (v1 == v2).all(), f"node sets do not match"
