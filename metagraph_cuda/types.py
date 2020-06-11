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
    Matrix,
    NodeSet,
    NodeMap,
    EdgeSet,
    EdgeMap,
)
from .registry import has_cudf, has_cugraph, has_cupy
from typing import List, Dict, Any

if has_cupy:

    import cupy as cp

    class CupyVector(Wrapper, abstract=Vector):
        def __init__(self, data, missing_mask=None):
            self._assert_instance(data, cp.ndarray)
            if len(data.shape) != 1:
                raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
            self.value = data
            self.missing_mask = missing_mask
            if missing_mask is not None:
                if missing_mask.dtype != bool:
                    raise ValueError("missing_mask must have boolean type")
                if missing_mask.shape != data.shape:
                    raise ValueError("missing_mask must be the same shape as data")

        def __len__(self):
            return len(self.value)

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_dense", "dtype"} - ret.keys():
                if prop == "is_dense":
                    ret[prop] = obj.missing_mask is None
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.value.dtype]
            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert (
                obj1.value.shape == obj2.value.shape
            ), f"{obj1.value.shape} != {obj2.value.shape}"
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            # Remove missing values
            d1 = (
                obj1.value
                if obj1.missing_mask is None
                else obj1.value[~obj1.missing_mask]
            )
            d2 = (
                obj2.value
                if obj2.missing_mask is None
                else obj2.value[~obj2.missing_mask]
            )
            assert d1.shape == d2.shape, f"{d1.shape} != {d2.shape}"
            # Check for alignment of missing masks
            if obj1.missing_mask is not None:
                mask_alignment = obj1.missing_mask == obj2.missing_mask
                assert mask_alignment.all(), f"{mask_alignment}"
            # Compare
            if issubclass(d1.dtype.type, cp.floating):
                assert cp.isclose(d1, d2, rtol=rel_tol, atol=abs_tol).all()
            else:
                assert (d1 == d2).all()

    class CupyMatrix(Wrapper, abstract=Matrix):
        def __init__(self, data, missing_mask=None):
            if type(data) is cp.ndarray:
                data = cp.array(data, copy=False)
            self._assert_instance(data, cp.ndarray)
            if len(data.shape) != 2:
                raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
            self.value = data
            self.missing_mask = missing_mask
            if missing_mask is not None:
                if missing_mask.dtype != bool:
                    raise ValueError("missing_mask must have boolean type")
                if missing_mask.shape != data.shape:
                    raise ValueError("missing_mask must be the same shape as data")

        @property
        def shape(self):
            return self.value.shape

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_dense", "is_square", "dtype"} - ret.keys():
                if prop == "is_dense":
                    ret[prop] = obj.missing_mask is None
                if prop == "is_square":
                    ret[prop] = obj.value.shape[0] == obj.value.shape[1]
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.value.dtype]

            # slow properties, only compute if asked
            for prop in props - ret.keys():
                if prop == "is_symmetric":
                    # TODO: make this dependent on the missing mask
                    ret[prop] = (
                        ret["is_square"] and (obj.value.T == obj.value).all().all()
                    )

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert (
                obj1.value.shape == obj2.value.shape
            ), f"{obj1.value.shape} != {obj2.value.shape}"
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            # Remove missing values
            d1 = (
                obj1.value
                if obj1.missing_mask is None
                else obj1.value[~obj1.missing_mask]
            )
            d2 = (
                obj2.value
                if obj2.missing_mask is None
                else obj2.value[~obj2.missing_mask]
            )
            assert d1.shape == d2.shape, f"{d1.shape} != {d2.shape}"
            # Check for alignment of missing masks
            if obj1.missing_mask is not None:
                mask_alignment = obj1.missing_mask == obj2.missing_mask
                assert mask_alignment.all().all(), f"{mask_alignment}"
                # Compare 1-D
                if issubclass(d1.dtype.type, cp.floating):
                    assert cp.isclose(d1, d2, rtol=rel_tol, atol=abs_tol).all()
                else:
                    assert (d1 == d2).all()
            else:
                # Compare 2-D
                if issubclass(d1.dtype.type, cp.floating):
                    assert cp.isclose(d1, d2, rtol=rel_tol, atol=abs_tol).all().all()
                else:
                    assert (d1 == d2).all().all()

    class CupyNodeMap(NodeMapWrapper, abstract=NodeMap):
        def __init__(self, data, *, missing_mask=None):
            self._assert_instance(data, cp.ndarray)
            if len(data.shape) != 1:
                raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
            self.value = data
            self.missing_mask = missing_mask
            if missing_mask is not None and missing_mask.shape != data.shape:
                raise ValueError("missing_mask must be the same shape as data")

        def __getitem__(self, node_id):
            if self.missing_mask:
                if self.missing_mask[node_id]:
                    raise ValueError(f"node {node_id} is not in the NodeMap")
            return self.value[node_id]

        def _determine_weights(self, dtype):
            if dtype == "str":
                return "any"
            values = (
                self.value
                if self.missing_mask is None
                else self.value[~self.missing_mask]
            )
            if dtype == "bool":
                return "non-negative"
            else:
                min_val = values.min()
                if min_val < 0:
                    return "any"
                elif min_val == 0:
                    return "non-negative"
                else:
                    return "positive"

        @property
        def num_nodes(self):
            if self.missing_mask is not None:
                # Count number of False in the missing mask
                return (~self.missing_mask).sum()
            return len(self.value)

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"dtype"} - ret.keys():
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.value.dtype]

            # slow properties, only compute if asked
            for prop in props - ret.keys():
                if prop == "weights":
                    ret[prop] = obj._determine_weights(ret["dtype"])

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert (
                obj1.num_nodes == obj2.num_nodes
            ), f"{obj1.num_nodes} != {obj2.num_nodes}"
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            # Remove missing values
            d1 = (
                obj1.value
                if obj1.missing_mask is None
                else obj1.value[~obj1.missing_mask]
            )
            d2 = (
                obj2.value
                if obj2.missing_mask is None
                else obj2.value[~obj2.missing_mask]
            )
            assert len(d1) == len(d2), f"{len(d1)} != {len(d2)}"
            # Compare
            if issubclass(d1.dtype.type, cp.floating):
                assert cp.isclose(d1, d2, rtol=rel_tol, atol=abs_tol).all()
            else:
                assert (d1 == d2).all()

    class CompactCupyNodeMap(NodeMapWrapper, abstract=NodeMap):
        # TODO: make this style more general with a separate mapper including array of node_ids plus dict of {node_id: pos}
        def __init__(self, data, node_lookup):
            self._assert_instance(data, cp.ndarray)
            if len(data.shape) != 1:
                raise TypeError(f"Invalid number of dimensions: {len(data.shape)}")
            self._assert_instance(node_lookup, dict)
            self._assert(
                len(data) == len(node_lookup), "size of data and node_lookup must match"
            )
            self.value = data
            self.lookup = node_lookup

        def __getitem__(self, node_id):
            pos = self.lookup[node_id]
            return self.value[pos]

        @property
        def num_nodes(self):
            return len(self.lookup)

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"dtype"} - ret.keys():
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[obj.value.dtype]

            # slow properties, only compute if asked
            for prop in props - ret.keys():
                if prop == "weights":
                    if ret["dtype"] == "str":
                        weights = "any"
                    elif ret["dtype"] == "bool":
                        weights = "non-negative"
                    else:
                        min_val = obj.value.min()
                        if min_val < 0:
                            weights = "any"
                        elif min_val == 0:
                            weights = "non-negative"
                        else:
                            weights = "positive"
                    ret[prop] = weights

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert (
                obj1.num_nodes == obj2.num_nodes
            ), f"{obj1.num_nodes} != {obj2.num_nodes}"
            assert len(obj1.value) == len(
                obj2.value
            ), f"{len(obj1.value)} != {len(obj2.value)}"
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            # Compare
            if issubclass(obj1.value.dtype.type, cp.floating):
                assert xp.isclose(
                    obj1.value, obj2.value, rtol=rel_tol, atol=abs_tol
                ).all()
            else:
                assert (obj1.value == obj2.value).all()


if has_cudf:
    import cudf

    def _cudf_series_is_close(s1: cudf.Series, s2: cudf.Series) -> np.array:
        if has_cupy:
            import cupy

            return cupy.isclose(s1, s2)
        abs_difference = s1.sub(s2).abs()
        tolerable_difference = s2.abs().mul(rel_tol).add(abs_tol)
        is_close = tolerable_difference.sub(abs_difference).gt(0)
        return is_close

    class CuDFType(ConcreteType, abstract=DataFrame):
        value_type = cudf.DataFrame

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            raise NotImplementedError

    class CuDFNodeMap(NodeMapWrapper, abstract=NodeMap):
        """
        CuDFNodeMap stores data in format where the node values are used as the index and the entries
        in the column with the name specified by value_label correspond to the mapped values.
        """

        def __init__(self, data, value_label):
            self._assert_instance(data, cudf.DataFrame)
            self.value = data
            self.value_label = value_label

        def __getitem__(self, node_id):
            return self.value.loc[node_id].loc[self.value]

        @property
        def num_nodes(self):
            return len(self.value)

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in props - ret.keys():
                if prop == "dtype":
                    ret[prop] = dtypes.dtypes_simplified[
                        self.value[self.value_label].dtype
                    ]

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            d1, d2 = obj1.value, obj2.value
            if props1.get("dtype") == "float":
                assert all(
                    _cudf_series_is_close(d1[obj1.value_label], d2[obj2.value_label])
                )
            else:
                print(f"d1 {repr(d1)}")
                print(f"d2 {repr(d2)}")
                print(
                    f"(d1[obj1.value_label] == d2[obj2.value_label]) {repr((d1[obj1.value_label] == d2[obj2.value_label]))}"
                )
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

        @classmethod
        def assert_equal(
            cls, obj1, obj2, props1, props2, *, rel_tol=None, abs_tol=None
        ):
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
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
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            g1 = obj1.value
            g2 = obj2.value
            assert len(g1) == len(g2), f"{len(g1)} != {len(g2)}"
            assert len(obj1.index & obj2.index) == len(
                obj1.index
            ), f"{len(obj1.index & obj2.index)} != {len(obj1.index)}"
            # Ensure dataframes are indexed the same
            if not (obj1.index == obj2.index).all():
                g2 = g2.set_index(obj2.index).reindex(obj1.index).reset_index(drop=True)
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

        @classmethod
        def assert_equal(
            cls, obj1, obj2, props1, props2, *, rel_tol=None, abs_tol=None
        ):
            v1, v2 = obj1.value, obj2.value
            assert len(v1) == len(v2), f"size mismatch: {len(v1)} != {len(v2)}"
            assert all(v1 == v2), f"node sets do not match"
            assert props1 == props2, f"property mismatch: {props1} != {props2}"


if has_cugraph:
    import cugraph
    import cudf

    class CuGraphEdgeSet(EdgeSetWrapper, abstract=EdgeSet):
        def __init__(self, graph):
            self.value = graph

        @classmethod
        def assert_equal(
            cls, obj1, obj2, props1, props2, *, rel_tol=None, abs_tol=None
        ):
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
            g1 = obj1.value
            g2 = obj2.value
            # Compare
            assert (
                g1.nodes() == g2.nodes()
            ), f"node mismatch: {g1.nodes()} != {g2.nodes()}"
            assert (
                g1.edges() == g2.edges()
            ), f"edge mismatch: {g1.edges()} != {g2.edges()}"

    class CuGraphEdgeMap(EdgeMapWrapper, abstract=EdgeMap):
        def __init__(
            self, graph, weight_label="weight",
        ):
            self.value = graph
            self.weight_label = weight_label
            self._assert_instance(graph, cugraph.Graph)

        def _determine_dtype(self, all_values):
            all_types = {type(v) for v in all_values}
            if not all_types or (all_types - {float, int, bool}):
                return "str"
            for type_ in (float, int, bool):
                if type_ in all_types:
                    return str(type_.__name__)

        @classmethod
        def _compute_abstract_properties(
            cls, obj, props: List[str], known_props: Dict[str, Any]
        ) -> Dict[str, Any]:
            ret = known_props.copy()

            # fast properties
            for prop in {"is_directed"} - ret.keys():
                if prop == "is_directed":
                    ret[prop] = obj.value.is_directed()

            # slow properties, only compute if asked
            slow_props = props - ret.keys()
            if "dtype" in slow_props or "weights" in slow_props:
                all_values = set()
                for edge in obj.value.edges(data=True):
                    e_attrs = edge[-1]
                    value = e_attrs[obj.weight_label]
                    all_values.add(value)
                if "dtype" in slow_props:
                    ret["dtype"] = obj._determine_dtype(all_values)
                if "weights" in slow_props:
                    if ret["dtype"] == "str":
                        weights = "any"
                    elif ret["dtype"] == "bool":
                        weights = "non-negative"
                    else:
                        min_val = min(all_values)
                        if min_val < 0:
                            weights = "any"
                        elif min_val == 0:
                            weights = "non-negative"
                        else:
                            weights = "positive"
                    ret["weights"] = weights

            return ret

        @classmethod
        def assert_equal(cls, obj1, obj2, props1, props2, *, rel_tol=1e-9, abs_tol=0.0):
            assert props1 == props2, f"property mismatch: {props1} != {props2}"
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
                            assert _cudf_series_is_close(g1_series == g2_series)
                        else:
                            assert all(
                                g1_series == g2_series
                            ), "g1 and g2 have different edges"
