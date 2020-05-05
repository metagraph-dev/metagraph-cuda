import numpy as np
from metagraph import ConcreteType, Wrapper, dtypes, IndexedNodes
from metagraph.types import DataFrame, Graph, DTYPE_CHOICES, WEIGHT_CHOICES, Nodes
from .registry import has_cudf, has_cugraph
from typing import List, Dict, Any

if has_cudf:
    import cudf

    class CuDFType(ConcreteType, abstract=DataFrame):
        value_type = cudf.DataFrame

    class CuDFNodes(Wrapper, abstract=Nodes):
        def __init__(
            self, data, key_label, value_label, *, weights=None, node_index=None,
        ):
            self.value = data
            self._assert_instance(data, cudf.DataFrame)
            self.key_label = key_label
            self.value_label = value_label
            self._dtype = dtypes.dtypes_simplified[data[self.value_label].dtype]
            self._weights = self._determine_weights(weights)
            self._node_index = node_index

        def __getitem__(self, label):
            if self._node_index is None:
                if self.value.index.name != self.key_label:
                    self.value = self.value.set_index(self.key_label)
                return self.value.loc[label].loc[self.value_label]
            return self.value.iloc[self._node_index.bylabel(label)].loc[
                self.value_label
            ]

        def _determine_weights(self, weights=None):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if self._dtype == "str":
                return "any"
            if self._dtype == "bool":
                if self.value.all():
                    return "unweighted"
                return "non-negative"
            else:
                min_val = self.value[self.value_label].min()
                if min_val < 0:
                    return "any"
                elif min_val == 0:
                    return "non-negative"
                else:
                    if (
                        self._dtype == "int"
                        and min_val == 1
                        and self.value[self.value_label].max() == 1
                    ):
                        return "unweighted"
                    return "positive"

        @property
        def num_nodes(self):
            return len(self.value.index)

        @property
        def node_index(self):
            if self._node_index is None:
                if self.value.index.name != self.key_label:
                    self.value = self.value.set_index(self.key_label)
                keys = self.value.index
                if keys.dtype == dtypes.int64:
                    self.value = self.value.sort_index()
                    keys = self.value.index
                self._node_index = IndexedNodes(keys)
            return self._node_index

        @classmethod
        def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
            cls._validate_abstract_props(props)
            return dict(dtype=obj._dtype, weights=obj._weights,)

        @classmethod
        def assert_equal(
            cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True
        ):
            assert (
                type(obj1) is cls.value_type
            ), f"obj1 must be CuDFNodes, not {type(obj1)}"
            assert (
                type(obj2) is cls.value_type
            ), f"obj2 must be CuDFNodes, not {type(obj2)}"

            assert (
                obj1.num_nodes == obj2.num_nodes
            ), f"{obj1.num_nodes} != {obj2.num_nodes}"
            if check_values:
                assert obj1._dtype == obj2._dtype, f"{obj1._dtype} != {obj2._dtype}"
                assert (
                    obj1._weights == obj2._weights
                ), f"{obj1._weights} != {obj2._weights}"
            # Convert to a common node indexing scheme
            d1 = obj1.value
            d2 = obj2.value
            assert len(d1) == len(d2), f"{len(d1)} != {len(d2)}"
            d1 = (
                d1[[obj1.key_label, obj1.value_label]]
                .rename(
                    {obj1.key_label: obj2.key_label, obj1.value_label: obj2.value_label}
                )
                .set_index(obj2.key_label)
                .sort_index(obj2.key_label)
            )
            d2 = (
                d2[[obj2.key_label, obj2.value_label]]
                .set_index(obj2.key_label)
                .sort_index(obj2.key_label)
            )
            assert d1.index.equals(d2.index), "Keys of d1 and d2 differ."
            # Compare
            if check_values:
                if obj1._dtype == "float":
                    v1 = d1[obj2.value_label]
                    v2 = d2[obj2.value_label]
                    abs_difference = v1.sub(v2).abs()
                    tolerable_difference = v2.abs().mul(rel_tol).add(abs_tol)
                    assert tolerable_difference.sub(abs_difference).min() >= 0
                else:
                    assert d1.equals(d2)

    class CuDFEdgeList(Wrapper, abstract=Graph):
        def __init__(
            self,
            df,
            src_label="source",
            dst_label="destination",
            weight_label=None,
            *,
            is_directed=True,
            weights=None,
            node_index=None,
        ):
            self._assert_instance(df, cudf.DataFrame)
            self.value = df
            self.is_directed = is_directed
            self._node_index = node_index
            self.src_label = src_label
            self.dst_label = dst_label
            self.weight_label = weight_label
            self._assert(src_label in df, f"Indicated src_label not found: {src_label}")
            self._assert(dst_label in df, f"Indicated dst_label not found: {dst_label}")
            if weight_label is not None:
                self._assert(
                    weight_label in df,
                    f"Indicated weight_label not found: {weight_label}",
                )
            self._dtype = self._determine_dtype()
            self._weights = self._determine_weights(weights)

        def _determine_dtype(self):
            if self.weight_label is None:
                return "bool"

            values = self.value[self.weight_label]
            return dtypes.dtypes_simplified[values.dtype]

        def _determine_weights(self, weights):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if self.weight_label is None:
                return "unweighted"

            if self._dtype == "str":
                return "any"
            values = self.value[self.weight_label]
            if self._dtype == "bool":
                if values.all():
                    return "unweighted"
                return "non-negative"
            else:
                min_val = values.min()
                if min_val < 0:
                    return "any"
                elif min_val == 0:
                    return "non-negative"
                else:
                    if self._dtype == "int" and min_val == 1 and values.max() == 1:
                        return "unweighted"
                    return "positive"

        @property
        def num_nodes(self):
            src_nodes, dst_nodes = self.index.levels
            return len(src_nodes | dst_nodes)

        @property
        def node_index(self):
            if self._node_index is None:
                src_col = self.value[self.src_label]
                dst_col = self.value[self.dst_label]
                all_nodes = cudf.concat([src_col, dst_col]).unique()
                if all_nodes.dtype == dtypes.int64:
                    all_nodes = all_nodes.sort_values()
                self._node_index = IndexedNodes(all_nodes)
            return self._node_index

        @classmethod
        def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
            cls._validate_abstract_props(props)
            return dict(
                is_directed=obj.is_directed, dtype=obj._dtype, weights=obj._weights,
            )

        @classmethod
        def assert_equal(
            cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True
        ):
            assert (
                type(obj1) is cls.value_type
            ), f"obj1 must be CuDFEdgeList, not {type(obj1)}"
            assert (
                type(obj2) is cls.value_type
            ), f"obj2 must be CuDFEdgeList, not {type(obj2)}"

            if check_values:
                assert obj1._dtype == obj2._dtype, f"{obj1._dtype} != {obj2._dtype}"
                assert (
                    obj1._weights == obj2._weights
                ), f"{obj1._weights} != {obj2._weights}"
            # Compare
            g1 = obj1.value
            g2 = obj2.value
            assert len(g1) == len(g2), f"{len(g1)} != {len(g2)}"
            obj1_columns = [obj1.src_label, obj1.dst_label]
            obj2_columns = [obj2.src_label, obj2.dst_label]
            renaming_dict = {
                obj1.src_label: obj2.src_label,
                obj1.dst_label: obj2.dst_label,
            }
            if obj1._weights != "unweighted" or obj2._weights != "unweighted":
                renaming_dict[obj1.weight_label] = obj2.weight_label
            g1 = (
                g1[[obj1.src_label, obj1.dst_label, obj1.weight_label]]
                if obj1.weight_label
                else g1[[obj1.src_label, obj1.dst_label]]
            )
            g1 = (
                g1.rename(renaming_dict)
                .set_index([obj2.src_label, obj2.dst_label])
                .sort_index([obj2.src_label, obj2.dst_label])
            )
            g2 = (
                g2[[obj2.src_label, obj2.dst_label, obj2.weight_label]]
                if obj2.weight_label
                else g2[[obj2.src_label, obj2.dst_label]]
            )
            g2 = g2.set_index([obj2.src_label, obj2.dst_label]).sort_index(
                [obj2.src_label, obj2.dst_label]
            )
            assert g1.index.equals(g2.index), "Srcs of g1 and g2 differ."
            if check_values and obj1._weights != "unweighted":
                v1 = g1[obj2.weight_label]
                v2 = g2[obj2.weight_label]
                if issubclass(v1.dtype.type, np.floating):
                    abs_difference = v1.sub(v2).abs()
                    tolerable_difference = v2.abs().mul(rel_tol).add(abs_tol)
                    assert tolerable_difference.sub(abs_difference).min() >= 0
                else:
                    assert v1.equals(v2)


if has_cugraph:
    import cugraph
    import cudf

    class CuGraph(Wrapper, abstract=Graph):
        def __init__(
            self, graph, *, weights=None, dtype=None, node_index=None,
        ):
            self._assert_instance(graph, cugraph.Graph)
            self.value = graph
            self.is_directed = isinstance(graph, cugraph.DiGraph)
            self._node_index = node_index
            self._assert(
                self.value.adjlist or self.value.edgelist, "Graph missing data."
            )
            self._dtype = self._determine_dtype(dtype)
            self._weights = self._determine_weights(weights)

        def _determine_dtype(self, dtype):
            if dtype is not None:
                if dtype not in DTYPE_CHOICES:
                    raise ValueError(f"Illegal dtype: {dtype}")
                return dtype

            if self.value.edgelist:
                edge_list = self.value.view_edge_list()

                if not "weights" in edge_list.columns:
                    return "bool"

                weights = edge_list["weights"]
                return dtypes.dtypes_simplified[weights.dtype]

            elif self.value.adjlist:
                adj_list = self.value.view_adj_list()
                weights = adj_list[2]

                if weights is None:
                    return "bool"

                return dtypes.dtypes_simplified[weights.dtype]

        def _determine_weights(self, weights):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if self._dtype == "str":
                return "any"

            if self.value.edgelist:
                edge_list = self.value.view_edge_list()
                if not "weights" in edge_list.columns:
                    return "unweighted"
                values = edge_list["weights"]
            elif self.value.adjlist:
                adj_list = self.value.view_adj_list()
                values = adj_list[2]

            if values is None:
                return "unweighted"

            if self._dtype == "bool":
                if values.all():
                    return "unweighted"
                return "non-negative"
            else:
                min_val = values.min()
                if min_val < 0:
                    return "any"
                elif min_val == 0:
                    return "non-negative"
                else:
                    if self._dtype == "int" and min_val == 1 and values.max() == 1:
                        return "unweighted"
                    return "positive"

        @property
        def num_nodes(self):
            edge_list = self.value.view_edge_list()
            all_nodes = cudf.concat([edge_list["src"], edge_list["dst"]]).unique()
            return len(all_nodes)

        @property
        def node_index(self):
            if self._node_index is None:
                if self.value.edgelist:
                    edge_list = self.value.view_edge_list()
                    all_nodes = cudf.concat(
                        [edge_list["src"], edge_list["dst"]]
                    ).unique()
                    if all_nodes.dtype == dtypes.int64:
                        all_nodes = all_nodes.sort_values()
                elif self.value.adjlist:
                    all_nodes = range(self.value.number_of_nodes())
                self._node_index = IndexedNodes(all_nodes)
            return self._node_index

        @classmethod
        def compute_abstract_properties(cls, obj, props: List[str]) -> Dict[str, Any]:
            cls._validate_abstract_props(props)
            return dict(
                is_directed=obj.is_directed, dtype=obj._dtype, weights=obj._weights,
            )

        @classmethod
        def assert_equal(
            cls, obj1, obj2, *, rel_tol=1e-9, abs_tol=0.0, check_values=True
        ):
            assert (
                type(obj1) is cls.value_type
            ), f"obj1 must be CuGraph, not {type(obj1)}"
            assert (
                type(obj2) is cls.value_type
            ), f"obj2 must be CuGraph, not {type(obj2)}"

            if check_values:
                assert obj1._dtype == obj2._dtype, f"{obj1._dtype} != {obj2._dtype}"
                assert (
                    obj1._weights == obj2._weights
                ), f"{obj1._weights} != {obj2._weights}"
            g1 = obj1.value
            g2 = obj2.value
            assert (
                g1.is_directed() == g2.is_directed()
            ), f"{g1.is_directed()} != {g2.is_directed()}"
            # Compare
            if self.value.edgelist:
                # TODO see if this can be optimized
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
                assert len(g1_edge_list) == len(g2_edge_list) and len(
                    g1_edge_list.merge(g2_edge_list, how="outer")
                ) == len(g1_edge_list), "g1 and g2 have different edges"
            else:
                assert (
                    g1.number_of_nodes() == g2.number_of_nodes()
                ), "g1 and g2 have different nodes"
                assert all(
                    g1.view_adj_list() == g2.view_adj_list()
                ), "g1 and g2 have different edges"

            if check_values and obj1._weights != "unweighted":
                if obj1._dtype == "float":
                    comp = partial(math.isclose, rel_tol=rel_tol, abs_tol=abs_tol)
                    compstr = "close to"
                else:
                    comp = operator.eq
                    compstr = "equal to"

                # TODO see if this can be optimized
                for e1, e2, d1 in g1.edges(data=True):
                    d2 = g2.edges[(e1, e2)]
                    val1 = d1[obj1.weight_label]
                    val2 = d2[obj2.weight_label]
                    assert comp(val1, val2), f"{(e1, e2)} {val1} not {compstr} {val2}"
