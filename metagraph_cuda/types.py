import numpy as np
from metagraph import ConcreteType, Wrapper, dtypes, IndexedNodes
from metagraph.types import DataFrame, Graph, WEIGHT_CHOICES, Nodes
from .registry import has_cudf, has_cugraph

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
        def get_type(cls, obj):
            """Get an instance of this type class that describes obj"""
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                ret_val.abstract_instance = cls.abstract(
                    dtype=obj._dtype, weights=obj._weights
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

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
        def get_type(cls, obj):
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                ret_val.abstract_instance = Graph(
                    dtype=obj._dtype, weights=obj._weights, is_directed=obj.is_directed
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")


if has_cugraph:
    import cugraph
    import cudf

    class CuGraph(Wrapper, abstract=Graph):
        def __init__(
            self, graph, *, weights=None, node_index=None,
        ):
            self._assert_instance(graph, cugraph.Graph)
            self.value = graph
            self.is_directed = isinstance(graph, cugraph.DiGraph)
            self._node_index = node_index
            self._assert(
                self.value.adjlist or self.value.edgelist, "Graph missing data."
            )
            self._dtype = self._determine_dtype()
            self._weights = self._determine_weights(weights)

        def _determine_dtype(self):
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

            if weights is None:
                return "unweighted"

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
        def get_type(cls, obj):
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                ret_val.abstract_instance = Graph(
                    dtype=obj._dtype, weights=obj._weights, is_directed=obj.is_directed
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")
