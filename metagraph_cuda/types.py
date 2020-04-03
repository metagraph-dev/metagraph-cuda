from metagraph import ConcreteType, Wrapper, dtypes, IndexedNodes
from metagraph.types import DataFrame, Graph
from .registry import has_cudf, has_cugraph

if has_cudf:
    import cudf

    class CuDFType(ConcreteType, abstract=DataFrame):
        value_type = cudf.DataFrame

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

    def _determine_dtype_from_cugraph_graph(g: cugraph.Graph) -> str:
        if g.edgelist:
            edge_list = g.view_edge_list()

            if not "weights" in edge_list.columns:
                return "bool"

            weights = edge_list["weights"]
            return dtypes.dtypes_simplified[weights.dtype]

        elif g.adjlist:
            adj_list = g.view_adj_list()
            weights = adj_list[2]

            if weights is None:
                return "bool"

            return dtypes.dtypes_simplified[weights.dtype]

    def _determine_weights_from_cugraph_graph(g: cugraph.Graph, dtype: str) -> str:
        if dtype == "str":
            return "any"

        if g.edgelist:
            edge_list = g.view_edge_list()
            if not "weights" in edge_list.columns:
                return "unweighted"
            values = edge_list["weights"]
        elif g.adjlist:
            adj_list = g.view_adj_list()
            values = adj_list[2]

        if dtype == "bool":
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
                if dtype == "int" and min_val == 1 and values.max() == 1:
                    return "unweighted"
                return "positive"

    class AutoCuGraphType(ConcreteType, abstract=Graph):
        value_type = cugraph.Graph

        @classmethod
        def get_type(cls, obj):
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                obj_dtype = _determine_dtype_from_cugraph_graph(obj)
                obj_weights = _determine_weights_from_cugraph_graph(obj, obj_dtype)
                obj_is_directed = isinstance(obj, cugraph.DiGraph)
                ret_val.abstract_instance = Graph(
                    dtype=obj_dtype, weights=obj_weights, is_directed=obj_is_directed
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

    class AutoCuDiGraphType(ConcreteType, abstract=Graph):
        value_type = cugraph.DiGraph

        @classmethod
        def get_type(cls, obj):
            if isinstance(obj, cls.value_type):
                ret_val = cls()
                obj_dtype = _determine_dtype_from_cugraph_graph(obj)
                obj_weights = _determine_weights_from_cugraph_graph(obj, obj_dtype)
                obj_is_directed = True
                ret_val.abstract_instance = Graph(
                    dtype=obj_dtype, weights=obj_weights, is_directed=obj_is_directed
                )
                return ret_val
            else:
                raise TypeError(f"object not of type {cls.__name__}")

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
            return _determine_dtype_from_cugraph_graph(self.value)

        def _determine_weights(self, weights):
            if weights is not None:
                if weights not in WEIGHT_CHOICES:
                    raise ValueError(f"Illegal weights: {weights}")
                return weights

            if weights is None:
                return "unweighted"

            return _determine_weights_from_cugraph_graph(self.value, self._dtype)

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
