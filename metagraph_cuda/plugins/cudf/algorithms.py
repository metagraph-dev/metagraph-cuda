import metagraph as mg
from metagraph import concrete_algorithm
from .. import has_cudf
from typing import Callable, Any

if has_cudf:
    import cupy
    import cudf
    from .types import CuDFVector, CuDFNodeSet, CuDFNodeMap, CuDFEdgeSet, CuDFEdgeMap

    # TODO cudf and cupy currently don't support sampling
    # @concrete_algorithm("util.nodeset.choose_random")
    # def cudf_nodeset_choose_random(x: CuDFNodeSet, k: int) -> CuDFNodeSet:
    #     pass

    @concrete_algorithm("util.nodeset.from_vector")
    def cudf_nodeset_from_vector(x: CuDFVector) -> CuDFNodeSet:
        return CuDFNodeSet(x.value.copy())

    @concrete_algorithm("util.nodemap.sort")
    def cudf_nodemap_sort(
        x: CuDFNodeMap, ascending: bool, limit: mg.Optional[int]
    ) -> CuDFVector:
        positions_of_sorted_values = x.value.values.argsort()
        if not ascending:
            positions_of_sorted_values = positions_of_sorted_values[::-1]
        if limit is not None:
            positions_of_sorted_values = positions_of_sorted_values[0:limit]
        data = x.value.iloc[positions_of_sorted_values].index.to_series(
            cudf.core.index.RangeIndex(0, len(positions_of_sorted_values))
        )
        return CuDFVector(data)

    @concrete_algorithm("util.nodemap.select")
    def cudf_nodemap_select(x: CuDFNodeMap, nodes: CuDFNodeSet) -> CuDFNodeMap:
        data = x.value.loc[nodes.value].copy()
        return CuDFNodeMap(data)

    @concrete_algorithm("util.nodemap.filter")
    def cudf_nodemap_filter(x: CuDFNodeMap, func: Callable[[Any], bool]) -> CuDFNodeSet:
        keep_mask = x.value.applymap(func).values
        nodes = x.value.iloc[keep_mask].index.to_series()
        return CuDFNodeSet(nodes)

    @concrete_algorithm("util.nodemap.apply")
    def cudf_nodemap_apply(x: CuDFNodeMap, func: Callable[[Any], Any]) -> CuDFNodeMap:
        data = x.value.applymap(func).set_index(x.value.index)
        return CuDFNodeMap(data)

    # TODO cupy.ufunc's reduce not supported https://docs.cupy.dev/en/stable/reference/ufunc.html#universal-functions-ufunc
    # @concrete_algorithm("util.nodemap.reduce")
    # def cudf_nodemap_reduce(x: CuDFNodeMap, func: Callable[[Any, Any], Any]) -> Any:
    #     pass

    @concrete_algorithm("util.edgemap.from_edgeset")
    def cudf_edge_map_from_edgeset(
        edgeset: CuDFEdgeSet, default_value: Any,
    ) -> CuDFEdgeMap:
        df = edgeset.value.copy()
        df["weight"] = cupy.full(len(df), default_value)
        return CuDFEdgeMap(
            df,
            edgeset.src_label,
            edgeset.dst_label,
            "weight",
            is_directed=edgeset.is_directed,
        )
