from metagraph import translator, dtypes
from metagraph.plugins import has_pandas, has_scipy
import numpy as np
from .. import has_cudf
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap, NumpyVectorType
from metagraph.plugins.python.types import (
    PythonNodeSetType,
    PythonNodeMapType,
    dtype_casting,
)

if has_cudf:
    import cudf
    import cupy
    from .types import (
        CuDFVector,
        CuDFNodeSet,
        CuDFNodeMap,
        CuDFEdgeSet,
        CuDFEdgeMap,
    )

    @translator
    def cudf_nodemap_to_nodeset(x: CuDFNodeMap, **props) -> CuDFNodeSet:
        return CuDFNodeSet(x.value.index.to_series())

    @translator
    def cudf_edgemap_to_edgeset(x: CuDFEdgeMap, **props) -> CuDFEdgeSet:
        data = x.value[[x.src_label, x.dst_label]].copy()
        return CuDFEdgeSet(data, x.src_label, x.dst_label, is_directed=x.is_directed)

    @translator
    def translate_nodes_cudfnodemap2pythonnodemap(
        x: CuDFNodeMap, **props
    ) -> PythonNodeMapType:
        python_dtype = dtype_casting[dtypes.dtypes_simplified[x.value.dtype]]
        data = x.value.astype(python_dtype).to_pandas().to_dict()
        return data

    @translator
    def translate_nodes_pythonnodemap2cudfnodemap(
        x: PythonNodeMapType, **props
    ) -> CuDFNodeMap:
        keys = x.keys()
        keys_are_compact = max(keys) - min(keys) == len(keys) - 1
        if keys_are_compact:
            keys = range(min(keys), max(keys) + 1)
        values = [x[key] for key in keys]
        if keys_are_compact:
            keys = range(min(keys), max(keys) + 1)
        data = cudf.Series(values, index=keys)
        return CuDFNodeMap(data)

    @translator
    def translate_nodes_cudfnodeset2pythonnodeset(
        x: CuDFNodeSet, **props
    ) -> PythonNodeSetType:
        return set(x.value.index.to_pandas())

    @translator
    def translate_nodes_pythonnodeset2cudfnodeset(
        x: PythonNodeSetType, **props
    ) -> CuDFNodeSet:
        return CuDFNodeSet(cudf.Series(x))

    @translator
    def translate_nodes_numpyvector2cudfvector(
        x: NumpyVectorType, **props
    ) -> CuDFVector:
        series = cudf.Series(x)
        return CuDFVector(series)

    @translator
    def translate_vector_cudfvector2numpyvector(
        x: CuDFVector, **props
    ) -> NumpyVectorType:
        np_vector = cupy.asnumpy(x.value.values)
        return np_vector

    @translator
    def translate_nodes_numpynodemap2cudfnodemap(
        x: NumpyNodeMap, **props
    ) -> CuDFNodeMap:
        series = cudf.Series(x.value).set_index(x.nodes)
        return CuDFNodeMap(series)

    @translator
    def translate_nodes_cudfnodemap2numpynodemap(
        x: CuDFNodeMap, **props
    ) -> NumpyNodeMap:
        return NumpyNodeMap(
            cupy.asnumpy(x.value.values), nodes=cupy.asnumpy(x.value.index.values)
        )

    @translator
    def translate_nodes_numpynodeset2cudfnodeset(
        x: NumpyNodeSet, **props
    ) -> CuDFNodeSet:
        data = cudf.Series(x.value)
        return CuDFNodeSet(data)

    @translator
    def translate_nodes_cudfnodeset2numpynodeset(
        x: CuDFNodeSet, **props
    ) -> NumpyNodeSet:
        return NumpyNodeSet(cupy.asnumpy(x.value.index.values))


if has_cudf and has_pandas:
    from metagraph.plugins.pandas.types import PandasEdgeSet, PandasEdgeMap

    @translator
    def translate_edgeset_pdedgeset2cudfedgeset(
        x: PandasEdgeSet, **props
    ) -> CuDFEdgeSet:
        df = cudf.from_pandas(x.value[[x.src_label, x.dst_label]])
        return CuDFEdgeSet(
            df, src_label=x.src_label, dst_label=x.dst_label, is_directed=x.is_directed
        )

    @translator
    def translate_edgemap_pdedgemap2cudfedgemap(
        x: PandasEdgeMap, **props
    ) -> CuDFEdgeMap:
        df = cudf.from_pandas(x.value[[x.src_label, x.dst_label, x.weight_label]])
        return CuDFEdgeMap(
            df,
            src_label=x.src_label,
            dst_label=x.dst_label,
            weight_label=x.weight_label,
            is_directed=x.is_directed,
        )

    @translator
    def translate_edgeset_cudfedgeset2pdedgeset(
        x: CuDFEdgeSet, **props
    ) -> PandasEdgeSet:
        pdf = x.value[[x.src_label, x.dst_label]].to_pandas()
        return PandasEdgeSet(
            pdf,
            src_label=x.src_label,
            dst_label=x.dst_label,
            is_directed=x.is_directed,
        )

    @translator
    def translate_edgemap_cudfedgemap2pdedgemap(
        x: CuDFEdgeMap, **props
    ) -> PandasEdgeMap:
        pdf = x.value[[x.src_label, x.dst_label, x.weight_label]].to_pandas()
        return PandasEdgeMap(
            pdf,
            src_label=x.src_label,
            dst_label=x.dst_label,
            weight_label=x.weight_label,
            is_directed=x.is_directed,
        )


if has_cudf and has_scipy:
    import scipy.sparse as ss
    from metagraph.plugins.scipy.types import ScipyEdgeSet, ScipyEdgeMap

    @translator
    def translate_edgeset_scipyedgeset2cudfedgeset(
        x: ScipyEdgeSet, **props
    ) -> CuDFEdgeSet:
        is_directed = ScipyEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        row_ids = x.node_list[coo_matrix.row]
        column_ids = x.node_list[coo_matrix.col]
        if not is_directed:
            mask = row_ids <= column_ids
            row_ids = row_ids[mask]
            column_ids = column_ids[mask]
        df = cudf.DataFrame({"source": row_ids, "target": column_ids})
        return CuDFEdgeSet(df, is_directed=is_directed)

    @translator
    def translate_edgemap_scipyedgemap2cudfedgemap(
        x: ScipyEdgeMap, **props
    ) -> CuDFEdgeMap:
        is_directed = ScipyEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = x.value.tocoo()
        row_ids = x.node_list[coo_matrix.row]
        column_ids = x.node_list[coo_matrix.col]
        weights = coo_matrix.data
        if not is_directed:
            mask = row_ids <= column_ids
            row_ids = row_ids[mask]
            column_ids = column_ids[mask]
            weights = weights[mask]
        df = cudf.DataFrame(
            {"source": row_ids, "target": column_ids, "weight": weights}
        )
        return CuDFEdgeMap(df, is_directed=is_directed)

    @translator
    def translate_edgeset_cudfedgeset2scipyedgeset(
        x: CuDFEdgeSet, **props
    ) -> ScipyEdgeSet:
        cdf = x.value
        is_directed = x.is_directed
        if not is_directed:
            self_loop_mask = cdf[x.src_label] == cdf[x.dst_label]
            self_loop_df = cdf[self_loop_mask]
            no_self_loop_df = cdf[~self_loop_mask]
            repeat_df = no_self_loop_df.rename(
                columns={x.src_label: x.dst_label, x.dst_label: x.src_label}
            )
            cdf = cudf.concat([no_self_loop_df, repeat_df, self_loop_df,])
        node_list = cupy.unique(cdf[[x.src_label, x.dst_label]].values.ravel())
        num_nodes = len(node_list)
        source_positions = cupy.searchsorted(node_list, cdf[x.src_label].values)
        target_positions = cupy.searchsorted(node_list, cdf[x.dst_label].values)

        node_list = cupy.asnumpy(node_list)
        source_positions = cupy.asnumpy(source_positions)
        target_positions = cupy.asnumpy(target_positions)
        matrix = ss.coo_matrix(
            (np.ones(len(source_positions)), (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeSet(matrix, node_list, aprops={"is_directed": is_directed})

    @translator
    def translate_edgemap_cudfedgemap2scipyedgemap(
        x: CuDFEdgeMap, **props
    ) -> ScipyEdgeMap:
        cdf = x.value
        is_directed = x.is_directed
        if not is_directed:
            self_loop_mask = cdf[x.src_label] == cdf[x.dst_label]
            self_loop_df = cdf[self_loop_mask]
            no_self_loop_df = cdf[~self_loop_mask]
            repeat_df = no_self_loop_df.rename(
                columns={x.src_label: x.dst_label, x.dst_label: x.src_label}
            )
            cdf = cudf.concat([no_self_loop_df, repeat_df, self_loop_df,])
        node_list = cupy.unique(cdf[[x.src_label, x.dst_label]].values.ravel())
        num_nodes = len(node_list)
        source_positions = cupy.searchsorted(node_list, cdf[x.src_label].values)
        target_positions = cupy.searchsorted(node_list, cdf[x.dst_label].values)

        node_list = cupy.asnumpy(node_list)
        source_positions = cupy.asnumpy(source_positions)
        target_positions = cupy.asnumpy(target_positions)
        weights = cupy.asnumpy(cdf[x.weight_label].values)
        matrix = ss.coo_matrix(
            (weights, (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeMap(matrix, node_list, aprops={"is_directed": is_directed})
