from metagraph import translator, dtypes
from metagraph.plugins import has_pandas, has_scipy
import numpy as np
from .. import has_cudf
from metagraph.plugins.numpy.types import NumpyNodeSet, NumpyNodeMap, NumpyVector
from metagraph.plugins.python.types import PythonNodeSet, PythonNodeMap, dtype_casting

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
    ) -> PythonNodeMap:
        cast = dtype_casting[dtypes.dtypes_simplified[x.value[x.value_label].dtype]]
        data = {
            i.item(): cast(x.value.loc[i.item()].loc[x.value_label])
            for i in x.value.index.values
        }
        return PythonNodeMap(data)

    @translator
    def translate_nodes_pythonnodemap2cudfnodemap(
        x: PythonNodeMap, **props
    ) -> CuDFNodeMap:
        keys, values = zip(*x.value.items())
        # TODO consider special casing the situation when all the keys form a compact range
        data = cudf.DataFrame({"value": values}, index=keys)
        return CuDFNodeMap(data, "value")

    @translator
    def translate_nodes_cudfnodeset2pythonnodeset(
        x: CuDFNodeSet, **props
    ) -> PythonNodeSet:
        return PythonNodeSet(set(x.value.index))

    @translator
    def translate_nodes_pythonnodeset2cudfnodeset(
        x: PythonNodeSet, **props
    ) -> CuDFNodeSet:
        return CuDFNodeSet(cudf.Series(x.value))

    @translator
    def translate_nodes_numpyvector2cudfvector(x: NumpyVector, **props) -> CuDFVector:
        if x.mask is not None:
            data = x.value[x.mask]
            series = cudf.Series(data, index=np.flatnonzero(x.mask))
        else:
            data = x.value
            series = cudf.Series(data)
        return CuDFVector(series)

    @translator
    def translate_vector_cudfvector2numpyvector(x: CuDFVector, **props) -> NumpyVector:
        is_dense = CuDFVector.Type.compute_abstract_properties(x, {"is_dense"})[
            "is_dense"
        ]
        if is_dense:
            np_vector = cupy.asnumpy(x.value.sort_index().values)
            mask = None
        else:
            series = x.value.sort_index()
            positions = series.index.to_array()
            np_vector = np.empty(len(x), dtype=series.dtype)
            np_vector[positions] = cupy.asnumpy(series.values)
            mask = np.zeros(len(x), dtype=bool)
            mask[positions] = True
        return NumpyVector(np_vector, mask=mask)

    @translator
    def translate_nodes_numpynodemap2cudfnodemap(
        x: NumpyNodeMap, **props
    ) -> CuDFNodeMap:
        if x.mask is not None:
            keys = np.flatnonzero(x.value)
            np_values = x.value[mask]
            # TODO make CuDFNodeMap store a Series instead of DataFrame to avoid making 2 copies here
            df = cudf.Series(np_values, index=keys).to_frame("value")
        elif x.pos2id is not None:
            # TODO make CuDFNodeMap store a Series instead of DataFrame to avoid making 2 copies here
            df = cudf.DataFrame({"value": x.value}, index=x.pos2id)
        else:
            df = cudf.DataFrame({"value": x.value})
        return CuDFNodeMap(df, "value")

    @translator
    def translate_nodes_cudfnodemap2numpynodemap(
        x: CuDFNodeMap, **props
    ) -> NumpyNodeMap:
        if isinstance(x.value.index, cudf.core.index.RangeIndex):
            x_index_min = x.value.index.start
            x_index_max = x.value.index.stop - 1
        else:
            x_index_min = x.value.index.min()
            x_index_max = x.value.index.max()
        x_density = (x_index_max + 1 - x_index_min) / (x_index_max + 1)
        if x_density == 1.0:
            data = np.empty(len(x.value), dtype=x.value[x.value_label].dtype)
            data[cupy.asnumpy(x.value.index.values)] = cupy.asnumpy(
                x.value[x.value_label].values
            )
            mask = None
            node_ids = None
        elif x_density > 0.5:  # TODO consider moving this threshold out to a global
            data = np.empty(x_index_max + 1, dtype=x.value[x.value_label].dtype)
            position_selector = cupy.asnumpy(x.value.index.values)
            data[position_selector] = cupy.asnumpy(x.value[x.value_label].values)
            mask = np.zeros(x_index_max + 1, dtype=bool)
            mask[position_selector] = True
            node_ids = None
        else:
            # O(n log n) sort, but n is small since not dense
            df_index_sorted = x.value.sort_index()
            data = cupy.asnumpy(df_index_sorted[x.value_label].values)
            node_ids = dict(map(reversed, enumerate(df_index_sorted.index)))
            mask = None
        return NumpyNodeMap(data, mask=mask, node_ids=node_ids)

    @translator
    def translate_nodes_numpynodeset2cudfnodeset(
        x: NumpyNodeSet, **props
    ) -> CuDFNodeSet:
        data = cudf.Series(x.nodes())
        return CuDFNodeSet(data)

    @translator
    def translate_nodes_cudfnodeset2numpynodeset(
        x: CuDFNodeSet, **props
    ) -> NumpyNodeSet:
        if isinstance(x.value.index, cudf.core.index.RangeIndex):
            x_index_min = x.value.index.start
            x_index_max = x.value.index.stop - 1
        else:
            x_index_min = x.value.index.min()
            x_index_max = x.value.index.max()
        x_density = (x_index_max + 1 - x_index_min) / (x_index_max + 1)
        node_positions = cupy.asnumpy(x.value.index.values)
        if x_density > 0.5:  # TODO consider moving this threshold out to a global
            mask = np.zeros(x_index_max + 1, dtype=bool)
            mask[node_positions] = True
            node_ids = None
        else:
            node_ids = node_positions
            mask = None
        return NumpyNodeSet(node_ids=node_ids, mask=mask)


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
    from metagraph.plugins.scipy.types import ScipyEdgeSet, ScipyEdgeMap, ScipyGraph

    @translator
    def translate_edgeset_scipyedgeset2cudfedgeset(
        x: ScipyEdgeSet, **props
    ) -> CuDFEdgeSet:
        is_directed = ScipyEdgeSet.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = (
            x.value.tocoo()
        )  # TODO consider handling CSR and COO cases separately
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rc_pairs = zip(row_ids, column_ids)
        if not is_directed:
            rc_pairs = filter(lambda pair: pair[0] < pair[1], rc_pairs)
        rc_pairs = list(rc_pairs)
        df = cudf.DataFrame(rc_pairs, columns=["source", "target"])
        return CuDFEdgeSet(df, is_directed=is_directed)

    @translator
    def translate_edgemap_scipyedgemap2cudfedgemap(
        x: ScipyEdgeMap, **props
    ) -> CuDFEdgeMap:
        is_directed = ScipyEdgeMap.Type.compute_abstract_properties(x, {"is_directed"})[
            "is_directed"
        ]
        coo_matrix = (
            x.value.tocoo()
        )  # TODO consider handling CSR and COO cases separately
        get_node_from_pos = lambda index: x.node_list[index]
        row_ids = map(get_node_from_pos, coo_matrix.row)
        column_ids = map(get_node_from_pos, coo_matrix.col)
        rcw_triples = zip(row_ids, column_ids, coo_matrix.data)
        if not is_directed:
            rcw_triples = filter(lambda triple: triple[0] < triple[1], rcw_triples)
        rcw_triples = list(rcw_triples)
        df = cudf.DataFrame(rcw_triples, columns=["source", "target", "weight"])
        return CuDFEdgeMap(df, is_directed=is_directed)

    @translator
    def translate_edgeset_cudfedgeset2scipyedgeset(
        x: CuDFEdgeSet, **props
    ) -> ScipyEdgeSet:
        is_directed = x.is_directed
        node_list = np.unique(
            cupy.asnumpy(x.value[[x.src_label, x.dst_label]].values).ravel("K")
        )
        node_list.sort()
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        source_positions = list(map(get_id_pos, x.value[x.src_label].values.tolist()))
        target_positions = list(map(get_id_pos, x.value[x.dst_label].values.tolist()))
        if not is_directed:
            source_positions, target_positions = (
                source_positions + target_positions,
                target_positions + source_positions,
            )
        source_positions = np.array(source_positions)
        target_positions = np.array(target_positions)
        matrix = ss.coo_matrix(
            (np.ones(len(source_positions)), (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeSet(matrix, node_list)

    @translator
    def translate_edgemap_cudfedgemap2scipyedgemap(
        x: CuDFEdgeMap, **props
    ) -> ScipyEdgeMap:
        is_directed = x.is_directed
        node_list = np.unique(
            cupy.asnumpy(x.value[[x.src_label, x.dst_label]].values).ravel("K")
        )
        node_list.sort()
        num_nodes = len(node_list)
        id2pos = dict(map(reversed, enumerate(node_list)))
        get_id_pos = lambda node_id: id2pos[node_id]
        source_positions = list(map(get_id_pos, x.value[x.src_label].tolist()))
        target_positions = list(map(get_id_pos, x.value[x.dst_label].tolist()))
        weights = cupy.asnumpy(x.value[x.weight_label].values)
        if not is_directed:
            source_positions, target_positions = (
                source_positions + target_positions,
                target_positions + source_positions,
            )
            weights = np.concatenate([weights, weights])
        matrix = ss.coo_matrix(
            (weights, (source_positions, target_positions)),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        return ScipyEdgeMap(matrix, node_list)
