############################
# Libraries used as plugins
############################

try:
    import cudf as _

    has_cudf = True
except ImportError:
    has_cudf = False

try:
    import cugraph as _

    has_cugraph = True
except ImportError:
    has_cugraph = False


import metagraph

# Use this as the entry_point object
registry = metagraph.PluginRegistry("metagraph_cuda")


def find_plugins():
    # Ensure we import all items we want registered
    from . import cudf, cugraph

    registry.register_from_modules(cudf, name="metagraph_cuda_cudf")
    registry.register_from_modules(cugraph, name="metagraph_cuda_cugraph")
    return registry.plugins
