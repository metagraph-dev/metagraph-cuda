from metagraph import PluginRegistry

# Use this as the entry_point object
registry = PluginRegistry()


def find_plugins():
    # Ensure we import all items we want registered
    from . import types, translators, algorithms

    registry.register_from_modules(types, translators, algorithms)
    return registry.plugins


################
# Import guards
################
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

try:
    import cupy as _

    has_cupy = True
except ImportError:
    has_cupy = False
