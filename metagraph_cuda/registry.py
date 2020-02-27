from metagraph import PluginRegistry

# Use this as the entry_point object
registry = PluginRegistry()


def find_plugins():
    # Ensure we import all items we want registered
    from . import wrappers, translators, algorithms

    registry.register_from_modules(wrappers, translators, algorithms)
    return registry.plugins



################
# Import guards
################
try:
    import cudf
except ImportError:
    cudf = None

try:
    import cugraph
except ImportError:
    cugraph = None

try:
    import pandas
except ImportError:
    pandas = None

try:
    import numpy
except ImportError:
    numpy = None
