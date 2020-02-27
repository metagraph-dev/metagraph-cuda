import metagraph_cuda as mg_cuda


def test_version():
    assert isinstance(mg_cuda.__version__, str)
