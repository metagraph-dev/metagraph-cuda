from setuptools import setup, find_packages
import versioneer

setup(
    name="metagraph-cuda",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="CUDA plugins for Metagraph",
    author="Anaconda, Inc.",
    packages=find_packages(include=["metagraph_cuda", "metagraph_cuda.*"]),
)
