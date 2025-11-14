from setuptools import setup, find_namespace_packages

setup(
    name="vis_foxglove",
    version="0.0.1",
    author="lhrrhl0419",
    description="visualization with foxglove",
    long_description=open("README.md").read(),
    packages=find_namespace_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "pin",
        "transforms3d",
        "trimesh",
        "foxglove-sdk",
        "foxglove-data-platform",
        "flask",
    ],
)
