from setuptools import setup, find_packages

setup(
    name="cmf_net",
    version="0.1.0",
    description="Cross-Modal Feature Fusion Network for Volumetric Medical Image Segmentation",
    author="Anonymous Authors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
    ],
)
