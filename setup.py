#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="llm_inference",
    version="0.0.1",
    author="Phillip Rust",
    author_email="plip.rust@gmail.com",
    url="https://github.com/xplip/llm-inference",
    description="",
    license="Apache 2.0",
    python_requires=">=3.8",
    install_requires=[
        "vllm>=0.4.3",
        "pandas",
        "hydra-core",
        "hydra-submitit-launcher",
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=True,
)
