#!/usr/bin/env python3
"""Packaging for pareto-dpo — scalarization-free multi-objective preference
alignment for generative genomic (sgRNA) design."""
from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


def _reqs():
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return [ln.strip() for ln in lines
            if ln.strip() and not ln.strip().startswith("#")]


setup(
    name="pareto-dpo",
    version="0.1.0",
    description="Scalarization-free multi-objective DPO for generative genomic design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pareto-DPO authors",
    license="MIT",
    url="https://github.com/malekpouri/pareto-dpo",
    packages=find_packages(include=["models", "models.*", "scripts", "scripts.*"]),
    py_modules=[],
    python_requires=">=3.10",
    install_requires=_reqs(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="DPO preference-alignment CRISPR sgRNA multi-objective Pareto genomics",
)
