#!/usr/bin/env python3
"""
Setup script for Requirements Engineering Multimedia Dataset
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="requirements-engineering-dataset",
    version="1.0.0",
    author="Cornelius Chimuanya Okechukwu",
    author_email="okechukwu@utb.cz",
    description="Dataset and framework for multimedia-enhanced requirements engineering research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/multimedia-re-study/dataset",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Requirements Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "generate-re-dataset=dataset_generator:main",
            "validate-re-dataset=validate_dataset:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/multimedia-re-study/dataset/issues",
        "Source": "https://github.com/multimedia-re-study/dataset",
        "Documentation": "https://github.com/multimedia-re-study/dataset/blob/main/docs/user_guide.md",
        "Paper": "https://doi.org/10.5281/zenodo.XXXXXX",
    },
    keywords=[
        "requirements-engineering",
        "multimedia-data",
        "empirical-software-engineering",
        "natural-language-processing",
        "computer-vision",
        "dataset",
        "research",
    ],
)
