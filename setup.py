from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README.md
with Path("README.md").open("r", encoding="utf-8") as f:
    long_description = f.read()

# Define package dependencies
install_requires = ["pytorch", "torchvision", "kagglehub", "tqdm"]

setup(
    name="transferbench",
    version="0.1.0",  # Follow semantic versioning (e.g., MAJOR.MINOR.PATCH)
    author="Fabio Brau",
    author_email="fabio.brau@unica.it",
    description="A benchmark toolkit for evaluating the transferability between models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fabiobrau/transfer-bench",
    packages=find_packages(include=["transferbench", "transferbench.*"]),
    python_requires=">=3.7",  # Minimum Python version
    install_requires=install_requires,
    extras_require={
        "dev": ["pytest>=7.0", "black>=22.0"],  # Dev dependencies
        "all": ["timm>=0.4.0", "transformers>=4.12.0"],  # Optional model backends
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "transferbench-cli=transferbench.cli:main",  # Optional CLI
        ],
    },
    include_package_data=True,  # For non-Python files (e.g., configs)
)
