"""
DeltaVLM Setup Script

Install with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="deltavlm",
    version="1.0.0",
    author="DeltaVLM Team",
    author_email="",
    description="Interactive Remote Sensing Image Change Analysis with Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hanlinwu/DeltaVLM",
    packages=find_packages(exclude=["tests", "docs", "data"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "pre-commit",
        ],
    },
)


