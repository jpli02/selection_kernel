from setuptools import setup, find_packages

setup(
    name="selection_kernel",
    version="0.0.1",
    description="A project implementing optimized attention mechanisms using Triton and PyTorch.",
    author="Neel Dani, Jianping Li",
    author_email="neeld2@illinois.edu, jli199@illinois.edu",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
