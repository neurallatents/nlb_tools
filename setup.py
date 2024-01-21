from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="nlb_tools",
    version="0.0.3",
    description="Python tools for participating in Neural Latents Benchmark '21",
    packages=find_packages(),
    install_requires=requirements,
    author="Felix Pei",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    extras_require={
        "dev": ["pytest", "dandi"],
    },
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    setup_requires=["setuptools>=61.0.0", "wheel"],
    url="https://github.com/neurallatents/nlb_tools",
)
