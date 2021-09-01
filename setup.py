from setuptools import setup, find_packages

setup(
    name='nlb_tools',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scipy',
        'numpy',
        'scikit-learn',
        'h5py<3,>=2.9',
        'pynwb',
    ],
    author="Neural Latents",
)