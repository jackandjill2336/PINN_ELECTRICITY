from setuptools import setup, find_packages

setup(
    name="pinn_electricity",
    version="0.1.0",
    description="Physics-Informed Neural Networks for Electrical Engineering",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
)
