# cem
Computational Electromagnetics

# Setup the development environment
1. Create a conda environment: `conda create --name cem python=3.7`
1. Activate the conda environment: `conda activate cem`
1. Install pytest: `conda install pytest`
1. Install yapf: `conda install yapf`
1. Setup package for development: `pip install -e .`
1. (Optional) GPU accelerate library CuPy: `conda install -c conda-forge cupy`, 
see more details in [CuPy](https://docs.cupy.dev/en/stable/index.html)