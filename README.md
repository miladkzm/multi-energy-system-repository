## Installation

HEMS has not yet been deployed to Python `pip` / `conda` package indexes, but can be installed in a local development environment as follows:

1. Install `conda`-based Python distribution¹ such as [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Miniforge](https://github.com/conda-forge/miniforge).
2In `conda`-enabled shell (e.g. Anaconda Prompt), run:
    - `cd path_to_hems_repository`
    - `conda create -n hems_pyomo_based -c conda-forge python=3.10 contextily numpy pandas scipy`
    - `conda activate hems`
    - `pip install -r requirements.txt`
