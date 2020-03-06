# A Global Inventory of Utility-Scale Solar Photovoltaic Generating Stations
Repository for machine learning and remote sensing pipeline described in Kruitwagen, L., Story, K., Friedrich, J, et. al. (2019) , used to produce a global inventory of utility-scale solar photvoltaic generating stations.

# Paper Summary



# Repository

## Setup

### Virtual Environment

We recommend using Conda for package and environment management. Create a new conda environment:

    conda create -n solar-pv python=3.6

### Clone Repository

Clone this repository using git:

    git clone

Add the directory root to the Python path environment variable:

    export PYTHONPATH=$(pwd):$PYTHONPATH

(optional) You may want to add this to a bash script for your environment:

    touch //path/to/conda/envs/solar-pv/etc/conda/activate.d/env_vars.sh
    nano //path/to/conda/envs/solar-pv/etc/conda/activate.d/env_vars.sh

Then input:

    export PYTHONPATH=$(pwd):$PYTHONPATH

and save and exit.

### Install Packages

Install Python packages via pip:

    pip install -r requirements.txt

### Descartes Labs

Descartes Lab alpha and Airbus SPOT6/7 access is required to run this repository.

### Google Basemap


## Directories and Scripts
- **training:**
  - `training_data.py`: scripts for assembling training data
  - `make_unet.py`
  - `train_model.py`
- **deployment:** Pytest tests.
- **analysis:** Exploratory Jupyter notebooks.
- `utils`: Shared utilities


## Managing Workflow

`manager.py`