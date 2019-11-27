# A Global Inventory of Utility-Scale Solar Photovoltaic Generating Stations
Repository for machine learning and remote sensing pipeline described in Kruitwagen, L., Story, K., Friedrich, J, et. al. (2019) , used to produce a global inventory of utility-scale solar photvoltaic generating stations.

# Paper Summary



# Repository

## Setup

- requirements
- google drive storage

### Virtual Environment

We recommend using Conda for package and environment management.

    conda create -n solar-pv python=3.6

### Descartes Labs

Descartes Lab alpha access is required


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