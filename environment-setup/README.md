# Instructions for setting up the environment
The dev-environment is anaconda-based and requires it installed. Download and installation should be made from [this source](https://www.anaconda.com/products/individual). Make sure that Anaconda is added to PATH environment variable such that the following command works: 
```console 
conda -V
```

## Create a conda environment, install python dependencies and activate/test the environment

1. Navigate to the repositorys root folder and run ```console conda env create -f environment-setup/conda-environment-<INSERT gpu OR cpu HERE>-win.yml ``` to install dependencies. 
2. Activate the environment by running ```conda activate dml-<INSERT gpu OR cpu HERE>```.
3. Run ```jupyter notebook``` to start Jupyter Notebook.
