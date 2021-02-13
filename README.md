# MembraneQuant

Functions for segmentation and accurate quantification of membrane and cytoplasmic protein concentrations from midplane confocal images of C. elegans zygotes


## Install instructions

You will need to install several packages in order to run this code. 
The easiest way to do this is with [Anaconda](https://docs.anaconda.com/anaconda/install/). 
Assuming Anaconda is already installed on your machine, you can set up an environment and install the necessary packages by navigating to this folder in terminal (MembraneQuant), and running the following commands in order:

    conda create -n membranequant python=3.7 anaconda=2020.11
    conda activate membranequant
    pip install -r requirements.txt

The python code will now be ready to run. 
You can return to this environment in the future simply by running:

    conda activate membranequant


## Running notebooks

The repository contains a series of notebooks which fully explain of the methods, and provide instructions for running analysis.
Open these notebooks from terminal by running:

    jupyter notebook notebooks/INDEX.ipynb


## Graphical user interface

Simple analysis can be performed with a graphical user interface. 
This has limited capabilities compared to scripting, but provides a quick and easy way to run simple analysis. 
Open the graphical user interface from terminal by running:

    python open_gui.py

A new window will pop up.
For help and instructions, click the Help button at the bottom of this window.