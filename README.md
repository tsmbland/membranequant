# MembraneQuant

Functions for segmentation and accurate quantification of membrane and cytoplasmic protein concentrations from midplane confocal images of C. elegans zygotes


## Install instructions

To install the necessary packages, navigate to MembraneQuant folder in terminal, and run the following commands in order:

    conda create -n membranequant python=3.7 anaconda=2020.11\
    conda activate membranequant\
    pip install -r requirements.txt


## Running notebooks

The repository contains a series of notebooks with an explanation of the methods, and instructions for running analysis.
Open these notebooks by running:

    jupyter notebook notebooks/INDEX.ipynb


## Graphical user interface

Simple analysis can be performed with a graphical user interface. 
This has limited capabilities compared to scripting, but provides a quick and easy way to run simple analysis. 
Open the graphical user interface by calling:

    python open_gui.py

For help and instructions, click the Help button at the bottom of the GUI window.