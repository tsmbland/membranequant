Early version of C. elegans membrane quantification code which has since evolved and been split into several separate repositories:

- The autofluorescence correction method evolved into SAIBR, see [here](https://github.com/goehringlab/saibr_fiji_plugin) and [here](https://github.com/goehringlab/saibr_python)
- The quantification/segmentation method has been split into two repositories which serve slightly different purposes:
  - [Segmentation and basic quantification](https://github.com/tsmbland/par-segmentation) (more general purpose)
  - [Advanced quantification](https://github.com/tsmbland/discco) (more niche for special applications)
  

Nevertheless, this repo still has several features which I have not yet incorporated into the new repositories, including detailed notebooks and a GUI, so may still be a useful resource.

---

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

(NB some notebooks are slightly out of date and don't account for some of the latest changes to the code)
