# 1-Norm_calculations

## Installation

To install the right libraries, start with a clean python 3.7 environment with numpy and matplotlib.

Install PySCF 1.7.1\
With pip:\
`pip install pyscf==1.7.1`\
Or conda:\
`conda install -c pyscf pyscf==1.7.1`

If you want to run CASCI calculations with this code, find your pyscf installation folder (usually in {Name of environment}/lib/python3.7/site-packages/pyscf), and please copy the file casci.py from this repository to the pyscf/mcscf/ folder and overwrite.

Now install the adapted code of OpenFermion and the OpenFermion-PySCF interface from my Fork repositories:
`git clone https://github.com/Emieeel/OpenFermion.git`

Go to the right branch (important!) and install:

`cd OpenFermion`\
`git checkout 1normcalc`\
`python -m pip install -e .`

Same with the PySCF interface:\
`git clone https://github.com/Emieeel/OpenFermion-PySCF.git`\
`cd OpenFermion-PySCF`\
`git checkout localorb`\
`python -m pip install -e .`

Finally if you want to visualize molecules in a jupyter notebook, you can install py3Dmol.

To install py3dmol with pip, run\
`pip install py3Dmol`\
To install py3dmol with conda, run\
`conda install -c conda-forge py3dmol`

Now you're ready to go!

For a tutorial to Calculating the 1-norm for different orbitals and molecules, see the notebook 1normcalc.ipynb in this repository.
