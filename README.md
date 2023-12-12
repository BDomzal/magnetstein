# Magnetstein
![main_workflow_final](https://github.com/BDomzal/magnetstein/assets/65540968/f3c35cfb-e996-4f2a-b01b-3007a7676bd0)

This repository contains software tools which allow to compare nuclear magnetic resonance (NMR) spectra and estimate proportions of components in mixture using the Wasserstein distance. 

Magnetstein is a modification of the algorithm from a Python3 package called `masserstein` (available [here](https://github.com/mciach/masserstein)). 

If you encounter any difficulties during installation or usage of these programs, or if you have any suggestions regarding their functionality, please post a GitHub issue or send an email to b.domzal@mimuw.edu.pl. 

# Installation

To be able to use the software provided in this repository, you will need to have a working Python3 distribution installed on your computer.  

To use Magnetstein, clone this repository. In the commandline, this can be done by typing:

```
git clone https://github.com/BDomzal/magnetstein.git
```

The above command will create a folder `magnetstein` in your current working directory. Go to this directory by typing

```
cd magnetstein/
```

in the commandline. Then, install the package by running the `setup.py` file:

```
python3 setup.py install --user
```

This will install the `masserstein` package for the current user (including NMR spectroscopy tool Magnetstein).  

You will also need to have the following packages installed (all availiable via pip):

* `IsoSpecPy`
* `numpy`
* `scipy`
* `PuLP`

(For example: if you would like to install pulp, you need to type

```
pip install PuLP
```

in the commandline.)

If you are a researcher, we strongly recommend using Gurobi (available for academics at no cost) as your solver in Magnetstein. For more information on license and installation, see [Gurobi website](https://www.gurobi.com/). Gurobi is a default solver. If you prefer to use Magnetstein without Gurobi, set solver=LpSolverDefault in estimate_proportions function. Note that using Magnetstein without Gurobi can result in long computation time and, in some cases, incorrect results.

# Examples

See estimation.ipynb in folder examples/.

# Acknowledgements

Powered by [© Gurobi.](https://www.gurobi.com/)

# Citing 

Article about Magnetstein is in preparation. If you use tools from this package, please include link to this repository in citation.

