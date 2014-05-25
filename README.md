Convex optimization for traffic demand estimation
====================


Setup
-----
Python dependencies (once only):

    sudo easy_install pip
    pip install -r requirements.txt

Matlab dependencies (must be run every time):

    setup.m

Running
-----
Run `main.m`.

Python Instructions
-------------------
To run, cd into the `traffic-estimation/python` directory, and run the `main.py`
script:
```
cd ~/traffic-estimation/python
python2 main.py ../data/stevesSmallData.mat --log=DEBUG
```
If the dataset you want to run is not in the data directory, symlink it in
from the main dataset.

Troubleshooting
--------
See if the examples provided by L1General run:

Compile mex files (not necessary on all systems): `mexAll`                

Runs a demo of the (older) L1General codes: `example_L1General`

Runs a demo of the (newer) L1General codes: `demo_L1General`

Runs a demo of computing a regularization path: `demo_L1Generalpath`

References
--------
Mark Schmidt's [L1General](http://www.di.ens.fr/~mschmidt/Software/L1General.html), a set of Matlab routines for solving L1-regularization problems. 
