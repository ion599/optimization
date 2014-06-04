Convex optimization for traffic assignment
==========================================

Setup
-----
Python dependencies (once only):

    sudo easy_install pip
    pip install -r requirements.txt

MATLAB dependencies (must be run every time MATLAB is started):

    setup.m

Running via MATLAB
-------------------
Run `main.m`.

Running via Python
-------------------
Run the python implementation from the `traffic-estimation/python` directory.

To run the main test, see these examples:
```
cd ~/traffic-estimation/python
python main.py --file data/stevesSmallData.mat --log=DEBUG --solver LBFGS
python main.py --file data/stevesSmallData.mat --log=DEBUG --solver BB
python main.py --file data/stevesSmallData.mat --log=DEBUG --solver DORE
```
If the dataset you want to run is not in the data directory, symlink it in
from the main dataset.

To run 3-fold cross validation test:
```
python CrossValidation.py --log=DEBUG
```

References
--------
Mark Schmidt's [L1General](http://www.di.ens.fr/~mschmidt/Software/L1General.html), a set of Matlab routines for solving L1-regularization problems. 
