import sys
import numpy
import matplotlib
import sklearn
import numba
import tqdm

print("Python version= {0}.{1}.{2}".format(*sys.version_info[:3]))
print("numpy version=", numpy.__version__)
print("matplotlib version=", matplotlib.__version__)
print("sklearn version=", sklearn.__version__)
print("numba version=", numba.__version__)
print("tqdm version=", tqdm.__version__)
print()
