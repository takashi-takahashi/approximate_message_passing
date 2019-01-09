# Approximate Message Passing(AMP) for LASSO
A Python implementation of Approximate Message Passing (AMP) algorithms for LASSO

## references
These algorithms are implemented based on the following papers 
* [Message-passing algorithms for compressed sensing
](https://www.pnas.org/content/106/45/18914.short)
* [Vector approximate message passing](https://ieeexplore.ieee.org/document/8006797)
    - and its [longer version](https://arxiv.org/abs/1610.03082)

## content
* ampy
    - approximate message passing solvers for the Standard Linear Model
* [0]AMP.ipynb
    - a demonstration notebook for AMP 
* [1]Self Averaging AMP.ipynb  
    - a demonstration notebook for Self Averaging AMP
* [2]Naive Self Averaging VAMP.ipynb
    - a demonstration notebook for Naive Self Averaging VAMP
    - *Naive* in the sense that matrix inverse of size N x N
* [3]Self Averaging VAMP.ipynb
    - a demonstration notebook for Self Averaging VAMP
    - in this version singular value decomposition is utilized to avoid computational cost of N^3 for each iteration
* [3-1]: same with the [3] except that the observation matrix is ​​drawn from random DCT matrix ensemble. 
    
## requirements
* Python version = 3.6.7
* numpy version = 1.15.4
* matplotlib version = 3.0.2
* sklean version = 0.20.1
* numba version = 0.41.0
* tqdm version = 4.28.1

