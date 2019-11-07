# Ordinal Optimisation Success Probabilities

This repository provides code implemented in Python and MATLAB for computing ordinal optimisation success probabilities for the Gaussian case, as described in [1].

[1] Chin, Robert, Jonathan E. Rowe, Iman Shames, Chris Manzie and Dragan Nešić. "Ordinal Optimisation for Continuous Problems under Gaussian Noise." [arXiv preprint arXiv:1911.01993](https://arxiv.org/abs/1911.01993) (2019)

Notes:

* Due to limitations with MATLAB's `mvncdf` function, `p_success_approx.m` only works for values of m up to 25. The Python version `p_success_approx.py` works for values of m up to 500.
* The MATLAB version `p_success_numerical.m` is recommended over the Python version `p_success_numerical.py` in terms of speed and numerical precision.