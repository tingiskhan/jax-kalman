# About the project
kaxman is yet another library that implements the [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter). The 
library is built on top of JAX and is designed to be fast and efficient. The library is still in its early stages and
is not yet feature complete. Some of the features include:
- JIT:able Kalman filter class.
- Support for fully/partially missing observations via inflation of variance.
- Support for time-varying state transition and observation matrices.
- Support for time-varying process and observation noise covariance matrices.
- Support for noise transform, e.g. having the same noise for multiple states.
- Rauch-Tung-Striebel smoother.


# Getting started
Follow the below instructions in order to get started with kaxman.

## Installation
The library is currently not available on pypi, so install it via
```
https://github.com/tingiskhan/kaxman
```

# Usage
TODO

# Disclaimers
Note that this project is not endorsed, affiliated or supported by Google/JAX, the name is just a mash-up of Kalman and JAX.