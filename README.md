# About the project
jaxman is yet another library that implements the [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter). This library is mainly intended as a self-study project in which to learn to [JAX](https://github.com/google/jax). The implementation is inspired by both [pykalman](https://pykalman.github.io/) and [simdkalman](https://github.com/oseiskar/simdkalman), and like simdkalman implements support for vectorized inference across multiple timeseries. As it's implemented in JAX and utilizes its JIT capabilities, we achieve blazing fast inference for 1,000s (sometimes even 100,000s) of parallel timeseries.

# Getting started
Follow the below instructions in order to get started with jaxman.

## Installation
The library is currently not available on pypi, and there are currently no plans on releasing it there, so install it via
```
https://github.com/tingiskhan/jaxman
```

# Usage
TODO

# Disclaimers
Note that this project is not endorsed, affiliated or supported by Google/JAX, the name is just a mash-up of Kalman and JAX.