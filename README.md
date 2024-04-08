# Reachy2 symbolic inverse kinematics

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![linter](https://github.com/pollen-robotics/reachy2_symbolic_ik/actions/workflows/lint.yml/badge.svg) 
![pytest](https://github.com/pollen-robotics/reachy2-sdk/actions/workflows/unit_tests.yml/badge.svg)


## Install

Use the following command to install:

```console
$ pip install -e .[dev]
```
The *[dev]* option includes tools for developers.


## Unit tests

To ensure everything is functioning correctly, run the unit tests.

```console
$ pytest 
```

Some unit tests need [Placo](https://github.com/pollen-robotics/reachy_placo) and some need reachy2_sdk.

You can decide which test you want to run with a flag.
- sdk : run tests with sdk
- placo : run tests with placo
- cicd : run tests using only reachy2_symbolic_ik

Example :

```console
$ pytest -m cicd
```


