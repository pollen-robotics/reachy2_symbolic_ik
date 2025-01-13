# Reachy2 symbolic inverse kinematics

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![linter](https://github.com/pollen-robotics/reachy2_symbolic_ik/actions/workflows/lint.yml/badge.svg) 
![pytest](https://github.com/pollen-robotics/reachy2-sdk/actions/workflows/unit_tests.yml/badge.svg)

## About
[![IK explained](./docs/img/thumbnail.jpeg)](https://youtu.be/FHZdJbMAmRA?si=wepM4vH2sNLo85QP&t=344 "IK explained")



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

## URDF

A URDF file is provided in 'src/config_files/reachy2.urdf'. This file is used if the user does not provide a URDF file when initializing the ControlIK class.

To regenerate the URDF file, you can use the following command from the root of the repository in the Docker container:

```console
$ xacro ../../reachy_ws/src/reachy2_core/reachy_description/urdf/reachy.urdf.xacro "use_fake_hardware:=true" > src/config_files/reachy2.urdf
```
