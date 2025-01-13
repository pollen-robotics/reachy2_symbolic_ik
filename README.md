# Reachy2 symbolic inverse kinematics

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![linter](https://github.com/pollen-robotics/reachy2_symbolic_ik/actions/workflows/lint.yml/badge.svg) 
![pytest](https://github.com/pollen-robotics/reachy2-sdk/actions/workflows/unit_tests.yml/badge.svg)

## About

<!-- Row 1: Text on the left, GIF 1 on the right -->
<div style="display: flex; align-items: center; margin-bottom: 20px;">
  <div style="flex: 1; padding-right: 10px;">
    <p>Top grasp is a fundamental robotic manipulation technique used to grip objects securely. 
    This demonstration shows its practical application on the real robot.</p>
  </div>
  <div style="flex: 1;">
    <img src="./docs/img/topgrasp.gif" alt="Top grasp demo" style="width: 100%; border: 1px solid #ddd;">
  </div>
</div>

<!-- Row 2: GIF 2 on the left, Text on the right -->
<div style="display: flex; align-items: center; margin-bottom: 20px;">
  <div style="flex: 1;">
    <img src="./docs/img/nullspace.gif" alt="Null space demo" style="width: 100%; border: 1px solid #ddd;">
  </div>
  <div style="flex: 1; padding-left: 10px;">
    <p>The null space control demo showcases how redundant degrees of freedom in a robot arm 
    can be exploited to achieve multiple objectives simultaneously.</p>
  </div>
</div>

<!-- Row 3: Text on the left, Thumbnail of Video on the right -->
<div style="display: flex; align-items: center; margin-bottom: 20px;">
  <div style="flex: 1; padding-right: 10px;">
    <p>This video explains the principles of inverse kinematics (IK) and how it is implemented in the Reachy2 robot.</p>
  </div>
  <div style="flex: 1;">
    <a href="https://youtu.be/FHZdJbMAmRA?si=wepM4vH2sNLo85QP&t=344" title="IK explained">
      <img src="./docs/img/thumbnail.jpeg" alt="IK explained" style="width: 100%; border: 1px solid #ddd;">
    </a>
  </div>
</div>

xxxxxxxx


<!-- Row 1: Text on the left, GIF 1 on the right -->
| Text | GIF |
|------|-----|
| Top grasp is a fundamental robotic manipulation technique used to grip objects securely. This demonstration shows its practical application on the real robot. | ![Top grasp demo](./docs/img/topgrasp.gif) |

<!-- Row 2: GIF 2 on the left, Text on the right -->
| GIF | Text |
|-----|------|
| ![Null space demo](./docs/img/nullspace.gif) | The null space control demo showcases how redundant degrees of freedom in a robot arm can be exploited to achieve multiple objectives simultaneously. |

<!-- Row 3: Text on the left, Thumbnail of Video on the right -->
| Text | Image |
|------|-------|
| This video explains the principles of inverse kinematics (IK) and how it is implemented in the Reachy2 robot. | [![IK explained](./docs/img/thumbnail.jpeg)](https://youtu.be/FHZdJbMAmRA?si=wepM4vH2sNLo85QP&t=344 "IK explained") |


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
