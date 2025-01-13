# Reachy2 symbolic inverse kinematics

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![linter](https://github.com/pollen-robotics/reachy2_symbolic_ik/actions/workflows/lint.yml/badge.svg) 
![pytest](https://github.com/pollen-robotics/reachy2-sdk/actions/workflows/unit_tests.yml/badge.svg)

**A kinematics library for Reachy2 7 DoF robotic arms**, providing precise and reliable tools for motion control.

## Key Features
1. **Symbolic Inverse Kinematics Solver**:
   - Provides exact solutions, free from numerical solver pitfalls like initial seed dependence or local minima.
   - Handles joint limits.
   - Solves reachability questions.
   - Provides symbolic expressions for the null space — a fancy way of saying we can choose where to place the elbow on a circle.
2. **Task-Space Control Algorithm**:
   - Ensures joint-space continuity, even for multi-turn joints (e.g., wrist_yaw).
   - Handles unreachable poses gracefully within trajectories.
   - Customizable workspace and configuration parameters.


| <p align="center"><img src="./docs/img/topgrasp.gif" alt="Top Grasp Demo" width="90%"></p> | <p align="center"><img src="./docs/img/nullspace.gif" alt="Null Space Visualization" width="100%"></p> |
|--------------------------------------------|-----------------------------------------------------|

## Table of Contents
- [Reachy2 symbolic inverse kinematics](#reachy2-symbolic-inverse-kinematics)
  - [Key Features](#key-features)
  - [Table of Contents](#table-of-contents)
  - [Understanding how it works](#understanding-how-it-works)
  - [Install](#install)
  - [Usage](#usage)
  - [Unit tests](#unit-tests)
  - [URDF](#urdf)
  - [License](#license)




## Understanding how it works
Learn the core concepts behind our symbolic inverse kinematics approach (French with English subtitles):

<p align="center">
  <a href="https://youtu.be/FHZdJbMAmRA?si=wepM4vH2sNLo85QP&t=344" title="IK explained">
    <img src="./docs/img/thumbnail.jpeg" alt="IK explained" width="60%">
  </a>
</p>




## Install

Use the following command to install:

```console
$ pip install -e .[dev]
```
The optional *[dev]* option includes tools for developers.

## Usage
Basic example of an inverse kinematics call. The input is a Pose of dimension 6, the output is the 7 joints of the arm:
```python
symbolic_ik = SymbolicIK(arm="l_arm")

# The input is a Pose = [position, orientation]
goal_position = [0.55, -0.3, -0.15]
goal_orientation = [0, -np.pi / 2, 0]
goal_pose = np.array([goal_position, goal_orientation])

# Check if the goal pose is reachable
is_reachable, interval, get_joints, _ = symbolic_ik_r.is_reachable(goal_pose)

if is_reachable:
    # Choose the elbow position inside the valid interval
    theta = interval[0]
    joints, elbow_position = get_joints(theta)
else:
    print("Pose not reachable")
```
Check the /src/example folder for complete examples.

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

## License

This project is licensed under the [Apache 2.0 License](LICENSE). See the LICENSE file for details.
