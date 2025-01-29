# Reachy2 symbolic inverse kinematics

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![linter](https://github.com/pollen-robotics/reachy2_symbolic_ik/actions/workflows/lint.yml/badge.svg) 
![pytest](https://github.com/pollen-robotics/reachy2_symbolic_ik/actions/workflows/pytest.yml/badge.svg)

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

To better understand the frame conventions used in the IK, the following figure illustrates the torso frame and the end-effector frame:

<table>
  <tr>
    <td align="center">
      <img src="./docs/img/reachy_frames1.png" alt="Top Grasp Demo" width="100%">
    </td>
    <td align="center">
      <img src="./docs/img/reachy_frames2.png" alt="Null Space Visualization" width="100%">
    </td>
  </tr>
</table>

For more detailed explanations, benchmarks, and discussions on specific issues, you can refer to our Notion page: [Notion IK Documentation](https://www.notion.so/pollen-robotics/Symbol-IK-27a420dfc4404c52a02fe93c10142229) (in French).

## Install

Use the following command to install:

```console
$ pip install -e .[dev]
```
The optional *[dev]* option includes tools for developers.

## Usage
Basics examples of an inverse kinematics call. The input is a Pose of dimension 6, the output is the 7 joints of the arm:

<details>
  <summary>Example with SymbolicIK</summary>


 ```python
import numpy as np
from reachy2_symbolic_ik.symbolic_ik import SymbolicIK

#Create the symbolic IK for the right arm
symbolic_ik = SymbolicIK(arm="r_arm")

# Define the goal position and orientation
goal_position = [0.55, -0.3, -0.15]
goal_orientation = [0, -np.pi / 2, 0]
goal_pose = np.array([goal_position, goal_orientation])

# Check if the goal pose is reachable
is_reachable, theta_interval, theta_to_joints_func, state = symbolic_ik.is_reachable(goal_pose)

# Get the joints for one elbow position, defined by the angle theta
if is_reachable:
    # Choose a theta in the interval
    # if theta_interval[0] < theta_interval[1], theta can be any value in the interval
    # else theta can be in the intervals [-np.pi, theta_interval[1]] or [theta_interval[0], np.pi]
    theta = theta_interval[0]

    # Get the joints 
    joints, elbow_position = theta_to_joints_func(theta)
    print(f"Pose is reachable \nJoints: {joints}")
else:
    print("Pose not reachable")
```
</details>

<details>
  <summary>Example with ControlIK</summary>

```python
import numpy as np
from reachy2_symbolic_ik.control_ik import ControlIK
from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix
from scipy.spatial.transform import Rotation as R

# Create the control IK for the right arm
control = ControlIK(urdf_path="../config_files/reachy2.urdf")

# Define the goal position and orientation
goal_position = [0.55, -0.3, -0.15]
goal_orientation = [0, -np.pi / 2, 0]
goal_pose = np.array([goal_position, goal_orientation])
goal_pose = make_homogenous_matrix_from_rotation_matrix(goal_position, R.from_euler("xyz", goal_orientation).as_matrix())

# Get joints for the goal pose
# The control type can be "discrete" or "continuous"
# If the control type is "discrete", the control will choose the best elbow position for the goal pose
# If the control type is "continuous", the control will choose a elbow position that insure continuity in the joints
control_type = "discrete"
joints, is_reachable, state = control.symbolic_inverse_kinematics("r_arm", goal_pose, control_type)
if is_reachable:
    print(f"Pose is reachable \nJoints: {joints}")
else:
    print("Pose not reachable")
```
</details>

<details>
  <summary>Example with SDK</summary>

For this example, you will need [Reachy2 SDK](https://github.com/pollen-robotics/reachy2-sdk)
```python
import time
import numpy as np
from reachy2_sdk import ReachySDK
from scipy.spatial.transform import Rotation as R
from reachy2_symbolic_ik.utils import make_homogenous_matrix_from_rotation_matrix

# Create the ReachySDK object
print("Trying to connect on localhost Reachy...")
reachy = ReachySDK(host="localhost")

time.sleep(1.0)
if reachy._grpc_status == "disconnected":
    print("Failed to connect to Reachy, exiting...")
    return

reachy.turn_on()

# Define the goal pose
goal_position = [0.55, -0.3, -0.15]
goal_orientation = [0, -np.pi / 2, 0]
goal_pose = np.array([goal_position, goal_orientation])
goal_pose = make_homogenous_matrix_from_rotation_matrix(goal_position, R.from_euler("xyz", goal_orientation).as_matrix())

# Get joints for the goal pose
joints = reachy.r_arm.inverse_kinematics(goal_pose)

# Go to the goal pose
reachy.r_arm.goto(joints, duration=4.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)
```
</details>

Check the `/src/example` folder for complete examples.

## Unit tests

To ensure everything is functioning correctly, run the unit tests.

```console
$ pytest 
```

Some unit tests need [Reachy2 SDK](https://github.com/pollen-robotics/reachy2-sdk).

You can decide which test you want to run with a flag.
- sdk : run tests with sdk
- cicd : run tests using only reachy2_symbolic_ik

Example :

```console
$ pytest -m cicd
```
or 
```console
$ python3 -m pytest -m cicd
```

## Documentation

The Documentation can be generated locally via pdoc with:

```console
pdoc reachy2_symbolic_ik --output-dir docs --logo "https://pollen-robotics.github.io/reachy2-sdk/pollen_logo.png" --logo-link "https://www.pollen-robotics.com" --docformat google
```


## URDF

A URDF file is provided in 'src/config_files/reachy2.urdf'. This file is used if the user does not provide a URDF file when initializing the ControlIK class.

To regenerate the URDF file, you can use the following command from the root of the repository in the Docker container:

```console
$ xacro ../../reachy_ws/src/reachy2_core/reachy_description/urdf/reachy.urdf.xacro "use_fake_hardware:=true" > src/config_files/reachy2.urdf
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE). See the LICENSE file for details.
