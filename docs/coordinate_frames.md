# Coordinate frames

The reproduction uses explicit frame contracts because the historical project
mixed simulator, Nerfstudio, body, camera, and Vicon conventions.

## Simulation state

The state is

```text
[x, y, z, vx, vy, vz, yaw]
```

in the track world frame, with metres, metres per second, radians, and a fixed
simulation step of 0.05 s unless overridden. Controller acceleration is rotated
from body frame to world frame before the dynamics update.

## Camera and GSplat

`quadpilot.datasets.generation.pose_to_camera_matrix` creates the legacy camera
pose. `quadpilot.perception.renderer.PoseTransform` then applies the exact
Nerfstudio dataparser transform and scale recorded beside each checkpoint.
Camera intrinsics are part of the dataset contract; the reproduced NPE datasets
use 640 x 480 RGB images.

## Hardware

Vicon and NeRF are not assumed to share scale, origin, or yaw. Hardware use
requires an evidence-backed similarity transform

```text
p_vicon = scale * R_vicon_from_nerf * p_nerf + translation
```

plus calibrated camera intrinsics and a body-from-camera extrinsic transform.
The offline calibrator writes the fitted transform and evidence hashes; the
readiness checker refuses to proceed if any frame, topic, unit, or evidence
contract is incomplete.
