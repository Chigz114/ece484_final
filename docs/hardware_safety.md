# Hardware safety boundary

Simulation success does not authorize flight. No propeller-on hardware command
is executed by the reproduction workflow.

## Required evidence

Before connecting the controller to a Crazyflie, all of the following must be
provided and verified:

1. Vicon-to-NeRF calibration correspondences and fitted similarity transform.
2. Calibrated camera intrinsics and distortion model.
3. Body-from-camera extrinsic transform.
4. Exact ROS topic, frame, unit, and message contracts.
5. Pose, odometry, command, and setpoint stale-data timeouts.
6. Geofence, acceleration, yaw-rate, and thrust limits.
7. A manually signed prop-off checklist covering signs, axes, emergency stop,
   takeoff, landing, occlusion, and shutdown behavior.

Start from `configs/hardware/preflight.template.json`. The template is
deliberately incomplete and must return `BLOCKED`:

```bash
quadpilot hardware preflight configs/hardware/preflight.template.json
```

Fit calibration only from captured, offline evidence:

```bash
quadpilot hardware calibrate correspondences.json --output vicon_from_nerf.json
```

Do not bypass a blocker by changing the template status. The evidence files and
their SHA-256 values must satisfy the checker.
