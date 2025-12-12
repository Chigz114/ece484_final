"""
File: ece484_vision_controller.py
Description: A Python module implementing a vision controller for a drone using 
Neural Pose Estimator (NPE). This script manages drone movements by predicting
the drone's pose from FPV images and using PD control to navigate through gates.

This module provides functions to calculate control inputs (acceleration in x, y, z 
axes and yaw rate) based on NPE pose prediction and state control strategy.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import json
import os
from scipy.spatial.transform import Rotation as R
import sys

# EKF for sensor fusion
from ekf_state_estimator import DroneEKF

# ============================================================================
# Track Configuration
# ============================================================================
# Gate positions in NeRF coordinate system (same as NPE training)
# Format: [x, y, z, yaw_deg]
# yaw_deg: The direction the gate faces (drone enters from opposite direction)
TRACK_CONFIG = {
    "circle": {
        "gates": {
            'Gate A': [-0.3, -3.8, -0.4, -90],   # yaw = -90°
            'Gate B': [-2.3, -6.0, -0.4, -180],  # yaw = -180°
            'Gate C': [-4.1, -3.9, -0.4, 90],    # yaw = 90°
            'Gate D': [-2.2, -1.7, -0.4, 0],     # yaw = 0°
        },
        "model_path": "npe_models/circle/best_npe.pth",
        "initial_pos": [-2.0, -3.5, 0.0],  # Center of training data range
    },
    "uturn": {
        "gates": {
            'Gate A': [-2.2, -6.1, -0.3, -180],  # yaw = -180°
            'Gate B': [-3.8, -4.6, -0.3, 90],    # yaw = 90°
            'Gate C': [-2.2, -3.4, -0.3, 0],     # yaw = 0°
            'Gate D': [-0.4, -1.6, -0.4, 90],    # yaw = 90°
        },
        "model_path": "npe_models/uturn/best_npe.pth",
        "initial_pos": [-2.0, -4.0, -0.3],  # Center of training data range
    },
    "lemniscate": {
        # Gate order: D -> A -> B -> C (lemniscate/figure-8 pattern)
        "gates": {
            'Gate A': [-0.8, -1.8, -0.4, 90],    # yaw = 90° (enter from south)
            'Gate B': [-3.5, -1.9, -0.4, -90],   # yaw = -90° (enter from north, reversed)
            'Gate C': [-0.9, -5.6, -0.4, -90],   # yaw = -90° (enter from north, reversed)
            'Gate D': [-3.4, -5.6, -0.4, 90],    # yaw = 90° (enter from south)
        },
        "gate_order": ['Gate D', 'Gate A', 'Gate B', 'Gate C'],  # Start with D (nearest to initial pos)
        "model_path": "npe_models/lemniscate/best_npe.pth",
        "initial_pos": [-3.4, -8.5, -0.4],  # Must match closed_loop starting position (~3m south of Gate D)
    },
}

# Current track (can be overridden via set_track() or command line)
_current_track = "circle"

def set_track(track_name):
    """Set the current track for vision controller."""
    global _current_track
    if track_name not in TRACK_CONFIG:
        raise ValueError(f"Unknown track: {track_name}. Available: {list(TRACK_CONFIG.keys())}")
    _current_track = track_name
    print(f"INFO: Vision controller track set to: {track_name}")

def get_track():
    """Get the current track name."""
    return _current_track

# Compatibility shim for torch.load
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat


class NPEModel(nn.Module):
    """Neural Pose Estimator model (must match training architecture)."""
    
    def __init__(self, backbone='resnet50'):
        super().__init__()
        
        if backbone == 'resnet50':
            base = models.resnet50(weights=None)
            feature_dim = 2048
        elif backbone == 'resnet34':
            base = models.resnet34(weights=None)
            feature_dim = 512
        else:
            base = models.resnet18(weights=None)
            feature_dim = 512
        
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # [x, y, z, sin(yaw), cos(yaw)]
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.regressor(features)


def _initialize_npe_controller():
    """Initialize the NPE model and gate data. Called once on first use."""
    global _npe_model, _device, _transform, _gates, _gate_order, _gate_normals
    global _initial_gate_sides, _gate_plane_sides
    global _current_trajectory, _lookahead_dist, _yaw_lookahead_dist, _filtered_yaw_rate
    global _current_track
    global _ekf, _use_ekf, _ekf_dynamics_noise
    
    # Check for track from command line args
    for i, arg in enumerate(sys.argv):
        if arg in ('--track', '-t') and i + 1 < len(sys.argv):
            _current_track = sys.argv[i + 1]
            break
    
    # Check for EKF options from command line
    _use_ekf = '--ekf' in sys.argv or '--use-ekf' in sys.argv
    _ekf_dynamics_noise = '--ekf-noise' in sys.argv or '--dynamics-noise' in sys.argv
    
    # Validate track
    if _current_track not in TRACK_CONFIG:
        print(f"WARNING: Unknown track '{_current_track}', defaulting to 'circle'")
        _current_track = "circle"
    
    track_cfg = TRACK_CONFIG[_current_track]
    
    # Device setup
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize trajectory globals
    _current_trajectory = None
    _lookahead_dist = 0.6  # Reduced look-ahead to force tighter path tracking (avoid cutting corners)
    _yaw_lookahead_dist = 2.0  # Keep yaw smooth
    _filtered_yaw_rate = 0.0   # For low-pass filtering
    
    # Initialize EKF for sensor fusion
    # Balance: trust dynamics for smoothness, but trust NPE enough for turns
    _ekf = DroneEKF(
        process_noise_pos=0.015,     # Medium - balance between smoothness and responsiveness
        process_noise_vel=0.05,      # Medium
        process_noise_yaw=0.03,      # Medium
        obs_noise_pos=0.12,          # Medium - trust NPE reasonably (actual error ~8cm)
        obs_noise_yaw=0.06,          # Medium
        dynamics_noise_enabled=_ekf_dynamics_noise,
        dynamics_noise_accel=0.5,    # Simulated real-world noise (when enabled)
        dynamics_noise_yaw_rate=0.15
    )
    
    if _use_ekf:
        noise_status = "WITH dynamics noise" if _ekf_dynamics_noise else "ideal dynamics"
        print(f"INFO: EKF sensor fusion ENABLED ({noise_status})")
    
    # Load NPE model (track-specific)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', track_cfg['model_path'])
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}, trying circle model as fallback")
        model_path = os.path.join(script_dir, '..', 'npe_models/circle/best_npe.pth')
    
    _npe_model = NPEModel(backbone='resnet50').to(_device)
    checkpoint = torch.load(model_path, map_location=_device)
    _npe_model.load_state_dict(checkpoint['model_state_dict'])
    _npe_model.eval()
    print(f"INFO: NPE model loaded from {model_path}")
    print(f"      (val_pos_err: {checkpoint.get('val_pos_err', 'N/A'):.1f} cm)")
    
    # Image transform
    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load gates from track config
    _gates = track_cfg['gates']
    # Use track-specific gate order if defined, otherwise default
    _gate_order = track_cfg.get('gate_order', ['Gate A', 'Gate B', 'Gate C', 'Gate D'])
    
    # Pre-calculate gate normals based on yaw (gate faces perpendicular to yaw direction)
    _gate_normals = []
    # Initial position (from track config)
    initial_pos = np.array(track_cfg['initial_pos'])
    _initial_gate_sides = []
    _gate_plane_sides = []
    
    for gate_name in _gate_order:
        gate_data = _gates[gate_name]
        gate_pos = np.array(gate_data[0:3])
        yaw_deg = gate_data[3]
        yaw_rad = np.radians(yaw_deg)
        
        # Gate normal points in the direction the gate faces (perpendicular to gate plane)
        # For a gate with yaw=0, it faces +X direction
        gate_normal = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
        _gate_normals.append(gate_normal)
        
        # Calculate initial side
        vec_to_gate = initial_pos - gate_pos
        dot_product = np.dot(vec_to_gate, gate_normal)
        sign = np.sign(dot_product)
        _initial_gate_sides.append(sign)
        _gate_plane_sides.append(sign)
    
    print(f"INFO: NPE Vision Controller initialized for track: {_current_track}")
    print(f"      {len(_gate_order)} gates (NeRF coords):")
    for name in _gate_order:
        g = _gates[name]
        print(f"        {name}: ({g[0]:.1f}, {g[1]:.1f}, {g[2]:.1f}) yaw={g[3]}°")


def generate_hermite_spline(p0, p1, m0, m1, num_points=50, scale_factor=0.5):
    """
    Generate points for a cubic Hermite spline.
    p0, p1: Start and end points (3D numpy arrays)
    m0, m1: Tangents at start and end (3D numpy arrays, normalized)
    """
    t = np.linspace(0, 1, num_points)
    
    # Calculate distance to scale tangents
    dist = np.linalg.norm(p1 - p0)
    scale = dist * scale_factor
    
    m0_scaled = m0 * scale
    m1_scaled = m1 * scale
    
    points = np.zeros((num_points, 3))
    
    # Hermite basis functions calculation vectorized
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    
    for i in range(3):
        points[:, i] = (h00 * p0[i] + 
                       h10 * m0_scaled[i] + 
                       h01 * p1[i] + 
                       h11 * m1_scaled[i])
        
    return points


def generate_arc(center, radius, start_angle, end_angle, z, num_points=30):
    """Generate arc points around a center point."""
    angles = np.linspace(start_angle, end_angle, num_points)
    points = np.zeros((num_points, 3))
    points[:, 0] = center[0] + radius * np.cos(angles)
    points[:, 1] = center[1] + radius * np.sin(angles)
    points[:, 2] = z
    return points


def generate_8segment_trajectory(start_pos, start_normal, end_pos, end_normal, 
                                  straight_dist=0.3, num_straight=10, num_curve=40,
                                  is_lap_transition=False, track="circle"):
    """
    Generate smooth trajectory for one gate-to-gate transition.
    
    For uturn track lap transitions (D->A), generates: 180° arc + straight + 90° arc
    Other tracks use standard 8-segment trajectory for all transitions.
    """
    # Calculate direction from start to end gate
    vec_to_next = end_pos - start_pos
    dist_to_next = np.linalg.norm(vec_to_next)
    dir_to_next = vec_to_next / dist_to_next if dist_to_next > 0.1 else start_normal
    
    # Special handling for uturn track lap transition (D->A): 180° arc + straight + 90° arc
    # This is ONLY for uturn track - circle and lemniscate use standard trajectory
    if is_lap_transition and track == "uturn" and dist_to_next > 4.0:
        # D gate exit direction (yaw=90° means +Y direction, going north)
        # First: large 180° arc to turn around (clockwise, to the right/+X)
        arc1_radius = 1.5
        
        # Arc center is to the RIGHT of exit direction (+X direction)
        arc1_center = start_pos + np.array([arc1_radius, 0, 0])
        
        # 180° arc: from 90° to 270° (clockwise)
        arc1 = generate_arc(arc1_center, arc1_radius, 
                           np.pi/2,       # Start at +Y (90°)
                           np.pi*3/2,     # End at -Y (270°)
                           start_pos[2], num_points=50)
        
        arc1_end = arc1[-1]  # End of 180° arc
        
        # Second: straight line going south towards A
        arc2_radius = 0.8
        # Arc2 center ABOVE A (+Y side), arc goes clockwise
        arc2_center = end_pos + np.array([0, arc2_radius, 0])  # Center above A
        arc2_start = arc2_center + np.array([arc2_radius, 0, 0])  # Start from +X side (east)
        
        # Straight line from arc1_end to arc2_start
        straight = np.linspace(arc1_end, arc2_start, 30)
        
        # Third: 90° arc, but stop early to leave room for straight approach
        # Only go to -45° instead of -90°, leaving a diagonal approach
        arc2 = generate_arc(arc2_center, arc2_radius,
                           0,             # Start at +X (0°)
                           -np.pi/4,      # End at -45°, leaves room for straight entry
                           end_pos[2], num_points=20)
        
        # Final straight approach from arc end to gate (going west-southwest)
        final_approach = np.linspace(arc2[-1], end_pos, 15)
        
        # Combine all segments
        trajectory = np.vstack([arc1, straight[1:], arc2[1:], final_approach[1:]])
        return trajectory
    
    # Normal trajectory generation for other transitions
    exit_dir = dir_to_next
    exit_dir = exit_dir / np.linalg.norm(exit_dir)
    
    # Exit point: short straight along exit direction  
    exit_point = start_pos + straight_dist * exit_dir
    
    # Approach point: straight along end gate's normal (opposite direction)
    approach_point = end_pos - straight_dist * end_normal
    
    # Part 1: Short straight line from start to exit point
    straight1 = np.linspace(start_pos, exit_point, num_straight)
    
    # Part 2: Curved segment connecting exit to approach
    curve = generate_hermite_spline(
        exit_point, approach_point,
        exit_dir,      # Exit direction
        end_normal,    # Approach direction (perpendicular to next gate)
        num_points=num_curve,
        scale_factor=0.6
    )
    
    # Part 3: Straight line from approach to end gate
    straight2 = np.linspace(approach_point, end_pos, num_straight)
    
    # Combine all segments (skip first point of each to avoid duplicates)
    trajectory = np.vstack([straight1, curve[1:], straight2[1:]])
    
    return trajectory


def vision_controller(image, vx=0.0, vy=0.0, vz=0.0):
    """
    NPE-based vision controller using trajectory tracking.
    
    Optional EKF fusion: Use --ekf flag to enable sensor fusion with dynamics model.
    Optional dynamics noise: Use --ekf-noise to add noise to dynamics (simulate real world).
    """
    global _npe_model, _device, _transform, _gates, _gate_order, _gate_normals
    global _initial_gate_sides, _gate_plane_sides
    global _current_trajectory, _lookahead_dist, _yaw_lookahead_dist, _filtered_yaw_rate
    global _ekf, _use_ekf, _ekf_dynamics_noise
    
    # Initialize on first call
    if not hasattr(vision_controller, '_initialized'):
        _initialize_npe_controller()
        vision_controller._initialized = True
        vision_controller._target_gate_index = 0
        vision_controller._lap_count = 0
        vision_controller._prev_yaw = 0.0
        vision_controller._debug_counter = 0
        vision_controller._prev_pos = None  # For velocity estimation
        vision_controller._dt = 0.05  # Time step
        vision_controller._last_control = [0, 0, 0, 0]  # For EKF prediction
    
    # Velocity Control Gains
    KP_VEL = np.array([4.0, 4.0, 4.0])  # Velocity P gain (accel to maintain velocity)
    TARGET_SPEED = 1.5  # Cruise speed in m/s
    KP_YAW = 2.5
    KD_YAW = 0.8
    
    # Limits
    MAX_ACCEL = 5.0
    MAX_YAW_RATE = 2.0
    GATE_RADIUS = 0.38
    TOTAL_LAPS = 2
    TOTAL_GATES = 4
    
    # ========== Step 1: NPE Pose Prediction ==========
    # Convert BGR to RGB if needed (OpenCV uses BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    with torch.no_grad():
        img_tensor = _transform(image_rgb).unsqueeze(0).to(_device)
        pred = _npe_model(img_tensor)[0].cpu().numpy()
    
    # Decode prediction: [x, y, z, sin(yaw), cos(yaw)] - already in NeRF coordinates
    pred_x, pred_y, pred_z = pred[0], pred[1], pred[2]
    pred_yaw = np.arctan2(pred[3], pred[4])
    
    # NPE raw observation
    npe_obs = np.array([pred_x, pred_y, pred_z, pred_yaw])
    
    # ========== EKF Sensor Fusion (if enabled) ==========
    if _use_ekf:
        # EKF Predict: use dynamics model with last control input
        _ekf.predict(vision_controller._last_control, vision_controller._dt)
        
        # EKF Update: fuse with NPE observation
        filtered_state = _ekf.update(npe_obs, outlier_threshold=3.0)
        
        # Use filtered estimates
        current_pos = filtered_state[:3]
        current_vel = filtered_state[3:6]  # EKF provides velocity estimate!
        current_yaw = filtered_state[6]
        
        # Store NPE and filtered for overlay
        vision_controller._last_prediction = (pred_x, pred_y, pred_z, np.degrees(pred_yaw))
        vision_controller._last_filtered = (current_pos[0], current_pos[1], current_pos[2], np.degrees(current_yaw))
    else:
        # No EKF: use raw NPE directly
        current_pos = np.array([pred_x, pred_y, pred_z])
        current_yaw = pred_yaw
        
        # Store last prediction for external access (e.g., video overlay)
        vision_controller._last_prediction = (pred_x, pred_y, pred_z, np.degrees(pred_yaw))
        
        # Estimate velocity from position change (if we have previous position)
        if vision_controller._prev_pos is not None:
            current_vel = (current_pos - vision_controller._prev_pos) / vision_controller._dt
        else:
            current_vel = np.array([vx, vy, vz])  # Use provided velocities initially
        vision_controller._prev_pos = current_pos.copy()
        
    # ========== Step 2: Check Race Completion ==========
    if vision_controller._lap_count >= TOTAL_LAPS:
        # Race complete: hover in place
        # Use simple velocity damping without relying on potentially drifting EKF
        # Just return zero control to let the simulation stop naturally
        vision_controller._last_control = [0, 0, 0, 0]
        return [0, 0, 0, 0]
        
    # ========== Step 3: Trajectory Management ==========
    target_idx = vision_controller._target_gate_index
    gate_name = _gate_order[target_idx]
    gate_data = _gates[gate_name]
    target_pos = np.array(gate_data[0:3])
        
    # Generate trajectory if not exists (First Run)
    if _current_trajectory is None:
        # For initial trajectory: straight towards gate, then align with gate normal
        vec_to_target = target_pos - current_pos
        dist = np.linalg.norm(vec_to_target)
        if dist > 0.1:
            start_dir = vec_to_target / dist
        else:
            start_dir = np.array([np.cos(current_yaw), np.sin(current_yaw), 0.0])
        
        # Use 8-segment style: start with direction to target, end with gate normal
        _current_trajectory = generate_8segment_trajectory(
            current_pos, start_dir,
            target_pos, _gate_normals[target_idx],
            straight_dist=0.6,
            track=_current_track
        )
        print(f"INFO: [Traj] Generated initial 8-segment trajectory to {gate_name}")
        
    # ========== Step 4: Gate Switching Logic ==========
    gate_normal = _gate_normals[target_idx]
    prev_side = _gate_plane_sides[target_idx]
        
    vec_to_gate = current_pos - target_pos
    dist_to_plane = np.dot(vec_to_gate, gate_normal)
    current_side = np.sign(dist_to_plane)
        
    # Check for plane crossing
    # Evaluator logic: crossed if side changes (product < 0) or we are exactly on plane (0)
    if (current_side != prev_side and prev_side != 0):
        # Check if we actually passed through the gate (distance to center < radius)
        # Project current pos onto plane to find intersection point approx
        dist_to_center = np.linalg.norm(vec_to_gate - dist_to_plane * gate_normal)
            
        if dist_to_center < GATE_RADIUS * 1.5: # slightly generous radius for detection
            print(f"INFO: [NPE] Detected PASS for {gate_name} (Dist: {dist_to_center:.2f}m)")
                
            # Update State
            _gate_plane_sides[target_idx] = current_side # Update side for current gate
                
            vision_controller._target_gate_index += 1
            if vision_controller._target_gate_index >= TOTAL_GATES:
                vision_controller._target_gate_index = 0
                vision_controller._lap_count += 1
                if vision_controller._lap_count >= TOTAL_LAPS:
                    return [0, 0, 0, 0]
                    
                # Reset plane sides for new lap
                print("INFO: [NPE] Completed lap, resetting.")
                for i in range(TOTAL_GATES):
                    _gate_plane_sides[i] = _initial_gate_sides[i]
                
            # Prepare new trajectory segment using 8-segment style
            prev_gate_idx = target_idx
            target_idx = vision_controller._target_gate_index # New target
                
            prev_gate_name = _gate_order[prev_gate_idx]
            curr_gate_name = _gate_order[target_idx]
                
            prev_pos = np.array(_gates[prev_gate_name][0:3])
            next_pos = np.array(_gates[curr_gate_name][0:3])
            
            # Use 8-segment trajectory:
            # - Straight exit from prev gate (along its normal)
            # - Curve to turn
            # - Straight approach to next gate (along its normal)
            # Special handling for uturn D->A (lap transition): 180° arc + straight + 90° arc
            is_lap_transition = (prev_gate_idx == TOTAL_GATES - 1 and target_idx == 0)
            _current_trajectory = generate_8segment_trajectory(
                prev_pos, _gate_normals[prev_gate_idx],  # Start: prev gate pos and normal
                next_pos, _gate_normals[target_idx],     # End: next gate pos and normal
                straight_dist=0.8,
                is_lap_transition=is_lap_transition,
                track=_current_track  # Pass track name for track-specific handling
            )
            # Only uturn uses arc-straight-arc for lap transition
            traj_type = "arc-straight-arc" if (is_lap_transition and _current_track == "uturn") else "8-segment"
            print(f"INFO: [Traj] Generated {traj_type} {prev_gate_name} -> {curr_gate_name}")
                
            # Update target vars
            gate_name = curr_gate_name
            gate_data = _gates[gate_name]
            target_pos = np.array(gate_data[0:3])
        else:
            # Missed pass
            print(f"WARNING: [NPE] Detected MISS for {gate_name} (Dist: {dist_to_center:.2f}m)")
            _gate_plane_sides[target_idx] = current_side
    elif current_side != 0:
        _gate_plane_sides[target_idx] = current_side
        
    # ========== Step 5: Trajectory Tracking (Carrot) ==========
    # Find closest point on trajectory
    dists = np.linalg.norm(_current_trajectory - current_pos, axis=1)
    closest_idx = np.argmin(dists)
        
    # Look ahead
    # Each point is roughly dist/50 apart.
    # Better: Find point at lookahead distance
    carrot_pos = _current_trajectory[-1] # Default to end
        
    # Search forward from closest_idx
    found_carrot = False
    for i in range(closest_idx, len(_current_trajectory)):
        dist_from_closest = np.linalg.norm(_current_trajectory[i] - _current_trajectory[closest_idx])
        if dist_from_closest >= _lookahead_dist:
            carrot_pos = _current_trajectory[i]
            found_carrot = True
            break
                
    # ===== Velocity Control =====
    # Step 1: Compute target velocity DIRECTION (towards carrot)
    vec_to_carrot = carrot_pos - current_pos
    dist_to_carrot = np.linalg.norm(vec_to_carrot)
    
    if dist_to_carrot > 0.01:
        direction = vec_to_carrot / dist_to_carrot
    else:
        direction = np.array([0.0, 0.0, 0.0])
    
    # Step 2: Compute target velocity (direction * cruise speed)
    target_vel = direction * TARGET_SPEED
    
    # Step 3: Compute velocity error (in WORLD frame)
    error_vel = target_vel - current_vel
    
    # Step 4: Compute acceleration to maintain target velocity (in WORLD frame)
    accel_world = KP_VEL * error_vel
        
    # Step 5: Convert world frame acceleration to BODY frame (same transform as before)
    cos_yaw = np.cos(current_yaw)
    sin_yaw = np.sin(current_yaw)
    ax_body = accel_world[0] * cos_yaw + accel_world[1] * sin_yaw
    ay_body = -accel_world[0] * sin_yaw + accel_world[1] * cos_yaw
    az_body = accel_world[2]
        
    ax = np.clip(ax_body, -MAX_ACCEL, MAX_ACCEL)
    ay = np.clip(ay_body, -MAX_ACCEL, MAX_ACCEL)
    az = np.clip(az_body, -MAX_ACCEL, MAX_ACCEL)
        
    # ========== Step 6: PD Yaw Control ==========
    # Use trajectory TANGENT at lookahead point (not direction TO lookahead)
    # This ensures yaw follows the curve direction, not the chord direction
    yaw_idx = len(_current_trajectory) - 1  # Default to end
    for i in range(closest_idx, len(_current_trajectory)):
        dist_from_closest = np.linalg.norm(_current_trajectory[i] - _current_trajectory[closest_idx])
        if dist_from_closest >= _yaw_lookahead_dist:
            yaw_idx = i
            break

    # Compute trajectory tangent at yaw_idx (use adjacent points)
    if yaw_idx < len(_current_trajectory) - 1:
        tangent = _current_trajectory[yaw_idx + 1] - _current_trajectory[yaw_idx]
    else:
        # At end of trajectory, use final segment direction
        tangent = _current_trajectory[-1] - _current_trajectory[-2]
    
    desired_yaw = np.arctan2(tangent[1], tangent[0])
    
    error_yaw = desired_yaw - current_yaw
    error_yaw = (error_yaw + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    
    yaw_rate_est = (current_yaw - vision_controller._prev_yaw) / vision_controller._dt
    error_yaw_rate = 0.0 - yaw_rate_est
    
    raw_yaw_cmd = KP_YAW * error_yaw + KD_YAW * error_yaw_rate
    
    # Low-pass filter for yaw rate
    alpha = 0.3  # Smoothing factor (0.0 = old, 1.0 = new)
    _filtered_yaw_rate = alpha * raw_yaw_cmd + (1 - alpha) * _filtered_yaw_rate
    
    yaw_rate_command = np.clip(_filtered_yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE)
    
    vision_controller._prev_yaw = current_yaw
    
    # ========== Debug Output ==========
    vision_controller._debug_counter += 1
    if vision_controller._debug_counter % 25 == 0:
        speed = np.linalg.norm(current_vel)
        print(f"  [NPE] Target: {gate_name} | Pos: ({pred_x:.2f},{pred_y:.2f},{pred_z:.2f}) | "
              f"Speed: {speed:.2f}m/s | Yaw: {np.degrees(current_yaw):.1f}°")
        print(f"        → a=[{ax:.2f},{ay:.2f},{az:.2f}] yaw_rate={yaw_rate_command:.2f}")
    
    control = [ax, ay, az, yaw_rate_command]
    
    # Store control for next EKF prediction step
    vision_controller._last_control = control
    
    return control

if __name__ == "__main__":
    image = np.zeros((480, 640, 3)) # PlaceHolder
    control = vision_controller(image)
    print(control)