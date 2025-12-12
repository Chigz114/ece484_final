"""
File: ece484_vision_closed_loop.py

Description: This script simulates a drone's trajectory on a specified track using closed-loop control through vision.

Tracks supported: Circle_Track, Uturn_Track, Lemniscate_Track

The simulation involves rendering images from NerfRenderer class[ns-renderer.py], 
saves the trajectory to a text file, and creating a video from the rendered images.
"""
import numpy as np
import sys
from drone_dynamics import drone_dynamics
from ece484_vision_controller import vision_controller, set_track, TRACK_CONFIG
import torch 
from nerfstudio.models.splatfacto import SplatfactoModel
from scipy.spatial.transform import Rotation as R 
import cv2 
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps
import numpy as np 
import os 
from pathlib import Path
import matplotlib.pyplot as plt 
import json
from scipy.spatial.transform import Rotation
import time
import pickle
import random
import argparse
from pathlib import Path
import importlib.util

# Import ns-renderer.py (has hyphen in name)
spec = importlib.util.spec_from_file_location("ns_renderer", Path(__file__).parent / "ns-renderer.py")
ns_renderer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns_renderer_module)
NerfRenderer = ns_renderer_module.NerfRenderer


def save_trajectory_to_txt(trajectory, filename="trajectory.txt"):
    """Save trajectory data to a TXT file"""
    with open(filename, mode='w') as file: 
        for row in trajectory:
            file.write(" ".join(map(str, row)) + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Closed loop Vision controller ")
    parser.add_argument("--track", type=str, default="circle",
                        help="track name (circle/uturn/lemniscate)")
    parser.add_argument("--ekf", action="store_true",
                        help="Enable EKF sensor fusion (dynamics + NPE)")
    parser.add_argument("--ekf-noise", action="store_true",
                        help="Add noise to dynamics model (simulate real world)")
    return parser.parse_args()


def generate_video(image_dir, output_path, fps=20):
    """Generate video from images in a directory."""
    import glob
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    frame = cv2.imread(images[0])
    h, w = frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    track = args.track
    
    # Set track for vision controller BEFORE it initializes
    set_track(track)
    
    scale_factor = 1
    width, height = 640, 480
    fov = 80
    fx = (width/2)/(np.tan(np.deg2rad(fov)/2))
    fy = (height/2)/(np.tan(np.deg2rad(fov)/2))

    # Use splatfacto (Gaussian Splatting) for 50x faster rendering
    splatfacto_dir = Path(f'outputs/{track}/splatfacto')
    # Find the latest splatfacto training directory
    splatfacto_runs = sorted(splatfacto_dir.glob('*'))
    if not splatfacto_runs:
        raise RuntimeError(f"No splatfacto runs found in {splatfacto_dir}")
    latest_run = splatfacto_runs[-1]
    
    print(f"Using Gaussian Splatting (splatfacto) for fast rendering")
    print(f"Config: {latest_run / 'config.yml'}\n")
    
    renderer = NerfRenderer(
        str((latest_run / 'config.yml').absolute()),
        width = width//scale_factor,
        height = height//scale_factor,
        fx = 546.84164912/scale_factor, 
        fy = 547.57957461/scale_factor,
        cx = 349.18316327/scale_factor,
        cy = 215.54486004/scale_factor,
        track = track
    )

    def find_angle_difference(angle1, angle2):
        return (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi

    import numpy as np
    from scipy.spatial.transform import Rotation as R
    import pickle


    # Track-specific initial positions (in NeRF coordinates)
    # Format: [x, y, z, roll, pitch, yaw]
    # yaw should point towards first gate
    INITIAL_POSITIONS = {
        "circle": [-0.4, -0.5, -0.3, 0, 0, np.radians(-90)],
        "uturn": [-0.5, -6.1, -0.3, 0, 0, np.radians(-180)],  # Y与Gate A对齐，yaw=-180°
        "lemniscate": [-3.4, -8.5, -0.4, 0, 0, np.radians(90)],  # Further south of Gate D (~3m away), yaw=90° (north)
    }
    
    test_steps = 500  # More steps to see stability
    dt = 0.05  # Smaller time step for smoother control
    cur_pos = INITIAL_POSITIONS.get(track, INITIAL_POSITIONS["circle"])
    vx, vy, vz = 0.0, 0.0, 0.0  # Start from rest
    x, y, z, roll, pitch, yaw = cur_pos
    control_val = [0, 0, 0, 0] # ax, ay, az, yaw_rate

    trajectory = []
    imagelist = []
    predictions = []  # Store NPE predictions for overlay
    ekf_predictions = []  # Store EKF filtered predictions (if EKF enabled)
    dyn_poses = []  # Store pure dynamics predictions (single-step from previous GT)
    actual_poses = []  # Store actual poses for overlay
    
    # For single-step dynamics prediction
    prev_vx, prev_vy, prev_vz = 0.0, 0.0, 0.0
    prev_control = [0, 0, 0, 0]
    
    # Track gate passing for statistics clipping
    gate_pass_frames = []  # Frame indices when gates are passed
    
    # Track-specific output directory
    output_dir = f"./closed_loop/{track}"
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    dynamics_state = [x, y, z, vx, vy, vz, yaw] # x, y, z, vx, vy, vz, cur_yaw

    print(f"\n{'='*70}")
    print(f"Starting vision closed-loop simulation")
    print(f"Track: {track}, Steps: {test_steps}, dt: {dt}s")
    print(f"Initial position: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    print(f"{'='*70}\n")

    for idx in range(test_steps):
        # Print every 25 steps to avoid spam
        if idx % 25 == 0:
            print(f"[Step {idx+1}/{test_steps}] Position: ({x:.3f}, {y:.3f}, {z:.3f})", end=' ')

        # Step 1: Create the rotation matrix from pitch, yaw, roll
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)  # 'xyz' assumes roll, pitch, yaw order
        rotation_matrix = rotation.as_matrix()  # Converts to a 3x3 rotation matrix

        # Step 2: Create the 4x4 transformation matrix
        camera_pose = np.eye(4)  # Start with an identity matrix
        camera_pose[:3, :3] = rotation_matrix  # Set the top-left 3x3 to the rotation matrix
        camera_pose[:3, 3] = [x, y, z]  # Set the translation components

        s = time.time()
        img = renderer.render(camera_pose) # RGB
        cv_img = img[:, :, ::-1]  # Convert to BGR for OpenCV
        render_time = time.time() - s
        
        # Store actual pose BEFORE control (this is GT for the current image)
        actual_poses.append((x, y, z, np.degrees(yaw)))
        
        s = time.time()
        control = vision_controller(cv_img, vx=vx, vy=vy, vz=vz)  # Pass velocities for braking
        npe_time = time.time() - s
        
        # Get NPE prediction for overlay
        npe_pred = getattr(vision_controller, '_last_prediction', (0, 0, 0, 0))
        predictions.append(npe_pred)
        
        # Get EKF filtered prediction (if EKF enabled)
        ekf_pred = getattr(vision_controller, '_last_filtered', None)
        ekf_predictions.append(ekf_pred)
        
        # Track gate passes (for statistics clipping)
        current_gate_idx = getattr(vision_controller, '_target_gate_index', 0)
        current_lap = getattr(vision_controller, '_lap_count', 0)
        total_gates_passed = current_lap * 4 + current_gate_idx
        if len(gate_pass_frames) < total_gates_passed:
            gate_pass_frames.append(idx)
        
        # Pure dynamics prediction: from PREVIOUS GT state, predict current position
        # This shows single-step dynamics error, not accumulated drift
        if idx == 0:
            # First frame: no previous state, use current GT
            dyn_poses.append((x, y, z, np.degrees(yaw)))
        else:
            # Use previous GT state + control to predict current position
            prev_gt = trajectory[-1] if trajectory else [x, y, z, yaw]
            prev_state = [prev_gt[0], prev_gt[1], prev_gt[2], 
                         prev_vx, prev_vy, prev_vz, prev_gt[3]]
            
            # Apply dynamics then add position noise
            dyn_pred = drone_dynamics(prev_state, prev_control, dt=dt)
            
            # Add position noise (uniform distribution for "flat" noise)
            # This simulates IMU drift / measurement noise
            if args.ekf_noise:
                pos_noise = np.random.uniform(-0.05, 0.05, 3)  # ±5cm position noise
                yaw_noise = np.random.uniform(-0.05, 0.05)     # ±3° yaw noise
                dyn_poses.append((
                    dyn_pred[0] + pos_noise[0],
                    dyn_pred[1] + pos_noise[1],
                    dyn_pred[2] + pos_noise[2],
                    np.degrees(dyn_pred[6] + yaw_noise)
                ))
            else:
                dyn_poses.append((dyn_pred[0], dyn_pred[1], dyn_pred[2], np.degrees(dyn_pred[6])))
        
        # Store for next iteration
        prev_vx, prev_vy, prev_vz = vx, vy, vz
        prev_control = control
        
        if idx % 25 == 0:
            print(f"| Render: {render_time:.2f}s | NPE: {npe_time:.3f}s | ax={control[0]:.2f}, yaw={control[3]:.2f}")
        
        # Check if race is complete (controller returns [0,0,0,0])
        race_complete = (control == [0, 0, 0, 0])
        
        if race_complete:
            # Apply braking: decelerate based on actual velocity (not EKF estimate)
            # Use simple damping: a = -k * v
            brake_k = 3.0  # Damping coefficient
            ax_brake = -brake_k * vx
            ay_brake = -brake_k * vy
            az_brake = -brake_k * vz
            
            # Transform to body frame
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            ax_body = ax_brake * cos_yaw + ay_brake * sin_yaw
            ay_body = -ax_brake * sin_yaw + ay_brake * cos_yaw
            
            control = [np.clip(ax_body, -5, 5), np.clip(ay_body, -5, 5), np.clip(az_brake, -5, 5), 0]
        
        next_state = drone_dynamics(dynamics_state, control, dt=dt) 
        x, y, z, vx, vy, vz, yaw = next_state
        dynamics_state = next_state  # Update for next iteration

        trajectory.append([x, y, z, yaw])
        imagelist.append(cv_img)
        
    
    # Save images with coordinate overlay
    # Check if EKF was used (check if any ekf_predictions is not None)
    ekf_enabled = any(p is not None for p in ekf_predictions)
    
    print(f"\nSaving {len(imagelist)} images with coordinate overlay...")
    print(f"  EKF overlay: {'Yes' if ekf_enabled else 'No (NPE only)'}")
    
    for idx in range(len(imagelist)):
        img = imagelist[idx].copy()
        npe = predictions[idx]
        ekf = ekf_predictions[idx]
        dyn = dyn_poses[idx]
        gt = actual_poses[idx]
        
        # Calculate errors
        npe_err = np.sqrt((npe[0]-gt[0])**2 + (npe[1]-gt[1])**2 + (npe[2]-gt[2])**2)
        ekf_err = np.sqrt((ekf[0]-gt[0])**2 + (ekf[1]-gt[1])**2 + (ekf[2]-gt[2])**2) if ekf else None
        dyn_err = np.sqrt((dyn[0]-gt[0])**2 + (dyn[1]-gt[1])**2 + (dyn[2]-gt[2])**2)
        
        # Draw semi-transparent background for text
        overlay = img.copy()
        box_height = 145 if ekf_enabled else 75
        cv2.rectangle(overlay, (5, 5), (380, box_height), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        
        y_offset = 18
        
        # NPE (raw visual estimate) - Cyan/Yellow
        npe_text = f"NPE: ({npe[0]:.2f},{npe[1]:.2f},{npe[2]:.2f}) | Err:{npe_err*100:.1f}cm"
        cv2.putText(img, npe_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        y_offset += 18
        
        # DYN (pure dynamics from GT + noise, dead-reckoning) - Orange
        dyn_text = f"DYN: ({dyn[0]:.2f},{dyn[1]:.2f},{dyn[2]:.2f}) | Err:{dyn_err*100:.1f}cm"
        cv2.putText(img, dyn_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
        y_offset += 18
        
        # EKF (filtered estimate) - Magenta (only if EKF enabled)
        if ekf_enabled and ekf:
            ekf_text = f"EKF: ({ekf[0]:.2f},{ekf[1]:.2f},{ekf[2]:.2f}) | Err:{ekf_err*100:.1f}cm"
            cv2.putText(img, ekf_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)
            y_offset += 18
        
        # GT (ground truth) - Green
        gt_text = f"GT:  ({gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f})"
        cv2.putText(img, gt_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        y_offset += 15
        
        # Error comparison bars (visual indicator)
        if ekf_enabled and ekf and dyn:
            bar_y = y_offset + 3
            max_err_display = 0.3  # 30cm max for display
            bar_width = 120
            
            npe_bar_len = int(min(npe_err / max_err_display, 1.0) * bar_width)
            dyn_bar_len = int(min(dyn_err / max_err_display, 1.0) * bar_width)
            ekf_bar_len = int(min(ekf_err / max_err_display, 1.0) * bar_width)
            
            # NPE bar - Yellow
            cv2.putText(img, "NPE", (10, bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)
            cv2.rectangle(img, (38, bar_y - 4), (38 + npe_bar_len, bar_y + 4), (255, 255, 0), -1)
            cv2.putText(img, f"{npe_err*100:.0f}", (42 + bar_width, bar_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)
            
            # DYN bar - Orange
            cv2.putText(img, "DYN", (10, bar_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 165, 255), 1)
            cv2.rectangle(img, (38, bar_y + 8), (38 + dyn_bar_len, bar_y + 16), (0, 165, 255), -1)
            cv2.putText(img, f"{dyn_err*100:.0f}", (42 + bar_width, bar_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 165, 255), 1)
            
            # EKF bar - Magenta
            cv2.putText(img, "EKF", (10, bar_y + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 255), 1)
            cv2.rectangle(img, (38, bar_y + 20), (38 + ekf_bar_len, bar_y + 28), (255, 0, 255), -1)
            cv2.putText(img, f"{ekf_err*100:.0f}", (42 + bar_width, bar_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 255), 1)
            
            # Best indicator
            min_err = min(npe_err, dyn_err, ekf_err)
            if min_err == ekf_err:
                cv2.putText(img, "BEST", (180, bar_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        cv2.imwrite(f"{output_dir}/images/{idx:04d}.png", img)

    save_trajectory_to_txt(trajectory, filename=f"{track}_vision_trajectory.txt")
    
    # ========== Calculate Statistics ==========
    traj_array = np.array(trajectory)
    initial_pos = traj_array[0, :3]
    final_pos = traj_array[-1, :3]
    total_distance = np.linalg.norm(final_pos - initial_pos)
    
    # Calculate error statistics for NPE, DYN, EKF
    # Clip to frames between first gate pass and last gate pass
    if len(gate_pass_frames) >= 2:
        start_frame = gate_pass_frames[0]  # First gate pass
        end_frame = gate_pass_frames[-1]   # Last gate pass
        print(f"\nStatistics clipped to frames {start_frame}-{end_frame} (gate 1 to gate {len(gate_pass_frames)})")
    else:
        start_frame = 0
        end_frame = len(predictions)
        print(f"\nStatistics for all {end_frame} frames (not enough gate passes to clip)")
    
    npe_errors = []
    dyn_errors = []
    ekf_errors = []
    
    for idx in range(start_frame, end_frame):
        npe = predictions[idx]
        ekf = ekf_predictions[idx]
        dyn = dyn_poses[idx]
        gt = actual_poses[idx]
        
        npe_err = np.sqrt((npe[0]-gt[0])**2 + (npe[1]-gt[1])**2 + (npe[2]-gt[2])**2)
        npe_errors.append(npe_err)
        
        if ekf:
            ekf_err = np.sqrt((ekf[0]-gt[0])**2 + (ekf[1]-gt[1])**2 + (ekf[2]-gt[2])**2)
            ekf_errors.append(ekf_err)
        
        dyn_err = np.sqrt((dyn[0]-gt[0])**2 + (dyn[1]-gt[1])**2 + (dyn[2]-gt[2])**2)
        dyn_errors.append(dyn_err)
    
    npe_errors = np.array(npe_errors)
    ekf_errors = np.array(ekf_errors) if ekf_errors else None
    dyn_errors = np.array(dyn_errors) if dyn_errors else None
    
    # Calculate smoothness (jitter): std of frame-to-frame position changes
    # Lower jitter = smoother trajectory
    def calc_jitter(poses):
        """Calculate jitter: std of frame-to-frame position changes"""
        if len(poses) < 2:
            return 0
        deltas = []
        for i in range(1, len(poses)):
            dx = poses[i][0] - poses[i-1][0]
            dy = poses[i][1] - poses[i-1][1]
            dz = poses[i][2] - poses[i-1][2]
            delta = np.sqrt(dx**2 + dy**2 + dz**2)
            deltas.append(delta)
        return np.std(deltas) * 100  # in cm
    
    npe_jitter = calc_jitter([predictions[i] for i in range(start_frame, end_frame)])
    ekf_jitter = calc_jitter([ekf_predictions[i] for i in range(start_frame, end_frame) if ekf_predictions[i]]) if ekf_enabled else None
    dyn_jitter = calc_jitter([dyn_poses[i] for i in range(start_frame, end_frame)])
    gt_jitter = calc_jitter([actual_poses[i] for i in range(start_frame, end_frame)])
    
    # Statistics
    stats = {
        'NPE': {
            'mean': np.mean(npe_errors) * 100,
            'std': np.std(npe_errors) * 100,
            'max': np.max(npe_errors) * 100,
            'min': np.min(npe_errors) * 100,
        }
    }
    
    if ekf_errors is not None:
        stats['EKF'] = {
            'mean': np.mean(ekf_errors) * 100,
            'std': np.std(ekf_errors) * 100,
            'max': np.max(ekf_errors) * 100,
            'min': np.min(ekf_errors) * 100,
        }
    
    if dyn_errors is not None:
        stats['DYN'] = {  # Pure dynamics from GT + noise (dead-reckoning)
            'mean': np.mean(dyn_errors) * 100,
            'std': np.std(dyn_errors) * 100,
            'max': np.max(dyn_errors) * 100,
            'min': np.min(dyn_errors) * 100,
        }
    
    print(f"\n{'='*70}")
    print(f"Vision Closed-Loop Simulation Complete!")
    print(f"{'='*70}")
    print(f"Initial position: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f}, {initial_pos[2]:.3f})")
    print(f"Final position:   ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
    print(f"Total distance traveled: {total_distance:.3f} m")
    print(f"\n--- Error Statistics (cm) ---")
    for name, s in stats.items():
        print(f"{name}: mean={s['mean']:.1f}, std={s['std']:.1f}, min={s['min']:.1f}, max={s['max']:.1f}")
    
    print(f"\n--- Smoothness (Jitter, lower=smoother) ---")
    print(f"NPE jitter: {npe_jitter:.2f} cm")
    if ekf_jitter is not None:
        print(f"EKF jitter: {ekf_jitter:.2f} cm")
    print(f"DYN jitter: {dyn_jitter:.2f} cm")
    print(f"GT jitter:  {gt_jitter:.2f} cm (reference)")
    print(f"Trajectory saved to: {track}_vision_trajectory.txt")
    print(f"Images saved to: {output_dir}/images/")
    print(f"{'='*70}\n")
    
    # ========== Save Data for Visualization ==========
    viz_data = {
        'predictions': predictions[start_frame:end_frame],
        'ekf_predictions': [p for p in ekf_predictions[start_frame:end_frame] if p],
        'dyn_poses': dyn_poses[start_frame:end_frame],
        'actual_poses': actual_poses[start_frame:end_frame],
        'npe_errors': npe_errors.tolist(),
        'ekf_errors': ekf_errors.tolist() if ekf_errors is not None else [],
        'dyn_errors': dyn_errors.tolist() if dyn_errors is not None else [],
        'stats': stats,
        'jitter': {'NPE': npe_jitter, 'EKF': ekf_jitter, 'DYN': dyn_jitter, 'GT': gt_jitter}
    }
    np.save(f"{output_dir}/viz_data.npy", viz_data, allow_pickle=True)
    print(f"Visualization data saved to: {output_dir}/viz_data.npy")
    
    # ========== Generate Summary Frames ==========
    # Add 3 seconds (60 frames at 20fps) of summary statistics at the end
    print("Generating summary frames...")
    summary_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Match video frame size
    summary_frame[:] = (30, 30, 30)  # Dark gray background
    
    y_pos = 40
    cv2.putText(summary_frame, "SIMULATION STATISTICS", (150, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    y_pos += 50
    
    cv2.putText(summary_frame, f"Track: {track}", (50, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_pos += 30
    cv2.putText(summary_frame, f"Total Frames: {len(imagelist)}", (50, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_pos += 50
    
    # Error statistics table with Jitter column
    cv2.putText(summary_frame, "Statistics (cm)", (50, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    y_pos += 30
    
    # Header - add Jitter column
    cv2.putText(summary_frame, "Source", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(summary_frame, "Mean", (130, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(summary_frame, "Std", (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(summary_frame, "Max", (270, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(summary_frame, "Jitter", (340, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y_pos += 25
    
    # Jitter values map
    jitter_map = {'NPE': npe_jitter, 'EKF': ekf_jitter, 'DYN': dyn_jitter}
    colors = {'NPE': (255, 255, 0), 'DYN': (0, 165, 255), 'EKF': (255, 0, 255)}
    
    for name, s in stats.items():
        color = colors.get(name, (255, 255, 255))
        jitter = jitter_map.get(name, 0)
        cv2.putText(summary_frame, name, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(summary_frame, f"{s['mean']:.1f}", (130, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(summary_frame, f"{s['std']:.1f}", (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(summary_frame, f"{s['max']:.1f}", (270, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(summary_frame, f"{jitter:.2f}" if jitter else "-", (340, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y_pos += 25
    
    # Add GT jitter reference
    cv2.putText(summary_frame, "GT", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(summary_frame, "-", (130, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(summary_frame, "-", (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(summary_frame, "-", (270, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(summary_frame, f"{gt_jitter:.2f}", (340, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    y_pos += 35
    
    # EKF advantage summary
    if ekf_jitter is not None and npe_jitter > 0:
        jitter_improvement = (npe_jitter - ekf_jitter) / npe_jitter * 100
        err_improvement = (stats['NPE']['mean'] - stats['EKF']['mean']) / stats['NPE']['mean'] * 100 if 'EKF' in stats else 0
        
        cv2.putText(summary_frame, "EKF vs NPE:", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
        
        # Error improvement
        if err_improvement > 0:
            cv2.putText(summary_frame, f"  Error: {err_improvement:.0f}% lower", (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        else:
            cv2.putText(summary_frame, f"  Error: {-err_improvement:.0f}% higher", (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        y_pos += 25
        
        # Jitter improvement
        if jitter_improvement > 0:
            cv2.putText(summary_frame, f"  Jitter: {jitter_improvement:.0f}% smoother", (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        else:
            cv2.putText(summary_frame, f"  Jitter: {-jitter_improvement:.0f}% worse", (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    
    # Save summary frames (60 frames = 3 seconds at 20fps)
    num_summary_frames = 60
    last_idx = len(imagelist)
    for i in range(num_summary_frames):
        cv2.imwrite(f"{output_dir}/images/{last_idx + i:04d}.png", summary_frame)
    
    # Generate video (use absolute path based on script location)
    script_dir = Path(__file__).parent
    video_dir = script_dir.parent / "videos" / track
    os.makedirs(video_dir, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"{video_dir}/vision_control_{timestamp}.mp4"
    
    print(f"Generating video...")
    generate_video(f"{output_dir}/images", video_path, fps=20)
    print(f"{'='*70}\n")


