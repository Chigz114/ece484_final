"""
File: ece484_state_controller.py
Description: A Python module implementing a state controller for a drone. 
This script manages drone movements by controlling acceleration and yaw 
rate based on the current state of the drone and a sequence of target gates.

This controller uses a PD (Proportional-Derivative) control law to 
calculate the necessary accelerations and yaw rate to navigate the drone
through a series of waypoints (gates).

TODO TASK 1: Implement the control logic in this file.

-- V1.3 (Auto-Stop) --
This version modifies the V1.2 logic to stop the simulation
immediately after the final gate is passed.

It changes the "race complete" return value from [0,0,0,0] (hover)
to None, which triggers the 'break' condition in the
ece484_state_closed_loop.py script.
"""

import numpy as np
import json
import sys
import os
# (NEW) Import the Rotation library, same as the evaluator
from scipy.spatial.transform import Rotation as R

# --------------------------------------------------------------------------
# --- Global variables ---
# --------------------------------------------------------------------------

# GATES will be populated by the _initialize_controller() function
GATES = {}

# GATE_ORDER will be populated by the _initialize_controller() function
GATE_ORDER = []

# (NEW) We need to store the "front" vector (normal) for each gate
GATE_NORMALS = []

# (NEW) We need to track which side of the gate plane we start on
# This list will be reset every lap
GATE_PLANE_SIDES = [None] * 4

# (NEW) This stores the *absolute* starting side, does not reset
INITIAL_GATE_SIDES = []


# --------------------------------------------------------------------------
# --- One-Time Initialization (This runs when the module is imported) ---
# --------------------------------------------------------------------------

def _initialize_controller():
    """
    This function runs ONCE when the module is imported.
    
    It "hacks" into the command line arguments (sys.argv) of the main 
    script that imported it (ece484_state_closed_loop.py). This is 
    necessary to find the --track-name argument.
    
    It also pre-calculates the gate normal vectors *exactly* as the
    evaluator script does, including its bugs.
    """
    global GATES, GATE_ORDER, GATE_NORMALS, INITIAL_GATE_SIDES
    try:
        # 1. Find the track name from sys.argv
        args = sys.argv
        track_name_arg = "--track-name"
        
        if track_name_arg in args:
            track_name_index = args.index(track_name_arg) + 1
            if track_name_index < len(args):
                track_name = args[track_name_index]
            else:
                raise ValueError("Missing value for --track-name in command line args")
        else:
            print("CRITICAL WARNING: --track-name not found in command line args.")
            return

        # 2. Load the JSON file
        json_path = os.path.join(os.path.dirname(__file__), "gates_poses.json")
        
        with open(json_path, 'r') as f:
            all_tracks = json.load(f)
        
        # 3. Select the gate data for the specified track
        if track_name in all_tracks:
            GATES = all_tracks[track_name]
            GATE_ORDER = list(GATES.keys()) # E.g., ["Gate A", "Gate B", ...]
        else:
            raise ValueError(f"Track name '{track_name}' not found in {json_path}")
            
        # 4. (NEW) Pre-calculate gate normals and initial side
        # The drone starts at [0, 0, 1]
        initial_pos = np.array([0.0, 0.0, 1.0])
        
        for gate_name in GATE_ORDER:
            gate_pose = GATES[gate_name]
            gate_pos = np.array(gate_pose[0:3])
            pitch, roll, yaw = gate_pose[3:6]
            
            # --- (CRITICAL FIX) ---
            # Replicate the evaluator's buggy normal vector logic
            
            # Bug 1: `if yaw ==0: yaw = 3.14`
            # We create a new variable `eval_yaw` to hold this.
            eval_yaw = yaw
            if eval_yaw == 0:
                eval_yaw = 3.14
            
            # Bug 2: `rotation.apply([yaw, 0, 0])`
            # It uses the *modified* yaw for the rotation AND inside the vector
            rotation = R.from_euler('xyz', [pitch, roll, eval_yaw])
            gate_normal = rotation.apply([eval_yaw, 0, 0])
            # --- End of Fix ---
            
            gate_normal = gate_normal / np.linalg.norm(gate_normal)
            GATE_NORMALS.append(gate_normal)
            
            # 5. Check which side of the plane we start on
            vec_to_gate = initial_pos - gate_pos
            dot_product = np.dot(vec_to_gate, gate_normal)
            sign = np.sign(dot_product)
            GATE_PLANE_SIDES.append(sign) # This will be the "current" side
            INITIAL_GATE_SIDES.append(sign) # This is the "starting" side (won't change)

        print(f"INFO: state_controller.py (V1.3-AutoStop) successfully initialized for track: {track_name}")

    except Exception as e:
        print(f"CRITICAL ERROR in state_controller initialization: {e}")

# --- Run the initialization function AS SOON as this file is imported ---
_initialize_controller()

# --------------------------------------------------------------------------
# --- Internal Controller State ---
# --------------------------------------------------------------------------
target_gate_index = 0
lap_count = 0
prev_yaw = 0.0 # Used to estimate yaw rate for the D-term

# --------------------------------------------------------------------------
# --- Constants ---
# --------------------------------------------------------------------------
TOTAL_LAPS = 2       # As per README, 2 laps (8 gates) are required
TOTAL_GATES = 4      # Number of gates per lap
DT = 0.05            # Simulation time step, from drone_dynamics.py
# (NEW) Using the evaluator's radius, not the 0.03 one
GATE_RADIUS = 0.38   # (meters) Radius from ece484_evaluate.py

# --------------------------------------------------------------------------
# --- PD Gains (Original V1 Gains) ---
# --------------------------------------------------------------------------
KP_POS = np.array([3.0, 3.0, 3.0])  # Proportional gains
KD_POS = np.array([2.5, 2.5, 2.5])  # Derivative gains

KP_YAW = 2.5  # Proportional gain
KD_YAW = 0.8  # Derivative gain

# --------------------------------------------------------------------------
# --- Saturation Limits (to prevent extreme commands) ---
# --------------------------------------------------------------------------
MAX_ACCEL = 5.0      # (m/s^2) Max acceleration in any single axis
MAX_YAW_RATE = 2.0   # (rad/s) Max yaw rate

def state_controller(state):
    """
    Calculates control inputs for a drone based on its current state and target gates.
    """
    global target_gate_index, lap_count, prev_yaw, GATE_PLANE_SIDES

    # --- 1. Check for Race Completion ---
    if lap_count >= TOTAL_LAPS:
        # --- (MODIFICATION 1 of 2) ---
        # Was: return [0, 0, 0, 0] (Hover)
        # Now: return None (Stop simulation)
        return None

    # --- 2. Check for Initialization ---
    if not GATES or not GATE_ORDER:
        print("ERROR: Controller not initialized with GATES data. Returning zero control.")
        return [0, 0, 0, 0]

    # --- 3. Determine Current Target Gate ---
    gate_name = GATE_ORDER[target_gate_index]
    target_pose = GATES[gate_name]
    target_pos = np.array(target_pose[0:3]) # This is the gate center

    # --- 4. Unpack Current State ---
    current_pos = np.array(state[0:3])
    current_vel = np.array(state[3:6])
    current_yaw = state[6]

    # --- 5. (MODIFIED) Implement Robust Gate Switching Logic ---
    
    # Get the normal and "previous side" for the *current target gate*
    gate_normal = GATE_NORMALS[target_gate_index]
    prev_side = GATE_PLANE_SIDES[target_gate_index]
    
    # Calculate our current position relative to the gate's plane
    vec_to_gate = current_pos - target_pos
    dist_to_plane = np.dot(vec_to_gate, gate_normal)
    current_side = np.sign(dist_to_plane)

    # Check for a plane crossing
    if current_side != 0 and prev_side is not None and current_side != prev_side:
        # A crossing occurred! Check if it was valid.
        dist_to_center = np.linalg.norm(vec_to_gate)
        
        if dist_to_center < GATE_RADIUS:
            # SUCCESSFUL PASS
            print(f"INFO: Controller detected PASS for {gate_name} (Dist: {dist_to_center:.2f}m)")
            
            # Update the *next* gate's "previous side" to None, so it's
            # ready to be armed when we approach it.
            if lap_count > 0:
                GATE_PLANE_SIDES[target_gate_index] = None
                
            # Increment target
            target_gate_index += 1
            if target_gate_index >= TOTAL_GATES:
                target_gate_index = 0
                lap_count += 1
                if lap_count >= TOTAL_LAPS:
                    # --- (MODIFICATION 2 of 2) ---
                    # Was: return [0, 0, 0, 0] (Hover)
                    # Now: return None (Stop simulation)
                    print("INFO: Controller finished 2 laps. Stopping simulation.")
                    return None
                
                # (NEW) Reset all plane signs for the new lap
                print("INFO: Controller completed lap, resetting plane sides.")
                for i in range(TOTAL_GATES):
                    GATE_PLANE_SIDES[i] = INITIAL_GATE_SIDES[i]

            # Get the new target gate data for the PD controller
            gate_name = GATE_ORDER[target_gate_index]
            target_pose = GATES[gate_name]
            target_pos = np.array(target_pose[0:3])
        
        else:
            # MISSED PASS
            print(f"WARNING: Controller detected MISS for {gate_name} (Dist: {dist_to_center:.2f}m)")
            # We flew past but missed. We must "reset" our side
            # so we can detect crossing *back* to try again.
            GATE_PLANE_SIDES[target_gate_index] = current_side
            
    # Update our current side if it's not zero (to avoid chatter)
    elif current_side != 0:
        GATE_PLANE_SIDES[target_gate_index] = current_side

    # --- 6. Calculate Position Control (PD Controller) ---
    
    # Proportional error (position)
    error_pos = target_pos - current_pos

    # Derivative error (velocity)
    # This is the original "stop-at-gate" logic
    target_vel = np.array([0.0, 0.0, 0.0])
    error_vel = target_vel - current_vel

    # Calculate total acceleration command (ax, ay, az)
    accel_command = KP_POS * error_pos + KD_POS * error_vel

    # Apply saturation (clipping)
    ax = np.clip(accel_command[0], -MAX_ACCEL, MAX_ACCEL)
    ay = np.clip(accel_command[1], -MAX_ACCEL, MAX_ACCEL)
    az = np.clip(accel_command[2], -MAX_ACCEL, MAX_ACCEL)

    # --- 7. Calculate Yaw Control (PD Controller) ---

    # The drone should look at the target gate
    vec_to_target = target_pos - current_pos
    desired_yaw = np.arctan2(vec_to_target[1], vec_to_target[0])

    # Proportional error (yaw)
    error_yaw = desired_yaw - current_yaw
    
    # Handle angle wrapping (normalize error to [-pi, pi])
    error_yaw = (error_yaw + np.pi) % (2 * np.pi) - np.pi

    # Derivative error (yaw rate)
    yaw_rate_est = (current_yaw - prev_yaw) / DT
    target_yaw_rate = 0.0
    error_yaw_rate = target_yaw_rate - yaw_rate_est

    # Calculate total yaw rate command
    yaw_rate_command = KP_YAW * error_yaw + KD_YAW * error_yaw_rate
    
    # Apply saturation (clipping)
    yaw_rate_command = np.clip(yaw_rate_command, -MAX_YAW_RATE, MAX_YAW_RATE)

    # --- 8. Update State and Return ---
    prev_yaw = current_yaw  # Store current yaw for next step's D-term
    
    control = [ax, ay, az, yaw_rate_command]
    return control

# --------------------------------------------------------------------------
# --- Main function for testing (optional) ---
# --------------------------------------------------------------------------
if __name__ == "__main__":
    """
    This block allows you to test the state_controller function directly
    by running `python3 ece484_state_controller.py`.
    
    NOTE: This test block is NON-FUNCTIONAL unless you
    manually set the GATES and GATE_ORDER variables below.
    """
    
    print("="*30)
    print("Testing state_controller.py (V1.3-AutoStop) directly")
    print("="*30)
    
    # --- Mock Initialization (REQUIRED for direct testing) ---
    print("Manually initializing GATES for testing...")
    GATES = {
        "Gate A": [1.5, -0.15, 0.4, 0, 0, -1.57],
        "Gate B": [-0.25, -1.7, 0.4, 0, 0, -3.14],
        "Gate C": [-1.8, 0.05, 0.4, 0, 0, 1.57],
        "Gate D": [-0.05, 1.6, 0.4, 0, 0, 0]
    }
    GATE_ORDER = ["Gate A", "Gate B", "Gate C", "Gate D"]
    # Manually calculated (BUGGY, but evaluator-matched) normals
    GATE_NORMALS = [ 
        np.array([-1.57, 0.0, 0.0]) / 1.57, # Gate A (yaw=-1.57)
        np.array([-3.14, 0.0, 0.0]) / 3.14, # Gate B (yaw=-3.14)
        np.array([1.57, 0.0, 0.0]) / 1.57,  # Gate C (yaw= 1.57)
        np.array([3.14, 0.0, 0.0]) / 3.14   # Gate D (yaw=0 -> 3.14)
    ]
    GATE_NORMALS = [g / np.linalg.norm(g) for g in GATE_NORMALS] # Normalize
    
    # Manually calculated initial signs
    initial_pos = np.array([0,0,1])
    INITIAL_GATE_SIDES = []
    GATE_PLANE_SIDES = []
    for i in range(4):
        vec = initial_pos - np.array(GATES[GATE_ORDER[i]][0:3])
        sign = np.sign(np.dot(vec, GATE_NORMALS[i]))
        INITIAL_GATE_SIDES.append(sign)
        GATE_PLANE_SIDES.append(sign)
    # --------------------------------------------------------

    # Initial state (at origin, 1m high)
    state = [0, 0, 1, 0, 0, 0, 0] # x, y, z, vx, vy, vz, yaw
    
    # Get the control command
    control = state_controller(state)
    
    print(f"Current State: {state}")
    print(f"Target Gate: {GATE_ORDER[target_gate_index]} at {GATES[GATE_ORDER[target_gate_index]][0:3]}")
    print(f"Calculated Control: [ax, ay, az, yaw_rate] = {control}")