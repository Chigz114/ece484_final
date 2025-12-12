import cv2
import os
import numpy as np
import torch
from ns_renderer_4_gates import SplatRenderer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import json 


# Compatibility shim: PyTorch >=2.6 defaults torch.load(weights_only=True),
# which breaks loading older Nerfstudio checkpoints that pickle numpy scalars.
# We trust our local checkpoints, so default to weights_only=False here.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = "./outputs/empty_arena/splatfacto/2025-05-08_195726/"
# model_path = "./outputs/one_gate/splatfacto/2025-05-08_204504/"
# model_path = "./outputs/circle/splatfacto/2025-05-09_144210/"
# model_path = "./outputs/uturn/splatfacto/2025-05-09_151825/"
model_path = "./outputs/lemniscate/splatfacto/2025-05-09_153156/"

out_img = os.path.join("images/","combo.png")

scale_factor = 1
width, height = 1280, 720

    # "w": 1280,
    # "h": 720,
    # "fl_x": 1008.57564,
    # "fl_y": 1014.23455,
    # "cx": 662.151392,
    # "cy": 245.460335,

renderer = SplatRenderer(
    model_path+"config.yml",
    model_path+"dataparser_transforms.json",
    # width = width//scale_factor,
    # height = height//scale_factor,
    # fx = 1008.57564/scale_factor, 
    # fy = 1014.23455/scale_factor,
    # cx = 662.151392/scale_factor,
    # cy = 245.460335/scale_factor

    # # Iphone Intrinsics
    # width = 1080,
    # height = 1920,
    # fx = 1634.4728852715373,
    # fy = 1644.960909102645,
    # cx = 536.3343420198621,
    # cy = 959.9484127789314,

    # ArduCam Intrinsics
    width = 640,
    height = 480,
    fx = 546.84164912, 
    fy = 547.57957461,
    cx = 349.18316327,
    cy = 215.54486004, 
)

# =====================
# Helper: Gate mask via geometric projection
# =====================

def build_gate_circle_points(center, yaw, diameter=1.0, n_points=16):
    """
    Construct points on a circular gate in world space.
    center: (x,y,z), yaw about +Z (radians), diameter in meters
    Returns np.ndarray shape (n_points, 3) of points on the circle perimeter.
    """
    cx, cy, cz = center
    radius = diameter / 2.0
    # Gate plane: u = width axis, v = height axis
    u = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    v = np.array([0.0, 0.0, 1.0])
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    points = []
    for angle in angles:
        offset = radius * (np.cos(angle) * u + np.sin(angle) * v)
        points.append(np.array([cx, cy, cz]) + offset)
    return np.stack(points, axis=0)

def project_points_to_image(points_world, camera_to_world, fx, fy, cx, cy, w_img, h_img, debug=False):
    """
    Project 3D world points to image using Nerfstudio camera conventions.
    
    Nerfstudio conventions:
    - Camera space: +X right, +Y up, +Z back (away from camera), -Z is forward
    - World space: +Z is up (sky direction)
    - camera_to_world: 4x4 transform matrix
    
    Args:
        points_world: Nx3 world coordinates
        camera_to_world: 4x4 camera-to-world matrix
        fx, fy, cx, cy: camera intrinsics
        w_img, h_img: image dimensions
    
    Returns:
        pts_pix: Nx2 pixel coordinates
        vis: N boolean mask for visibility (points in front of camera)
    """
    # Convert to homogeneous coordinates
    N = points_world.shape[0]
    points_world_h = np.concatenate([points_world, np.ones((N, 1))], axis=1)  # Nx4
    
    # Get world_to_camera transform (inverse of camera_to_world)
    world_to_camera = np.linalg.inv(camera_to_world)
    
    # Transform points to camera space
    pts_cam_h = (world_to_camera @ points_world_h.T).T  # Nx4
    pts_cam = pts_cam_h[:, :3]  # Nx3
    
    if debug:
        print(f"    Point in camera coords: {pts_cam[0]}")
    
    # In Nerfstudio/OpenGL convention, -Z is forward, so points with z < 0 are in front
    z = pts_cam[:, 2]
    vis = z < -1e-6  # Points with negative z are visible
    
    # Perspective projection (pinhole camera model)
    # Use negative z because forward is -Z
    z_safe = np.where(np.abs(z) < 1e-6, -1e-6, z)
    xs = fx * (pts_cam[:, 0] / -z_safe) + cx  # Note: divide by -z
    ys = fy * (pts_cam[:, 1] / -z_safe) + cy
    
    pts_pix = np.stack([xs, ys], axis=1)
    
    # Clip to image bounds
    pts_pix[:, 0] = np.clip(pts_pix[:, 0], 0, w_img-1)
    pts_pix[:, 1] = np.clip(pts_pix[:, 1], 0, h_img-1)
    
    return pts_pix.astype(np.int32), vis

def make_gate_mask(camera_pose, intrinsics, image_size, gates, debug=False):
    """
    camera_pose: 4x4 camera-to-world (same as used for rendering)
    intrinsics: (fx, fy, cx, cy)
    image_size: (width, height)
    gates: list of dicts with keys {center: (x,y,z), yaw: float, diameter: float}
    Returns HxW uint8 mask (0/255) with filled circles for visible gates.
    """
    w_img, h_img = image_size
    fx, fy, cx, cy = intrinsics
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    
    if debug:
        print(f"\n=== Mask Generation Debug ===")
        print(f"Camera position (world): {camera_pose[:3, 3]}")
        print(f"Image size: {w_img}x{h_img}")
        print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
    for idx, g in enumerate(gates):
        # Project gate center only
        center_w = np.array(g["center"]).reshape(1, 3)
        
        if debug:
            print(f"\nGate {idx}: center={g['center']}")
            print(f"  World coords: {center_w[0]}")
        
        center_pix, center_vis = project_points_to_image(center_w, camera_pose, fx, fy, cx, cy, w_img, h_img, debug=debug)
        
        if debug:
            print(f"  Projected pixel: {center_pix[0]}, visible: {center_vis[0]}")
        
        if center_vis[0]:
            # Draw a fixed-size circle at gate center (ignoring perspective)
            # Use a moderate radius that's visible (e.g., 30 pixels)
            radius_pix = 30
            center_tuple = tuple(center_pix[0].astype(int))
            cv2.circle(mask, center_tuple, radius_pix, 255, -1)
            
            if debug:
                print(f"  ✓ Drew circle at {center_tuple}, radius={radius_pix}")
        elif debug:
            print(f"  ✗ Skipped (center not visible)")
    
    return mask

# Load gate positions from JSON
with open("scripts/gates_poses.json", "r") as f:
    gates_data = json.load(f)

# Sim(3) pre-alignment parameters (scale, rotation, translation)
# These align the gate coordinates from gates_poses.json to the GSplat scene
# Adjust these parameters to match the actual gate positions in the rendered RGB
GATE_PREALIGN = {
    "scale": 2.0,  # Adjust manually if needed
    "rotation": [0, 0, 0.15],  # [roll, pitch, yaw] - adjust manually
    "translation": [0, 0, -1.0]
}

def apply_sim3_prealignment(gate_pos, scale, rotation_euler, translation):
    """
    Apply Sim(3) transformation: X' = s * R * X + t
    This pre-aligns gate coordinates before applying the renderer's transform chain.
    """
    # Apply scale
    pos = np.array(gate_pos) * scale
    
    # Apply rotation
    if np.any(rotation_euler):
        R_pre = R.from_euler('xyz', rotation_euler).as_matrix()
        pos = R_pre @ pos
    
    # Apply translation
    pos = pos + np.array(translation)
    
    return pos

# Convert Lemniscate gates from JSON format [x, y, z, roll, pitch, yaw] to our format
LEMNISCATE_GATES = []
for gate_name, pose in gates_data["Lemniscate_Track"].items():
    x, y, z, roll, pitch, yaw = pose
    
    # Apply Sim(3) pre-alignment
    aligned_center = apply_sim3_prealignment(
        [x, y, z],
        GATE_PREALIGN["scale"],
        GATE_PREALIGN["rotation"],
        GATE_PREALIGN["translation"]
    )
    
    LEMNISCATE_GATES.append({
        "center": tuple(aligned_center),
        "yaw": yaw,
        "diameter": 1.0 * GATE_PREALIGN["scale"]  # Scale diameter too
    })

# Camera pose in FalconGym convention: [x, y, z, roll, pitch, yaw]
# Matching official ns-renderer.py format
pose = np.array([-3.5, -7, -0.2, 0, 0, 1.2])
# pose = np.array([x, y, z, roll, pitch, yaw])

x, y, z, roll, pitch, yaw = pose

# Use official Euler angle convention: 'xyz' order
rot = R.from_euler("xyz", (roll, pitch, yaw))

# Analog
# rot = R.from_euler("ZYX", (yaw, 0.1, 0.15))

# Get the 3x3 rotation matrix
R_matrix = rot.as_matrix()

# Step 2: Create the 4x4 transformation matrix
camera_pose = np.eye(4)  # Start with an identity matrix
camera_pose[:3, :3] = R_matrix  # Set the top-left 3x3 to the rotation matrix

camera_pose[:3, 3] = [x, y, z]  # Set the translation components


img, transformed_pose, dp_transform, dp_scale = renderer.render(camera_pose) # RGB and transforms
cv_img = img[:, :, ::-1]
cv2.imwrite(out_img, cv_img)

# Transform gate positions through ALL the same coordinate transformations as camera
# This matches the exact sequence in ns_renderer_4_gates.py lines 177-191
def transform_gate_position(gate_pos, dp_transform, dp_scale):
    """Apply same coordinate transforms to gate position as applied to camera."""
    # Start with gate position as 4x4 matrix (position only, no rotation)
    gate_mat = np.eye(4)
    gate_mat[:3, 3] = gate_pos
    
    # Step 1: Euler rotation (same as camera)
    tmp = R.from_euler('zyx', [-np.pi/2, np.pi/2, 0]).as_matrix()
    gate_mat[:3, :3] = gate_mat[:3, :3] @ tmp
    
    # Step 2: COLMAP coordinate frame conversion
    gate_mat[0:3, 1:3] *= -1
    gate_mat = gate_mat[np.array([0, 2, 1, 3]), :]
    gate_mat[2, :] *= -1
    
    # Step 3: Dataparser transform and scale
    gate_mat = dp_transform @ gate_mat
    gate_mat[:3, 3] *= dp_scale
    
    return gate_mat[:3, 3]

transformed_gates = []
for g in LEMNISCATE_GATES:
    transformed_center = transform_gate_position(g["center"], dp_transform, dp_scale)
    transformed_gates.append({
        "center": tuple(transformed_center),
        "yaw": g["yaw"],
        "diameter": g["diameter"]  # Keep original for now
    })

# Generate corresponding gate mask via projection using the SAME transformed pose
intrinsics = (546.84164912, 547.57957461, 349.18316327, 215.54486004)
image_size = (640, 480)
mask = make_gate_mask(transformed_pose, intrinsics, image_size, transformed_gates, debug=False)

# Save to a dedicated folder
save_dir = os.path.join("dataset_demo", "lemniscate")
os.makedirs(save_dir, exist_ok=True)
rgb_path = os.path.join(save_dir, "demo_rgb.png")
mask_path = os.path.join(save_dir, "demo_mask.png")
cv2.imwrite(rgb_path, cv_img)
cv2.imwrite(mask_path, mask)

# Create overlay visualization for verification
overlay = cv_img.copy()
# Find contours in mask and draw them on RGB
mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# Make mask semi-transparent red
mask_overlay = np.zeros_like(cv_img)
mask_overlay[mask > 0] = [0, 0, 255]  # Red in BGR
overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
overlay_path = os.path.join(save_dir, "demo_overlay.png")
cv2.imwrite(overlay_path, overlay)

print(f"Saved RGB to {rgb_path}\nSaved mask to {mask_path}\nSaved overlay to {overlay_path}")

plt.imshow(img)
plt.show()
# cv2.imshow("demo", cv_img)
# cv2.waitKey(0)

# # closing all open windows
# cv2.destroyAllWindows()