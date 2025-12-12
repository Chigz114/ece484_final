"""
Automatic dataset generator for NPE (Neural Pose Estimator) training.
Generates images with pose labels by randomly sampling within valid render bounds.
"""
import cv2
import os
import numpy as np
import torch
from ns_renderer_4_gates import SplatRenderer
from scipy.spatial.transform import Rotation as R
import json
import argparse
from tqdm import tqdm


# Compatibility shim for PyTorch >= 2.6
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat


class AutoDatasetGenerator:
    """
    Automatically generate dataset for NPE training.
    Samples random poses within valid render bounds and saves images with pose labels.
    """
    
    # Valid render bounds (from interactive tool trackbar values)
    # Trackbar to real coordinate conversion:
    #   x = (x_scaled - 100) / 10.0
    #   y = (y_scaled - 100) / 10.0
    #   z = (z_scaled - 50) / 10.0
    
    # Track configurations: bounds and default splatfacto model paths
    TRACK_CONFIG = {
        "circle": {
            # Trackbar values: x=53~100, y=33~89, z=42~54
            "bounds": {
                "x_min": (53 - 100) / 10.0,   # -4.7
                "x_max": (100 - 100) / 10.0,  # 0.0
                "y_min": (33 - 100) / 10.0,   # -6.7
                "y_max": (89 - 100) / 10.0,   # -1.1
                "z_min": (42 - 50) / 10.0,    # -0.8
                "z_max": (54 - 50) / 10.0,    # 0.4
            },
            "model_path": "./outputs/circle/splatfacto/2025-05-09_144210/",
            "output_dir": "npe_datasets/circle",
        },
        "uturn": {
            # Trackbar values: x=50~110, y=28~90, z=42~55
            "bounds": {
                "x_min": (50 - 100) / 10.0,   # -5.0
                "x_max": (110 - 100) / 10.0,  # 1.0
                "y_min": (28 - 100) / 10.0,   # -7.2
                "y_max": (90 - 100) / 10.0,   # -1.0
                "z_min": (42 - 50) / 10.0,    # -0.8
                "z_max": (55 - 50) / 10.0,    # 0.5
            },
            "model_path": "./outputs/uturn/splatfacto/2025-05-09_151825/",
            "output_dir": "npe_datasets/uturn",
        },
        "lemniscate": {
            # User-specified bounds: x=-4.5~0, y=-7~-0.5, z=-1~0.2
            "bounds": {
                "x_min": -4.5,
                "x_max": 0.0,
                "y_min": -7.0,
                "y_max": -0.5,
                "z_min": -1.0,
                "z_max": 0.2,
            },
            "model_path": "./outputs/lemniscate/splatfacto/2025-05-09_153156/",
            "output_dir": "npe_datasets/lemniscate",
        },
    }
    
    # Backward compatibility alias
    BOUNDS = {k: v["bounds"] for k, v in TRACK_CONFIG.items()}
    
    def __init__(self, model_path, output_dir, track="circle"):
        self.output_dir = output_dir
        self.track = track
        self.images_dir = os.path.join(output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Get bounds for this track
        if track not in self.BOUNDS:
            raise ValueError(f"Unknown track: {track}. Available: {list(self.BOUNDS.keys())}")
        self.bounds = self.BOUNDS[track]
        
        print(f"Track: {track}")
        print(f"Bounds: x=[{self.bounds['x_min']:.2f}, {self.bounds['x_max']:.2f}], "
              f"y=[{self.bounds['y_min']:.2f}, {self.bounds['y_max']:.2f}], "
              f"z=[{self.bounds['z_min']:.2f}, {self.bounds['z_max']:.2f}]")
        
        # Initialize renderer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.renderer = SplatRenderer(
            model_path + "config.yml",
            model_path + "dataparser_transforms.json",
            width=640,
            height=480,
            fx=546.84164912,
            fy=547.57957461,
            cx=349.18316327,
            cy=215.54486004,
        )
        
        # Intrinsics for metadata
        self.intrinsics = {
            "fx": 546.84164912,
            "fy": 547.57957461,
            "cx": 349.18316327,
            "cy": 215.54486004
        }
        
        self.saved_poses = []
    
    def sample_random_pose(self):
        """Sample a random pose within valid bounds."""
        x = np.random.uniform(self.bounds["x_min"], self.bounds["x_max"])
        y = np.random.uniform(self.bounds["y_min"], self.bounds["y_max"])
        z = np.random.uniform(self.bounds["z_min"], self.bounds["z_max"])
        yaw = np.random.uniform(-np.pi, np.pi)  # Full 360 degree range
        
        # roll and pitch fixed to 0
        return [x, y, z, 0.0, 0.0, yaw]
    
    def pose_to_camera_matrix(self, pose):
        """Convert [x, y, z, roll, pitch, yaw] to 4x4 camera matrix."""
        x, y, z, roll, pitch, yaw = pose
        rot = R.from_euler("xyz", (roll, pitch, yaw))
        R_matrix = rot.as_matrix()
        
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R_matrix
        camera_pose[:3, 3] = [x, y, z]
        
        return camera_pose
    
    def render_pose(self, pose):
        """Render image at given pose."""
        camera_pose = self.pose_to_camera_matrix(pose)
        img, _, _, _ = self.renderer.render(camera_pose)
        
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        cv_img = np.ascontiguousarray(img[:, :, ::-1], dtype=np.uint8)
        return cv_img
    
    def generate_dataset(self, n_samples):
        """Generate n_samples images with random poses."""
        print(f"\nGenerating {n_samples} samples...")
        
        for i in tqdm(range(n_samples), desc="Generating"):
            # Sample random pose
            pose = self.sample_random_pose()
            
            # Render image
            try:
                img = self.render_pose(pose)
            except Exception as e:
                print(f"\nRender error at sample {i}: {e}")
                continue
            
            # Save image
            img_path = os.path.join(self.images_dir, f"frame_{i:05d}.png")
            cv2.imwrite(img_path, img)
            
            # Store pose
            self.saved_poses.append(pose)
        
        # Save metadata
        self.save_metadata()
        
        print(f"\n✓ Generated {len(self.saved_poses)} samples")
        print(f"  Images: {self.images_dir}/")
        print(f"  Metadata: {os.path.join(self.output_dir, 'metadata.json')}")
    
    def save_metadata(self):
        """Save poses and metadata to JSON."""
        metadata = {
            "n_frames": len(self.saved_poses),
            "track": self.track,
            "bounds": self.bounds,
            "poses": self.saved_poses,
            "pose_format": ["x", "y", "z", "roll", "pitch", "yaw"],
            "image_size": [640, 480],
            "intrinsics": self.intrinsics
        }
        
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    # Get available tracks
    available_tracks = list(AutoDatasetGenerator.TRACK_CONFIG.keys())
    
    parser = argparse.ArgumentParser(description='Auto dataset generator for NPE training')
    parser.add_argument('--track', type=str, default='circle',
                        choices=available_tracks,
                        help='Track type (determines default model_path and output_dir)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to GSplat model (optional, uses track default if not specified)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (optional, uses track default if not specified)')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Get track config
    track_config = AutoDatasetGenerator.TRACK_CONFIG[args.track]
    
    # Use track defaults if not specified
    model_path = args.model_path or track_config["model_path"]
    output_dir = args.output_dir or track_config["output_dir"]
    
    print(f"\n{'='*60}")
    print(f"NPE Dataset Generator")
    print(f"{'='*60}")
    print(f"Track: {args.track}")
    print(f"Model path: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Samples: {args.n_samples}")
    print(f"{'='*60}\n")
    
    # Set random seed
    np.random.seed(args.seed)
    
    generator = AutoDatasetGenerator(
        model_path=model_path,
        output_dir=output_dir,
        track=args.track
    )
    
    generator.generate_dataset(args.n_samples)


if __name__ == '__main__':
    main()
