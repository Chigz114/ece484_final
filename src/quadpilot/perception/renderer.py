"""Render-only GSplat adapter that does not require the original training set."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets.generation import CameraIntrinsics, normalize_rgb


class PoseTransform:
    """Map project/body camera poses into normalized Nerfstudio coordinates."""

    def __init__(self, dataparser_transform_path: Path) -> None:
        info = json.loads(Path(dataparser_transform_path).read_text(encoding="utf-8"))
        self.dataparser_transform = np.asarray(info["transform"], dtype=np.float64)
        self.scale = float(info["scale"])
        if self.dataparser_transform.shape != (3, 4):
            raise ValueError("dataparser transform must have shape (3,4)")
        if (
            not np.isfinite(self.dataparser_transform).all()
            or not np.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("invalid dataparser transform or scale")
        dataparser_rotation = self.dataparser_transform[:, :3]
        if not np.allclose(
            dataparser_rotation.T @ dataparser_rotation,
            np.eye(3),
            rtol=0.0,
            atol=1e-5,
        ) or not np.isclose(
            np.linalg.det(dataparser_rotation), 1.0, rtol=0.0, atol=1e-5
        ):
            raise ValueError("dataparser transform must contain a proper rotation")

        # Rotation.from_euler("zyx", [-pi/2, pi/2, 0]).as_matrix(), kept
        # dependency-free so coordinate tests run without SciPy/Nerfstudio.
        self.body_from_opencv = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

    def to_nerfstudio_c2w(self, camera_to_world: np.ndarray) -> np.ndarray:
        camera = np.array(camera_to_world, dtype=np.float64, copy=True)
        if camera.shape != (4, 4):
            raise ValueError("camera_to_world must have shape (4,4)")
        if not np.isfinite(camera).all():
            raise ValueError("camera_to_world contains NaN or infinity")
        if not np.allclose(
            camera[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1e-9
        ):
            raise ValueError("camera_to_world must be a homogeneous transform")
        rotation = camera[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6
        ) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("camera_to_world must contain a proper rotation")

        camera[:3, :3] = camera[:3, :3] @ self.body_from_opencv
        camera[:3, 1:3] *= -1.0
        camera = camera[[0, 2, 1, 3], :]
        camera[2, :] *= -1.0
        camera = self.dataparser_transform @ camera
        camera[:3, 3] *= self.scale
        return camera[:3, :].astype(np.float32)


class RenderOnlySplatRenderer:
    """Load only Gaussian parameters and render RGB without a Datamanager."""

    GAUSSIAN_KEYS = {
        "gauss_params.features_dc",
        "gauss_params.features_rest",
        "gauss_params.means",
        "gauss_params.opacities",
        "gauss_params.quats",
        "gauss_params.scales",
    }

    def __init__(
        self,
        checkpoint_path: Path,
        dataparser_transform_path: Path,
        *,
        intrinsics: CameraIntrinsics = CameraIntrinsics(),
        device: str = "cuda:0",
        expected_step: int | None = None,
        expected_gaussians: int | None = None,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            import torchmetrics.image.lpip as lpip_module
            from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
            from nerfstudio.cameras.cameras import Cameras, CameraType
            from nerfstudio.data.scene_box import SceneBox
            from nerfstudio.models.splatfacto import SplatfactoModelConfig
        except ImportError as exc:
            raise RuntimeError(
                "renderer dependencies are incomplete; use the pinned WSL "
                "environment documented in REPRODUCTION.md"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("GSplat rendering requires a CUDA-capable device")
        self.torch = torch
        self.Cameras = Cameras
        self.CameraType = CameraType
        self.device = torch.device(device)
        self.intrinsics = intrinsics
        self.pose_transform = PoseTransform(dataparser_transform_path)

        config = SplatfactoModelConfig(
            sh_degree=3,
            sh_degree_interval=1000,
            background_color="random",
            rasterize_mode="classic",
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            use_bilateral_grid=False,
            output_depth_during_training=False,
            random_init=False,
        )
        dummy_xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        dummy_rgb = torch.empty((0, 3), dtype=torch.uint8)

        class UnusedLearnedMetric(nn.Module):
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                super().__init__()

            def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
                return torch.zeros((), dtype=torch.float32)

        original_lpips = lpip_module.LearnedPerceptualImagePatchSimilarity
        lpip_module.LearnedPerceptualImagePatchSimilarity = UnusedLearnedMetric
        try:
            model = config.setup(
                scene_box=SceneBox(
                    aabb=torch.tensor(
                        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                        dtype=torch.float32,
                    )
                ),
                num_train_data=1,
                metadata={},
                device="cpu",
                grad_scaler=None,
                seed_points=(dummy_xyz, dummy_rgb),
            )
        finally:
            lpip_module.LearnedPerceptualImagePatchSimilarity = original_lpips

        load_kwargs: dict[str, Any] = {"map_location": "cpu"}
        load_parameters = inspect.signature(torch.load).parameters
        if "weights_only" in load_parameters:
            load_kwargs["weights_only"] = False
        if "mmap" in load_parameters:
            load_kwargs["mmap"] = True
        checkpoint = torch.load(str(Path(checkpoint_path)), **load_kwargs)
        step = int(checkpoint["step"])
        if expected_step is not None and step != expected_step:
            raise RuntimeError(
                f"checkpoint step {step} does not match expected {expected_step}"
            )

        gaussian_state: dict[str, Any] = {}
        for key, value in checkpoint["pipeline"].items():
            if key.startswith("_model.module.gauss_params."):
                gaussian_state[key.removeprefix("_model.module.")] = value
            elif key.startswith("_model.gauss_params."):
                gaussian_state[key.removeprefix("_model.")] = value
        if set(gaussian_state) != self.GAUSSIAN_KEYS:
            raise RuntimeError(
                "unexpected Gaussian state keys: " + ", ".join(sorted(gaussian_state))
            )
        gaussian_count = int(gaussian_state["gauss_params.means"].shape[0])
        if expected_gaussians is not None and gaussian_count != expected_gaussians:
            raise RuntimeError(
                f"Gaussian count {gaussian_count} does not match "
                f"expected {expected_gaussians}"
            )

        incompatible = model.load_state_dict(gaussian_state, strict=False)
        if incompatible is not None and incompatible.unexpected_keys:
            raise RuntimeError(
                f"unexpected model state keys: {incompatible.unexpected_keys}"
            )
        allowed_missing_prefixes = (
            "lpips.",
            "psnr.",
            "ssim.",
            "camera_optimizer.",
        )
        if incompatible is not None:
            disallowed_missing = [
                key
                for key in incompatible.missing_keys
                if key != "device_indicator_param"
                and not key.startswith(allowed_missing_prefixes)
            ]
            if disallowed_missing:
                raise RuntimeError(f"missing model state keys: {disallowed_missing}")
        for key, expected_tensor in gaussian_state.items():
            parameter_name = key.removeprefix("gauss_params.")
            actual_tensor = model.gauss_params[parameter_name]
            if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
                raise RuntimeError(
                    f"loaded parameter shape mismatch for {key}: "
                    f"{tuple(actual_tensor.shape)} != {tuple(expected_tensor.shape)}"
                )

        model.lpips = nn.Identity()
        model.psnr = nn.Identity()
        model.ssim = nn.Identity()
        model.step = step
        model.eval()
        model.to(self.device)
        self.model = model
        self.checkpoint_step = step
        self.gaussian_count = gaussian_count

    def render_rgb(self, camera_to_world: np.ndarray) -> np.ndarray:
        c2w = self.pose_transform.to_nerfstudio_c2w(camera_to_world)
        camera = self.Cameras(
            camera_to_worlds=self.torch.from_numpy(c2w)[None],
            fx=self.intrinsics.fx,
            fy=self.intrinsics.fy,
            cx=self.intrinsics.cx,
            cy=self.intrinsics.cy,
            width=self.intrinsics.width,
            height=self.intrinsics.height,
            distortion_params=None,
            camera_type=self.CameraType.PERSPECTIVE,
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model.get_outputs_for_camera(camera)
        if "rgb" not in outputs:
            raise KeyError(f"renderer output has no rgb key: {sorted(outputs)}")
        rgb = outputs["rgb"].detach().to("cpu").numpy()
        if rgb.shape != (
            self.intrinsics.height,
            self.intrinsics.width,
            3,
        ):
            raise RuntimeError(f"unexpected rendered RGB shape: {rgb.shape}")
        return rgb

    def render_rgb_u8(self, camera_to_world: np.ndarray) -> np.ndarray:
        return normalize_rgb(
            self.render_rgb(camera_to_world),
            width=self.intrinsics.width,
            height=self.intrinsics.height,
            minimum_dynamic_range=20,
        )
