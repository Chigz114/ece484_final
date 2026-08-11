# Quad Pilots reproduction log

This document records the complete three-track simulation reproduction and the
remaining hardware-only boundary. It deliberately separates verified results
from claims that still require physical calibration or manual safety evidence.

Install the package from the repository root before running the commands below:

```bash
python -m pip install -e '.[ml]'
```

The separate renderer environment is pinned in `configs/environments/`.

## Milestone 1: visual-control core with an oracle pose observation

The first recovered milestone tests the part downstream of the neural pose
estimator.  An observation provider supplies exact `[x, y, z, yaw]`; the final
submission's December 2025 trajectory planner and controller then produce body
frame acceleration and yaw-rate commands.  The recovered dynamics explicitly
rotate body acceleration into the NeRF world frame before integration.

Run from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
quadpilot simulate oracle --track all --max-steps 1200
python -m unittest discover -s tests -v
```

Strict evaluation uses the course gate radius of 0.38 m, requires the gates in
track order and in the correct crossing direction, and requires two laps/eight
crossings.  It does **not** use the submitted controller's permissive 1.5x pass
threshold.  The deterministic baseline at 50 ms is:

| Track | Gates | Success rate | Mean gate error | Steps | Mission time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circle | 8/8 | 100% | 2.55 cm | 356 | 17.76 s |
| U-turn | 8/8 | 100% | 2.62 cm | 331 | 16.53 s |
| Lemniscate | 8/8 | 100% | 2.20 cm | 454 | 22.68 s |

The machine-readable reference is
`results/baselines/oracle_strict.json`. Generated trajectories and detailed gate
crossings are written under `repro_outputs/oracle/` and are intentionally
ignored by Git.

## Coordinate and interface contract

- State: `[x, y, z, vx, vy, vz, yaw]` in the project's NeRF world frame, metres,
  metres/second and radians.
- Pose observation: `[x, y, z, yaw]` in the same frame.
- Controller output: `[ax_body, ay_body, az_body, yaw_rate]` in metres/second
  squared and radians/second.
- Dynamics: body acceleration is rotated into the world frame using current yaw.
- These gate poses are not the FalconGym evaluator poses and are not Vicon
  coordinates. A calibrated transform is required before hardware deployment.

## What this milestone proves—and what it does not

It proves that the recovered final trajectory planner, control law, frame
conversion and an order-aware evaluator can complete all three courses when pose
input is correct.  It does not reproduce neural localization yet and does not
validate a real camera or Crazyflie.

At the start of recovery, the public repository omitted the three rendered
GSplat/NeRF scenes, rendered NPE datasets, trained NPE checkpoints and
gate-focused training assets. The following milestones guided the work:

1. recover or regenerate the three renderable scenes and camera calibration;
2. make dataset generation deterministic and validate pose labels visually;
3. train and validate one NPE per track, including gate-local error slices;
4. inject recorded NPE errors into this harness, validate the EKF, then run the
   renderer/NPE/EKF/controller loop end to end;
5. only after offline safety tests, adapt the frame and message contracts for the
   Crazyflie/Vicon hardware stack.

Recovered/expected scene files and their hashes are recorded in
`configs/assets/manifest.json`. Check presence quickly, or verify full hashes, with:

```powershell
quadpilot verify assets --track all
quadpilot verify assets --track circle --hash
```

The untouched Circle archive has been recovered locally and its extracted copy
matches all three recorded file hashes. The exact FalconGym-linked Lemniscate and
U-turn source profiles have also been recovered and receipt-verified. Their
locally trained Splatfacto artifacts are runtime half-resolution variants, not
the missing original full-resolution runs; the precise source, training and
renderer status is recorded in Milestone 3B.

Do not run the current hardware repository with propellers installed: its public
ROS path does not yet provide stale-data timeouts, geofencing or a verified
emergency-stop sequence.

## Milestone 2: synthetic NPE-like observation stress test

At this historical milestone the checkpoints had not yet been recovered, so the
result was explicitly a synthetic interface test rather than a neural-network
result. Independent 5 cm
per-axis Gaussian position noise produces about 8 cm mean three-dimensional
error, close to the scale printed in the final teaser. Run the raw-pose and EKF
comparison with fixed seeds using:

```powershell
quadpilot simulate synthetic --track all --seeds 20
```

The frozen 50-seed aggregate is stored in
`results/baselines/synthetic_pose_seed0_49.json`. Under this deliberately simple
noise model, raw pose control completed 47/50 Circle, 41/50 U-turn and 44/50
Lemniscate runs. EKF plus hysteresis completed 50/50 on every track. The filtered
mean pose error is unrealistically low (~3.2 cm) because the simulated plant and
filter model match exactly; this number must not be presented as an NPE result.

The robust profile adds a 5 cm gate-plane hysteresis. This prevents one noisy
sign flip from advancing the mission state before the vehicle has crossed the
gate. Set `--crossing-hysteresis 0` to reproduce the submitted instantaneous
sign-change behavior.

The teaser and final PDF must be treated as two separate historical targets:

| Profile | NPE mean/jitter | EKF mean/jitter | DYN mean/jitter |
| --- | --- | --- | --- |
| Teaser noisy | 7.9 / 3.87 cm | 7.2 / 1.37 cm | 8.3 / 3.34 cm |
| Final PDF clean | 8.2 / 3.95 cm | 7.3 / 1.29 cm | 6.9 / 0.81 cm |

The old `DYN` curve is only a one-step prediction restarted from ground truth on
every frame, and the old jitter is `std(norm(position[t]-position[t-1]))`; it is
not accumulated inertial dead reckoning or a pure high-frequency-noise metric.
New runs preserve those legacy numbers for comparison and also report residual
step jitter and control-command change.

## Milestone 3: recovered Circle GSplat renderer

The untouched local `circle.zip` contains the exact final Circle Splatfacto run:
checkpoint step 29,999 with 308,832 Gaussians. Its Dataparser transform was
independently recomputed from the public Drive's 2,002 referenced Circle poses;
the maximum matrix difference was `3.34e-6` and the scale difference was
`5.61e-8`. This ties the local checkpoint to the public source capture rather
than merely matching a filename.

The checkpoint configuration fingerprints Nerfstudio 1.1.4 and its pinned
`gsplat==1.0.0`. A render-only loader now bypasses Nerfstudio's Datamanager, so
the original 3.60 GiB of Circle training photographs are not required for
inference. It loads only the six Gaussian tensors and does not download LPIPS or
ImageNet metric weights.

The previously missing `ns_renderer_4_gates.py` was located in FalconGym's
historical `Stock_Pavilion` branch at commit `b684f56`. It confirms the original
constructor, intrinsics and coordinate chain, but is not drop-in compatible:
its `render()` returns one image while both submitted callers unpack four
values, and its `eval_setup()` path still requires the source dataset. The new
adapter preserves the confirmed coordinate math while correcting those two
interfaces and avoiding in-place pose mutation.

The verified WSL setup is deliberately isolated from the existing Conda env:

```text
overlay venv: /home/chi/UAV/envs/quadpilot-render
Python:       3.10.20
PyTorch:      2.1.0+cu118
torchvision:  0.16.0+cu118
Nerfstudio:   1.1.4
gsplat:       1.0.0
CUDA toolkit: /home/chi/UAV/cuda-11.8 (nvcc/cudart 11.8.89)
GPU arch:     8.9 (RTX 4060)
```

`MAX_JOBS=1` is intentional. Parallel nvcc jobs exceeded the WSL memory limit
during the first build. The helper configures CUDA, the Conda GCC/G++ 11.2 host
compiler and overlay-local Ninja before gsplat is imported.

Run the deterministic Gate A smoke test from PowerShell:

```powershell
wsl.exe -d Ubuntu-22.04 -- /home/chi/UAV/envs/quadpilot-render/bin/python `
  -m quadpilot.cli.render_smoke `
  --cuda-home /home/chi/UAV/cuda-11.8
```

Verified output: 640x480 RGB `uint8`, range 2–255, image standard deviation
57.19, and a visually centered Gate A. The first run compiles the CUDA extension;
later processes reuse the build cache. Generated smoke files live under
`repro_outputs/renderer/circle/`.

Generate a label-safe dataset on WSL's ext4 storage with:

```powershell
wsl.exe -d Ubuntu-22.04 -- /home/chi/UAV/envs/quadpilot-render/bin/python `
  -m quadpilot.cli.data_generate_uniform `
  --track circle --samples 10000 --seed 42 `
  --output-dir /home/chi/UAV/quadpilot-data/npe_datasets/circle/uniform_seed42_10000_v1 `
  --cuda-home /home/chi/UAV/cuda-11.8
```

Unlike the submitted generator, failed renders do not create holes between
image filenames and pose labels. PNGs are written atomically, round-tripped,
and paired with a continuous `samples.jsonl`; progress and RNG state are saved
after every attempt. The historical axis-aligned bounds and full-yaw sampling
are retained for the base dataset. Gate-focused sampling remains a separate
fine-tuning milestone.

The public GSplat Drive is currently readable without authentication. A complete
three-track mirror is 30.37 GiB, but only 9.58 GiB across 4,995 source images is
actually referenced by the three `transforms.json` files. The exact folder IDs,
byte counts and recovered hashes are recorded in `configs/assets/manifest.json`.
The referenced Lemniscate and U-turn images have been recovered for the two local
training variants; unused originals, downscales, COLMAP databases and
calibration-selection copies were deliberately excluded.

The maintained package is developed on `codex/visual-reproduction`; reviewed
changes are published through a GitHub pull request.

## Milestone 3B: recovered Lemniscate and U-turn GSplat variants

The two source directories are exact, fail-closed profiles rather than a search
for any folder with a compatible name. Every receipt entry was streamed and
SHA-256 checked before a container could train, and the dataparser then confirmed
the camera and sparse-point counts:

| Track | Receipt SHA-256 | Entries / images | Image bytes / verified bytes | Cameras / sparse points |
| --- | --- | ---: | ---: | ---: |
| Lemniscate | `6614c5be765ab7456eac95403af4b2c6fb34e757afc263ba3aa7b9f075cd356a` | 1,555 / 1,553 | 3,362,065,056 / 3,370,611,629 | 1,553 / 183,994 |
| U-turn | `a42c422dc084375e7f2bf5ef530ac7a5409e9abc0d6c5b3fa90ccd840beb6023` | 1,442 / 1,440 | 3,062,618,402 / 3,070,717,413 | 1,440 / 175,292 |

Both profiles report zero download failures, missing images, receipt size errors
or receipt hash errors. The pinned runtime is the immutable
`dromni/nerfstudio` image digest
`ff0107a7db96bb8ee29c638729328b832b268b890c50f2a2ff25988bb84d4f75`:
Nerfstudio 1.1.4, `gsplat==1.0.0`, PyTorch 2.1.2+cu118,
torchvision 0.16.2+cu118 and Viser 0.2.3. The CPU gate also validates the known
`pip check` deviations, audits and disables nine pinned external method entry
points, confirms 43 built-in methods including Splatfacto, and verifies the
244,408,911-byte LPIPS AlexNet cache at
`7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`
before a GPU mode can start.

The 30k command uses seed 42, `camera_res_scale_factor=0.5`, dataparser
`downscale_factor=1`, `num_downscales=2`, `resolution_schedule=3000`, no
periodic evaluation, and latest-only checkpoints every 2,000 steps. Therefore
the raster scale relative to the source is 1/8 through step 2,999, 1/4 from
steps 3,000 through 5,999 and 1/2 thereafter. This is a runtime **0.5 linear-resolution
variant (0.25 of the source pixels), not the original full-resolution
experiment**; the source images themselves were not rewritten.

The final successful standalone gates are retained under unique run IDs. Earlier
diagnostic failures remain in their original directories and are not relabeled:

| Track | CPU preflight | One-step trainer smoke | 101-step trainer smoke |
| --- | --- | --- | --- |
| Lemniscate | `preflight_method_guard_20260810_v2`: PASS | `smoke1_ns114_halfres_20260810_v4`: PASS; step-0 checkpoint `a2f002c139e4cc2f6431781bb5b7e2d4a95f54e17b79d78e11e832f5b2ab1d0e` | `smoke101_ns114_halfres_20260810_v1`: PASS; step-100 checkpoint `ca5428ca5c3b6716d700f3fd86b4db993c6be573ef869a4b37ed9257604eece2` |
| U-turn | `repro_ns114_uturn_cpu_preflight_20260809T200529Z`: PASS | `smoke1_ns114_uturn_halfres_20260810_v1`: PASS; step-0 checkpoint `2069ef8521f8ba3802f7d30d03434a30f7fb112d4d7f7c2d36aa042f18d25c4d` | `smoke101_ns114_uturn_halfres_20260810_v1`: PASS; step-100 checkpoint `43b0dfb75b8cb1c69f59804d60143af159b2861a0b1927d668f430c694e24c33` |

Those are trainer startup/densification gates, not renderer smoke tests. The
locked 30k artifacts are:

| Track | Original wrapper/postflight truth | Step / Gaussians / finite | Config / transform / checkpoint SHA-256 |
| --- | --- | --- | --- |
| Lemniscate | Original `status.env`: `exit_code=2`, `result=failed`; `overall_success=false`. Independent `recovered-postflight.json` classifies only the artifacts as `WRAPPER_FAILED_TRAINING_ARTIFACTS_VERIFIED`. | 29,999 / 394,366 / yes | `e8bdab1e1914b466edc3741ee4c6bac416b6035e85ee281b438c91f13de607aa` / `d5be6872b9a89c07547bff962ce32f5513b66fec175eef4ac7585cacbeb46333` / `a8a1064a1d95a9bdc642c1ad540c8dcd2b00b28680c23978f6c47e258b611a32` |
| U-turn | Wrapper postflight: `exit_code=0`, `result=success`, artifact manifest PASS and `overall_success=true`. | 29,999 / 437,285 / yes | `13d08ac255c0f75921c1f7e212796afb5d5a67583c506b47ea28c6db6540c78a` / `abddf07924fa64e3ba57376ea27dfc67e8ff483a730d82005122f962b9f4324f` / `c3a884a5765ed86789facca4648416c01141afc93a684dcab724a5df8613b5b7` |

The Lemniscate independent postflight file has SHA-256
`f2658bdacf22b5faa1818347ba1f2a291110471eed22cf255bada83c56f83000`.
It CPU-loads the unique final checkpoint, checks all six Gaussian tensors for
finite values, verifies the source/preflight/plugin/image/cache evidence, and
deliberately preserves the wrapper's failed status. It does not turn that run
into an overall PASS. U-turn completed the wrapper guards and artifact manifest;
the Gaussian count and finite result in the table were then independently
confirmed with a read-only CPU load.

Verify either selected run directly, without silently falling back to the old
historical run names, using `--run-dir`:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- python3 `
  -m quadpilot.cli.verify_assets --track lemniscate `
  --run-dir /home/chi/UAV/quadpilot-data/gsplat_outputs/lemniscate/train-30k/repro_ns114_gs100_halfres_seed42_v1/training-output/lemniscate/splatfacto/repro_ns114_gs100_halfres_seed42_v1 `
  --hash

wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- python3 `
  -m quadpilot.cli.verify_assets --track uturn `
  --run-dir /home/chi/UAV/quadpilot-data/gsplat_outputs/uturn/train-30k/repro_ns114_uturn_gs100_halfres_seed42_v1/training-output/uturn/splatfacto/repro_ns114_uturn_gs100_halfres_seed42_v1 `
  --hash
```

Both commands return `STATUS READY` for all three complete artifact files named
by the manifest. That means exact artifact hashes match; it does not override the
Lemniscate wrapper failure recorded above.

The Lemniscate renderer has a separate 32-frame smoke dataset at
`/home/chi/UAV/quadpilot-data/npe_datasets/lemniscate/smoke_uniform_seed42_32_v1`.
It produced 32/32 RGB 640x480 PNGs with zero render failures from the locked
step-29,999/394,366-Gaussian artifacts. A full-byte dataset index passed with
fingerprint
`1ca476163856336dd85eb2b11b4a011091a77d7d6b8c64af80f22844357f8d7e`;
`metadata.json` and `samples.jsonl` hash to
`cd706a41f367c72300da5530779b3f506fa78cc1f4aecfe346de1b5e80f317b2`
and `43b69aae3f69eb2cc056b157d921e55416610c8c871d9c7b5668a0819f57bfe6`.
This renderer smoke is PASS. No equivalent U-turn renderer smoke dataset has
been completed, so U-turn renderer readiness must not be marked PASS merely
because its trainer smoke and 30k artifact checks succeeded.

These two half-resolution scenes are prerequisites, not three-track visual
reproduction completion. Lemniscate and U-turn still lack locked NPE training,
frozen TEST evaluation and real rendered NPE/EKF/controller closed loops; U-turn
also lacks renderer smoke. Only Circle has the completed neural closed-loop
evidence in Milestones 4--5.

## Milestone 4: deterministic NPE training chain

The submitted ResNet architecture is retained (ResNet-50 plus 512/256/five-value
regression head), but the training contract is now explicit:

- the first three outputs are training-split-normalized NeRF `x/y/z`;
- yaw remains `sin(yaw), cos(yaw)` and includes a small unit-norm penalty;
- inference must pass through `decode_predictions()` before control;
- train/validation/test keys are frozen in a source-balanced split manifest;
- the default full dataset fingerprint hashes every PNG byte;
- all metrics are accumulated per sample rather than averaging batch means;
- checkpoints contain preprocessing, normalizer, split, dataset, software,
  optimizer, scheduler, AMP scaler, RNG, history and repository provenance;
- ImageNet initialization is opt-in. The historical V1 weight was explicitly
  cached and verified as 102,530,333 bytes with SHA-256
  `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.

The recovered Circle renderer produced the base dataset at:

```text
/home/chi/UAV/quadpilot-data/npe_datasets/circle/uniform_seed42_10000_v1
```

It contains exactly 10,000 RGB PNGs and 10,000 JSONL records, no temporary
files, no render failures, and occupies about 2.5 GiB. Generation took 1,459.3
seconds on the RTX 4060. Its full content fingerprint is
`7cc2b7f4c258e3736e4b9ddc3077a17922fa50ed0898cc3a8ed8c0cfe3709454`;
the 80/10/10 split manifest is
`ea536723c73a7f039d239d65e3a4508786826c69c2fb471e836ee8b66f4505f9`.

The formal base run uses:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  /home/chi/miniconda3/envs/mvot-mmaction/bin/python -u `
  -m quadpilot.cli.train_npe `
  --data-dir /home/chi/UAV/quadpilot-data/npe_datasets/circle/uniform_seed42_10000_v1 `
  --output-dir /home/chi/UAV/quadpilot-data/npe_models/circle/base_resnet50_seed42_v1 `
  --weights imagenet1k_v1 --backbone resnet50 --epochs 100 `
  --batch-size 8 --accumulation-steps 4 --lr 0.0001 `
  --weight-decay 0.0001 --max-grad-norm 5 `
  --num-workers 4 --amp on --device cuda --save-every 0 `
  --fingerprint-mode full --seed 42
```

### Locked Circle base result

The base run completed all 100 epochs. Validation-only model selection chose
display epoch 96 (`checkpoint["epoch"] == 95`, 96 completed epochs): 8.147 cm
mean position error and 1.106 degrees mean yaw error on the 1,000-sample frozen
validation split. The selected checkpoint is
`base_resnet50_seed42_v1/best_npe.pth`.

The 1,000-sample frozen base test was evaluated only after training and model
selection had finished:

| Model/split | Position mean | Median | p95 | Maximum | Yaw mean | Yaw p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base checkpoint / base test | 9.232 cm | 7.224 cm | 21.618 cm | 302.714 cm | 1.156 deg | 3.019 deg |

The large maximum is reported rather than trimmed. The corresponding mean is
below the historical 10.3 cm base result, but the yaw mean is above the
historical 1.0 degree value.

### Gate-focused dataset and fine-tuning

The gate sampler selects a gate before rejection sampling, places the camera on
its incoming side, and records the gate and all offsets in each JSONL row. The
locked 4,000-view Circle dataset was generated with:

```powershell
wsl.exe -d Ubuntu-22.04 -- /home/chi/UAV/envs/quadpilot-render/bin/python -u `
  -m quadpilot.cli.data_generate_gate `
  --track circle --samples 4000 --seed 4242 `
  --run-dir /mnt/f/UAV/ece484_final/outputs/circle/splatfacto/2025-05-09_144210 `
  --output-dir /home/chi/UAV/quadpilot-data/npe_datasets/circle/gate_seed4242_4000_v1 `
  --cuda-home /home/chi/UAV/cuda-11.8 `
  --min-distance 0.35 --max-distance 2.0 `
  --lateral 0.55 --vertical 0.32 --yaw-jitter-deg 25 `
  --image-margin-px 32 --maximum-sampling-rejections 100 `
  --maximum-failures 200
```

It contains 4,000 640x480 PNG/JSONL pairs from 4,000 attempts, with zero render
failures and no temporary files. The accepted gate counts are A/B/C/D =
997/1,018/983/1,002. `metadata.json` records renderer step 29,999, 308,832
Gaussians, seed 4,242, the dataparser transform, and every sampling bound.

Fine-tuning used the base source first and gate source second so source IDs stay
stable. It inherited the base checkpoint normalizer byte-for-value and started a
fresh optimizer and scheduler:

```powershell
wsl.exe -d Ubuntu-22.04 -- /home/chi/miniconda3/envs/mvot-mmaction/bin/python -u `
  -m quadpilot.cli.train_npe `
  --data-dir /home/chi/UAV/quadpilot-data/npe_datasets/circle/uniform_seed42_10000_v1 `
  --data-dir /home/chi/UAV/quadpilot-data/npe_datasets/circle/gate_seed4242_4000_v1 `
  --output-dir /home/chi/UAV/quadpilot-data/npe_models/circle/gate_finetune_seed42_init_base_v1 `
  --init-checkpoint /home/chi/UAV/quadpilot-data/npe_models/circle/base_resnet50_seed42_v1/best_npe.pth `
  --weights none --backbone resnet50 --epochs 30 `
  --batch-size 8 --accumulation-steps 4 --lr 1e-5 `
  --weight-decay 1e-4 --max-grad-norm 5 --num-workers 4 `
  --amp on --device cuda --save-every 0 --fingerprint-mode full --seed 42
```

The combined frozen split has 11,200/1,400/1,400 train/validation/test records:
8,000/1,000/1,000 base and 3,200/400/400 gate records. The initialization
checkpoint scored 8.491 cm and 1.131 degrees on the new combined validation
split. The initialization was retained as the best candidate until a strict
validation improvement occurred. After all 30 epochs, display epoch 27
(`checkpoint["epoch"] == 26`) was selected at 7.231 cm, 0.971 degrees and
16.294 cm position p95. The final epoch was 7.490 cm, so `best_npe.pth`, not
`last_npe.pth` or `final_npe.pth`, is the locked model.

### Frozen Circle test results

The combined manifest preserves every base record's original split membership.
Exact source filtering was applied only after selecting its frozen test keys; it
does not create a new split or include gate train/validation records.

| Checkpoint / frozen test slice | n | Position mean | Median | p95 | Maximum | Yaw mean | Yaw p95 | Yaw maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base / base | 1,000 | 9.232 cm | 7.224 cm | 21.618 cm | 302.714 cm | 1.156 deg | 3.019 deg | 17.951 deg |
| Fine / combined | 1,400 | 7.932 cm | 6.439 cm | 18.023 cm | 312.812 cm | 1.049 deg | 2.690 deg | 20.749 deg |
| Fine / base source | 1,000 | 8.838 cm | 7.021 cm | 19.917 cm | 312.812 cm | 1.151 deg | 3.063 deg | 20.749 deg |
| Fine / gate source | 400 | 5.668 cm | 5.125 cm | 11.276 cm | 18.741 cm | 0.795 deg | 1.953 deg | 3.236 deg |

The fine combined position mean beats the historical 8.9 cm target. Its exact
yaw mean is 1.048799 degrees, slightly above rather than equal to the historical
1.0 degree target. The combined and fine/base rows deliberately retain the
312.811920 cm base-source maximum. No sample was removed, winsorized, or hidden;
the tail therefore remains visible in the standard deviation and RMSE.

All hyperparameters and checkpoint selection were locked from train/validation
data before these test files were read. The three prespecified fine-model slices
form one frozen reporting stage; they were not used to tune, choose another
epoch, alter the source mixture, or suppress the 312.812 cm case. A future
experiment must use a new run identifier and make its decisions on frozen
validation data while preserving these results.

### Locked Circle NPE provenance

The dataset and split entries below are semantic content hashes; file entries
are byte-for-byte SHA-256 values.

| Object | SHA-256 |
| --- | --- |
| Circle renderer checkpoint | `af37b9e28b033d0b21a47d26e56b0479649ba0fc092a97979c031b8217767069` |
| Circle dataparser transform | `c43166261f14fa78e3c9c8134dd16e716b2a1977adfeba08d0dc6b942740b874` |
| Base dataset full fingerprint | `7cc2b7f4c258e3736e4b9ddc3077a17922fa50ed0898cc3a8ed8c0cfe3709454` |
| Base split manifest | `ea536723c73a7f039d239d65e3a4508786826c69c2fb471e836ee8b66f4505f9` |
| Base best checkpoint | `457c03c5976c2913f44119b065855591a0690fd2cd0a6982543b06aab2720da5` |
| Gate `metadata.json` | `2b0fc6f063ffd7e1d3bc1fe42ff96d6355025a41b6faabdc30562a616a80dff9` |
| Gate `samples.jsonl` | `264149f284d466c23427e70a0a9fb81837607c60c87a59c525b7c024ebc40c72` |
| Combined dataset full fingerprint | `2eb6bf16fe7264031b1e9734976d715b8c9b69b57238026719c00abfae536f10` |
| Combined split manifest | `d0ce84d901aab3992c7aee27322cc7859c54fd584d72d394c808f83f2223e755` |
| Fine best checkpoint | `bbc63703481556814b2d8419e5404d69428d5a4b280b26d4a3b5db9b20da4332` |
| Base test JSON | `814d2f1eed1e7aaf42b112d69dde1324759b94bf284ef2f9077cee22b672f4e6` |
| Fine combined/base/gate test JSONs | `64b35bcc7244f252983ed3248d0a205a98e45e986fdc1d8072b753cc63708d50` / `2dc799145ec93ec4f31abedd13eff7853c6bd00b665b7d7d911549972a7467a9` / `d21f4afae0eeb0383f8e214f3199f945e4b11977ee9a4159850f9e8eaf3b9523` |

Both model checkpoints record commit
`f0232f67fd22cd7646a57b906d316bdf8c71ef2e` on
`codex/visual-reproduction` with a dirty worktree, so the commit alone is not a
complete code identity. The base run records `quadpilot_repro/npe.py` /
`scripts/train_repro_npe.py` hashes
`3e0b6feb60563702393d2bbc087f140b66756ddb4ed8c7fd1a537aec1ea5a5f7` /
`3f3ae8990e3040a6b76a3215d5dab012f64a495cb35adb890a98dcb3e88be627`;
the fine run records
`e346f1c1d71f11327777eb27c04daeae16fba26c04d0d54e61f04d8972486860` /
`f138f69f86406de7d07afa963fa958c7164edb7181163f2fb3b126b5245a626d`.

The old `ece484_vision_controller.py` interprets its first three network outputs
as raw coordinates and therefore must not load a new normalized checkpoint
directly. The reproduction closed loop uses the schema-checked loader and
decoder; legacy raw-output export is available only as an explicit affine
conversion.

### Locked Lemniscate NPE datasets and current training boundary

The half-resolution Lemniscate Splatfacto artifact in Milestone 3B produced two
formal NPE datasets. Both use the same explicit renderer run rather than a
historical default path:

```text
/home/chi/UAV/quadpilot-data/gsplat_outputs/lemniscate/train-30k/repro_ns114_gs100_halfres_seed42_v1/training-output/lemniscate/splatfacto/repro_ns114_gs100_halfres_seed42_v1
```

The formal uniform 10,000-view invocation was:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 `
  /home/chi/UAV/envs/quadpilot-render/bin/python -u `
  -m quadpilot.cli.data_generate_uniform `
  --track lemniscate --samples 10000 --seed 42 `
  --run-dir /home/chi/UAV/quadpilot-data/gsplat_outputs/lemniscate/train-30k/repro_ns114_gs100_halfres_seed42_v1/training-output/lemniscate/splatfacto/repro_ns114_gs100_halfres_seed42_v1 `
  --output-dir /home/chi/UAV/quadpilot-data/npe_datasets/lemniscate/uniform_seed42_10000_v1 `
  --cuda-home /home/chi/UAV/cuda-11.8 `
  --maximum-failures 200 --resume
```

The axis-aligned Lemniscate bounds and full-yaw distribution are the generator
defaults. The separate default gate-focused 4,000-view invocation was:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 `
  /home/chi/UAV/envs/quadpilot-render/bin/python -u `
  -m quadpilot.cli.data_generate_gate `
  --track lemniscate --samples 4000 --seed 4242 `
  --run-dir /home/chi/UAV/quadpilot-data/gsplat_outputs/lemniscate/train-30k/repro_ns114_gs100_halfres_seed42_v1/training-output/lemniscate/splatfacto/repro_ns114_gs100_halfres_seed42_v1 `
  --output-dir /home/chi/UAV/quadpilot-data/npe_datasets/lemniscate/gate_seed4242_4000_v1 `
  --cuda-home /home/chi/UAV/cuda-11.8 `
  --min-distance 0.35 --max-distance 2.0 `
  --lateral 0.55 --vertical 0.32 --yaw-jitter-deg 25 `
  --image-margin-px 32 --maximum-sampling-rejections 100 `
  --maximum-failures 200 --resume
```

`--cuda-home` makes the process set `CUDA_HOME` to
`/home/chi/UAV/cuda-11.8`, prepend its `bin` directory to `PATH`, and prepend
its `lib` and `lib64` directories to `LD_LIBRARY_PATH`. The two locked output
directories must not be reused for a new experiment.

Both datasets passed the full-content, full-PNG verifier. The byte counts below
are the sums of the PNG file sizes, not filesystem allocation estimates:

| Dataset | Attempts / records | Render failures | PNG bytes | Full dataset fingerprint |
| --- | ---: | ---: | ---: | --- |
| Uniform seed 42 | 10,000 / 10,000 | 0 | 2,600,681,834 | `43b570c144a8bc508fef7a3084877927dd71ef52028c44e8b2f695c4fa8025fd` |
| Gate seed 4,242 | 4,000 / 4,000 | 0 | 1,220,604,028 | `9434aa042135e50b8b6d20b064fe1171583f4b0e414520a683a34723e5fbff28` |

| Receipt | Uniform SHA-256 | Gate SHA-256 |
| --- | --- | --- |
| `metadata.json` | `c3a0d178b8748db5321f8d3fa52989d2811730fefa92bd9858ca0e1b40258698` | `b2f626102c128b78c4de2db52e4044d92b3c938252da3cc8a58ec58ab9c25986` |
| `samples.jsonl` | `33c94f0335069e17cf64934de364ccd87085eb8a5a403752de70df94065676a1` | `a367dfc2cfd11ed8b0cf5ddbe6ccfabce7512e7a1f3661d14e17817aad82489f` |
| `progress.json` | `a5c84ae97fca37cfd3c482e2ac80624f9954eb8804960660b17ad2fea43cef72` | `11ac1ef83586cc0fc3d6942891a5169d39dafe1784b0b147e0f6a69ab37c1af2` |

The gate verifier independently reconstructed every position and wrapped yaw,
reprojected every selected gate center, and replayed all 4,000 samples from
seed 4,242. Geometry, projection and deterministic sampler replay each passed
4,000/4,000. The accepted distribution and rejection statistics were:

| Gate A | Gate B | Gate C | Gate D | Rejection min / max | Mean / total | Nonzero |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,034 | 987 | 1,018 | 961 | 0 / 22 | 1.01175 / 4,047 | 1,799 |

All four gate counts are inside the prespecified 850--1,150 acceptance interval
for a 4,000-frame dataset. Reproduce the two read-only acceptance gates without
starting a renderer or exposing a GPU with:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= `
  /home/chi/UAV/envs/quadpilot-render/bin/python `
  -m quadpilot.cli.verify_dataset `
  /home/chi/UAV/quadpilot-data/npe_datasets/lemniscate/uniform_seed42_10000_v1 `
  --track lemniscate --seed 42 --expected-frames 10000 `
  --expected-checkpoint-step 29999 --expected-gaussians 394366 `
  --expected-checkpoint-sha256 a8a1064a1d95a9bdc642c1ad540c8dcd2b00b28680c23978f6c47e258b611a32 `
  --expected-transform-sha256 d5be6872b9a89c07547bff962ce32f5513b66fec175eef4ac7585cacbeb46333

wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= `
  /home/chi/UAV/envs/quadpilot-render/bin/python `
  -m quadpilot.cli.verify_dataset `
  /home/chi/UAV/quadpilot-data/npe_datasets/lemniscate/gate_seed4242_4000_v1 `
  --track lemniscate --seed 4242 --expected-frames 4000 `
  --expected-checkpoint-step 29999 --expected-gaussians 394366 `
  --expected-checkpoint-sha256 a8a1064a1d95a9bdc642c1ad540c8dcd2b00b28680c23978f6c47e258b611a32 `
  --expected-transform-sha256 d5be6872b9a89c07547bff962ce32f5513b66fec175eef4ac7585cacbeb46333 `
  --expect-gate-focused
```

Current generators hash the selected checkpoint, dataparser transform and all
generation-critical source files before rendering. Those values are embedded
in provenance and therefore in the immutable `progress.json` generation
contract; a clean-boundary `--resume` rejects asset, code, seed, bounds,
intrinsics, sampler, target or failure-budget drift. The completed gate receipt
contains both the automatic asset hashes and the six-file code-hash map. The
uniform run completed during the transition after asset hashing was added but
before `generation_code_sha256` was added, so its receipt locks the checkpoint
and transform hashes but must not be described as carrying the later automatic
code-hash map. Its file receipts and full-content fingerprint remain locked.

The six source hashes embedded in the completed gate contract are shown below.
The paths are historical provenance identifiers from the pre-`src/` layout;
their exact files are preserved by Git commit `e77ab34` and the code-lock JSONs.

| Generation-critical file | SHA-256 |
| --- | --- |
| `scripts/generate_repro_gate_dataset.py` | `df5c42ae447aba608e4d4a79ca363e180cd31b939e9a317338459b487d6e68ee` |
| `quadpilot_repro/data_generation.py` | `042dc891a18b4d9038c69486ca1aeb734f02bda489b9455f23dbb871d46f9869` |
| `quadpilot_repro/environment.py` | `55c9298b7c1bf654cc50bc1eb1a85ca6466cb51a7a9b88cf59a3d68d9e7d68e4` |
| `quadpilot_repro/gate_sampling.py` | `b89b84da39c809507d800049d17b298c49a5c1311b8131f42c4d3620ae0dff70` |
| `quadpilot_repro/renderer.py` | `bfbd6c3a02e42cd3ddf4e030b1684744a9d04e903ee0bec60e019cd30ceaba58` |
| `quadpilot_repro/tracks.py` | `0428387c07ef283964a2764a78df93dfa2f37fa04e1662e9fc9b157a34d0bec5` |

Resume is fail-closed but is not a multi-file transaction. A process or power
loss between PNG rename, JSONL append/fsync and progress replacement can leave
an orphan image or a JSONL/progress count mismatch. Such a state is rejected
rather than guessed or repaired; do not hand-splice it. Restart into a new,
unique output directory. This is the remaining P1 availability and wasted-work
risk even though it does not silently relabel a sample.

The Lemniscate base, gate-fine-tuned, and launch-corridor-fine-tuned NPE runs
are complete. Their validation-selected checkpoints, once-only frozen TEST
results, launch-corridor preflight, and raw/EKF closed-loop evidence are locked
in `configs/assets/manifest.json`. No TEST result was used to select or tune a
checkpoint.

## Milestone 5: real rendered Circle NPE closed loop

This milestone uses a newly rendered GSplat image and a real NPE inference at
every observation step. It is a visual simulation result, not a synthetic-noise
injection and not a physical Crazyflie flight. Raw and EKF each start with a
fresh controller, plant and estimator state while sharing the locked assets and
seed. The formal float32 invocation is:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  /home/chi/UAV/envs/quadpilot-render/bin/python -u `
  -m quadpilot.cli.simulate_closed_loop `
  --track circle --estimator both `
  --npe-checkpoint /home/chi/UAV/quadpilot-data/npe_models/circle/gate_finetune_seed42_init_base_v1/best_npe.pth `
  --renderer-checkpoint /mnt/f/UAV/ece484_final/outputs/circle/splatfacto/2025-05-09_144210/nerfstudio_models/step-000029999.ckpt `
  --dataparser-transform /mnt/f/UAV/ece484_final/outputs/circle/splatfacto/2025-05-09_144210/dataparser_transforms.json `
  --device cuda:0 --cuda-home /home/chi/UAV/cuda-11.8 `
  --seed 42 --max-steps 1200 --dt 0.05 --laps 2 `
  --gate-radius 0.38 --crossing-hysteresis-m 0.05 `
  --ekf-outlier-threshold 4 --expected-renderer-step 29999 `
  --expected-gaussians 308832 --snapshot-every 0 `
  --output-dir /home/chi/UAV/quadpilot-data/closed_loop/circle_gate_finetune_seed42_both_s1200_l2_v1
```

Do not rerun into this locked directory; use a new output directory for a new
experiment. Both controllers completed two strict laps in gate order and inside
the 0.38 m radius:

| Controller input | Steps | Strict crossings | Mean gate error | Mission time |
| --- | ---: | ---: | ---: | ---: |
| Raw NPE | 368 | 8/8 | 14.797 cm | 18.322 s |
| EKF estimate | 355 | 8/8 | 8.246 cm | 17.694 s |

Because the two controllers follow different paths, their rendered NPE sample
sets are not identical. Filtering effects must therefore be compared within the
EKF run, not by subtracting the two runs' raw-observation means:

| Run/signal | Samples | Position mean | Position std | Position maximum | Yaw mean | Position step jitter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw run / raw NPE and controller input | 369 | 8.229 cm | 8.374 cm | 68.051 cm | 0.729 deg | 3.970 cm |
| EKF run / raw NPE observations | 356 | 8.151 cm | 8.340 cm | 68.051 cm | 0.660 deg | 3.348 cm |
| EKF run / controller estimate | 356 | 7.591 cm | 8.362 cm | 67.646 cm | 0.478 deg | 1.445 cm |

All 356 EKF updates were accepted and none rejected. On that same trajectory,
the EKF reduced position step jitter from 3.348 cm to 1.445 cm. Truth step
jitter was 1.098 cm. These values use the explicit metric definitions in
Milestone 2; they are not interchangeable with the teaser's 500-frame
Lemniscate statistics.

The locked artifact hashes are:

| Artifact | SHA-256 |
| --- | --- |
| `circle_raw.json` | `7619cc1e449e371fe2d692211b3099befc12ba0c7224727bc284f3fb7fc37ba0` |
| `circle_raw.npz` | `7effac4ba64f954c081282a8a52a9b741c39062510a96ecea254641647931442` |
| `circle_ekf.json` | `33c20616e4bf58b3079f5b9dc5892fbbd1b79040ea0627e31bace868ea4c9c0f` |
| `circle_ekf.npz` | `a8ceb6f485d9492ac3cfd9a713829b2fe455d900e1f62541fe900d29929b7599` |

Verify the four files without using a GPU or rerunning the experiment:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  env CUDA_VISIBLE_DEVICES= `
  /home/chi/miniconda3/envs/mvot-mmaction/bin/python `
  -m quadpilot.cli.verify_closed_loop `
  /home/chi/UAV/quadpilot-data/closed_loop/circle_gate_finetune_seed42_both_s1200_l2_v1 `
  --npe-checkpoint /home/chi/UAV/quadpilot-data/npe_models/circle/gate_finetune_seed42_init_base_v1/best_npe.pth `
  --expected-npe-sha256 bbc63703481556814b2d8419e5404d69428d5a4b280b26d4a3b5db9b20da4332 `
  --manifest /mnt/f/UAV/ece484_final/configs/assets/manifest.json `
  --asset-root /mnt/f/UAV/ece484_final/outputs `
  --expected-seed 42 --expected-device cuda:0 `
  --expected-max-steps 1200 --expected-dt 0.05 --expected-laps 2 `
  --expected-gate-radius 0.38 --expected-hysteresis 0.05 `
  --expected-renderer-step 29999 --expected-gaussians 308832
```

The verifier returned `status: PASS`. It checks the NPE, renderer and transform
hashes; exact output membership; JSON/NPZ equality; finite shapes and sample
alignment; saved metrics against recomputation; ordered strict crossings; and
the complete raw/EKF metadata contract.

## Milestone 5B: teaser-compatible Lemniscate comparison

The teaser at <https://www.youtube.com/watch?v=8l80orgLiXs> displays 500 total
frames, but the historical implementation computes its summary statistics only
over `range(first_gate_pass, last_gate_pass)`. The reproducible comparison uses
that same clipping rule, the same jitter definition, and the same historical
DYN construction on the locked Lemniscate EKF trajectory. The DYN noise is now
explicitly seeded with 42; the historical script did not persist its RNG seed.

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  env CUDA_VISIBLE_DEVICES= `
  /home/chi/miniconda3/envs/mvot-mmaction/bin/python `
  -m quadpilot.cli.compare_teaser `
  /home/chi/UAV/quadpilot-data/closed_loop/lemniscate_launch_finetune_seed42_both_s1200_l2_v1 `
  --track lemniscate --dyn-seed 42 `
  --output /mnt/f/UAV/ece484_final/results/baselines/teaser_lemniscate_comparison.json
```

The comparable interval is 417 observations, from controller pass step 42
inclusive to step 459 exclusive. This is the correct metric window even though
the saved run naturally completed two laps after 460 observations instead of
padding its display to 500 frames.

| Source/metric | Teaser | Reproduction | Delta |
| --- | ---: | ---: | ---: |
| NPE mean | 7.9 cm | 6.191 cm | -21.63% |
| NPE std | 4.0 cm | 3.842 cm | -3.94% |
| NPE max | 25.3 cm | 34.587 cm | +36.71% |
| NPE jitter | 3.87 cm | 3.755 cm | -2.97% |
| EKF mean | 7.2 cm | 4.830 cm | -32.91% |
| EKF std | 3.7 cm | 2.361 cm | -36.20% |
| EKF max | 19.2 cm | 9.765 cm | -49.14% |
| EKF jitter | 1.37 cm | 1.229 cm | -10.30% |
| DYN mean | 8.3 cm | 8.311 cm | +0.13% |
| DYN std | 2.5 cm | 2.630 cm | +5.19% |
| DYN max | 14.2 cm | 14.348 cm | +1.04% |
| DYN jitter | 3.34 cm | 3.477 cm | +4.10% |
| GT jitter | 0.95 cm | 0.631 cm | -33.62% |

The teaser reports EKF error 9% lower and jitter 65% smoother than NPE. The
reproduction gives 21.98% lower mean error and 67.27% smoother jitter. NPE has
one larger tail error, but its mean/std/jitter are close or better, EKF
suppresses the tail to 9.765 cm, and the locked controller still completes all
eight strict crossings. The engineering assessment is therefore
`SIMULATION_REPRODUCTION_SUCCESS`, not merely an approximate visual match.

The locked comparison report is
`results/baselines/teaser_lemniscate_comparison.json`, SHA-256
`de6ed9ad88638de5419e63e1eb511fd0b368e6e9ed37cfe261b8df9d70a3faa5`.

## Milestone 6: U-turn NPE and real rendered closed loop

U-turn uses the same canonical directory contract as the other tracks. Formal
artifacts live only under these four roots:

```text
/home/chi/UAV/quadpilot-data/npe_datasets/uturn/
/home/chi/UAV/quadpilot-data/npe_models/uturn/
/home/chi/UAV/quadpilot-data/gsplat_outputs/uturn/
/home/chi/UAV/quadpilot-data/closed_loop/
```

The two renderer-derived training datasets passed full, per-image verification:

| Dataset | Frames | Failures | Full fingerprint | Additional gate |
| --- | ---: | ---: | --- | --- |
| `uniform_seed42_10000_v1` | 10,000 | 0 | `0e147b2f05a6876224bd739c26a34cbb0697f676de3ce65c5282c14a433d93cb` | 10,000 PNG/SHA/pose records verified |
| `gate_seed4242_4000_v1` | 4,000 | 0 | `76f743453046cb13227bb5a83f0af6225907aa14f2a562998ea3ab17e81d742a` | A/B/C/D = 999/997/1041/963; geometry, projection, and seed replay PASS |

The ResNet-50 base run completed 100 epochs. Validation-only selection chose
display epoch 99 with 9.639 cm position error. Its once-only 1,000-sample frozen
TEST was 10.651 cm mean, 24.054 cm p95, and 1.389 degrees yaw mean. The
gate-focused fine-tune completed 30 epochs and selected display epoch 29 with
8.485 cm combined validation error. Its checkpoint SHA-256 is
`8fc5065cb8db0b904cd1b802e3c0cea2ff07777dff079078f30dcd84a07dfef7`.

The three once-only fine-tuned TEST slices were independently recomputed from
their prediction JSONL files:

| Frozen TEST slice | Samples | Position mean | p95 | Maximum | Yaw mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combined | 1,400 | 9.101 cm | 21.449 cm | 131.439 cm | 1.209 deg |
| Base source | 1,000 | 10.345 cm | 23.827 cm | 131.439 cm | 1.366 deg |
| Gate source | 400 | 5.991 cm | 11.381 cm | 18.001 cm | 0.818 deg |

The combined TEST is slightly above the historical 8.9 cm target and is kept
as observed. It was not used for another tuning round. The base slice improved
over the base model's 10.651 cm TEST, while the gate slice confirms that the
critical crossing region is substantially more accurate.

The locked visual closed loop was generated with:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  /home/chi/UAV/envs/quadpilot-render/bin/python -u `
  -m quadpilot.cli.simulate_closed_loop `
  --track uturn --estimator both `
  --npe-checkpoint /home/chi/UAV/quadpilot-data/npe_models/uturn/gate_finetune_seed42_init_base_v1/best_npe.pth `
  --renderer-checkpoint /home/chi/UAV/quadpilot-data/gsplat_outputs/uturn/train-30k/repro_ns114_uturn_gs100_halfres_seed42_v1/training-output/uturn/splatfacto/repro_ns114_uturn_gs100_halfres_seed42_v1/nerfstudio_models/step-000029999.ckpt `
  --dataparser-transform /home/chi/UAV/quadpilot-data/gsplat_outputs/uturn/train-30k/repro_ns114_uturn_gs100_halfres_seed42_v1/training-output/uturn/splatfacto/repro_ns114_uturn_gs100_halfres_seed42_v1/dataparser_transforms.json `
  --device cuda:0 --cuda-home /home/chi/UAV/cuda-11.8 `
  --seed 42 --max-steps 1200 --dt 0.05 --laps 2 `
  --gate-radius 0.38 --crossing-hysteresis-m 0.05 `
  --ekf-outlier-threshold 4 --expected-renderer-step 29999 `
  --expected-gaussians 437285 --snapshot-every 0 `
  --output-dir /home/chi/UAV/quadpilot-data/closed_loop/uturn_gate_finetune_seed42_both_s1200_l2_v1
```

| Controller input | Steps | Strict crossings | Mean gate error |
| --- | ---: | ---: | ---: |
| Raw NPE | 362 | 8/8 | 10.147 cm |
| EKF estimate | 347 | 8/8 | 5.157 cm |

The CPU verifier returned `status: PASS`, including A-B-C-D x2 strict order,
JSON/NPZ equality, sample alignment, finite values, and exact NPE/renderer/
transform hashes. The four locked artifact hashes are recorded in
`configs/assets/manifest.json`.

## Current simulation boundary

Circle, Lemniscate, and U-turn now each have renderer-derived NPE data,
validation-selected neural checkpoints, frozen TEST evidence, and successful
raw/EKF two-lap visual simulation. This completes the three-track simulation
reproduction scope. It does not yet validate Vicon-to-NeRF calibration, camera
extrinsics, ROS freshness timeouts, radio timing, emergency stop, onboard
compute, or propeller-on Crazyflie safety. Hardware integration remains a
separate safety-gated milestone and must not be inferred from simulation PASS.

## Hardware handoff gate

All hardware-independent safety code is now present. Estimate the required
`vicon_world <- nerf_world` mapping from manually collected corresponding
points with:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  /home/chi/miniconda3/envs/mvot-mmaction/bin/python `
  -m quadpilot.cli.hardware_calibrate `
  /home/chi/UAV/quadpilot-data/hardware-evidence/vicon_nerf_correspondences.json `
  --output /home/chi/UAV/quadpilot-data/hardware-evidence/vicon_from_nerf.json `
  --maximum-rmse-m 0.03
```

The estimator uses an Umeyama similarity transform, rejects non-finite,
collinear, or underspecified points, records the input SHA-256, and exits 2 if
the fitted RMSE exceeds the declared limit. Camera intrinsics and the rigid
`body_from_camera` transform are separate evidence files; they must not be
inferred from the GSplat dataparser transform.

Before any ROS or radio command, copy
`configs/hardware/preflight.template.json` to a run-specific local config,
fill the three evidence paths and SHA-256 values, and complete the manual
propellers-removed checks. Then run:

```powershell
wsl.exe --cd /mnt/f/UAV/ece484_final -d Ubuntu-22.04 -- `
  env CUDA_VISIBLE_DEVICES= `
  /home/chi/miniconda3/envs/mvot-mmaction/bin/python `
  -m quadpilot.cli.hardware_preflight `
  /home/chi/UAV/quadpilot-data/hardware-evidence/hardware-preflight.json
```

The committed template currently returns `BLOCKED` and exit 2, as required.
Its unresolved gates are the Vicon/NeRF calibration, camera intrinsics,
camera/body extrinsics, and nine human-confirmed prop-off safety checks. The
checker validates exact evidence hashes, ROS topic uniqueness, frame contracts,
freshness timeouts, geofence size, acceleration/yaw-rate limits, and never
executes a hardware command. No propeller-on stage is enabled in this code.
