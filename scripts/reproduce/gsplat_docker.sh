#!/usr/bin/env bash
# Reproduce a pinned Quad Pilots Splatfacto scene in a non-networked container.
#
# This wrapper deliberately never pulls an image.  Obtain the exact image out of
# band, then run `preflight`; every execution is tied to the immutable digest
# below and recorded under the selected output root.

set -Eeuo pipefail
IFS=$'\n\t'

readonly IMAGE_REF='dromni/nerfstudio@sha256:ff0107a7db96bb8ee29c638729328b832b268b890c50f2a2ff25988bb84d4f75'
readonly IMAGE_EDITABLE_SOURCE='/home/user/nerfstudio'
readonly IMAGE_USER_SITE='/home/user/.local/lib/python3.10/site-packages'
readonly IMAGE_PYTHONPATH="$IMAGE_EDITABLE_SOURCE:$IMAGE_USER_SITE"
readonly IMAGE_PATH='/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/user/.local/bin'
readonly SOURCE_RECEIPT_NAME='.quadpilot_source_receipt.json'
readonly LEMNISCATE_RECEIPT_SHA256='6614c5be765ab7456eac95403af4b2c6fb34e757afc263ba3aa7b9f075cd356a'
readonly LEMNISCATE_RECEIPT_FILES='1555'
readonly LEMNISCATE_RECEIPT_IMAGES='1553'
readonly LEMNISCATE_RECEIPT_IMAGE_BYTES='3362065056'
readonly LEMNISCATE_RECEIPT_TOTAL_BYTES='3370611629'
readonly LEMNISCATE_SPARSE_POINTS='183994'
readonly UTURN_RECEIPT_SHA256='a42c422dc084375e7f2bf5ef530ac7a5409e9abc0d6c5b3fa90ccd840beb6023'
readonly UTURN_RECEIPT_FILES='1442'
readonly UTURN_RECEIPT_IMAGES='1440'
readonly UTURN_RECEIPT_IMAGE_BYTES='3062618402'
readonly UTURN_RECEIPT_TOTAL_BYTES='3070717413'
readonly UTURN_SPARSE_POINTS='175292'
readonly LPIPS_ALEXNET_RELATIVE_PATH='torch/hub/checkpoints/alexnet-owt-7be5be79.pth'
readonly LPIPS_ALEXNET_SIZE_BYTES='244408911'
readonly LPIPS_ALEXNET_SHA256='7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'
readonly HALF_RES_SCALE='0.5'
readonly DEFAULT_MAX_GPU_UTIL='5'
readonly DEFAULT_MAX_GPU_MEMORY_MIB='1024'
readonly DEFAULT_SHM_SIZE='8g'
readonly TRAIN_MAX_JOBS='4'

usage() {
  cat <<'EOF'
Usage:
  reproduce_gsplat_docker.sh MODE --data DIR --output-root DIR --cache DIR [OPTIONS]

Modes (all training modes load/cache at scale 0.5 in width and height):
  preflight    Run CPU-only version, known pip-deviation, and dataparser gates in
               the locally cached pinned image. It never requests a GPU.
  smoke-1      Run Splatfacto with --max-num-iterations 1.
  smoke-101    Run Splatfacto with --max-num-iterations 101.
  train-30k    Run Splatfacto with --max-num-iterations 30000.

Required paths:
  --data DIR          Nerfstudio dataset containing transforms.json and images/.
                      It is mounted at /data READ-ONLY.
  --output-root DIR   Audit records and training outputs (read-write).
  --cache DIR         Dedicated model/runtime cache (read-write).
  --track NAME        Required source profile: lemniscate or uturn. The data
                      directory basename and receipt track must match exactly.

Options:
  --gpu INDEX                  GPU index (default: 0).
  --run-id NAME                Unique run label (default: UTC timestamp plus PID).
  --max-gpu-util PERCENT       Busy threshold (default: 5).
  --max-gpu-memory-mib MIB     Busy threshold (default: 1024).
  --allow-busy-gpu             Explicitly override the fail-closed busy-GPU gate.
  --dry-run                    Record commands only; start no container or GPU job.
  -h, --help                   Show this help.

Important resolution note:
  "half-res" means the loaded/cached camera and images are 0.5 on each linear
  dimension (one quarter of the source pixels). Smoke disables Splatfacto's
  additional progressive downscaling. The 30k run preserves its default
  num_downscales=2, resolution_schedule=3000: relative to the source, raster
  resolution is 1/8 for steps 0-2999, 1/4 for 3000-5999, then 1/2. This is NOT
  the original full-resolution experiment; the source dataset is never rewritten.

Safety properties:
  * image is fixed by digest and Docker is invoked with --pull=never;
  * container networking is disabled;
  * data, output and cache paths must be pairwise non-overlapping;
  * every receipt entry in the explicitly selected source profile is
    size/SHA-256 verified before training;
  * external method plugins are disabled only after their pinned metadata and
    Nerfstudio registry sources pass an exact built-in-only policy audit;
  * the LPIPS AlexNet weight is size/SHA-256 verified before a GPU run and is
    over-mounted read-only; network download fallback remains impossible;
  * the container can write only training-output/, not pre-run audit evidence;
  * periodic evaluation is disabled to match the original viewer-only semantics;
  * compute processes, utilization, or memory above the configured limits make
    the run fail unless --allow-busy-gpu is explicitly recorded;
  * nvidia-smi, image inspection, provenance and the exact command are retained.
  * success requires the exact final-step checkpoint and recorded artifact hashes.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

note() {
  printf '%s\n' "$*" >&2
}

require_value() {
  local option=$1
  local value=${2-}
  [[ -n "$value" ]] || die "$option requires a value"
}

is_uint() {
  [[ $1 =~ ^[0-9]+$ ]]
}

is_number() {
  [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]]
}

trim() {
  local value=$1
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

paths_overlap() {
  local first=$1
  local second=$2
  [[ "$first" == "$second" || "$first" == "$second"/* || "$second" == "$first"/* ]]
}

safe_label() {
  [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

record_value() {
  local key=$1
  local value=$2
  printf '%s=%q\n' "$key" "$value" >>"$PROVENANCE_FILE"
}

write_command_file() {
  local destination=$1
  shift
  {
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\nexec '
    printf '%q ' "$@"
    printf '\n'
  } >"$destination"
  chmod 700 "$destination"
}

MODE=${1-}
if [[ "$MODE" == '-h' || "$MODE" == '--help' || -z "$MODE" ]]; then
  usage
  [[ -n "$MODE" ]] && exit 0
  exit 2
fi
shift

case "$MODE" in
  preflight)
    MAX_ITERATIONS=''
    ;;
  smoke-1)
    MAX_ITERATIONS='1'
    ;;
  smoke-101)
    MAX_ITERATIONS='101'
    ;;
  train-30k)
    MAX_ITERATIONS='30000'
    ;;
  *)
    usage >&2
    die "unknown mode: $MODE"
    ;;
esac

DATA_ARG=''
OUTPUT_ARG=''
CACHE_ARG=''
GPU_INDEX='0'
TRACK=''
RUN_ID=''
MAX_GPU_UTIL=$DEFAULT_MAX_GPU_UTIL
MAX_GPU_MEMORY_MIB=$DEFAULT_MAX_GPU_MEMORY_MIB
ALLOW_BUSY_GPU=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --data)
      require_value "$1" "${2-}"
      DATA_ARG=$2
      shift 2
      ;;
    --output-root)
      require_value "$1" "${2-}"
      OUTPUT_ARG=$2
      shift 2
      ;;
    --cache)
      require_value "$1" "${2-}"
      CACHE_ARG=$2
      shift 2
      ;;
    --gpu)
      require_value "$1" "${2-}"
      GPU_INDEX=$2
      shift 2
      ;;
    --track)
      require_value "$1" "${2-}"
      TRACK=$2
      shift 2
      ;;
    --run-id)
      require_value "$1" "${2-}"
      RUN_ID=$2
      shift 2
      ;;
    --max-gpu-util)
      require_value "$1" "${2-}"
      MAX_GPU_UTIL=$2
      shift 2
      ;;
    --max-gpu-memory-mib)
      require_value "$1" "${2-}"
      MAX_GPU_MEMORY_MIB=$2
      shift 2
      ;;
    --allow-busy-gpu)
      ALLOW_BUSY_GPU=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "$DATA_ARG" ]] || die '--data is required'
[[ -n "$OUTPUT_ARG" ]] || die '--output-root is required'
[[ -n "$CACHE_ARG" ]] || die '--cache is required'
is_uint "$GPU_INDEX" || die '--gpu must be a non-negative integer'
is_number "$MAX_GPU_UTIL" || die '--max-gpu-util must be a non-negative number'
is_number "$MAX_GPU_MEMORY_MIB" || die '--max-gpu-memory-mib must be a non-negative number'
[[ -n "$TRACK" ]] || die '--track is required; choose exactly lemniscate or uturn'
case "$TRACK" in
  lemniscate)
    SOURCE_PROFILE='lemniscate'
    SOURCE_RECEIPT_SHA256=$LEMNISCATE_RECEIPT_SHA256
    SOURCE_RECEIPT_FILES=$LEMNISCATE_RECEIPT_FILES
    SOURCE_RECEIPT_IMAGES=$LEMNISCATE_RECEIPT_IMAGES
    SOURCE_RECEIPT_IMAGE_BYTES=$LEMNISCATE_RECEIPT_IMAGE_BYTES
    SOURCE_RECEIPT_TOTAL_BYTES=$LEMNISCATE_RECEIPT_TOTAL_BYTES
    SOURCE_SPARSE_POINTS=$LEMNISCATE_SPARSE_POINTS
    ;;
  uturn)
    SOURCE_PROFILE='uturn'
    SOURCE_RECEIPT_SHA256=$UTURN_RECEIPT_SHA256
    SOURCE_RECEIPT_FILES=$UTURN_RECEIPT_FILES
    SOURCE_RECEIPT_IMAGES=$UTURN_RECEIPT_IMAGES
    SOURCE_RECEIPT_IMAGE_BYTES=$UTURN_RECEIPT_IMAGE_BYTES
    SOURCE_RECEIPT_TOTAL_BYTES=$UTURN_RECEIPT_TOTAL_BYTES
    SOURCE_SPARSE_POINTS=$UTURN_SPARSE_POINTS
    ;;
  *)
    die '--track must be exactly lemniscate or uturn'
    ;;
esac
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
safe_label "$RUN_ID" || die '--run-id must contain only letters, digits, dot, underscore, or dash'

[[ -d "$DATA_ARG" ]] || die "data directory does not exist: $DATA_ARG"
[[ -r "$DATA_ARG/transforms.json" ]] || die 'data directory must contain a readable transforms.json'
[[ -r "$DATA_ARG/sparse_pc.ply" ]] || die 'data directory must contain a readable sparse_pc.ply'
[[ -r "$DATA_ARG/$SOURCE_RECEIPT_NAME" ]] || \
  die "data directory must contain a readable $SOURCE_RECEIPT_NAME"
[[ -d "$DATA_ARG/images" && -r "$DATA_ARG/images" && -x "$DATA_ARG/images" ]] || \
  die 'data directory must contain a readable/searchable images/ directory'

REALPATH_BIN=$(command -v realpath || true)
[[ -n "$REALPATH_BIN" ]] || die 'GNU realpath is required for safe mount-path validation'
DATA_DIR=$("$REALPATH_BIN" -e -- "$DATA_ARG")
OUTPUT_ROOT=$("$REALPATH_BIN" -m -- "$OUTPUT_ARG")
CACHE_ROOT=$("$REALPATH_BIN" -m -- "$CACHE_ARG")
DATA_BASENAME=$(basename -- "$DATA_DIR")
[[ "$DATA_BASENAME" == "$SOURCE_PROFILE" ]] || \
  die "data directory basename '$DATA_BASENAME' does not match --track profile '$SOURCE_PROFILE'"

for path in "$DATA_DIR" "$OUTPUT_ROOT" "$CACHE_ROOT"; do
  [[ "$path" != '/' ]] || die 'refusing to expose the filesystem root'
  [[ "$path" != *','* && "$path" != *$'\n'* ]] || die 'mount paths may not contain commas or newlines'
done
paths_overlap "$DATA_DIR" "$OUTPUT_ROOT" && die 'data and output paths must not overlap'
paths_overlap "$DATA_DIR" "$CACHE_ROOT" && die 'data and cache paths must not overlap'
paths_overlap "$OUTPUT_ROOT" "$CACHE_ROOT" && die 'output and cache paths must not overlap'

# Creating writable directories only after canonical overlap checks prevents a
# rejected invocation from writing even an empty directory inside the dataset.
mkdir -p -- "$OUTPUT_ROOT" "$CACHE_ROOT"
OUTPUT_ROOT=$("$REALPATH_BIN" -e -- "$OUTPUT_ROOT")
CACHE_ROOT=$("$REALPATH_BIN" -e -- "$CACHE_ROOT")
paths_overlap "$DATA_DIR" "$OUTPUT_ROOT" && die 'data and output paths must not overlap after creation'
paths_overlap "$DATA_DIR" "$CACHE_ROOT" && die 'data and cache paths must not overlap after creation'
paths_overlap "$OUTPUT_ROOT" "$CACHE_ROOT" && die 'output and cache paths must not overlap after creation'
[[ -w "$OUTPUT_ROOT" ]] || die "output root is not writable: $OUTPUT_ROOT"
[[ -w "$CACHE_ROOT" ]] || die "cache root is not writable: $CACHE_ROOT"

RUN_PARENT="$OUTPUT_ROOT/$TRACK/$MODE"
RUN_DIR="$RUN_PARENT/$RUN_ID"
mkdir -p -- "$RUN_PARENT"
if ! mkdir -- "$RUN_DIR"; then
  die "run directory already exists; choose a new --run-id: $RUN_DIR"
fi

CACHE_DIR="$CACHE_ROOT/nerfstudio-ff0107a7db96"
mkdir -p -- \
  "$CACHE_DIR/home" \
  "$CACHE_DIR/xdg" \
  "$CACHE_DIR/torch" \
  "$CACHE_DIR/huggingface"
LPIPS_ALEXNET_PATH="$CACHE_DIR/$LPIPS_ALEXNET_RELATIVE_PATH"

PROVENANCE_FILE="$RUN_DIR/provenance.env"
STATUS_FILE="$RUN_DIR/status.env"
STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

finalize() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "${RUN_DIR-}" && -d "${RUN_DIR-}" ]]; then
    if [[ -n "${NVIDIA_SMI_BIN-}" && -x "${NVIDIA_SMI_BIN-}" ]]; then
      "$NVIDIA_SMI_BIN" -q -i "${GPU_INDEX-0}" >"$RUN_DIR/nvidia-smi.after.txt" 2>&1 || true
    fi
    {
      printf 'finished_utc=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'exit_code=%q\n' "$exit_code"
      if ((exit_code == 0)); then
        printf 'result=%q\n' 'success'
      else
        printf 'result=%q\n' 'failed'
      fi
    } >"$STATUS_FILE"
  fi
  exit "$exit_code"
}
trap finalize EXIT

{
  printf '# shell-escaped key/value audit record\n'
} >"$PROVENANCE_FILE"
record_value schema_version '1'
record_value started_utc "$STARTED_UTC"
record_value mode "$MODE"
record_value image_ref "$IMAGE_REF"
record_value track "$TRACK"
record_value run_id "$RUN_ID"
record_value data_dir "$DATA_DIR"
record_value data_mount_mode 'readonly'
record_value source_profile "$SOURCE_PROFILE"
record_value source_receipt_name "$SOURCE_RECEIPT_NAME"
record_value expected_source_receipt_sha256 "$SOURCE_RECEIPT_SHA256"
record_value expected_source_receipt_files "$SOURCE_RECEIPT_FILES"
record_value expected_source_receipt_images "$SOURCE_RECEIPT_IMAGES"
record_value expected_source_receipt_image_bytes "$SOURCE_RECEIPT_IMAGE_BYTES"
record_value expected_source_receipt_total_bytes "$SOURCE_RECEIPT_TOTAL_BYTES"
record_value expected_source_sparse_points "$SOURCE_SPARSE_POINTS"
record_value output_root "$OUTPUT_ROOT"
record_value run_dir "$RUN_DIR"
record_value cache_root "$CACHE_ROOT"
record_value cache_dir "$CACHE_DIR"
record_value gpu_index "$GPU_INDEX"
record_value max_gpu_util_percent "$MAX_GPU_UTIL"
record_value max_gpu_memory_mib "$MAX_GPU_MEMORY_MIB"
record_value allow_busy_gpu "$ALLOW_BUSY_GPU"
record_value dry_run "$DRY_RUN"
record_value half_res_linear_scale "$HALF_RES_SCALE"
record_value half_res_pixel_fraction '0.25'
record_value full_resolution_reproduction 'false'
record_value hostname "$(hostname)"
record_value uid "$(id -u)"
record_value gid "$(id -g)"
record_value lpips_alexnet_cache_path "$LPIPS_ALEXNET_PATH"
record_value lpips_alexnet_container_path "/cache/$LPIPS_ALEXNET_RELATIVE_PATH"
record_value lpips_alexnet_expected_size_bytes "$LPIPS_ALEXNET_SIZE_BYTES"
record_value lpips_alexnet_expected_sha256 "$LPIPS_ALEXNET_SHA256"
if [[ "$MODE" == 'preflight' ]]; then
  record_value lpips_alexnet_cache_required 'false'
  record_value lpips_alexnet_cache_verified 'not-applicable'
elif ((DRY_RUN)); then
  # A dry run starts no container. Keep command construction inspectable even
  # when the operator has not populated the dedicated cache yet.
  record_value lpips_alexnet_cache_required 'true'
  record_value lpips_alexnet_cache_verified 'false-dry-run'
else
  record_value lpips_alexnet_cache_required 'true'
  [[ -f "$LPIPS_ALEXNET_PATH" && -r "$LPIPS_ALEXNET_PATH" ]] || \
    die "missing readable LPIPS AlexNet cache file: $LPIPS_ALEXNET_PATH"
  LPIPS_ALEXNET_REAL_PATH=$(
    "$REALPATH_BIN" -e -- "$LPIPS_ALEXNET_PATH"
  ) || die 'cannot resolve LPIPS AlexNet cache file'
  [[ "$LPIPS_ALEXNET_REAL_PATH" == "$LPIPS_ALEXNET_PATH" ]] || \
    die 'LPIPS AlexNet cache path must not contain symlinks'
  if ! LPIPS_ALEXNET_ACTUAL_SIZE=$(stat -c '%s' -- "$LPIPS_ALEXNET_PATH"); then
    die 'cannot read LPIPS AlexNet cache size'
  fi
  [[ "$LPIPS_ALEXNET_ACTUAL_SIZE" == "$LPIPS_ALEXNET_SIZE_BYTES" ]] || \
    die "LPIPS AlexNet size mismatch: $LPIPS_ALEXNET_ACTUAL_SIZE != $LPIPS_ALEXNET_SIZE_BYTES"
  if ! LPIPS_ALEXNET_ACTUAL_SHA256=$(
    sha256sum -- "$LPIPS_ALEXNET_PATH" | awk '{print $1}'
  ); then
    die 'cannot hash LPIPS AlexNet cache file'
  fi
  [[ "$LPIPS_ALEXNET_ACTUAL_SHA256" == "$LPIPS_ALEXNET_SHA256" ]] || \
    die "LPIPS AlexNet SHA-256 mismatch: $LPIPS_ALEXNET_ACTUAL_SHA256"
  record_value lpips_alexnet_actual_size_bytes "$LPIPS_ALEXNET_ACTUAL_SIZE"
  record_value lpips_alexnet_actual_sha256 "$LPIPS_ALEXNET_ACTUAL_SHA256"
  record_value lpips_alexnet_cache_verified 'true'
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  record_value git_commit "$(git -C "$REPO_ROOT" rev-parse HEAD)"
  if git -C "$REPO_ROOT" diff --quiet --ignore-submodules -- && \
     git -C "$REPO_ROOT" diff --cached --quiet --ignore-submodules --; then
    record_value git_tracked_worktree_dirty 'false'
  else
    record_value git_tracked_worktree_dirty 'true'
  fi
fi

DOCKER_BIN=$(command -v docker || true)
[[ -n "$DOCKER_BIN" ]] || die 'docker is not installed or not on PATH'
record_value docker_bin "$DOCKER_BIN"

if ! "$DOCKER_BIN" version --format '{{.Server.Version}}' >"$RUN_DIR/docker-version.txt" 2>&1; then
  die 'Docker engine is unavailable; see docker-version.txt'
fi

# `image inspect` is local-only.  It cannot pull, and subsequent execution also
# uses --pull=never.  Failure therefore means the exact digest is not cached.
if ! "$DOCKER_BIN" image inspect "$IMAGE_REF" >"$RUN_DIR/docker-image-inspect.json" 2>&1; then
  die "pinned image is not present locally; this script will not pull it: $IMAGE_REF"
fi

# The pinned image contains several unrelated external method plugins. One is
# an editable bionerf install whose .pth finder is not executed after HOME is
# redirected to the dedicated cache; loading all entry points therefore fails
# before Splatfacto is selected. Enabling bionerf alone merely advances CPU
# import into unrelated kplanes/tinycudann failures. The formal experiment uses
# the built-in Splatfacto method, so audit the complete immutable plugin set and
# then hide only this entry-point group for the lifetime of the current Python
# process. No package/file is installed, removed, or rewritten.
METHOD_PLUGIN_GUARD_PY=$(cat <<'PY'
import hashlib as _qp_hashlib
import importlib.metadata as _qp_metadata
import json as _qp_json
from pathlib import Path as _QPPath


_QP_METHOD_GROUP = "nerfstudio.method_configs"
_QP_ENTRY_FIELDS = (
    "name",
    "value",
    "distribution",
    "version",
    "dist_path",
    "entry_points_sha256",
)
_QP_EXPECTED_METHOD_ENTRY_POINTS = (
    (
        "bionerf",
        "bionerf.bionerf_config:bionerf_method",
        "bionerf",
        "1.0",
        "/home/user/.local/lib/python3.10/site-packages/bionerf-1.0.dist-info",
        "8dd1975af4901f2d5c8e0f1ec9401e4bae1f016d36c84b049c80f46de0f204f6",
    ),
    (
        "igs2gs",
        "igs2gs.igs2gs_config:igs2gs_method",
        "igs2gs",
        "0.1.0",
        "/home/user/.local/lib/python3.10/site-packages/igs2gs-0.1.0.dist-info",
        "cbdea16aec02408ea1850f18d8c3d21d6a16aeee0e4d759ae255edff724f7a21",
    ),
    (
        "kplanes",
        "kplanes.kplanes_configs:kplanes_method",
        "kplanes_nerfstudio",
        "0.5.2",
        "/home/user/.local/lib/python3.10/site-packages/kplanes_nerfstudio-0.5.2.dist-info",
        "aa826943bd6514ee0a263148382780968e73b5ef39870b2ef740cc3e7ddf2759",
    ),
    (
        "kplanes_dynamic",
        "kplanes.kplanes_configs:kplanes_dynamic_method",
        "kplanes_nerfstudio",
        "0.5.2",
        "/home/user/.local/lib/python3.10/site-packages/kplanes_nerfstudio-0.5.2.dist-info",
        "aa826943bd6514ee0a263148382780968e73b5ef39870b2ef740cc3e7ddf2759",
    ),
    (
        "lerf",
        "lerf.lerf_config:lerf_method",
        "lerf",
        "0.1.1",
        "/home/user/.local/lib/python3.10/site-packages/lerf-0.1.1.dist-info",
        "45a6092164be0da69a528d3d7ec95d01f5e8b82e748ede4ed4a73b8189720507",
    ),
    (
        "lerf_big",
        "lerf.lerf_config:lerf_method_big",
        "lerf",
        "0.1.1",
        "/home/user/.local/lib/python3.10/site-packages/lerf-0.1.1.dist-info",
        "45a6092164be0da69a528d3d7ec95d01f5e8b82e748ede4ed4a73b8189720507",
    ),
    (
        "lerf_lite",
        "lerf.lerf_config:lerf_method_lite",
        "lerf",
        "0.1.1",
        "/home/user/.local/lib/python3.10/site-packages/lerf-0.1.1.dist-info",
        "45a6092164be0da69a528d3d7ec95d01f5e8b82e748ede4ed4a73b8189720507",
    ),
    (
        "nerfplayer_nerfacto",
        "nerfplayer.nerfplayer_config:nerfplayer_nerfacto",
        "nerfplayer",
        "0.0.1",
        "/home/user/.local/lib/python3.10/site-packages/nerfplayer-0.0.1.dist-info",
        "1c89b0236ca4a9811c6d52c4c72b9ea967e20d92d1d8480150c838d1c4c7144b",
    ),
    (
        "nerfplayer_ngp",
        "nerfplayer.nerfplayer_config:nerfplayer_ngp",
        "nerfplayer",
        "0.0.1",
        "/home/user/.local/lib/python3.10/site-packages/nerfplayer-0.0.1.dist-info",
        "1c89b0236ca4a9811c6d52c4c72b9ea967e20d92d1d8480150c838d1c4c7144b",
    ),
)
_QP_EXPECTED_NERFSTUDIO_SOURCES = {
    "/home/user/nerfstudio/nerfstudio/plugins/registry.py":
        "2c955c48ff6b42e7823c90fc36dca9344fc6010c511238337b1528aa36c6930f",
    "/home/user/nerfstudio/nerfstudio/configs/method_configs.py":
        "b004bcf9e7ba5de52d94138a86aae260c58dbd751eb901ae41f0ab3f75a22718",
    "/home/user/nerfstudio/nerfstudio/scripts/train.py":
        "2a3b31c832427ca6c56b068a9b18039ea616834da5046852e0231c9df1b6d3c9",
}


def _qp_sha256(path):
    path = _QPPath(path)
    if not path.is_file():
        raise RuntimeError(f"pinned method-plugin audit file is missing: {path}")
    return _qp_hashlib.sha256(path.read_bytes()).hexdigest()


_qp_source_hashes = {
    path: _qp_sha256(path) for path in _QP_EXPECTED_NERFSTUDIO_SOURCES
}
if _qp_source_hashes != _QP_EXPECTED_NERFSTUDIO_SOURCES:
    raise RuntimeError(
        "Nerfstudio method registry source hashes changed: "
        f"{_qp_source_hashes!r}"
    )

_qp_original_entry_points = _qp_metadata.entry_points
_qp_discovered = _qp_original_entry_points(group=_QP_METHOD_GROUP)
_qp_actual = []
for _qp_entry_point in _qp_discovered:
    _qp_dist_path = _QPPath(_qp_entry_point.dist._path)
    _qp_metadata_path = _qp_dist_path / "entry_points.txt"
    _qp_actual.append(
        (
            _qp_entry_point.name,
            _qp_entry_point.value,
            _qp_entry_point.dist.metadata.get("Name"),
            _qp_entry_point.dist.version,
            str(_qp_dist_path),
            _qp_sha256(_qp_metadata_path),
        )
    )
_qp_actual = tuple(sorted(_qp_actual))
if _qp_actual != _QP_EXPECTED_METHOD_ENTRY_POINTS:
    raise RuntimeError(
        "external Nerfstudio method entry points changed; refusing the "
        f"built-in-only policy: {_qp_actual!r}"
    )

print(
    "METHOD_PLUGIN_AUDIT "
    + _qp_json.dumps(
        {
            "policy": "built-in-only",
            "disabled_group": _QP_METHOD_GROUP,
            "disabled_entry_points": [
                dict(zip(_QP_ENTRY_FIELDS, row)) for row in _qp_actual
            ],
            "nerfstudio_source_sha256": _qp_source_hashes,
        },
        sort_keys=True,
    ),
    flush=True,
)

_qp_empty_entry_points = _qp_metadata.EntryPoints(())


def _qp_builtin_only_entry_points(*args, **kwargs):
    if kwargs.get("group") == _QP_METHOD_GROUP:
        return _qp_empty_entry_points
    return _qp_original_entry_points(*args, **kwargs)


_qp_metadata.entry_points = _qp_builtin_only_entry_points
PY
)
METHOD_PLUGIN_GUARD_SHA256=$(printf '%s' "$METHOD_PLUGIN_GUARD_PY" | sha256sum | awk '{print $1}')
record_value method_plugin_policy 'built-in-only'
record_value method_plugin_guard_sha256 "$METHOD_PLUGIN_GUARD_SHA256"

# The image has two experimentally confirmed, immutable `pip check` deviations.
# Accept exactly those two lines; a missing line or any additional problem fails.
# Do not use `ns-train splatfacto --help` as a CPU gate: in this image its eager
# plugin discovery imports kplanes/tinycudann and fails merely because no CUDA
# device is exposed. The first GPU smoke is the authoritative CLI/GPU gate.
PREFLIGHT_BODY_PY=$(cat <<'PY'
import hashlib
import json
import os
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path, PurePosixPath


def verify_source_receipt(
    root,
    *,
    expected_track,
    expected_receipt_sha256,
    expected_file_count,
    expected_image_count,
    expected_image_bytes,
    expected_total_bytes,
):
    """Stream and verify every source byte named by the pinned receipt."""
    root = Path(root).resolve(strict=True)
    receipt_path = root / ".quadpilot_source_receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha256 != expected_receipt_sha256:
        raise RuntimeError(
            f"receipt SHA-256 mismatch: {receipt_sha256} != {expected_receipt_sha256}"
        )
    receipt = json.loads(receipt_bytes)
    if receipt.get("schema_version") != 1 or receipt.get("track") != expected_track:
        raise RuntimeError("receipt schema/track mismatch")
    files = receipt.get("files")
    if not isinstance(files, dict) or len(files) != expected_file_count:
        raise RuntimeError(
            f"receipt file count mismatch: {len(files) if isinstance(files, dict) else -1}"
        )
    image_count = sum(name.startswith("images/") for name in files)
    if image_count != expected_image_count:
        raise RuntimeError(f"receipt image count mismatch: {image_count}")
    if "transforms.json" not in files or "sparse_pc.ply" not in files:
        raise RuntimeError("receipt omits transforms.json or sparse_pc.ply")

    verified_bytes = 0
    verified_image_bytes = 0
    for index, (name, metadata) in enumerate(sorted(files.items()), start=1):
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise RuntimeError("receipt path/metadata has an invalid type")
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in name
        ):
            raise RuntimeError(f"unsafe receipt path: {name!r}")
        candidate = root.joinpath(*relative.parts)
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise RuntimeError(f"receipt path traverses a symlink: {name}")
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(f"receipt file missing: {name}") from error
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise RuntimeError(f"receipt path escapes data root or is not a file: {name}")

        expected_size = metadata.get("size_bytes")
        expected_sha256 = metadata.get("sha256")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise RuntimeError(f"invalid receipt size for {name}")
        if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise RuntimeError(f"invalid receipt SHA-256 for {name}")
        actual_size = resolved.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"size mismatch for {name}: {actual_size} != {expected_size}")
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {name}: {actual_sha256} != {expected_sha256}"
            )
        verified_bytes += actual_size
        if name.startswith("images/"):
            verified_image_bytes += actual_size
        if index % 100 == 0 or index == expected_file_count:
            print(f"RECEIPT_PROGRESS {index}/{expected_file_count}", flush=True)
    if verified_bytes != expected_total_bytes:
        raise RuntimeError(
            f"receipt byte total mismatch: {verified_bytes} != {expected_total_bytes}"
        )
    if verified_image_bytes != expected_image_bytes:
        raise RuntimeError(
            f"receipt image byte total mismatch: {verified_image_bytes} != {expected_image_bytes}"
        )
    return {
        "source_profile": expected_track,
        "receipt_sha256": receipt_sha256,
        "verified_files": expected_file_count,
        "verified_images": expected_image_count,
        "verified_image_bytes": verified_image_bytes,
        "verified_bytes": verified_bytes,
    }

expected_versions = {
    "nerfstudio": "1.1.4",
    "gsplat": "1.0.0",
    "torch": "2.1.2+cu118",
    "torchvision": "0.16.2+cu118",
    "viser": "0.2.3",
}
versions = {name: version(name) for name in expected_versions}
print("VERSIONS", json.dumps(versions, sort_keys=True), flush=True)
if versions != expected_versions:
    print("VERSION_GATE_FAILED", file=sys.stderr, flush=True)
    raise SystemExit(10)

pip_check = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
)
actual_deviations = sorted(
    line.strip() for line in pip_check.stdout.splitlines() if line.strip()
)
expected_deviations = sorted(
    [
        "rawpy 0.22.0 has requirement numpy>=2.0, but you have numpy 1.26.4.",
        "ninja 1.11.1.1 is not supported on this platform",
    ]
)
print(
    "PIP_CHECK",
    json.dumps(
        {"returncode": pip_check.returncode, "known_deviations": actual_deviations},
        sort_keys=True,
    ),
    flush=True,
)
if pip_check.returncode != 1 or actual_deviations != expected_deviations:
    print("PIP_CHECK_UNKNOWN_OR_CHANGED_DEVIATION", file=sys.stderr, flush=True)
    raise SystemExit(11)

source_profile = os.environ["QUADPILOT_SOURCE_PROFILE"]
expected_receipt_sha256 = os.environ["QUADPILOT_SOURCE_RECEIPT_SHA256"]
expected_file_count = int(os.environ["QUADPILOT_SOURCE_RECEIPT_FILES"])
expected_image_count = int(os.environ["QUADPILOT_SOURCE_RECEIPT_IMAGES"])
expected_image_bytes = int(os.environ["QUADPILOT_SOURCE_RECEIPT_IMAGE_BYTES"])
expected_total_bytes = int(os.environ["QUADPILOT_SOURCE_RECEIPT_TOTAL_BYTES"])
expected_sparse_points = int(os.environ["QUADPILOT_SOURCE_SPARSE_POINTS"])

receipt_summary = verify_source_receipt(
    Path("/data"),
    expected_track=source_profile,
    expected_receipt_sha256=expected_receipt_sha256,
    expected_file_count=expected_file_count,
    expected_image_count=expected_image_count,
    expected_image_bytes=expected_image_bytes,
    expected_total_bytes=expected_total_bytes,
)
print("RECEIPT_OK", json.dumps(receipt_summary, sort_keys=True), flush=True)

from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    NerfstudioDataParserConfig,
)

root = Path("/data")
outputs = NerfstudioDataParserConfig(
    data=root,
    load_3D_points=True,
    eval_mode="all",
    downscale_factor=1,
).setup().get_dataparser_outputs(split="train")
image_count = len(outputs.image_filenames)
camera_count = len(outputs.cameras)
missing_images = sum(not path.is_file() for path in outputs.image_filenames)
points = outputs.metadata.get("points3D_xyz")
point_count = int(points.shape[0]) if points is not None else -1
dataset_summary = {
    "source_profile": source_profile,
    "images": image_count,
    "cameras": camera_count,
    "missing_images": missing_images,
    "sparse_points": point_count,
    "dataparser_downscale_factor": 1,
}
print("DATASET", json.dumps(dataset_summary, sort_keys=True), flush=True)
if dataset_summary != {
    "source_profile": source_profile,
    "images": expected_image_count,
    "cameras": expected_image_count,
    "missing_images": 0,
    "sparse_points": expected_sparse_points,
    "dataparser_downscale_factor": 1,
}:
    print("DATASET_GATE_FAILED", file=sys.stderr, flush=True)
    raise SystemExit(12)

from nerfstudio.configs.method_configs import all_methods

builtin_method_summary = {
    "method_count": len(all_methods),
    "splatfacto_present": "splatfacto" in all_methods,
}
print(
    "BUILTIN_METHOD_CONFIGS_OK "
    + json.dumps(builtin_method_summary, sort_keys=True),
    flush=True,
)
if builtin_method_summary != {
    "method_count": 43,
    "splatfacto_present": True,
}:
    print("BUILTIN_METHOD_CONFIGS_GATE_FAILED", file=sys.stderr, flush=True)
    raise SystemExit(13)

print("PREFLIGHT_OK", flush=True)
PY
)
PREFLIGHT_PY="$METHOD_PLUGIN_GUARD_PY"$'\n'"$PREFLIGHT_BODY_PY"

TRAIN_LAUNCHER_BODY_PY=$(cat <<'PY'
import sys

sys.argv[0] = "ns-train"
from nerfstudio.scripts.train import entrypoint

raise SystemExit(entrypoint())
PY
)
TRAIN_LAUNCHER_PY="$METHOD_PLUGIN_GUARD_PY"$'\n'"$TRAIN_LAUNCHER_BODY_PY"
TRAIN_LAUNCHER_SHA256=$(printf '%s' "$TRAIN_LAUNCHER_PY" | sha256sum | awk '{print $1}')

PREFLIGHT_COMMAND=(
  "$DOCKER_BIN" run
  --rm
  --pull=never
  --network none
  --read-only
  --tmpfs /tmp:rw,nosuid,nodev,size=2g
  --cap-drop ALL
  --security-opt no-new-privileges
  --user "$(id -u):$(id -g)"
  --mount "type=bind,src=$DATA_DIR,dst=/data,readonly"
  --mount "type=bind,src=$CACHE_DIR,dst=/cache"
  --env HOME=/cache/home
  --env XDG_CACHE_HOME=/cache/xdg
  --env TORCH_HOME=/cache/torch
  --env HF_HOME=/cache/huggingface
  --env "QUADPILOT_SOURCE_PROFILE=$SOURCE_PROFILE"
  --env "QUADPILOT_SOURCE_RECEIPT_SHA256=$SOURCE_RECEIPT_SHA256"
  --env "QUADPILOT_SOURCE_RECEIPT_FILES=$SOURCE_RECEIPT_FILES"
  --env "QUADPILOT_SOURCE_RECEIPT_IMAGES=$SOURCE_RECEIPT_IMAGES"
  --env "QUADPILOT_SOURCE_RECEIPT_IMAGE_BYTES=$SOURCE_RECEIPT_IMAGE_BYTES"
  --env "QUADPILOT_SOURCE_RECEIPT_TOTAL_BYTES=$SOURCE_RECEIPT_TOTAL_BYTES"
  --env "QUADPILOT_SOURCE_SPARSE_POINTS=$SOURCE_SPARSE_POINTS"
  --env "PYTHONPATH=$IMAGE_PYTHONPATH"
  --env "PATH=$IMAGE_PATH"
  --env PYTHONDONTWRITEBYTECODE=1
  --env NVIDIA_VISIBLE_DEVICES=void
  --env CUDA_VISIBLE_DEVICES=
  --workdir /tmp
  "$IMAGE_REF"
  python3.10 -c "$PREFLIGHT_PY"
)
write_command_file "$RUN_DIR/preflight-command.sh" "${PREFLIGHT_COMMAND[@]}"

if [[ "$MODE" != 'preflight' ]]; then
  case "$MODE" in
    smoke-1|smoke-101)
      NUM_DOWNSCALES='0'
      ;;
    train-30k)
      NUM_DOWNSCALES='2'
      ;;
  esac
  case "$MODE" in
    smoke-1) FINAL_STEP='0' ;;
    smoke-101) FINAL_STEP='100' ;;
    train-30k) FINAL_STEP='29999' ;;
  esac
  printf -v EXPECTED_CHECKPOINT_NAME 'step-%09d.ckpt' "$FINAL_STEP"
  record_value max_num_iterations "$MAX_ITERATIONS"
  record_value expected_final_step "$FINAL_STEP"
  record_value expected_checkpoint_name "$EXPECTED_CHECKPOINT_NAME"
  record_value splatfacto_num_downscales "$NUM_DOWNSCALES"
  record_value splatfacto_resolution_schedule '3000'
  record_value periodic_evaluation_enabled 'false'
  record_value training_max_jobs "$TRAIN_MAX_JOBS"
  record_value training_launcher_sha256 "$TRAIN_LAUNCHER_SHA256"
  TRAINING_OUTPUT_DIR="$RUN_DIR/training-output"
  mkdir -- "$TRAINING_OUTPUT_DIR"
  record_value training_output_dir "$TRAINING_OUTPUT_DIR"

  DOCKER_COMMAND=(
    "$DOCKER_BIN" run
    --rm
    --pull=never
    --gpus "device=$GPU_INDEX"
    --network none
    --shm-size "$DEFAULT_SHM_SIZE"
    --cap-drop ALL
    --security-opt no-new-privileges
    --user "$(id -u):$(id -g)"
    --mount "type=bind,src=$DATA_DIR,dst=/data,readonly"
    --mount "type=bind,src=$TRAINING_OUTPUT_DIR,dst=/outputs"
    --mount "type=bind,src=$CACHE_DIR,dst=/cache"
    --mount "type=bind,src=$LPIPS_ALEXNET_PATH,dst=/cache/$LPIPS_ALEXNET_RELATIVE_PATH,readonly"
    --env HOME=/cache/home
    --env XDG_CACHE_HOME=/cache/xdg
    --env TORCH_HOME=/cache/torch
    --env HF_HOME=/cache/huggingface
    --env "MAX_JOBS=$TRAIN_MAX_JOBS"
    --env "PYTHONPATH=$IMAGE_PYTHONPATH"
    --env "PATH=$IMAGE_PATH"
    --env WANDB_MODE=offline
    --workdir /outputs
    "$IMAGE_REF"
    python3.10 -c "$TRAIN_LAUNCHER_PY"
    splatfacto
    --data /data
    --output-dir /outputs
    --experiment-name "$TRACK"
    --timestamp "$RUN_ID"
    --max-num-iterations "$MAX_ITERATIONS"
    --vis tensorboard
    --steps-per-eval-batch 0
    --steps-per-eval-image 0
    --steps-per-eval-all-images 0
    --machine.num-devices 1
    --pipeline.datamanager.camera-res-scale-factor "$HALF_RES_SCALE"
    --pipeline.model.num-downscales "$NUM_DOWNSCALES"
    --pipeline.model.resolution-schedule 3000
    nerfstudio-data
    --downscale-factor 1
  )
  write_command_file "$RUN_DIR/command.sh" "${DOCKER_COMMAND[@]}"
fi

if ((DRY_RUN)); then
  record_value cpu_preflight_executed 'false'
  note "Dry run: no container was started. CPU preflight command: $RUN_DIR/preflight-command.sh"
  if [[ "$MODE" != 'preflight' ]]; then
    note "Dry run: GPU training command: $RUN_DIR/command.sh"
    printf 'DRY RUN: '
    printf '%q ' "${DOCKER_COMMAND[@]}"
    printf '\n'
  fi
  exit 0
fi

record_value cpu_preflight_executed 'true'
note 'Running pinned CPU-only version/pip/dataparser preflight (no GPU requested).'
if ! "${PREFLIGHT_COMMAND[@]}" 2>&1 | tee "$RUN_DIR/preflight-container.log"; then
  die 'CPU preflight failed; see preflight-container.log'
fi
if ! grep '^RECEIPT_OK ' "$RUN_DIR/preflight-container.log" \
  | sed 's/^RECEIPT_OK //' >"$RUN_DIR/receipt-verification.json"; then
  die 'CPU preflight omitted the receipt verification record'
fi
[[ -s "$RUN_DIR/receipt-verification.json" ]] || die 'receipt verification record is empty'
if ! grep '^METHOD_PLUGIN_AUDIT ' "$RUN_DIR/preflight-container.log" \
  | sed 's/^METHOD_PLUGIN_AUDIT //' >"$RUN_DIR/method-plugin-audit.json"; then
  die 'CPU preflight omitted the method-plugin audit record'
fi
[[ -s "$RUN_DIR/method-plugin-audit.json" ]] || die 'method-plugin audit record is empty'
if ! grep '^BUILTIN_METHOD_CONFIGS_OK ' "$RUN_DIR/preflight-container.log" \
  | sed 's/^BUILTIN_METHOD_CONFIGS_OK //' >"$RUN_DIR/builtin-method-configs.json"; then
  die 'CPU preflight omitted the built-in method-config diagnostic'
fi
[[ -s "$RUN_DIR/builtin-method-configs.json" ]] || \
  die 'built-in method-config diagnostic is empty'
record_value method_plugin_audit_path "$RUN_DIR/method-plugin-audit.json"
record_value builtin_method_configs_path "$RUN_DIR/builtin-method-configs.json"

if [[ "$MODE" == 'preflight' ]]; then
  note "CPU-only preflight passed. GPU availability is intentionally deferred to smoke-1. Audit: $RUN_DIR"
  exit 0
fi

NVIDIA_SMI_BIN=$(command -v nvidia-smi || true)
[[ -n "$NVIDIA_SMI_BIN" ]] || die 'nvidia-smi is not installed or not on PATH'
record_value nvidia_smi_bin "$NVIDIA_SMI_BIN"

if ! "$NVIDIA_SMI_BIN" -q -i "$GPU_INDEX" >"$RUN_DIR/nvidia-smi.before.txt" 2>&1; then
  die "nvidia-smi could not inspect GPU $GPU_INDEX"
fi
if ! "$NVIDIA_SMI_BIN" \
  --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits >"$RUN_DIR/gpu-query.csv" 2>"$RUN_DIR/gpu-query.stderr.txt"; then
  die 'nvidia-smi GPU query failed closed'
fi

GPU_ROW=$(awk -F, -v target="$GPU_INDEX" '
  {
    idx=$1
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", idx)
    if (idx == target) { print; exit }
  }
' "$RUN_DIR/gpu-query.csv")
[[ -n "$GPU_ROW" ]] || die "GPU index $GPU_INDEX was not reported by nvidia-smi"

IFS=',' read -r GPU_INDEX_ACTUAL GPU_UUID GPU_NAME GPU_MEMORY_TOTAL GPU_MEMORY_USED GPU_UTIL <<<"$GPU_ROW"
GPU_INDEX_ACTUAL=$(trim "$GPU_INDEX_ACTUAL")
GPU_UUID=$(trim "$GPU_UUID")
GPU_NAME=$(trim "$GPU_NAME")
GPU_MEMORY_TOTAL=$(trim "$GPU_MEMORY_TOTAL")
GPU_MEMORY_USED=$(trim "$GPU_MEMORY_USED")
GPU_UTIL=$(trim "$GPU_UTIL")
[[ "$GPU_INDEX_ACTUAL" == "$GPU_INDEX" ]] || die 'nvidia-smi returned an inconsistent GPU index'
[[ -n "$GPU_UUID" && "$GPU_UUID" != 'N/A' ]] || die 'nvidia-smi did not provide a usable GPU UUID'
is_number "$GPU_MEMORY_USED" || die 'nvidia-smi returned non-numeric used GPU memory'
is_number "$GPU_UTIL" || die 'nvidia-smi returned non-numeric GPU utilization'
record_value gpu_uuid "$GPU_UUID"
record_value gpu_name "$GPU_NAME"
record_value gpu_memory_total_mib "$GPU_MEMORY_TOTAL"
record_value gpu_memory_used_mib "$GPU_MEMORY_USED"
record_value gpu_util_percent "$GPU_UTIL"

if ! "$NVIDIA_SMI_BIN" \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits >"$RUN_DIR/compute-apps.csv" 2>"$RUN_DIR/compute-apps.stderr.txt"; then
  die 'nvidia-smi compute-process query failed closed'
fi

ACTIVE_COMPUTE=$(awk -F, -v target="$GPU_UUID" '
  {
    uuid=$1
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
    if (uuid == target) print
  }
' "$RUN_DIR/compute-apps.csv")

BUSY_REASONS=()
if [[ -n "$ACTIVE_COMPUTE" ]]; then
  BUSY_REASONS+=('active compute process(es)')
fi
if awk -v actual="$GPU_UTIL" -v maximum="$MAX_GPU_UTIL" 'BEGIN { exit !(actual > maximum) }'; then
  BUSY_REASONS+=("utilization ${GPU_UTIL}% exceeds ${MAX_GPU_UTIL}%")
fi
if awk -v actual="$GPU_MEMORY_USED" -v maximum="$MAX_GPU_MEMORY_MIB" 'BEGIN { exit !(actual > maximum) }'; then
  BUSY_REASONS+=("used memory ${GPU_MEMORY_USED} MiB exceeds ${MAX_GPU_MEMORY_MIB} MiB")
fi

if ((${#BUSY_REASONS[@]})); then
  BUSY_TEXT=$(IFS='; '; printf '%s' "${BUSY_REASONS[*]}")
  record_value gpu_busy_detected 'true'
  record_value gpu_busy_reasons "$BUSY_TEXT"
  if ((ALLOW_BUSY_GPU == 0)); then
    die "GPU $GPU_INDEX is busy ($BUSY_TEXT); refusing by default"
  fi
  note "WARNING: explicitly overriding busy GPU gate: $BUSY_TEXT"
else
  record_value gpu_busy_detected 'false'
fi

if [[ "$MODE" == 'train-30k' ]]; then
  note 'HALF-RES CACHE WARNING: source scale is 0.5; progressive raster is source 1/8 (steps 0-2999), 1/4 (3000-5999), then 1/2. This is not full resolution.'
else
  note 'HALF-RES WARNING: scale 0.5 in width/height (0.25 pixels), with no extra model downscale. This is not the original full-resolution experiment.'
fi

note "Starting pinned $MODE run; audit and outputs: $RUN_DIR"
"${DOCKER_COMMAND[@]}" 2>&1 | tee "$RUN_DIR/docker.log"

# A successful training process must prove that the same audited guard actually
# ran in the GPU container, not merely in the earlier CPU preflight.
if ! grep '^METHOD_PLUGIN_AUDIT ' "$RUN_DIR/docker.log" \
  | sed 's/^METHOD_PLUGIN_AUDIT //' >"$RUN_DIR/training-method-plugin-audit.json"; then
  die 'training container omitted the method-plugin audit record'
fi
[[ -s "$RUN_DIR/training-method-plugin-audit.json" ]] || \
  die 'training method-plugin audit record is empty'
if ! cmp -s \
  "$RUN_DIR/method-plugin-audit.json" \
  "$RUN_DIR/training-method-plugin-audit.json"; then
  die 'training method-plugin audit differs from CPU preflight'
fi
record_value training_method_plugin_audit_path \
  "$RUN_DIR/training-method-plugin-audit.json"

# Nerfstudio 1.1.4 can swallow KeyboardInterrupt in its single-device launcher.
# Do not equate container exit zero with a completed experiment: require the
# exact final-step checkpoint and config before the EXIT trap records success.
TRAIN_RUN_DIR="$TRAINING_OUTPUT_DIR/$TRACK/splatfacto/$RUN_ID"
CONFIG_PATH="$TRAIN_RUN_DIR/config.yml"
CHECKPOINT_DIR="$TRAIN_RUN_DIR/nerfstudio_models"
CHECKPOINT_PATH="$CHECKPOINT_DIR/$EXPECTED_CHECKPOINT_NAME"
[[ -s "$CONFIG_PATH" ]] || die "training exited without a non-empty config.yml: $CONFIG_PATH"
[[ -s "$CHECKPOINT_PATH" ]] || \
  die "training exited without the expected final checkpoint: $CHECKPOINT_PATH"
mapfile -d '' CHECKPOINT_FILES < <(
  find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'step-*.ckpt' -print0
)
if ((${#CHECKPOINT_FILES[@]} != 1)) || [[ "${CHECKPOINT_FILES[0]}" != "$CHECKPOINT_PATH" ]]; then
  die 'checkpoint directory must contain exactly the expected final-step checkpoint'
fi
(
  cd -- "$TRAIN_RUN_DIR"
  sha256sum "config.yml" "nerfstudio_models/$EXPECTED_CHECKPOINT_NAME"
) >"$RUN_DIR/training-artifacts.sha256"
record_value verified_config_path "$CONFIG_PATH"
record_value verified_checkpoint_path "$CHECKPOINT_PATH"
record_value verified_checkpoint_size_bytes "$(stat -c '%s' "$CHECKPOINT_PATH")"
