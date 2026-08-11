"""Order- and direction-aware gate evaluation for reproduction runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .tracks import TrackConfig, get_track


@dataclass(frozen=True)
class GateCrossing:
    gate: str
    lap: int
    time_seconds: float
    radial_error_m: float
    position: tuple[float, float, float]


@dataclass(frozen=True)
class EvaluationResult:
    track: str
    required_crossings: int
    successful_crossings: int
    success_rate: float
    mean_gate_error_m: float | None
    mission_time_s: float | None
    first_to_last_gate_time_s: float | None
    completed: bool
    crossings: tuple[GateCrossing, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["crossings"] = [asdict(item) for item in self.crossings]
        return result


def _side(distance: float, fallback: float = 1.0) -> float:
    sign = float(np.sign(distance))
    return sign if sign != 0.0 else fallback


def evaluate_ordered_gates(
    states: np.ndarray,
    track: str | TrackConfig,
    *,
    dt: float = 0.05,
    laps: int = 2,
    gate_radius: float = 0.38,
) -> EvaluationResult:
    """Evaluate only ordered, correctly directed, interpolated crossings.

    Unlike the course evaluator, this function cannot double-count arbitrary
    infinite-plane crossings and its success rate is bounded by 100 percent.
    """

    config = get_track(track) if isinstance(track, str) else track
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError("states must have shape (N, >=3)")
    if len(states) < 2:
        raise ValueError("at least two states are required")
    if dt <= 0 or laps <= 0 or gate_radius <= 0:
        raise ValueError("dt, laps and gate_radius must be positive")

    sequence = config.ordered_gates() * laps
    target_index = 0
    crossings: list[GateCrossing] = []
    first_gate_time: float | None = None
    last_gate_time: float | None = None

    required_side = float(config.incoming_gate_sides[0])

    for sample_index in range(len(states) - 1):
        if target_index >= len(sequence):
            break
        gate = sequence[target_index]
        normal = gate.normal
        center = np.asarray(gate.center)
        p0 = states[sample_index, :3]
        p1 = states[sample_index + 1, :3]
        d0 = float(np.dot(p0 - center, normal))
        d1 = float(np.dot(p1 - center, normal))
        side0 = _side(d0, fallback=required_side)
        side1 = _side(d1, fallback=-side0)

        if side0 == side1 or side0 != required_side:
            continue

        denominator = d0 - d1
        alpha = 0.5 if abs(denominator) < 1e-12 else d0 / denominator
        alpha = float(np.clip(alpha, 0.0, 1.0))
        intersection = p0 + alpha * (p1 - p0)
        relative = intersection - center
        radial_error = float(
            np.linalg.norm(relative - np.dot(relative, normal) * normal)
        )
        if radial_error > gate_radius:
            continue

        crossing_time = (sample_index + alpha) * dt
        lap = target_index // len(config.gate_order) + 1
        crossings.append(
            GateCrossing(
                gate=gate.name,
                lap=lap,
                time_seconds=crossing_time,
                radial_error_m=radial_error,
                position=tuple(float(value) for value in intersection),
            )
        )
        if first_gate_time is None:
            first_gate_time = crossing_time
        last_gate_time = crossing_time
        target_index += 1

        if target_index < len(sequence):
            required_side = float(
                config.incoming_gate_sides[
                    target_index % len(config.incoming_gate_sides)
                ]
            )

    errors = [crossing.radial_error_m for crossing in crossings]
    completed = len(crossings) == len(sequence)
    mission_time = last_gate_time if completed else None
    first_to_last = (
        last_gate_time - first_gate_time
        if completed and first_gate_time is not None and last_gate_time is not None
        else None
    )
    return EvaluationResult(
        track=config.name,
        required_crossings=len(sequence),
        successful_crossings=len(crossings),
        success_rate=len(crossings) / len(sequence),
        mean_gate_error_m=float(np.mean(errors)) if errors else None,
        mission_time_s=mission_time,
        first_to_last_gate_time_s=first_to_last,
        completed=completed,
        crossings=tuple(crossings),
    )
