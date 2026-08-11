"""Authoritative NeRF-coordinate track definitions used by the final project."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Gate:
    """A gate center and its forward normal in the NeRF world frame."""

    name: str
    center: tuple[float, float, float]
    yaw_deg: float

    @property
    def normal(self) -> np.ndarray:
        yaw = np.deg2rad(self.yaw_deg)
        return np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)


@dataclass(frozen=True)
class TrackConfig:
    """Track geometry and the initial simulation state.

    State order is ``[x, y, z, vx, vy, vz, yaw]``.  Positions and gate
    directions intentionally preserve the coordinate system used to train the
    original NPE models; they are not FalconGym/Vicon coordinates.

    ``incoming_gate_sides[i]`` is the required sign of
    ``dot(position - gate.center, gate.normal)`` immediately before crossing
    ``gate_order[i]``.  Direction is therefore a per-gate contract rather than
    something inferred from an arbitrary shared reference position.
    """

    name: str
    gates: Mapping[str, Gate]
    gate_order: tuple[str, ...]
    initial_state: tuple[float, float, float, float, float, float, float]
    incoming_gate_sides: tuple[int, ...]
    model_path: str

    def __post_init__(self) -> None:
        if not self.gate_order:
            raise ValueError("gate_order must contain at least one gate")
        if not isinstance(self.incoming_gate_sides, tuple):
            raise ValueError("incoming_gate_sides must be an immutable tuple")
        if len(self.incoming_gate_sides) != len(self.gate_order):
            raise ValueError(
                "incoming_gate_sides must contain exactly one side for each "
                f"ordered gate; got {len(self.incoming_gate_sides)} sides for "
                f"{len(self.gate_order)} gates"
            )
        invalid_sides = [
            side
            for side in self.incoming_gate_sides
            if isinstance(side, (bool, np.bool_))
            or not isinstance(side, Integral)
            or int(side) not in (-1, 1)
        ]
        if invalid_sides:
            raise ValueError(
                "incoming_gate_sides values must be integer -1 or +1; "
                f"got {invalid_sides}"
            )

    def ordered_gates(self) -> tuple[Gate, ...]:
        return tuple(self.gates[name] for name in self.gate_order)


def _gates(**items: tuple[tuple[float, float, float], float]) -> Mapping[str, Gate]:
    return MappingProxyType(
        {
            name.replace("_", " "): Gate(
                name=name.replace("_", " "), center=center, yaw_deg=yaw_deg
            )
            for name, (center, yaw_deg) in items.items()
        }
    )


TRACKS: Mapping[str, TrackConfig] = MappingProxyType(
    {
        "circle": TrackConfig(
            name="circle",
            gates=_gates(
                Gate_A=((-0.3, -3.8, -0.4), -90.0),
                Gate_B=((-2.3, -6.0, -0.4), -180.0),
                Gate_C=((-4.1, -3.9, -0.4), 90.0),
                Gate_D=((-2.2, -1.7, -0.4), 0.0),
            ),
            gate_order=("Gate A", "Gate B", "Gate C", "Gate D"),
            initial_state=(-0.4, -0.5, -0.3, 0.0, 0.0, 0.0, -pi / 2.0),
            incoming_gate_sides=(-1, -1, -1, -1),
            model_path="npe_models/circle/best_npe.pth",
        ),
        "uturn": TrackConfig(
            name="uturn",
            gates=_gates(
                Gate_A=((-2.2, -6.1, -0.3), -180.0),
                Gate_B=((-3.8, -4.6, -0.3), 90.0),
                Gate_C=((-2.2, -3.4, -0.3), 0.0),
                Gate_D=((-0.4, -1.6, -0.4), 90.0),
            ),
            gate_order=("Gate A", "Gate B", "Gate C", "Gate D"),
            initial_state=(-0.5, -6.1, -0.3, 0.0, 0.0, 0.0, -pi),
            incoming_gate_sides=(-1, -1, -1, -1),
            model_path="npe_models/uturn/best_npe.pth",
        ),
        "lemniscate": TrackConfig(
            name="lemniscate",
            gates=_gates(
                Gate_A=((-0.8, -1.8, -0.4), 90.0),
                Gate_B=((-3.5, -1.9, -0.4), -90.0),
                Gate_C=((-0.9, -5.6, -0.4), -90.0),
                Gate_D=((-3.4, -5.6, -0.4), 90.0),
            ),
            gate_order=("Gate D", "Gate A", "Gate B", "Gate C"),
            initial_state=(-3.4, -8.5, -0.4, 0.0, 0.0, 0.0, pi / 2.0),
            incoming_gate_sides=(-1, -1, -1, -1),
            model_path="npe_models/lemniscate/best_npe.pth",
        ),
    }
)


def get_track(name: str) -> TrackConfig:
    try:
        return TRACKS[name.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(TRACKS))
        raise ValueError(f"Unknown track {name!r}; expected one of: {choices}") from exc
