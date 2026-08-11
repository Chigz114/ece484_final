#!/usr/bin/env python3
"""Fail-closed offline gate before any Crazyflie hardware command is allowed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

from quadpilot.hardware.readiness import check_hardware_readiness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Crazyflie readiness gate")
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    report = check_hardware_readiness(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY_FOR_PROP_OFF_BENCH" else 2


if __name__ == "__main__":
    raise SystemExit(main())
