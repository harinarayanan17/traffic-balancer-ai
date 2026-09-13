"""
feed.py — Synthetic Traffic Feed Generator
===========================================
Advances zone state each tick with controlled randomness + scripted scenarios.
"""

from __future__ import annotations

import math
import random
from datetime import datetime


# ---------------------------------------------------------------------------
# Scripted scenario seeds (guaranteed interesting outputs for the live demo)
# ---------------------------------------------------------------------------

SCRIPTED_SCENARIOS: dict[str, list[dict]] = {
    # Z01: accident clears, congestion eases
    "Z01": [
        {"current_speed_kmph": 8, "vehicle_density_pct": 91, "incident_flag": "accident"},
        {"current_speed_kmph": 12, "vehicle_density_pct": 85, "incident_flag": "accident"},
        {"current_speed_kmph": 20, "vehicle_density_pct": 70, "incident_flag": "none"},
        {"current_speed_kmph": 32, "vehicle_density_pct": 52, "incident_flag": "none"},
    ],
    # Z05: Silk Board — chronic gridlock, then slowly clears
    "Z05": [
        {"current_speed_kmph": 5, "vehicle_density_pct": 95, "incident_flag": "breakdown"},
        {"current_speed_kmph": 7, "vehicle_density_pct": 90, "incident_flag": "breakdown"},
        {"current_speed_kmph": 10, "vehicle_density_pct": 82, "incident_flag": "none"},
        {"current_speed_kmph": 16, "vehicle_density_pct": 70, "incident_flag": "none"},
    ],
    # Z20: Republic Day — acute peak then slow release
    "Z20": [
        {"current_speed_kmph": 7, "vehicle_density_pct": 93, "incident_flag": "none"},
        {"current_speed_kmph": 5, "vehicle_density_pct": 97, "incident_flag": "none"},
        {"current_speed_kmph": 8, "vehicle_density_pct": 88, "incident_flag": "none"},
        {"current_speed_kmph": 15, "vehicle_density_pct": 72, "incident_flag": "none"},
    ],
}

_scenario_tick: dict[str, int] = {}   # per-zone tick counter for scripted scenarios


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _advance_event(ev: dict) -> dict:
    """Decrement event timers each tick (1 tick ≈ some minutes in sim time)."""
    if ev.get("expected_footfall", "none") == "none":
        return ev
    ev = dict(ev)
    tick_min = random.randint(5, 12)   # each tick = 5-12 sim-minutes
    if ev["starts_in_min"] > 0:
        ev["starts_in_min"] = max(0, ev["starts_in_min"] - tick_min)
    else:
        ev["ends_in_min"] = max(-1, ev["ends_in_min"] - tick_min)
    return ev


def simulate_tick(zones: list[dict]) -> list[dict]:
    """
    Advance all zones by one tick.
    Scripted zones follow their scenario; others get plausible random drift.
    Returns a new list of updated zone dicts.
    """
    updated = []
    for zone in zones:
        zone = dict(zone)
        zid = zone["zone_id"]

        if zid in SCRIPTED_SCENARIOS:
            tick_idx = _scenario_tick.get(zid, 0)
            scenario = SCRIPTED_SCENARIOS[zid]
            if tick_idx < len(scenario):
                patch = scenario[tick_idx]
                zone.update(patch)
            _scenario_tick[zid] = tick_idx + 1

        else:
            # Drift: speed moves ±10-20% with regression toward historical avg
            hist = zone.get("historical_avg_speed", zone["free_flow_speed_kmph"] * 0.7)
            cur = zone["current_speed_kmph"]
            # Mean-reverting random walk
            drift = random.uniform(-6, 6)
            reversion = (hist - cur) * 0.12
            new_speed = _clamp(cur + drift + reversion, 4, zone["free_flow_speed_kmph"])
            zone["current_speed_kmph"] = round(new_speed, 1)

            # Density inversely correlated with speed + noise
            speed_ratio = new_speed / zone["free_flow_speed_kmph"]
            base_density = (1.0 - speed_ratio) * 90 + speed_ratio * 20
            noise = random.uniform(-5, 5)
            zone["vehicle_density_pct"] = int(_clamp(base_density + noise, 10, 99))

        # Advance event timers
        zone["nearby_event"] = _advance_event(zone.get("nearby_event", {}))
        zone["last_updated"] = datetime.now().isoformat(timespec="seconds")

        updated.append(zone)

    return updated


def reset_scenario_counters():
    """Call when resetting to seed data."""
    _scenario_tick.clear()
