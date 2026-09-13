"""
agent.py — Smart City Traffic Balancer: Reasoning Engine
=========================================================
Implements multi-factor scoring + strategy selection with explicit reasoning traces.
No hardcoded action-per-threshold — every decision is derived from computed scores.

Second-order effects:
  When a route_notification is issued, diverted traffic is modeled as a density
  increase on neighboring zones (see propagate_spillover). This is a first-order
  approximation — real traffic would require a graph-flow model — but it surfaces
  the core failure mode (local fix → downstream jam) visibly in the dashboard.

Assumption audit:
  Each reasoning trace labels which assumptions are LOAD-BEARING (if false, the
  action is likely ineffective) vs COSMETIC (relaxable without changing the decision).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Constants / tuneable weights
# ---------------------------------------------------------------------------

RUSH_HOUR_WINDOWS = [
    (7, 10),   # morning peak
    (17, 20),  # evening peak
]

FOOTFALL_PRESSURE = {
    "very_high": 1.0,
    "high": 0.75,
    "medium": 0.45,
    "low": 0.2,
    "none": 0.0,
}

TOLL_STEPS = [1.0, 1.25, 1.5, 1.75, 2.0]      # available multipliers
SIGNAL_EXTENSION_S = [10, 15, 20, 30]           # green-phase extension seconds


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Scores:
    """Intermediate factor scores for a zone."""
    congestion_score: float = 0.0
    event_pressure: float = 0.0
    impact_of_diversion: float = 0.0
    time_sensitivity: float = 0.0
    speed_deficit_pct: float = 0.0

    # Strategy fit scores (0-1)
    route_notification_fit: float = 0.0
    toll_adjustment_fit: float = 0.0
    signal_timing_fit: float = 0.0


@dataclass
class ActionPayload:
    action_type: str                    # "route_notification" | "toll_adjustment" | "signal_timing" | "none"
    display_label: str
    details: dict[str, Any] = field(default_factory=dict)
    payload_text: str = ""


@dataclass
class Decision:
    zone_id: str
    timestamp: str
    scores: Scores
    chosen_actions: list[ActionPayload]
    reasoning_trace: str
    overridden: bool = False
    override_reason: str = ""


# ---------------------------------------------------------------------------
# Step 1 — Factor computation
# ---------------------------------------------------------------------------

def _is_rush_hour(dt: datetime) -> bool:
    h = dt.hour
    return any(lo <= h < hi for lo, hi in RUSH_HOUR_WINDOWS)


def compute_congestion_score(zone: dict) -> float:
    """
    Blended congestion: 60% speed deficit + 40% density.
    Returns 0–1.
    """
    speed_ratio = zone["current_speed_kmph"] / max(zone["free_flow_speed_kmph"], 1)
    speed_deficit = 1.0 - min(speed_ratio, 1.0)

    density_score = zone["vehicle_density_pct"] / 100.0

    return round(0.6 * speed_deficit + 0.4 * density_score, 4)


def compute_event_pressure(zone: dict) -> float:
    """
    High when an event starts imminently or is already underway with high footfall.
    Decays as ends_in_min shrinks (event nearly over → low pressure).
    """
    ev = zone.get("nearby_event", {})
    starts = ev.get("starts_in_min", -1)
    ends = ev.get("ends_in_min", -1)
    footfall = ev.get("expected_footfall", "none")

    if starts == -1 or footfall == "none":
        return 0.0

    base = FOOTFALL_PRESSURE.get(footfall, 0.0)

    # Imminence factor (event hasn't started yet)
    if starts > 0:
        # 0→30 min away: high pressure; 30→120 min: moderate; beyond 120: low
        if starts <= 30:
            imminence = 1.0
        elif starts <= 120:
            imminence = 1.0 - (starts - 30) / 90 * 0.5   # decays from 1→0.5
        else:
            imminence = 0.3
    else:
        # Event already started
        # Pressure fades as event ends
        if ends <= 0:
            imminence = 0.0
        elif ends <= 30:
            imminence = 0.3   # almost over
        elif ends <= 90:
            imminence = 0.6
        else:
            imminence = 1.0   # long way to go

    return round(base * imminence, 4)


def compute_impact_of_diversion(zone: dict) -> float:
    """
    How bad is it to divert traffic away from this zone?
    High impact → alt route is nearly full OR drivers hate diversions.
    Returns 0–1 (higher = diversion is more harmful).
    """
    alt_capacity_free = zone["alt_route_capacity_pct"] / 100.0
    alt_congestion = 1.0 - alt_capacity_free          # how full is the alt route
    sensitivity = zone["sensitivity_score"]
    dci = zone["driver_complaint_index"]

    # Weighted blend: 50% alt route fullness, 30% zone sensitivity, 20% complaint history
    return round(0.5 * alt_congestion + 0.3 * sensitivity + 0.2 * dci, 4)


def compute_time_sensitivity(zone: dict, now: datetime) -> float:
    """
    Boosts urgency during rush hour or when an event is about to start.
    """
    rush_boost = 0.3 if _is_rush_hour(now) else 0.0

    ev = zone.get("nearby_event", {})
    starts = ev.get("starts_in_min", -1)
    event_boost = 0.0
    if 0 < starts <= 20:
        event_boost = 0.4    # imminent start — act NOW
    elif 20 < starts <= 60:
        event_boost = 0.25   # act soon, some lead time

    return round(min(rush_boost + event_boost, 1.0), 4)


# ---------------------------------------------------------------------------
# Step 2 — Strategy fit scoring
# ---------------------------------------------------------------------------

def score_route_notification(scores: Scores, zone: dict) -> float:
    """
    Best when: congestion moderate-high, alt route has spare capacity,
    situation isn't so acute that faster relief is needed.
    Penalised when: alt route nearly full, or event already started (committed drivers).
    """
    ev = zone.get("nearby_event", {})
    starts = ev.get("starts_in_min", -1)

    congestion_fit = 0.0
    if 0.45 <= scores.congestion_score <= 0.90:
        congestion_fit = scores.congestion_score
    elif scores.congestion_score > 0.90:
        congestion_fit = 0.5   # too acute; notification alone insufficient

    alt_capacity = zone["alt_route_capacity_pct"] / 100.0
    alt_fit = alt_capacity   # more spare cap → better diversion potential

    # Penalty when event already started (drivers committed to their route)
    event_started = (starts is not None and starts <= 0 and ev.get("expected_footfall", "none") != "none")
    event_penalty = 0.25 if event_started else 0.0

    # Penalty when alt route is dangerously full
    alt_full_penalty = 0.4 if alt_capacity < 0.15 else 0.0

    score = 0.45 * congestion_fit + 0.45 * alt_fit - event_penalty - alt_full_penalty
    return round(max(score, 0.0), 4)


def score_toll_adjustment(scores: Scores, zone: dict) -> float:
    """
    Best when: congestion is building (not yet gridlocked), there's lead time before
    peak (discretionary traffic can reroute in advance), tier supports tolls.
    Poor when: event already started (demand committed), local streets, or incident-blocked.
    """
    ev = zone.get("nearby_event", {})
    starts = ev.get("starts_in_min", -1)
    footfall = ev.get("expected_footfall", "none")

    # Toll is most effective on arterial/highway with discretionary traffic
    tier_fit = {"highway": 1.0, "arterial": 0.7, "local": 0.3}.get(zone["tier"], 0.5)

    # Works best when congestion is building but not yet gridlocked
    cong = scores.congestion_score
    if cong < 0.3:
        cong_fit = 0.2   # no real need
    elif cong < 0.65:
        cong_fit = 0.8   # sweet spot — pre-emptive
    elif cong < 0.85:
        cong_fit = 0.5   # still usable
    else:
        cong_fit = 0.2   # too late for demand management

    # Lead-time bonus: toll works best when there's time to shift demand
    if starts > 30:
        lead_bonus = 0.3
    elif 10 < starts <= 30:
        lead_bonus = 0.1
    else:
        lead_bonus = 0.0   # event started or imminent → drivers committed

    # Penalty for incidents (traffic stuck due to incident, not demand)
    incident_penalty = 0.3 if zone["incident_flag"] not in ("none", "") else 0.0

    score = 0.4 * tier_fit + 0.4 * cong_fit + lead_bonus - incident_penalty
    return round(max(score, 0.0), 4)


def score_signal_timing(scores: Scores, zone: dict) -> float:
    """
    Best when: congestion is acute/local, diversion is not viable (alt route full or
    sensitivity high), need immediate throughput boost.
    Works across all tiers; strong when incident is blocking intersection.
    """
    cong = scores.congestion_score
    diversion_blocked = scores.impact_of_diversion > 0.65  # diversion is harmful
    alt_capacity = zone["alt_route_capacity_pct"] / 100.0

    # Signal timing most effective at high congestion
    if cong > 0.80:
        cong_fit = 1.0
    elif cong > 0.60:
        cong_fit = 0.75
    elif cong > 0.45:
        cong_fit = 0.45
    else:
        cong_fit = 0.1

    # Stronger when diversion is not viable
    necessity_boost = 0.35 if diversion_blocked else 0.1

    # Works on local intersections where signals exist
    tier_fit = {"highway": 0.4, "arterial": 0.8, "local": 0.9}.get(zone["tier"], 0.6)

    # Incident-clearing bonus
    incident_bonus = 0.2 if zone["incident_flag"] not in ("none", "") else 0.0

    score = 0.45 * cong_fit + necessity_boost + 0.25 * tier_fit + incident_bonus
    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Step 3 — Combination logic + action selection
# ---------------------------------------------------------------------------

def _select_actions(scores: Scores, zone: dict) -> list[ActionPayload]:
    """
    Compatibility rules applied on top of fit scores.
    Returns the final action list (may be empty = hold/monitor).
    """
    ev = zone.get("nearby_event", {})
    starts = ev.get("starts_in_min", -1)
    footfall = ev.get("expected_footfall", "none")
    alt_cap = zone["alt_route_capacity_pct"]

    actions: list[ActionPayload] = []

    # ---- Route Notification ----
    if scores.route_notification_fit >= 0.35 and alt_cap >= 20:
        # Build notification payload
        alt_speed_est = int(zone["free_flow_speed_kmph"] * 0.6)
        payload = (
            f"[NOTIFICATION] Zone {zone['zone_id']} — {zone['zone_name']}\n"
            f"  Current avg speed: {zone['current_speed_kmph']} km/h "
            f"(free-flow: {zone['free_flow_speed_kmph']} km/h)\n"
            f"  Congestion: {int(scores.congestion_score*100)}%  |  "
            f"Alt route capacity free: {alt_cap}%\n"
            f"  ➜ Recommend using alternate route. Est. time saving: "
            f"~{max(2, int((zone['free_flow_speed_kmph']-zone['current_speed_kmph'])/zone['free_flow_speed_kmph']*15))} min\n"
            f"  Alternate route avg speed est: {alt_speed_est} km/h"
        )
        actions.append(ActionPayload(
            action_type="route_notification",
            display_label="🔔 Route Notification",
            details={"alt_route_capacity_pct": alt_cap, "est_alt_speed_kmph": alt_speed_est},
            payload_text=payload,
        ))

    # ---- Toll Adjustment ----
    if scores.toll_adjustment_fit >= 0.35 and zone["tier"] in ("arterial", "highway"):
        # Pick multiplier proportional to congestion
        cong = scores.congestion_score
        if cong < 0.5:
            mult = 1.25
        elif cong < 0.7:
            mult = 1.5
        elif cong < 0.85:
            mult = 1.75
        else:
            mult = 2.0
        payload = (
            f"[TOLL ADJUSTMENT] Zone {zone['zone_id']} — {zone['zone_name']}\n"
            f"  Congestion: {int(cong*100)}%  |  Tier: {zone['tier']}\n"
            f"  Previous multiplier: {zone['toll_multiplier']}x  "
            f"→  New multiplier: {mult}x\n"
            f"  Effective immediately. Expected demand reduction: "
            f"~{int((mult-1.0)*25)}% discretionary traffic"
        )
        actions.append(ActionPayload(
            action_type="toll_adjustment",
            display_label="💰 Toll Adjustment",
            details={"new_toll_multiplier": mult, "prev_multiplier": zone["toll_multiplier"]},
            payload_text=payload,
        ))

    # ---- Signal Timing ----
    if scores.signal_timing_fit >= 0.45:
        cong = scores.congestion_score
        # Extension proportional to severity
        if cong > 0.85:
            ext = 30
        elif cong > 0.70:
            ext = 20
        elif cong > 0.55:
            ext = 15
        else:
            ext = 10
        new_plan = f"extended_green_{ext}s"
        payload = (
            f"[SIGNAL TIMING CHANGE] Zone {zone['zone_id']} — {zone['zone_name']}\n"
            f"  Congestion: {int(cong*100)}%  |  Alt route cap: {alt_cap}%  "
            f"(diversion impact: {scores.impact_of_diversion:.2f})\n"
            f"  Current plan: {zone['signal_plan']}  "
            f"→  New plan: {new_plan}\n"
            f"  Green-phase extended by +{ext}s on main approach.\n"
            f"  Est. throughput gain: ~{int(ext * 2.5)} veh/cycle"
        )
        actions.append(ActionPayload(
            action_type="signal_timing",
            display_label="🚦 Signal Timing",
            details={"signal_plan": new_plan, "green_extension_s": ext},
            payload_text=payload,
        ))

    # ---- Compatibility guards (post-selection) ----
    # If notification is in list but alt route is actually full, drop it
    if any(a.action_type == "route_notification" for a in actions) and alt_cap < 20:
        actions = [a for a in actions if a.action_type != "route_notification"]

    # If toll + signal both selected and congestion is truly gridlocked (>90%) → keep signal only
    if (scores.congestion_score > 0.90
            and any(a.action_type == "toll_adjustment" for a in actions)
            and any(a.action_type == "signal_timing" for a in actions)):
        actions = [a for a in actions if a.action_type != "toll_adjustment"]

    # No action if everything below threshold or the zone is healthy
    if not actions:
        actions = [ActionPayload(
            action_type="none",
            display_label="✅ Monitor Only",
            payload_text=(
                f"[HOLD — NO ACTION] Zone {zone['zone_id']} — {zone['zone_name']}\n"
                f"  Congestion {int(scores.congestion_score*100)}% within acceptable bounds "
                f"or contextual factors prevent effective intervention at this time.\n"
                f"  Continue monitoring."
            )
        )]

    return actions


# ---------------------------------------------------------------------------
# Spillover propagation (second-order effects)
# ---------------------------------------------------------------------------

# Compliance rates are the single most load-bearing assumption in the system.
# These values are our modeling choices — conservative, not validated against field data.
# We chose 25% for notifications and 30% for tolls as a deliberate underestimate
# (if anything, real-world compliance would make the actions more effective, not less).
# If compliance is near zero, route_notification and toll_adjustment do nothing —
# that's the falsification condition for those two action types.
COMPLIANCE_RATE = {
    "route_notification": 0.25,   # conservative modeling choice: 1 in 4 drivers reroutes
    "toll_adjustment": 0.30,      # conservative modeling choice: 3 in 10 drivers change route
    "signal_timing": 0.0,         # signals don't rely on driver compliance
}


def propagate_spillover(zones: list[dict], decisions: list) -> list[dict]:
    """
    After all zone decisions are made, push diverted traffic onto neighboring zones.
    This models the second-order effect: fixing zone A may congest zone B.

    For each zone that issued a route_notification:
      - Estimate diverted vehicles = density_pct * compliance_rate * 0.5 (partial load)
      - Split that load equally across neighboring zones
      - Increase their vehicle_density_pct accordingly (capped at 99)
      - Record spillover_received_pct on each affected zone

    NOTE: This is a deliberate simplification. A proper model would use
    O-D matrices and link-flow equations. This surfaces the effect qualitatively.
    """
    zone_by_id = {z["zone_id"]: z for z in zones}

    # Reset spillover from previous tick
    for z in zones:
        z["spillover_received_pct"] = 0

    for decision in decisions:
        if decision is None:
            continue
        action_types = {a.action_type for a in decision.chosen_actions}
        if "route_notification" not in action_types:
            continue

        source_zone = zone_by_id.get(decision.zone_id)
        if not source_zone:
            continue

        neighbors = source_zone.get("neighboring_zones", [])
        if not neighbors:
            continue

        compliance = COMPLIANCE_RATE["route_notification"]
        # Diverted vehicles as % of source zone's current density
        diverted_density = source_zone["vehicle_density_pct"] * compliance
        per_neighbor_load = diverted_density / len(neighbors)

        for nid in neighbors:
            neighbor = zone_by_id.get(nid)
            if not neighbor:
                continue
            # Scale by neighbor's road capacity relative to source
            capacity_ratio = source_zone["road_capacity_vph"] / max(neighbor["road_capacity_vph"], 1)
            load = round(per_neighbor_load * capacity_ratio * 0.5, 1)  # 0.5 = partial absorption
            neighbor["vehicle_density_pct"] = int(min(99, neighbor["vehicle_density_pct"] + load))
            neighbor["spillover_received_pct"] = round(
                neighbor.get("spillover_received_pct", 0) + load, 1
            )
            # Spill slightly degrades speed too
            speed_loss = load * 0.08
            neighbor["current_speed_kmph"] = round(
                max(4, neighbor["current_speed_kmph"] - speed_loss), 1
            )
            # Reduce alt_route_capacity_pct of the neighbor (it's absorbing traffic)
            neighbor["alt_route_capacity_pct"] = max(
                0, neighbor["alt_route_capacity_pct"] - int(load * 0.4)
            )

    return zones


# ---------------------------------------------------------------------------
# Step 4 — Reasoning trace builder
# ---------------------------------------------------------------------------

def _build_reasoning_trace(zone: dict, scores: Scores, actions: list[ActionPayload], now: datetime) -> str:
    ev = zone.get("nearby_event", {})
    ev_name = ev.get("name", "None")
    starts = ev.get("starts_in_min", -1)
    ends = ev.get("ends_in_min", -1)
    footfall = ev.get("expected_footfall", "none")

    # Describe event status
    if starts is None or starts == -1 or footfall == "none":
        ev_desc = "No nearby event."
    elif starts > 0:
        ev_desc = f"'{ev_name}' starts in {starts} min (footfall: {footfall})."
    elif ends > 0:
        ev_desc = f"'{ev_name}' is ONGOING, ends in {ends} min (footfall: {footfall})."
    else:
        ev_desc = f"'{ev_name}' has ended."

    speed_deficit_pct = round((1 - zone["current_speed_kmph"] / max(zone["free_flow_speed_kmph"], 1)) * 100, 1)
    hist_avg = zone.get("historical_avg_speed", zone["free_flow_speed_kmph"])

    # ---- Baseline: what congestion looks like without any intervention ----
    baseline = zone.get("baseline_congestion")
    baseline_str = f"{int(baseline*100)}%" if baseline is not None else "(not yet measured)"
    if baseline is not None:
        delta = scores.congestion_score - baseline
        delta_str = (f"+{int(delta*100)}pp worse" if delta > 0.01
                     else f"{abs(int(delta*100))}pp better" if delta < -0.01
                     else "unchanged")
    else:
        delta_str = "—"

    # ---- Spillover received from upstream diversions ----
    spillover = zone.get("spillover_received_pct", 0)

    lines = [
        f"━━━ ZONE {zone['zone_id']} — {zone['zone_name']} ━━━",
        f"",
        f"📊 FACTOR ANALYSIS",
        f"  • Speed: {zone['current_speed_kmph']} km/h vs free-flow {zone['free_flow_speed_kmph']} km/h "
        f"(−{speed_deficit_pct}% deficit | hist. avg: {hist_avg} km/h)",
        f"  • Density: {zone['vehicle_density_pct']}% of road capacity",
        f"  • Congestion Score: {scores.congestion_score:.3f} "
        f"({'SEVERE' if scores.congestion_score>0.80 else 'HIGH' if scores.congestion_score>0.60 else 'MODERATE' if scores.congestion_score>0.40 else 'LOW'})",
        f"  • Event Pressure: {scores.event_pressure:.3f}  →  {ev_desc}",
        f"  • Impact of Diversion: {scores.impact_of_diversion:.3f} "
        f"(alt-route free: {zone['alt_route_capacity_pct']}% | sensitivity: {zone['sensitivity_score']} | DCI: {zone['driver_complaint_index']})",
        f"  • Time Sensitivity: {scores.time_sensitivity:.3f} "
        f"({'rush-hour' if _is_rush_hour(now) else 'off-peak'} + event timing boost)",
        f"  • Incident: {zone['incident_flag']} | Construction: {zone['construction_active']} | Tier: {zone['tier']}",
        f"  • Baseline congestion (pre-agent): {baseline_str}  |  vs now: {delta_str}",
        f"  • Spillover received from upstream diversions: +{spillover}% density",
        f"",
        f"📋 STRATEGY FIT SCORES",
        f"  • Route Notification :  {scores.route_notification_fit:.3f}",
        f"  • Toll Adjustment    :  {scores.toll_adjustment_fit:.3f}",
        f"  • Signal Timing      :  {scores.signal_timing_fit:.3f}",
        f"",
    ]

    # Explain each strategy's outcome (why selected or rejected)
    def _why_not_notification():
        if zone["alt_route_capacity_pct"] < 20:
            return f"alt route only {zone['alt_route_capacity_pct']}% free — diverting would shift the jam, not resolve it"
        ev = zone.get("nearby_event", {})
        if (ev.get("starts_in_min", -1) <= 0 and ev.get("expected_footfall", "none") != "none"):
            return "event already started — drivers committed to current route, notification impact near-zero"
        return f"fit score {scores.route_notification_fit:.3f} below threshold"

    def _why_not_toll():
        if zone["tier"] == "local":
            return "local street — toll adjustment not applicable"
        ev = zone.get("nearby_event", {})
        if ev.get("starts_in_min", -1) <= 10 and ev.get("starts_in_min", -1) > -1:
            return "event imminent — drivers already committed, demand-side nudge too late"
        if zone["incident_flag"] not in ("none", ""):
            return f"incident ({zone['incident_flag']}) causing jam, not excess demand — toll ineffective"
        return f"fit score {scores.toll_adjustment_fit:.3f} below threshold"

    def _why_not_signal():
        return f"fit score {scores.signal_timing_fit:.3f} below threshold (congestion not acute enough for signal change)"

    action_types = {a.action_type for a in actions}

    lines.append("🔍 DECISION RATIONALE")
    if "route_notification" in action_types:
        lines.append(f"  ✅ ROUTE NOTIFICATION selected: congestion {int(scores.congestion_score*100)}%, alt route has {zone['alt_route_capacity_pct']}% spare capacity — diversion viable and worthwhile.")
    else:
        lines.append(f"  ❌ Route Notification skipped: {_why_not_notification()}.")

    if "toll_adjustment" in action_types:
        lines.append(f"  ✅ TOLL ADJUSTMENT selected: tier '{zone['tier']}', congestion building ({int(scores.congestion_score*100)}%), lead time available — demand management effective now.")
    else:
        lines.append(f"  ❌ Toll Adjustment skipped: {_why_not_toll()}.")

    if "signal_timing" in action_types:
        ext_s = next((a.details.get("green_extension_s", "?") for a in actions if a.action_type == "signal_timing"), "?")
        lines.append(f"  ✅ SIGNAL TIMING selected: acute congestion, diversion impact {scores.impact_of_diversion:.2f} (high) → extend green +{ext_s}s to push throughput.")
    else:
        lines.append(f"  ❌ Signal Timing skipped: {_why_not_signal()}.")

    if action_types == {"none"}:
        lines.append(f"  ⏸  HOLD/MONITOR: No intervention meets threshold. "
                     f"Congestion {int(scores.congestion_score*100)}% is manageable or contextual factors "
                     f"reduce effectiveness of all strategies.")

    # ---- Assumption audit ----
    lines += [
        "",
        "⚠️  ASSUMPTION AUDIT",
        "  LOAD-BEARING (if false → action is likely ineffective):",
        f"    [LB-1] Drivers comply with rerouting at ~25% rate "
        f"(conservative modeling choice, not a validated field number — "
        f"{'assumed viable' if zone['alt_route_capacity_pct'] >= 20 else 'MOOT anyway — alt route too full to absorb diverted traffic'})",
        f"    [LB-2] Event timing data is accurate (starts_in_min ±5 min acceptable)",
        f"    [LB-3] Toll elasticity applies — discretionary trips on '{zone['tier']}' tier "
        f"({'plausible' if zone['tier'] in ('arterial','highway') else 'weak — local streets have low discretionary traffic'})",
        "  COSMETIC (relaxable without changing the core decision):",
        "    [C-1] Exact scoring weights (0.6/0.4 blend) — directionally stable ±20%",
        "    [C-2] Historical avg speed as baseline — imprecise but sufficient for delta",
        "    [C-3] Spillover load split equally across neighbors — real split depends on O-D matrix",
        "",
        "  WHAT WOULD FALSIFY THIS DECISION:",
    ]
    for action in actions:
        if action.action_type == "route_notification":
            lines.append(
                f"    → If alt route fills up before diverted drivers arrive, "
                f"notification made things worse (monitor {', '.join(zone.get('neighboring_zones',[]))} next tick)"
            )
        elif action.action_type == "toll_adjustment":
            lines.append(
                f"    → If congestion score rises next tick despite toll increase, "
                f"demand is inelastic here and toll should be reversed"
            )
        elif action.action_type == "signal_timing":
            lines.append(
                f"    → If density_pct doesn't drop within 2 ticks, "
                f"signal change is insufficient and incident clearance is needed"
            )
        elif action.action_type == "none":
            lines.append(
                f"    → If congestion_score exceeds 0.75 next tick without intervention, "
                f"this hold decision was wrong"
            )

    lines += ["", f"⏰ Evaluated at: {now.strftime('%H:%M:%S')} | Zone tier: {zone['tier']}"]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_zone(zone: dict, now: datetime | None = None) -> Decision:
    """
    Run the full agent reasoning pipeline for one zone.
    Returns a Decision object with scores, actions, and reasoning trace.
    """
    if now is None:
        now = datetime.now()

    scores = Scores()
    scores.congestion_score = compute_congestion_score(zone)
    scores.event_pressure = compute_event_pressure(zone)
    scores.impact_of_diversion = compute_impact_of_diversion(zone)
    scores.time_sensitivity = compute_time_sensitivity(zone, now)

    scores.route_notification_fit = score_route_notification(scores, zone)
    scores.toll_adjustment_fit = score_toll_adjustment(scores, zone)
    scores.signal_timing_fit = score_signal_timing(scores, zone)

    actions = _select_actions(scores, zone)
    reasoning = _build_reasoning_trace(zone, scores, actions, now)

    return Decision(
        zone_id=zone["zone_id"],
        timestamp=now.isoformat(timespec="seconds"),
        scores=scores,
        chosen_actions=actions,
        reasoning_trace=reasoning,
    )


def apply_decision_to_zone(zone: dict, decision: Decision) -> dict:
    """
    Mutate zone record in place based on chosen actions.
    Returns the updated zone dict.
    """
    zone = dict(zone)   # shallow copy
    action_labels = []

    for action in decision.chosen_actions:
        if action.action_type == "toll_adjustment":
            zone["toll_multiplier"] = action.details.get("new_toll_multiplier", zone["toll_multiplier"])
        elif action.action_type == "signal_timing":
            zone["signal_plan"] = action.details.get("signal_plan", zone["signal_plan"])
        if action.action_type != "none":
            action_labels.append(action.display_label)

    zone["last_action"] = ", ".join(action_labels) if action_labels else "Monitor Only"
    zone["last_action_reason"] = decision.reasoning_trace[:200] + "…"
    zone["last_updated"] = decision.timestamp
    return zone
