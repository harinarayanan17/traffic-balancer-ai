"""
app.py — Smart City Traffic Balancer
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

def clean_html(html: str) -> str:
    return "\n".join(line.strip() for line in html.strip().split("\n"))

from agent import (
    evaluate_zone,
    apply_decision_to_zone,
    compute_congestion_score,
    propagate_spillover,
)
from feed import simulate_tick, reset_scenario_counters

ROOT      = Path(__file__).parent
SEED_FILE = ROOT / "data" / "zones_seed.json"
STATE_FILE= ROOT / "data" / "zones_live.json"

st.set_page_config(page_title="City Traffic Control", page_icon="🚦",
                   layout="wide", initial_sidebar_state="expanded")

# ── Design system ────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #0c0e14;
  --surface:   #131720;
  --elevated:  #1b2035;
  --border:    #252b40;
  --border-sub:#1a1f30;
  --text:      #e2e8f6;
  --text-2:    #8892aa;
  --text-3:    #434c66;
  --accent:    #6366f1;
  --accent-dim:#6366f118;
  --severe:    #ef4444;
  --high:      #f97316;
  --moderate:  #eab308;
  --low:       #22c55e;
  --severe-bg: #ef444412;
  --high-bg:   #f9731612;
  --moderate-bg:#eab30812;
  --low-bg:    #22c55e12;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--text);
}
[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
.block-container { padding: 1.5rem 2rem 3rem; }
h1,h2,h3,h4 { font-family: 'Inter', sans-serif; font-weight: 600; }

/* ── KPI grid ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 28px;
}
.kpi {
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: 2px solid var(--accent);
  border-radius: 10px;
  padding: 18px 20px 16px;
}
.kpi-label  { font-size: 10px; font-weight: 600; letter-spacing: .08em;
               text-transform: uppercase; color: var(--text-3); margin-bottom: 8px; }
.kpi-value  { font-size: 2rem; font-weight: 700; line-height: 1; }
.kpi-sub    { font-size: 11px; color: var(--text-3); margin-top: 6px; }

/* ── Zone table ── */
.zone-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.zone-table th {
  font-size: 10px; font-weight: 600; letter-spacing: .07em;
  text-transform: uppercase; color: var(--text-3);
  padding: 0 12px 10px; text-align: left; border-bottom: 1px solid var(--border);
}
.zone-table td {
  padding: 10px 12px; vertical-align: middle;
  border-bottom: 1px solid var(--border-sub);
}
.zone-table tr:last-child td { border-bottom: none; }
.zone-table tr:hover td { background: var(--elevated); }

.zone-id   { font-family: 'JetBrains Mono', monospace; font-size: 12px;
              font-weight: 500; color: var(--text-2); }
.zone-name { font-weight: 500; color: var(--text); }
.zone-tier { font-size: 11px; color: var(--text-3); }

/* Severity left border */
.sev-severe   td:first-child { box-shadow: inset 3px 0 0 var(--severe); }
.sev-high     td:first-child { box-shadow: inset 3px 0 0 var(--high); }
.sev-moderate td:first-child { box-shadow: inset 3px 0 0 var(--moderate); }
.sev-low      td:first-child { box-shadow: inset 3px 0 0 var(--low); }

/* Congestion bar */
.cong-wrap  { display: flex; align-items: center; gap: 10px; }
.cong-bar   { flex: 1; height: 4px; background: var(--border); border-radius: 2px; min-width: 80px; }
.cong-fill  { height: 4px; border-radius: 2px; transition: width .3s; }
.cong-pct   { font-size: 12px; font-weight: 600; min-width: 32px; text-align: right;
               font-family: 'JetBrains Mono', monospace; }

/* Speed display */
.speed-wrap { display: flex; align-items: center; gap: 8px; }
.speed-val  { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text); }
.speed-ff   { font-size: 11px; color: var(--text-3); }

/* Action chips */
.chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip  { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 11px; font-weight: 500; white-space: nowrap; }
.chip-notify   { background: #1e3a5f; color: #93c5fd; }
.chip-toll     { background: #3d2e0e; color: #fcd34d; }
.chip-signal   { background: #14321e; color: #6ee7b7; }
.chip-monitor  { background: var(--elevated); color: var(--text-3); }
.chip-override { background: #3d1c0e; color: #fb923c; }

/* Spillover pill */
.spill { display: inline-block; padding: 1px 7px; border-radius: 4px;
         font-size: 11px; font-weight: 500;
         background: #f9731614; color: var(--high); border: 1px solid #f9731630; }

/* Event text */
.ev-upcoming { color: var(--moderate); font-size: 12px; }
.ev-ongoing  { color: var(--high); font-size: 12px; }
.ev-none     { color: var(--text-3); font-size: 12px; }

/* ── Analysis tab ── */
.metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0 24px; }
.metric-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px 18px;
}
.metric-card .m-label { font-size: 10px; font-weight: 600; letter-spacing: .07em;
                         text-transform: uppercase; color: var(--text-3); margin-bottom: 6px; }
.metric-card .m-value { font-size: 1.5rem; font-weight: 700; }
.metric-card .m-delta { font-size: 11px; margin-top: 4px; }
.delta-up   { color: var(--severe); }
.delta-down { color: var(--low); }
.delta-flat { color: var(--text-3); }

/* ── Reasoning sections ── */
.reason-block {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 16px 18px; margin-bottom: 10px;
}
.reason-block .r-title {
  font-size: 10px; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--text-3); margin-bottom: 10px;
}
.reason-block pre {
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  line-height: 1.7; color: var(--text-2); margin: 0; white-space: pre-wrap;
}

/* ── History table ── */
.hist-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.hist-table th {
  font-size: 10px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
  color: var(--text-3); padding: 0 10px 8px; text-align: left;
  border-bottom: 1px solid var(--border);
}
.hist-table td { padding: 8px 10px; border-bottom: 1px solid var(--border-sub);
                  vertical-align: middle; }
.hist-table tr:last-child td { border-bottom: none; }
.hist-table tr:hover td { background: var(--elevated); }

/* Streamlit overrides */
[data-testid="stTabs"] button {
  font-size: 13px; font-weight: 500;
  color: var(--text-3); border-radius: 0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--text); border-bottom-color: var(--accent);
}
[data-testid="stSidebar"] button[kind="primary"] {
  background: var(--accent); border: none; font-weight: 600;
  font-size: 13px; width: 100%; padding: 10px;
  border-radius: 6px; letter-spacing: .02em;
}
[data-testid="stSidebar"] .stSlider { padding: 0; }
div[data-testid="stExpander"] > div { background: var(--surface); }
</style>
""", unsafe_allow_html=True)

# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_seed():
    with open(SEED_FILE) as f: return json.load(f)

def load_live():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f: return json.load(f)
    return load_seed()

def save_live(zones):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f: json.dump(zones, f, indent=2)

def init_state():
    if "zones"    not in st.session_state: st.session_state.zones     = load_live()
    if "decisions"not in st.session_state: st.session_state.decisions = []
    if "tick"     not in st.session_state: st.session_state.tick      = 0
    if "selected" not in st.session_state: st.session_state.selected  = None

# ── Visual helpers ────────────────────────────────────────────────────────────

def sev_class(s):
    if s > .80: return "sev-severe"
    if s > .60: return "sev-high"
    if s > .40: return "sev-moderate"
    return "sev-low"

def sev_color(s):
    if s > .80: return "var(--severe)"
    if s > .60: return "var(--high)"
    if s > .40: return "var(--moderate)"
    return "var(--low)"

def sev_label(s):
    if s > .80: return "Severe"
    if s > .60: return "High"
    if s > .40: return "Moderate"
    return "Low"

def cong_bar_html(pct, color):
    return (f'<div class="cong-wrap">'
            f'<div class="cong-bar"><div class="cong-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="cong-pct" style="color:{color}">{pct}%</span>'
            f'</div>')

def action_chips_html(last_action):
    if not last_action or last_action == "Monitor Only":
        return '<div class="chips"><span class="chip chip-monitor">Monitor</span></div>'
    chips = []
    if "Notification" in last_action: chips.append('<span class="chip chip-notify">Notification</span>')
    if "Toll"         in last_action: chips.append('<span class="chip chip-toll">Toll Adj.</span>')
    if "Signal"       in last_action: chips.append('<span class="chip chip-signal">Signal</span>')
    if "Override"     in last_action or "Overridden" in last_action:
        chips.append('<span class="chip chip-override">Overridden</span>')
    return f'<div class="chips">{"".join(chips) or chr(8212)}</div>'

def event_html(ev):
    name    = ev.get("name", "None")
    starts  = ev.get("starts_in_min", -1)
    ends    = ev.get("ends_in_min", -1)
    foot    = ev.get("expected_footfall", "none")
    if not name or name == "None" or foot == "none":
        return '<span class="ev-none">—</span>'
    if starts and starts > 0:
        return f'<span class="ev-upcoming">{name[:22]} &nbsp;·&nbsp; in {starts} min</span>'
    if ends and ends > 0:
        return f'<span class="ev-ongoing">{name[:22]} &nbsp;·&nbsp; ongoing</span>'
    return f'<span class="ev-none">{name[:22]} · ended</span>'

# ── Agent pass ────────────────────────────────────────────────────────────────

def run_agent_pass(zones):
    now = datetime.now()
    all_decisions = []

    for z in zones:
        z["baseline_congestion"] = compute_congestion_score(z)

    for z in zones:
        dec = evaluate_zone(z, now=now)
        all_decisions.append(dec if not z.get("override_active") else None)

    zones = propagate_spillover(list(zones), all_decisions)

    updated, log = [], []
    for z, dec in zip(zones, all_decisions):
        if z.get("override_active") or dec is None:
            if dec is None:
                dec = evaluate_zone(z, now=now)
            updated.append(z)
        else:
            updated.append(apply_decision_to_zone(z, dec))

        cong = compute_congestion_score(z)
        action_label = ("Overridden" if (z.get("override_active") or dec is None)
                        else ", ".join(a.display_label for a in dec.chosen_actions))

        log.append({
            "tick":        st.session_state.tick,
            "ts":          now.isoformat(timespec="seconds"),
            "zone_id":     z["zone_id"],
            "zone_name":   z["zone_name"],
            "cong_pct":    int(cong * 100),
            "cong_score":  cong,
            "baseline_pct":int((z.get("baseline_congestion") or cong) * 100),
            "spill_pct":   z.get("spillover_received_pct", 0),
            "action":      action_label,
            "override":    z.get("override_active", False),
            "override_reason": z.get("override_reason", ""),
            "reasoning":   dec.reasoning_trace,
            "scores": {
                "congestion":             dec.scores.congestion_score,
                "event_pressure":         dec.scores.event_pressure,
                "diversion_impact":       dec.scores.impact_of_diversion,
                "time_sensitivity":       dec.scores.time_sensitivity,
                "route_notification_fit": dec.scores.route_notification_fit,
                "toll_adjustment_fit":    dec.scores.toll_adjustment_fit,
                "signal_timing_fit":      dec.scores.signal_timing_fit,
            },
            "payloads": [
                {"type": a.action_type, "label": a.display_label, "text": a.payload_text}
                for a in dec.chosen_actions
            ],
        })

    return updated, log

# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown("### City Traffic Control")
    st.sidebar.caption("Autonomous congestion management · 20 zones")
    st.sidebar.markdown("---")

    tick_btn  = st.sidebar.button("Simulate Tick", type="primary", use_container_width=True)
    st.sidebar.markdown("")
    reset_btn = st.sidebar.button("Reset to seed data", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filter**")
    tiers = st.sidebar.multiselect(
        "Tier", ["arterial", "highway", "local"],
        default=["arterial", "highway", "local"],
        label_visibility="collapsed",
    )
    min_c = st.sidebar.slider("Min congestion %", 0, 100, 0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Tick **{st.session_state.tick}** &nbsp;·&nbsp; "
                        f"{len(st.session_state.decisions)} decisions", unsafe_allow_html=True)

    return tick_btn, reset_btn, tiers, min_c

# ── KPI row ───────────────────────────────────────────────────────────────────

def render_kpis(zones):
    scores  = [compute_congestion_score(z) for z in zones]
    severe  = sum(1 for s in scores if s > .80)
    high    = sum(1 for s in scores if .60 < s <= .80)
    avg_c   = sum(scores) / len(scores) * 100 if scores else 0
    active  = sum(1 for z in zones
                  if z.get("last_action") and z["last_action"] not in ("Monitor Only", None))

    st.markdown(clean_html(f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Severe zones</div>
        <div class="kpi-value" style="color:var(--severe)">{severe}</div>
        <div class="kpi-sub">Congestion above 80%</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">High zones</div>
        <div class="kpi-value" style="color:var(--high)">{high}</div>
        <div class="kpi-sub">Congestion 60 – 80%</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Average congestion</div>
        <div class="kpi-value" style="color:var(--accent)">{avg_c:.1f}%</div>
        <div class="kpi-sub">Across all 20 zones</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Active actions</div>
        <div class="kpi-value" style="color:var(--low)">{active}</div>
        <div class="kpi-sub">Agent intervening now</div>
      </div>
    </div>
    """), unsafe_allow_html=True)

# ── Tab 1 — Live Overview ─────────────────────────────────────────────────────

def render_overview(zones, tiers, min_c):
    filtered = sorted(
        [z for z in zones
         if z["tier"] in tiers
         and compute_congestion_score(z) * 100 >= min_c],
        key=compute_congestion_score, reverse=True
    )

    # Congestion bar chart
    chart_df = pd.DataFrame({
        "Congestion %": [round(compute_congestion_score(z) * 100, 1) for z in filtered]
    }, index=[z["zone_id"] for z in filtered])
    st.markdown("**Congestion distribution**")
    st.bar_chart(chart_df, color="#6366f1", height=180)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if not filtered:
        st.info("No zones match the current filter.")
        return

    # Build the HTML zone table
    rows = []
    for z in filtered:
        cong  = compute_congestion_score(z)
        pct   = int(cong * 100)
        color = sev_color(cong)
        cls   = sev_class(cong)
        spill = z.get("spillover_received_pct", 0)
        spill_html = (f'<span class="spill">+{spill}%</span>' if spill > 2 else "")

        rows.append(f"""
        <tr class="{cls}">
          <td>
            <div class="zone-id">{z['zone_id']}</div>
            <div class="zone-tier">{z['tier']}</div>
          </td>
          <td><div class="zone-name">{z['zone_name']}</div></td>
          <td>{cong_bar_html(pct, color)}</td>
          <td>
            <div class="speed-wrap">
              <span class="speed-val">{z['current_speed_kmph']}</span>
              <span class="speed-ff">/ {z['free_flow_speed_kmph']} km/h</span>
            </div>
          </td>
          <td>{event_html(z.get('nearby_event', {}))}</td>
          <td>
            {action_chips_html(z.get('last_action'))}
            {spill_html}
          </td>
        </tr>""")

    table_html = f"""
    <table class="zone-table">
      <thead><tr>
        <th>Zone</th>
        <th>Name</th>
        <th style="min-width:160px">Congestion</th>
        <th>Speed</th>
        <th>Event</th>
        <th>Action</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""

    st.markdown(clean_html(table_html), unsafe_allow_html=True)

    # Zone selector below the table (can't mix buttons inside HTML)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.caption("To inspect a zone's reasoning, go to the Zone Analysis tab.")

# ── Tab 2 — Zone Analysis ─────────────────────────────────────────────────────

def render_analysis(zones):
    sorted_ids   = [z["zone_id"] for z in sorted(zones, key=compute_congestion_score, reverse=True)]
    name_map     = {z["zone_id"]: z["zone_name"] for z in zones}

    sel = st.selectbox(
        "Zone",
        sorted_ids,
        index=sorted_ids.index(st.session_state.selected) if st.session_state.selected in sorted_ids else 0,
        format_func=lambda zid: f"{zid}  ·  {name_map[zid]}",
        label_visibility="collapsed",
    )
    st.session_state.selected = sel
    zone = next((z for z in zones if z["zone_id"] == sel), None)
    if not zone:
        return

    log = next((e for e in reversed(st.session_state.decisions) if e["zone_id"] == sel), None)

    if not log:
        st.info("Run at least one tick to see analysis for this zone.")
        return

    sc       = log["scores"]
    baseline = log.get("baseline_pct", log["cong_pct"])
    current  = log["cong_pct"]
    spill    = log.get("spill_pct", 0)
    delta    = current - baseline

    if delta > 1:
        delta_cls, delta_str = "delta-up",   f"+{delta}pp"
    elif delta < -1:
        delta_cls, delta_str = "delta-down", f"{delta}pp"
    else:
        delta_cls, delta_str = "delta-flat", "unchanged"

    st.markdown(clean_html(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="m-label">Baseline congestion</div>
        <div class="m-value" style="color:var(--text)">{baseline}%</div>
        <div class="m-delta" style="color:var(--text-3)">Before agent acted</div>
      </div>
      <div class="metric-card">
        <div class="m-label">After agent</div>
        <div class="m-value" style="color:{sev_color(current/100)}">{current}%</div>
        <div class="m-delta {delta_cls}">{delta_str}</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Spillover absorbed</div>
        <div class="m-value" style="color:{'var(--high)' if spill > 3 else 'var(--text)'}">{spill}%</div>
        <div class="m-delta" style="color:var(--text-3)">From upstream diversions</div>
      </div>
    </div>
    """), unsafe_allow_html=True)

    if spill > 3:
        st.warning(f"This zone absorbed **+{spill}%** extra density from upstream diversions. "
                   f"Monitor neighboring zones next tick.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Factor scores**")
        st.bar_chart(pd.DataFrame({
            "Score": [sc["congestion"], sc["event_pressure"],
                      sc["diversion_impact"], sc["time_sensitivity"]]
        }, index=["Congestion", "Event Pressure", "Diversion Impact", "Time Sensitivity"]),
        color="#6366f1", height=220)

    with col_r:
        st.markdown("**Strategy fit**")
        st.bar_chart(pd.DataFrame({
            "Fit": [sc["route_notification_fit"],
                    sc["toll_adjustment_fit"],
                    sc["signal_timing_fit"]]
        }, index=["Notification", "Toll Adj.", "Signal Timing"]),
        color="#22c55e", height=220)

    # Action payloads
    if log["payloads"]:
        st.markdown("**Actions taken**")
        for p in log["payloads"]:
            if p["type"] != "none":
                with st.expander(p["label"]):
                    st.code(p["text"], language="text")
            else:
                st.markdown(
                    '<div class="reason-block"><div class="r-title">No intervention</div>'
                    '<pre>Congestion within acceptable bounds or no effective strategy available.</pre>'
                    '</div>', unsafe_allow_html=True
                )

    # Reasoning sections
    st.markdown("**Reasoning**")
    raw = log["reasoning"]
    sections = [
        ("Decision rationale",    "DECISION RATIONALE"),
        ("Factor analysis",       "FACTOR ANALYSIS"),
        ("Strategy fit scores",   "STRATEGY FIT SCORES"),
        ("Assumption audit",      "ASSUMPTION AUDIT"),
        ("Falsification",         "WHAT WOULD FALSIFY"),
    ]
    for title, heading in sections:
        content = _extract(raw, heading)
        if content:
            with st.expander(title, expanded=(heading == "DECISION RATIONALE")):
                st.markdown(
                    f'<div class="reason-block"><pre>{content}</pre></div>',
                    unsafe_allow_html=True
                )

    # Override
    st.markdown("---")
    st.markdown("**Operator override**")
    if zone.get("override_active"):
        st.error(f"Override active — {zone.get('override_reason', '')}")
        if st.button("Clear override"):
            for z in st.session_state.zones:
                if z["zone_id"] == sel:
                    z["override_active"] = False
                    z["override_reason"] = None
                    break
            save_live(st.session_state.zones)
            st.rerun()
    else:
        c1, c2, c3 = st.columns([2, 1.2, 0.8])
        reason = c1.text_input("Reason", placeholder="e.g. Officer on-site", label_visibility="collapsed")
        action = c2.selectbox("Action", ["— select —", "Monitor Only", "Route Notification",
                                          "Signal Timing", "Toll Adjustment"],
                              label_visibility="collapsed")
        if c3.button("Apply", type="primary"):
            if action != "— select —":
                for z in st.session_state.zones:
                    if z["zone_id"] == sel:
                        z["override_active"]  = True
                        z["override_reason"]  = f"{reason} · Forced: {action}"
                        z["last_action"]      = action
                        z["last_updated"]     = datetime.now().isoformat(timespec="seconds")
                        break
                save_live(st.session_state.zones)
                st.rerun()
            else:
                st.error("Select an action.")


def _extract(text, heading):
    known = ["FACTOR ANALYSIS", "STRATEGY FIT SCORES", "DECISION RATIONALE",
             "ASSUMPTION AUDIT", "WHAT WOULD FALSIFY", "Evaluated at"]
    capturing, result = False, []
    for line in text.split("\n"):
        if heading in line:
            capturing = True
            continue
        if capturing:
            if any(h in line for h in known) and line.strip():
                break
            result.append(line)
    return "\n".join(result).strip()


# ── Tab 3 — History ───────────────────────────────────────────────────────────

def render_history():
    if not st.session_state.decisions:
        st.info("No decisions yet.")
        return

    recent = list(reversed(st.session_state.decisions[-100:]))

    rows = []
    for e in recent:
        color = sev_color(e["cong_score"])
        ov    = ('<span style="color:var(--high);font-size:11px">Yes</span>'
                 if e.get("override") else "—")
        rows.append(f"""
        <tr>
          <td style="color:var(--text-3)">{e['tick']}</td>
          <td style="color:var(--text-3);font-family:'JetBrains Mono',monospace;font-size:11px">{e['ts'].split('T')[1]}</td>
          <td style="font-family:'JetBrains Mono',monospace;font-weight:500;font-size:12px">{e['zone_id']}</td>
          <td>{e['zone_name']}</td>
          <td style="color:{color};font-family:'JetBrains Mono',monospace;font-weight:600;font-size:12px">{e['cong_pct']}%</td>
          <td>{action_chips_html(e['action'])}</td>
          <td>{ov}</td>
        </tr>""")

    st.markdown(clean_html(f"""
    <table class="hist-table">
      <thead><tr>
        <th>Tick</th><th>Time</th><th>Zone</th><th>Name</th>
        <th>Cong.</th><th>Action</th><th>Override</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""), unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_state()
    tick_btn, reset_btn, tiers, min_c = sidebar()

    if reset_btn:
        reset_scenario_counters()
        st.session_state.zones     = load_seed()
        st.session_state.decisions = []
        st.session_state.tick      = 0
        st.session_state.selected  = None
        save_live(st.session_state.zones)
        st.rerun()

    if tick_btn:
        st.session_state.tick += 1
        st.session_state.zones = simulate_tick(st.session_state.zones)
        updated, log = run_agent_pass(st.session_state.zones)
        st.session_state.zones = updated
        st.session_state.decisions.extend(log)
        save_live(st.session_state.zones)
        st.rerun()

    st.markdown(
        '<h1 style="font-size:1.6rem;font-weight:700;margin-bottom:4px">City Traffic Control</h1>'
        '<p style="color:var(--text-3);font-size:13px;margin-bottom:24px">'
        'Autonomous agent · 20 zones · Dynamic congestion management</p>',
        unsafe_allow_html=True
    )

    if st.session_state.tick == 0:
        st.info("Press Simulate Tick in the sidebar to start.")

    render_kpis(st.session_state.zones)

    tab1, tab2, tab3 = st.tabs(["Live Overview", "Zone Analysis", "Decision History"])
    with tab1: render_overview(st.session_state.zones, tiers, min_c)
    with tab2: render_analysis(st.session_state.zones)
    with tab3: render_history()


if __name__ == "__main__":
    main()
