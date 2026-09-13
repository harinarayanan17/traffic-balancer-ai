"""
test_agent.py — Quick smoke test for the agent reasoning engine.
Run: python test_agent.py
Prints decisions for 4 contrasting zones so you can verify reasoning diversity.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from agent import evaluate_zone

SEED = Path("data/zones_seed.json")

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    with open(SEED) as f:
        zones = json.load(f)

    # Hand-pick 4 interesting zones for the comparison
    interesting_ids = ["Z01", "Z05", "Z07", "Z04", "Z11", "Z20"]
    zones_by_id = {z["zone_id"]: z for z in zones}

    now = datetime.now()

    for zid in interesting_ids:
        zone = zones_by_id.get(zid)
        if not zone:
            continue
        decision = evaluate_zone(zone, now=now)
        print("\n" + "=" * 70)
        print(f"ZONE: {zid} — {zone['zone_name']}")
        print(f"Actions: {', '.join(a.display_label for a in decision.chosen_actions)}")
        print("-" * 70)
        print(decision.reasoning_trace)

    print("\n" + "=" * 70)
    print("✅ Smoke test complete.")

if __name__ == "__main__":
    main()
