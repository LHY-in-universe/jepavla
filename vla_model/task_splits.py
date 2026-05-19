from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


# Default hard split aligned with the Evo-1 MetaWorld evaluation setup.
EVO1_HARD_TASKS: List[str] = [
    "nut-assembly-v3",
    "hand-insert-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "push-v3",
    "push-back-v3",
]


def load_evo1_level_tasks(mt50_order_json: str, level: str = "hard") -> List[str]:
    """
    Parse MetaWorld task split from Evo-1 style `mt50_order.json`.

    Expected structure (keys may vary):
    - `idx_to_slug`: {"0":"reach-v3", ...}
    - `per_group_indices`: {"hard":[...], "easy":[...], ...}
    """
    path = Path(mt50_order_json)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    idx_to_slug: Dict[str, str] = data.get("idx_to_slug", {})
    groups: Dict[str, List[int]] = data.get("per_group_indices", {})
    if not idx_to_slug or not groups:
        raise ValueError(f"Unexpected mt50_order format: missing idx_to_slug/per_group_indices in {path}")

    if level not in groups:
        raise ValueError(f"Level `{level}` not found in per_group_indices. Available: {sorted(groups.keys())}")

    out = []
    for idx in groups[level]:
        key = str(idx)
        if key not in idx_to_slug:
            raise KeyError(f"Index {idx} from level `{level}` missing in idx_to_slug.")
        out.append(str(idx_to_slug[key]))
    return out
