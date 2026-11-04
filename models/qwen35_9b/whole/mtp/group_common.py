"""Small shared helpers for the Qwen3.5-9B MTP build/check scripts."""
from __future__ import annotations

import json


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
