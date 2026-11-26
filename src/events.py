"""obkit phase events (obkit-events.jsonl) -> one row per phase.

obkit writes a start and an end record per phase and leaves pairing to the
reader: pair them by (pid, event), innermost first, and take the end record's
attrs (e.g. num_runs and resolution of a clustering search)."""

import json
from datetime import datetime


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def phases(path):
    open_ = {}
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["pid"], rec["event"])
            if rec["phase"] == "start":
                open_.setdefault(key, []).append(rec)
                continue
            starts = open_.get(key)
            if not starts:
                continue  # an end without a start: skip
            start = starts.pop()
            row = {
                "phase": rec["event"],
                "start": start["ts"],
                "end": rec["ts"],
                "seconds": (_ts(rec["ts"]) - _ts(start["ts"])).total_seconds(),
                "host": rec.get("host"),
            }
            row.update(rec.get("attrs") or {})
            rows.append(row)
    # phases that never ended (the run failed inside them)
    for (pid, event), starts in open_.items():
        for start in starts:
            rows.append({"phase": event, "start": start["ts"], "end": None,
                         "seconds": None, "host": start.get("host")})
    return rows
