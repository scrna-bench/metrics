"""Reading an omnibenchmark gather manifest (`--from-files`, lineage.json v2,
ob api 0.8). Every member lists its files by output id and its full lineage,
so nothing here parses ob's output paths.

Adapted from omni-scrna/metrics src/fanin.py.
"""

import json

import polars as pl


def manifest_members(path, *ids):
    """Members that carry every output id in `ids`, in manifest order."""
    with open(path) as fh:
        members = json.load(fh)["members"]
    return [m for m in members if all(i in m["files"] for i in ids)]


def cell(v):
    """A parameter value as one table cell: scalars as they are, lists and
    mappings as compact JSON text."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return json.dumps(v, separators=(",", ":"))


def node_label(entry):
    """`<module>.<params hash>`, or `<module>` for a node without parameters."""
    return entry["module"] if entry.get("params") is None else f"{entry['module']}.{entry['params']}"


def member_lineage_cols(member, as_):
    """One member's lineage as named columns:
      dataset                 the DATA module id
      <stage>                 node_label() of every other upstream stage
      <as_>_stage, <as_>, <as_>_params
                              the member's own node: stage, module, params hash
      <stage>_<parameter>     every parameter value along the lineage
    """
    lin = member["lineage"]
    own = lin[-1]
    data = next((e for e in lin if e["stage"] == "DATA"), None)
    cols = {"dataset": data["module"] if data else None}
    for e in lin[:-1]:
        if e["stage"] != "DATA":
            cols[e["stage"].lower()] = node_label(e)
    cols[f"{as_}_stage"] = own["stage"]
    cols[as_] = own["module"]
    cols[f"{as_}_params"] = own.get("params")
    for e in lin:
        for k, v in (e.get("parameters") or {}).items():
            cols[f"{e['stage'].lower()}_{k}"] = cell(v)
    return cols


def frame(rows, tail, types=None):
    """Rows (dicts: lineage columns + `tail`) -> a typed table, lineage columns
    first, then `tail` in order. `types` fixes the dtype of named columns.
    Parameter columns differ between members (a module without the parameter
    leaves it null); a column whose values mix types becomes text, ints mixed
    with floats become floats."""
    types = types or {}
    keys = list(dict.fromkeys(k for r in rows for k in r))
    keys = [k for k in keys if k not in tail] + [k for k in tail if k in keys]
    cols = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        if k in types:
            cols[k] = pl.Series(k, vals, dtype=types[k])
            continue
        kinds = {type(v) for v in vals if v is not None}
        if kinds == {int, float}:
            vals = [None if v is None else float(v) for v in vals]
        elif len(kinds) > 1:
            vals = [None if v is None else str(v) for v in vals]
        cols[k] = pl.Series(k, vals)
    return pl.DataFrame(cols)
