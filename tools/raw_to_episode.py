"""Convert raw sandbox runner rows into the release episode schema (data/episodes) and test the mapping.

`python raw_to_episode.py --test <release_root>` re-derives every sandbox episode file from its raw counterpart and
reports byte-level agreement.
"""
import gzip, io, json, sys
from pathlib import Path
import numpy as np


def episode_row(r):
    wins = r.get("windows") or []
    prov = r.get("provenance") or []
    lat = [float(p.get("latency_s") or 0.0) for p in prov]
    return {
        "decisions": [{"fallback": bool(w.get("fallback")), "seed": r["seed"], "tau": w.get("tau"),
                       "update_reward": w.get("update_reward")} for w in wins],
        "fallback": any(bool(w.get("fallback")) for w in wins),
        "latency": float(np.mean(lat)) if lat else 0.0,
        "n_dec": len(wins),
        "provenance": prov,
        "seed": r["seed"],
        "success": r.get("success"),
        "tau": wins[0].get("tau") if wins else None,
        "total_reward": r.get("total_reward"),
        "type": r.get("type"),
        "value": r.get("total_reward"),
    }


def clean_obj(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): clean_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_obj(v) for v in obj]
    return obj


def llm_call_rows(row: dict) -> list:
    out = []
    for i, prov in enumerate(row.get("provenance") or [], 1):
        extra = prov.get("extra") or {}
        if not extra.get("prompt") and not extra.get("raw") and "usage" not in extra:
            continue
        out.append(
            {
                "seed": row.get("seed"),
                "episode_index": row.get("episode_index", row.get("episode")),
                "attempt": extra.get("attempts", i),
                "prompt": extra.get("prompt"),
                "raw_completion": extra.get("raw"),
                "parsed_strategy": prov.get("tau"),
                "fallback": bool(prov.get("fallback") or extra.get("source") == "fallback"),
                "usage": extra.get("usage"),
                "call_latency_s": extra.get("call_latency_s", prov.get("latency_s")),
                "latency_s": prov.get("latency_s"),
                "temperature": extra.get("temperature"),
                "max_tokens": extra.get("max_tokens"),
                "extra_body": extra.get("extra_body"),
                "system_fingerprint": extra.get("system_fingerprint"),
            }
        )
    return clean_obj(out)


def gz_bytes(rows):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return buf.getvalue()


def read_rows(p):
    p = Path(p)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def overlay(raw_path, rel_root, cell, arm):
    """Write the raw, episode and LLM-call files for one arm into a release tree."""
    rows = read_rows(raw_path)
    d = Path(rel_root) / "data"
    out = {}
    out["raw_episodes"] = gz_bytes(rows)
    out["episodes"] = gz_bytes([episode_row(r) for r in rows])
    if arm.startswith("llm__"):
        calls = []
        for r in rows:
            calls.extend(llm_call_rows(r))
        out["llm_calls"] = gz_bytes(calls)
    for kind, data in out.items():
        p = d / kind / "sandbox" / cell / f"{arm}.jsonl.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return len(rows)


if __name__ == "__main__":
    if sys.argv[1] == "--test":
        rel = Path(sys.argv[2]) / "data"
        ok = bad = content_ok = 0
        for raw in sorted((rel / "raw_episodes" / "sandbox").rglob("*.jsonl.gz")):
            ep = rel / "episodes" / raw.relative_to(rel / "raw_episodes")
            new = gz_bytes([episode_row(r) for r in read_rows(raw)])
            old = ep.read_bytes()
            if new == old:
                ok += 1
            else:
                bad += 1
                a = [json.loads(l) for l in gzip.decompress(new).decode().splitlines()]
                b = read_rows(ep)
                if a == b:
                    content_ok += 1
                elif bad - content_ok <= 3:
                    print("DIFF", ep, [k for k in b[0] if a[0].get(k) != b[0].get(k)])
        print("bytes equal", ok, "differ", bad, "of which content-equal", content_ok)
