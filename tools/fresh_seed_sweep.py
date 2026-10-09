"""Fixed-strategy sweep of a typed MAgent cell on fresh seeds, with the released framework and the cell's lock.

At a single decision point the per-window oracle is the per-seed maximum over the fixed strategies, so this sweep reads
V* and H_D on any seed set. Used to re-read the headline cells on seeds that neither calibration nor deployment used.

usage: python tools/fresh_seed_sweep.py --domain battle|combined --lock <lock.json> --seeds 5000-5024 --out <file.json>
       (run from the framework root that contains experiments0106b/, v8_lib/)
"""
import argparse
import json
import sys
from pathlib import Path


def parse_range(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["battle", "combined"], required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve() / "experiments0106b"
    sys.path[:0] = [str(root), str(root.parent / "v8_lib")]
    lock = json.load(open(args.lock, encoding="utf-8"))
    seeds = parse_range(args.seeds)
    assert not set(seeds) & set(lock["calibration_seeds"]), "fresh seeds overlap calibration"
    assert min(seeds) >= 300, "fresh seeds must lie outside the registered deployment seeds 0-299"
    if args.domain == "battle":
        from scripts.rs.calibrate_rs_magent import _schedule_to_switch
        from scripts.rs.probes_rs import build_episode_manifest, fixed_sweep
        env_cfg = {"map_size": lock["map_size"], "max_cycles": lock["max_cycles"], "type_set": lock["types"],
                   "pool": tuple(lock["pool"]), "default": lock["default"], "k": lock["k"],
                   "switch": _schedule_to_switch(lock["schedule"]), "delta_scale": lock["delta_scale"],
                   "held_out": lock["held_out"], "force_fake": False}
        manifest = build_episode_manifest(env_cfg, seeds)
        sweep = fixed_sweep(env_cfg, manifest, env_cfg["pool"])
    else:
        from scripts.rs.calibrate_rs_combined import _schedule_to_switch
        from scripts.rs.combined_env import CombinedArmsRS
        from scripts.rs.probes_rs_common import build_episode_manifest, fixed_sweep
        env_cfg = {"map_size": lock["map_size"], "max_cycles": lock["max_cycles"], "n_melee": lock["n_melee"],
                   "n_ranged": lock["n_ranged"], "type_set": lock["types"], "pool": tuple(lock["pool"]),
                   "default": lock["default"], "k": lock["k"], "switch": _schedule_to_switch(lock["schedule"]),
                   "delta_scale": lock["delta_scale"], "held_out": lock["held_out"], "force_fake": False}
        manifest = build_episode_manifest(CombinedArmsRS, env_cfg, seeds)
        sweep = fixed_sweep(CombinedArmsRS, env_cfg, manifest, env_cfg["pool"])
    out = {"domain": args.domain, "lock": Path(args.lock).name, "seeds": seeds, "manifest": sweep["manifest"],
           "per_episode": sweep["per_episode"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"))
    print("wrote", args.out, len(seeds), "seeds")


if __name__ == "__main__":
    main()
