# Briefing line: regenerate the analyses

Run from the repository root with Python 3.12 and NumPy (`pip install -r environment/requirements-analysis.txt`). The
scripts find their own library (`code/briefing/v8_lib`, `code/briefing/v8_sandbox`); no `PYTHONPATH` is needed.

```bash
python code/briefing/v8_sandbox/analyze_briefing.py --root .                        # analysis/briefing/all_types.json
python code/briefing/v8_sandbox/analyze_briefing.py --root . --deploy-types heldout  # analysis/briefing/new_agents.json
python code/briefing/v8_sandbox/analyze_briefing.py --root . --superseded           # analysis/briefing/bare_prompt_first_format.json
python code/briefing/assemble_task_prompt.py --root .                               # build/briefing_cells (working copy)
python code/briefing/v8_sandbox/analyze_task_prompt.py --root .                     # analysis/briefing/task_prompt.json, task_prompt.ledger.txt
sha256sum -c SHA256SUMS
```

Each step reads only the repository; each analysis takes one to three minutes on a four-core laptop. Every bootstrap
draws from a seeded NumPy stream (B = 10,000) in a fixed order: cells sorted by name, arms sorted by file name. The
outputs are written with `\n` line endings on every platform, so a rerun reproduces the shipped files byte for byte and
the checksum test passes.

## What each step reads and writes

| step | reads | writes |
| --- | --- | --- |
| all-types deployment | `data/briefing/all/<cell>/<arm>.jsonl.gz` | `all_types.json`: per group (base cell under one held-out set) the reference (V\*, H_D, per-type best responses), the exact reference and prefix ceilings, every arm's capture, margin, strata, per-type misreading and loss, curves and verdict, and the predictions P-A to P-F; `P_E` holds the primary and Holm-corrected verdicts |
| held-out-only deployment | `data/briefing/heldout/`, and `data/briefing/all/` for P-I | `new_agents.json`: the same per group on 300 headline seeds, the held-out-only reference (`reference_0b`) and P-E′, P-G, P-H, P-I |
| bare-prompt LLM arms (first prompt format) | `data/briefing/superseded/` laid over `data/briefing/all/` (the non-LLM arms are shared) | `bare_prompt_first_format.json`, the all-types analysis of those arms, which carries their fallback rates |
| assembly | `data/briefing/task_prompt/{all,heldout,classic}`, `data/briefing/all`, `data/briefing/heldout`, and the sandbox records `data/raw_episodes/sandbox` (fixed arms and bare-prompt zero-shot arms of the eight sandbox cells) | `build/briefing_cells/{s0,s0b,classic}`, removed and rebuilt on every run |
| task-stating prompt | `build/briefing_cells` | `task_prompt.json`: `s0` (all-types deployment, 18 types, noise 0, `react`), `s0b` (held-out-only `react4` and `react8`), `classic` (the eight sandbox cells), the predictions P-J to P-N and the decision rule; `task_prompt.ledger.txt` is a one-line-per-prediction summary |

The scripts also take explicit paths (`--results`, `--out`, `--headline-n`, `--stage0-cells`, `--overlay` for
`analyze_briefing.py`; `--runs-0c`, `--stage0-cells`, `--s0b-cells`, `--release-raw`, `--out` for `assemble_task_prompt.py`;
`--cells`, `--out` for `analyze_task_prompt.py`).

## Names

- Cells: `sandbox-traits-M{9|18}-K{5|10}-sharp0.8-noise{0|0.5}-{react|diag|react8}-{nonllm|briefing|briefing_prefix|semantic}[-{sameword|newword}]`.
  The held-out set is `react2`/`react4` for `react`, `diag3`/`diag6` for `diag` (9/18 types), and `react8`.
- Arms: `fixed__<strategy>`, `type_oracle`, `random`, `scripted`, `fewshot__1`, `plastic`, `linucb`, `lints`, `ctxucb`, `mucb`,
  `ppo`, `scripted_text`, `text_bow`, `text_embed`, `text_linucb`, `text_bow_matched__nshot<R>`, `text_embed_matched__nshot<R>`,
  and `llm__<model>__<single|hypothesis_first>__nshot<N>[__<variant>]`. No variant is the bare prompt; `v2` is the task-stating
  prompt, `v2-shuf` the same with the pool order shuffled per episode, `v2-med` and `v2-high` gpt-oss reasoning effort medium and
  high, `v2-think` Qwen thinking on (4,096 completion tokens), `-long` the same under 32,768-token serving with 16,384
  completion tokens. Under the task-stating prompt `nshot42` (`nshot30` for `react8`) is three examples per seen type.
- Each record is one episode as the runner wrote it: seed, opponent type, chosen strategy, reward, fallback count, and for LLM
  arms the call record in `provenance[0].extra` (prompt, system prompt, completion, reasoning text where the server returns it,
  token counts, latency, finish reason, request parameters, serving build).

## Re-running the experiments

`v8_sandbox/run_sandbox.py` runs one arm of one cell. The exact command of every task, with its seeds, is in
`data/briefing/tasks/{all,heldout,task_prompt,superseded}` (task lists, executed commands under `cmds/`, exit status per
task, fallback and merge reports, run and serving logs). Paths there are execution paths on the experiment server; the
labels `s0` and `s0b` in them are the all-types and held-out-only deployments. `make_jobs.py`, `make_jobs_new_agents.py`,
`make_jobs_task_prompt.py` and `make_jobs_task_prompt_long.py` write those lists, `assemble_results.py` copies runs shared between held-out
sets into every cell that uses them (`data/briefing/tasks/all/shared_map.json`), `merge_shards_task_prompt.py` joins seed shards,
and `fallback_rerun_task_prompt.py` applies the rerun rule for arms above 5% fallback. `calibrate_briefing.py`,
`calibrate_new_agents.py` and `closed_form.py` rebuild the locks and exact references in `data/briefing/locks`.

The text readers need no model: `v8_sandbox/sandbox/embed_cache_bge-small-en-v1.5.npz` holds the embedding of every
renderable briefing, keyed by the SHA-256 of its text. To embed new text, point `OGHP_EMBED_MODEL` at a local copy of
`BAAI/bge-small-en-v1.5`. Serving is described in `environment/serving.md`.
