# Simulator composer (FCR-sym): reproduce

Everything runs on CPU with Python 3.12 (verified with 3.12.10, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0) from the
repository root. No model call, no GPU, no network; the embedding sensitivity reads the cached bge-small-en-v1.5
vectors (`code/briefing/v8_sandbox/sandbox/embed_cache_bge-small-en-v1.5.npz`), and the FCR's own trait readers and
settings are imported from `code/briefing/fcr/` (not copied).

```bash
python code/briefing/fcr_sym/run_all.py --root .     # about 7 minutes on four cores
```

`run_all.py` first checks that the FCR modules and sandbox/v8lib modules it imports carry the hashes in
`SOURCES.json`, then runs, in order:

| step | script (in `code/briefing/fcr_sym/`) | writes |
|---|---|---|
| 1 | `check_sources.py` | nothing; checks that `code/briefing/fcr/{fcr_common,fcr_model,evaluate}.py`, `code/briefing/fcr/calib/chosen.json`, the sandbox modules (`briefing.py`, `game.py`, `prompt_v2.py`, `env.py`) and the v8lib modules (`__init__.py`, `context.py`, `encoders.py`) the composer needs, and `code/briefing/fcr_sym/sym_composer.py` itself, carry the hashes in `SOURCES.json` |
| 2 | `calib_check.py` | a temporary file, compared byte for byte with `calib/sym_calib_check.json`: the composer, given the true traits of every seen type, against the best response and value row of all 18 briefing locks; the run stops if it is not reproduced |
| 3 | `evaluate_sym.py --placebo` | `analysis/briefing/placebo/placebo_check.json`: the machinery test (every FCR-sym arm replaced by its FCR twin; the FCR evaluation reproduced against the published `fcr.json`) |
| 4 | `evaluate_sym.py` | `analysis/briefing/fcr_sym.json`; refuses to run unless `preregistration/briefing/PREREG_SYM.md` carries its frozen row and `code/briefing/fcr/calib/chosen.json` matches the hash quoted there |
| 5 | `make_tables_sym.py` | `analysis/briefing/fcr_sym_tables.md` (all tables) and `analysis/briefing/fcr_sym_ledger.json` (the predictions, each with its value and verdict) |

`--skip-checks` runs steps 4 and 5 only. Expected sha256:

| file | sha256 |
|---|---|
| `analysis/briefing/fcr_sym.json` | `8a8909cad10e8bac2b06331295908dcd2001c5d6a89fe086f2cecbf634e5176e` |
| `analysis/briefing/fcr_sym_tables.md` | `2423af86581b6efb80c6823e81416c4db8e0e1c5444f786af37e746a9b6c86e0` |
| `analysis/briefing/fcr_sym_ledger.json` | `c1f42dad322b3e8e85a0b17b3b2e6c3be42c593966d9818078ec8e2bb9804fc4` |
| `analysis/briefing/placebo/placebo_check.json` | `27bfbaef554c67d56447bbddee5623e45ce2e80a5f8e1b9b70b389195eb32f88` |
| `calib/sym_calib_check.json` | `76e00417f1938b36696a422dc8e7b71c8d7b276b4af9c062dd301fd0d9135956` (reference file) |

## Determinism

- The trait parsers are the FCR's (TF-IDF or cached embedding, `LogisticRegression(random_state=0, max_iter=5000)`),
  refit from `code/briefing/fcr/calib/chosen.json` on every run; no pickled model is used.
- The symbolic composer (`code/briefing/fcr_sym/sym_composer.py`) runs the sandbox's own `SandboxEnv` for 64 rollouts
  per (trait tuple, strategy) on the fixed simulator seeds 900,000,000 + r; the same seeds serve every strategy.
  Choices are memoised per game cell (M, K, σ), which does not change them.
- The bootstrap is the FCR experiment's: `numpy.random.default_rng([20261006, deployment number, crc32 of the seed
  set])`, one index matrix per deployment and seed set, B = 10,000, deployments in the FCR's order. Every number that
  does not involve FCR-sym equals the FCR's; step 4 re-runs the FCR evaluation and records that it equals
  `analysis/briefing/fcr.json` in all 36 deployments.
- JSON and Markdown are written with `\n` line endings; two consecutive runs give byte-identical outputs, and so does
  a run in a fresh copy of the repository.
- Scripts run with `python -B`, so no bytecode cache is written to this tree.

## Layout

| path | content |
|---|---|
| `preregistration/briefing/PREREG_SYM.md` | the design, tests, outcome rules and predictions, with the revision record |
| `code/briefing/fcr_sym/sym_composer.py` | the symbolic composer: given a predicted trait tuple, simulates every pool strategy against the sandbox's own scripted opponent for that tuple and picks the best (its hash is checked by `check_sources.py` against `SOURCES.json`) |
| `code/briefing/fcr_sym/calib_check.py` | the calibration-only check |
| `code/briefing/fcr_sym/evaluate_sym.py` | deployment replay, statistics, verdicts with FCR-sym, primary tests T1/T2′/T3′, outcome letters, secondary families, integrity checks, prediction ledger; `--placebo` for the machinery test |
| `code/briefing/fcr_sym/make_tables_sym.py` | tables and the ledger file, read from `analysis/briefing/fcr_sym.json` |
| `code/briefing/fcr_sym/check_sources.py` | hashes of every module FCR-sym reads from elsewhere in the repository |
| `code/briefing/fcr_sym/SOURCES.json` | those hashes (copied into `fcr_sym.json`, `meta.vendor_sources`) |
| `code/briefing/fcr_sym/calib/` | `sym_calib_check.json` (calibration check), `freeze.json` (freeze timestamp and hashes) |
| `analysis/briefing/{fcr_sym.json, fcr_sym_tables.md, fcr_sym_ledger.json, placebo/placebo_check.json}` | the outputs |

`PREREG_SYM.md` names files by the paths of the original working layout: `code/<script>.py` is
`code/briefing/fcr_sym/<script>.py` here, `calib/` is `code/briefing/fcr_sym/calib/`, `analysis/fcr_sym.json`,
`fcr_sym_tables.md` and `fcr_sym_ledger.json` are in `analysis/briefing/`, and `analysis/placebo/placebo_check.json` is
`analysis/briefing/placebo/placebo_check.json`. FCR-sym keeps no copies of the FCR's code and settings
(`code/fcr_common.py`, `fcr_model.py`, `fcr_evaluate.py`, `calib/fcr_chosen.json`) or of the sandbox and v8lib modules
(`code/vendor/`): it imports `code/briefing/fcr/`, `code/briefing/v8_sandbox/` and `code/briefing/v8_lib/` directly.
`evaluate_sym.py`'s freeze guard checks only `code/briefing/fcr/calib/chosen.json` (pure data) against the hash
`PREREG_SYM.md` quotes; the composer's own identity is guarded by `check_sources.py` against `SOURCES.json`.

## `analysis/briefing/fcr_sym.json`

- `meta`: freeze record (frozen UTC time and the hash of `code/briefing/fcr/calib/chosen.json`), bootstrap seed, B,
  TIE_TOL, composer settings (rollouts, seed base, tie rule), the arm definitions, `vendor_sources` (the hashes in
  `SOURCES.json`).
- `checks`: `fcr_reproduction` (the FCR evaluation re-run here against `analysis/briefing/fcr.json`),
  `composer_tables` (the composer's choice, means and gap for every trait tuple of every game cell, and its agreement
  with every lock's best responses), `simulator_fidelity` (recorded fixed-arm returns replayed by the composer's
  simulator).
- `primary_tests`: T1 (verdicts with FCR-sym), T2′ (FCR-sym-full − TF-IDF), T3′ (one-example LLM arm − FCR-sym-N on the
  same rows), each with bound, p-value, uncorrected verdict and a Holm reading within the 21 primary tests.
- `outcomes`: per primary deployment, L*, the headline letter, the FCR evaluation's letter and the letter of every
  primary LLM arm.
- `secondary`: S1′ (verdicts with FCR-sym of every other LLM arm, Holm before and after) and S2′ (FCR-sym-full − TF-IDF
  in the other 33 deployments, Holm).
- `changed_verdicts`: every LLM arm in all 36 deployments whose verdict differs from the FCR evaluation's.
- `strongest_changes`: every deployment whose strongest non-LLM arm changes.
- `predictions_ledger`: the predictions with values and verdicts.
- `deployments/<all|heldout>/<group>/<wording>`: `meta`, `reference` (V*, H_D, best fixed), `strongest` (baselines,
  before = with the FCR, after = with FCR-sym, with only the primary FCR-sym arms), `arms` (every FCR-sym arm and
  trait-oracle-sym: capture with interval, margin bound, counter accuracy, hit rate, trait and tuple accuracies,
  strata seen / held-out with the reactive-type (reactivity, timing) joint accuracy, held-out choices), `comparators`
  (every baseline and FCR arm of the deployment as the FCR evaluation computed it), `decomposition` (parsing vs
  composition against trait-oracle-sym), `paired` (FCR-sym arm − comparators and FCR twins), `llm` (every LLM arm:
  baseline, before and after verdicts, its paired difference against every FCR-sym arm and every FCR arm).
