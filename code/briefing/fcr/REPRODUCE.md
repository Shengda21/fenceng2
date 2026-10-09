# Factorized compositional reader (FCR): reproduce

Everything runs on CPU with Python 3.12 (verified with 3.12.10, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0) from the
repository root. No model call, no GPU, no network; the embedding sensitivity reads the cached bge-small-en-v1.5
vectors (`code/briefing/v8_sandbox/sandbox/embed_cache_bge-small-en-v1.5.npz`).

```bash
python code/briefing/fcr/run_all.py --root .     # about 2 minutes on four cores
```

`run_all.py` first checks that the sandbox modules it imports (`code/briefing/v8_sandbox/sandbox/briefing.py`, `game.py`,
`prompt_v2.py`) carry the hashes in `SOURCES.json`, then runs, in order:

| step | script (in `code/briefing/fcr/`) | writes |
|---|---|---|
| 1 | `survey_checks.py` | `analysis/briefing/fcr_survey_checks.json`: lock identity across σ and deployments, the N = 1 prompt rows rebuilt and compared with all 21,600 recorded prompts, exact replay of 147,300 recorded episodes |
| 2 | `evaluate.py --selftest` | `analysis/briefing/fcr_selftest.json`: the evaluation machinery reproduces the capture point estimates of the main analysis (871 arms) and strongest non-LLM arms (36 deployments) |
| 3 | `calib_cv.py` | the calibration-only leave-one-combination-out CV, written to a temporary folder and compared byte for byte with the stored `calib/loco_cv.json` and `calib/chosen.json`; the run stops if the stored settings are not reproduced |
| 4 | `evaluate.py` | `analysis/briefing/fcr.json`, from `calib/chosen.json`; refuses to run unless `preregistration/briefing/PREREG_FCR.md` records the hash of that settings file and it matches |
| 5 | `make_tables.py` | `analysis/briefing/fcr_tables.md` (all tables) and `analysis/briefing/fcr_ledger.json` (the predictions, each with its value and verdict) |

`--skip-checks` runs steps 4 and 5 only. Expected: `analysis/briefing/fcr.json` sha256
`3968ebc709b99065190593381eb54b708fafbccde9f00fcbb39169787d8cfdeb`, `calib/chosen.json` sha256
`93328267e05e99cfd90a252c02e0fb11424c852cdd1ab517575190d327597332`.

## Determinism

- The FCR models are deterministic (TF-IDF, `LogisticRegression(random_state=0, max_iter=5000)`, `Ridge`); they are refit
  from `calib/chosen.json` on every run, no pickled model is used.
- The bootstrap uses `numpy.random.default_rng([20261006, deployment number, crc32 of the seed set])`, one index matrix
  per deployment and seed set, shared by all arms of that deployment, with B = 10,000; deployments are visited in sorted
  order. Every number about an existing arm that does not involve the FCR (capture, interval, margin bound, verdict,
  paired difference against the strongest non-LLM arm) is read from the main analysis in `analysis/briefing/*.json`.
- JSON is written with `\n` line endings; two consecutive runs give byte-identical `analysis/briefing/fcr.json`.

## Layout

| path | content |
|---|---|
| `preregistration/briefing/PREREG_FCR.md` | the design, tests, outcome rules A–D and predictions; `evaluate.py` reads the hash of `calib/chosen.json` recorded there |
| `fcr_common.py` | conventions of the main analysis (headline seeds, V*, H_D, TIE_TOL, Holm), record loading, calibration settings, the N = 1 rows |
| `fcr_model.py` | the trait readers (TF-IDF, embedding), the ridge composer (main, pairwise), the FCR pipeline |
| `calib_cv.py` | leave-one-combination-out CV inside calibration and the selection rules |
| `evaluate.py` | deployment replay, statistics, verdicts with the FCR arms, primary tests, outcome letters, secondary families |
| `survey_checks.py`, `make_tables.py` | the survey checks; tables and the prediction ledger |
| `SOURCES.json` | the sandbox modules the FCR imports, with their sha256 (copied into `fcr.json`, `meta.vendor_sources`) |
| `calib/` | `loco_cv.json`, `chosen.json` (selected hyperparameters), `freeze.json` (timestamp and hashes) |

`PREREG_FCR.md` names files by their original layout: `code/<script>.py` is `code/briefing/fcr/<script>.py` here,
`calib/` is `code/briefing/fcr/calib/`, `analysis/fcr.json`, `fcr_tables.md` and `fcr_ledger.json` are in
`analysis/briefing/`, `analysis/survey_checks.json` and `analysis/selftest.json` are
`analysis/briefing/fcr_survey_checks.json` and `fcr_selftest.json`, and `code/vendor/SOURCES.json` is `SOURCES.json`.

## `analysis/briefing/fcr.json`

- `meta`: calibration record, bootstrap seed, B, TIE_TOL, the arm definitions.
- `calibration`: the selected settings and a summary of the CV (full CV in `calib/loco_cv.json`).
- `primary_tests`: T1 (verdicts with the FCR), T2 (FCR-full − TF-IDF), T3 (1/type LLM arm − FCR-N on the same rows), each
  with its bound, p-value, uncorrected verdict and a Holm reading within the 21 primary tests.
- `outcomes`: per primary deployment, L*, the headline letter and the letter of every primary LLM arm.
- `secondary`: S1 (verdicts with the FCR of every other LLM arm, Holm) and S2 (FCR-full − TF-IDF in the other 33
  deployments, Holm).
- `deployments/<all|heldout>/<group>/<wording>`: `meta` (including the briefing check against the recorded prompts),
  `reference` (V*, H_D, best fixed), `strongest` (main analysis, with FCR, online bandit, FCR-full*, oracle*), `arms` (every FCR
  arm and every comparator: capture with interval, margin bound, counter accuracy, hit rate; FCR arms also trait and tuple
  accuracies and the seen / held-out strata; comparators also the numbers of the main analysis), `decomposition` (parsing vs
  composition per FCR arm), `paired` (FCR arm − non-LLM comparator), `llm` (every LLM arm: its paired difference against
  every FCR arm and its verdict with the FCR arms added).
