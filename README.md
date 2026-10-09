# fenceng2: code and prompts for "When Does LLM Strategy Selection Pay in Multi-Agent Games?"

This repository holds the code and the prompts behind my paper *When Does LLM Strategy Selection Pay in Multi-Agent
Games? The Value of Reading New Agents*. The paper and its supplementary material are not part of the repository.

## What the work is about

A hierarchical agent in a multi-agent game picks one strategy from a finite pool, and a fixed executor carries it out.
For a selector that decides once per episode, the margin over the best fixed strategy equals the game's deployment
headroom (what a selector that sees the whole decision state gains over the best fixed strategy) minus the selector's
regret. On top of this identity I test three conditions under which a language-model selector beats both the best fixed
strategy and the strongest tested cheaper selector: there is headroom, the model can read the other agents from its
input, and no cheaper selector with the same information catches the model within the deployment horizon.

The experiments run in typed cells of two MAgent games (`battle_v4` and `combined_arms_v6`), in an analytic typed
rock-paper-scissors sandbox, in a single-type slice (Hanabi, SMAC, MAgent, Overcooked) and in a transfer to the HLA
Overcooked agent. A further line gives every sandbox opponent a written scouting briefing and tests held-out opponent
types. Three open-weights models (`gpt-oss-20b`, `gpt-oss-120b`, `Qwen3.8-27B`) and non-LLM selectors (scripted rule,
blind and contextual bandits, PLASTIC, few-shot classifier, PPO scheduler) take the selector slot under one executor and
one seed list.

## Layout

| path | contents |
| --- | --- |
| `code/oghp/` | experiment frameworks, the selector library (`v8_lib`), the sandbox (`v8_sandbox`) and the MAgent runners (`experiments0106b`) |
| `code/briefing/` | the briefing line: job generators, run scripts, the text readers, `fcr/` (factorized compositional reader) and `fcr_sym/` (its simulator composer) |
| `code/calib_verdict/` | the check that reads the legibility and comparative-advantage verdicts on calibration seeds alone |
| `code/single_type/`, `code/hla_transfer/` | the single-type slice and the HLA transfer |
| `code/analyze_*.py`, `code/replay.py` | the analysis: paired bootstrap, Holm correction, exact replay, checks |
| `code/make_*.py`, `code/briefing_c3.py`, `code/check_numbers.py` | table and figure generators and the shared verdict rule |
| `prompts/` | every prompt the language models saw, with real rendered examples (see `prompts/PROMPTS.md`) |
| `tools/` | helpers for the single-type runs and for packing raw records |
| `environment/` | pinned requirements and the model-serving notes |

## Prompts

`prompts/PROMPTS.md` lists the system instruction for each mode, the user-prompt templates, the encodings of the
other agents' types, the few-shot format, the sampling settings and how an answer is parsed. `prompts/examples/` holds
one real prompt for each family and encoding, together with the model's raw completion. The prompts of the single-type slice are not stored with the records, so `PROMPTS.md` documents the HLA transfer as a template only and says where the rest could not be recovered.

## Environment

The analysis scripts need Python 3.12 and the versions in `environment/requirements-analysis.txt`:

```bash
pip install -r environment/requirements-analysis.txt
```

Re-running the experiments needs Linux x86-64, Python 3.12 and `environment/requirements-experiments.txt`. The models are
served with vLLM behind an OpenAI-compatible endpoint, as described in `environment/serving.md`. The clients read
`LLM_API_BASE`, `LLM_API_KEY` and `LLM_MODEL` from the environment; no credential is stored in the code.

## Data

The per-episode records, the model-call logs and the calibration locks are about 130 MB and are not in this
repository. The analysis and table scripts expect them in a `data/` directory next to `code/`, and they write their
outputs to `analysis/`. I will release the records together with the paper.

## Running

With the data in place, from the repository root:

```bash
python code/analyze_main.py --root . --out analysis
python code/analyze_extras.py --root .
python code/analyze_checks.py --root .
python code/make_tables.py --root .
python code/make_main_figures.py --root .
```

`code/briefing/REPRODUCE.md` lists the steps of the briefing line, and `code/briefing/fcr/REPRODUCE.md` and
`code/briefing/fcr_sym/REPRODUCE.md` those of the two readers. To rerun an experiment against a served model, start from
`code/oghp/v8_sandbox/run_sandbox.py` (sandbox), `code/oghp/experiments0106b/scripts/rs/` (MAgent battle and combined)
or the job generators in `code/briefing/`.

## License and citation

The code is under the MIT license (`LICENSE`). `CITATION.cff` has the citation entry for the paper.
