# E0 Analytic Sandbox Specification

This sandbox is a typed repeated RPS-style game for exercising the selectors in
`../v8_lib`.  It provides deterministic seeding, an environment implementing the
`v8lib.probes.EpisodeEnv` protocol, Monte Carlo closed forms, calibration
artifacts, a CLI runner, and focused tests.

## Game Model

- Primitive actions are `ROCK`, `PAPER`, and `SCISSORS`.
- Strategy pools contain the primitive strategies for `K=3`; `K=5` appends
  `MIX_RP` and `MIX_PS`, which execute uniform mixtures over their named
  primitive actions.
- Raw payoffs are cyclic RPS: win `+1`, loss `-1`, tie `0`.
- The payoff matrix is scaled by `delta / 0.7`.  `delta` is only a payoff
  scale; it is not the sandbox difficulty knob.
- Lover-style policies use `sharpness=p`: the favourite primitive is played
  with probability `p`, and each other primitive with probability `(1-p)/2`.
  Lower `p` makes type identification harder and shrinks the best-response gap.
- `reward_noise=sigma` adds independent zero-mean Gaussian noise to each
  returned per-round payoff.  This noise is not scaled by the payoff matrix.
- Each episode draws one hidden type uniformly from the active type set using
  `numpy.random.default_rng(seed)`.  `M=3` uses the lover types and `M=6` uses
  all task types.
- `switch="mid"` redraws the type at round `H/2`; the CLI schedule `mid` is a
  per-round deployment with this switch enabled.

## Opponent Types

- `ROCK_LOVER`, `PAPER_LOVER`, `SCISSORS_LOVER`: favourite with probability
  `sharpness`, each other primitive with probability `(1-sharpness)/2`.
- `BEST_RESPONDER`: first round uniform, then plays the RPS best response to
  our previous primitive action.
- `FLIPPER_L`: uses lover distributions whose favourite cycles every `L`
  rounds, default `L=5`.
- `GULLIBLE`: first round uniform, then copies our previous primitive action.

## Schedules

- `single`: reset executes `k` default-strategy observation rounds, then exposes
  one decision context at round `k`.  `step_window(tau)` holds the selected
  strategy through round `H-1`.
- `per_round`: reset exposes round `0`; each `step_window(tau)` advances one
  round.
- `mid`: same as `per_round` with a hidden-type redraw at `H/2`.

Reward returned by `step_window` includes expected payoff for the selected
window under the current stochastic policies plus optional unscaled Gaussian
reward noise.  Primitive samples are still drawn to update histories and
observations.  Reward noise uses a separate seeded RNG, so cloned oracle
branches see the same exogenous noise path and cannot choose an arm by peeking
at random noise.

## Observation Contract

Raw observations expose:

- counts of opponent primitive actions so far,
- fraction of rounds where the opponent best-responded to our previous action,
- fraction of rounds where the opponent copied our previous action,
- booleans for whether the opponent best-responded or copied on the last round,
- run length of the opponent's current most common action,
- current round index,
- hidden type only inside `Context.extra["raw"]` for encoders and calibration.

The visible feature vector is numeric and deterministic.  `features="basic"`
uses normalized opponent action counts, best-response/copy fractions, common-run
fraction, and round index.  `features="rich"` appends one-hot features for the
opponent's most frequent action so far, whether it copied us on the last round,
and whether it best-responded to us on the last round.  The semantic template
renders one or two concise sentences.  Encoder modes are supplied by
`v8lib.encoders.make_encoder`; opaque text contains only deterministic tokens,
and permuted text uses a deterministic derangement of type names.

## Calibration

`sandbox.calibrate.build_lock` uses calibration seeds, excluding the configured
held-out type when requested, to write `sandbox_lock.json` with:

- best-response table,
- few-shot feature matrix and labels,
- basic-feature calibration rows for `ctxucb` tercile bins,
- PLASTIC type log-likelihood models,
- per-type value table,
- the registered calibration seed block used for offline PPO training.

Tests use these in-memory builders; the CLI writes the lock file before running
calibrated arms.

## Closed Forms and Probes

`closed_form.py` estimates `V(tau)`, `V*`, `V_default`,
`E[max_tau Q(o,tau)]`, `H_D`, `Delta_min`, `sigma_eff`, `snr`, and predicted
`T*` by deterministic Monte Carlo over seeded episodes.  For each type,
`Delta_min` is the expected best-response payoff minus the expected
second-best payoff at the decision context; the reported value is the minimum
over active types:

```text
Delta_min = min_m [ max_tau E[r | m,tau] - second_max_tau E[r | m,tau] ].
```

`sigma_eff` is the Monte Carlo standard deviation of realized per-round reward
under the best response, including stochastic opponent actions and optional
Gaussian reward noise:

```text
sigma_eff = std( payoff(a_tau*, a_opp) + Normal(0, reward_noise) ).
snr = Delta_min / sigma_eff.
predicted_T*_tabular = M * K / snr^2.
predicted_T*_known_type = K / snr^2.
```

The implementation reports the sampling count used.  Replay probes in
`sandbox.probes` exercise the same branch logic as clone probes without relying
on object copying.

## CLI Outputs

`run_sandbox.py` accepts the task flags and writes JSONL records produced by
`v8lib.runner.run_deployment`.  The default deployment seed manifest is
`0-99,120-319`, which stays disjoint from the default registered calibration
block `100-119`; the runner receives both lists and validates this separation.
The `ctxucb` arm fits tercile context keys from the basic calibration features.
The `ppo` arm is trained offline on the registered calibration block before any
deployment selection, matching the library requirement that offline PPO must be
trained or loaded before use.

`analyze_sandbox.py` reads JSONL files and emits tables for value, capture, and
persistent crossover horizon against `--target-capture` (default `0.8`) or a
JSON map supplied through `--llm-captures`.  `T*` is the first prefix after
which all later prefixes remain at or above target.  `"never"` denotes a
right-censored run through the observed horizon and is accompanied by
`T_star_status`, `T_star_censored`, `T_star_ci`, and `crossing_probability`.
Capture values include `capture_status`; undefined `H_D` is reported as `NaN`
with a non-`ok` status.
