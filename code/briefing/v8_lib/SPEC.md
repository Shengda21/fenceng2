# v8lib Specification

## Package Scope

`v8lib` provides environment-agnostic strategy selectors, observation encoders,
oracle probes, deployment logging, and analysis helpers for strategy-selection
experiments. A selector chooses exactly one strategy name `tau` from a finite
`pool: list[str]` at each decision window. Executors and game dynamics are owned
by the caller.

All stochastic components accept deterministic seeds and use
`numpy.random.default_rng(seed)` where practical. Tests must not use network
access. LLM credentials are read only from `LLM_API_BASE`, `LLM_API_KEY`, and
`LLM_MODEL`.

## Core API

### `Context`

```python
@dataclass
class Context:
    pool: list[str]
    features: np.ndarray
    text: str
    episode_index: int
    window_index: int
    default: str
    type_label: str | None = None
    extra: dict = field(default_factory=dict)
```

`type_label` is hidden truth. Only selectors with `reads_type_label=True` may
receive it in the runner.

### `Selector`

```python
class Selector(ABC):
    name: str
    role: str
    reads_type_label: bool = False

    def reset_run(self, seed: int) -> None: ...
    def reset_episode(self, ctx0: Context) -> None: ...
    def select(self, ctx: Context) -> str: ...
    def update(self, ctx: Context, tau: str, reward: float, done: bool, info: dict) -> None: ...
    def state_summary(self) -> dict: ...
```

Valid roles are `"instrument"`, `"floor"`, `"zero_shot"`, `"online"`, and
`"llm"`.

### Provenance JSON

Every selection can be described as:

```json
{
  "arm": "selector-name",
  "episode": 0,
  "window": 0,
  "tau": "STRATEGY",
  "in_pool": true,
  "fallback": false,
  "latency_s": 0.0,
  "extra": {}
}
```

Selectors expose their most recent provenance through `last_provenance`.

## Arms

The following selectors live under `v8lib.arms`:

- `RandomSelector(seed=None)`: floor selector, uniform over `ctx.pool`.
- `FixedSelector(tau)`: instrument selector, returns `tau` if present, otherwise
  `ctx.default`.
- `TypeOracleSelector(best_response)`: instrument selector, reads
  `ctx.type_label`, returns `best_response[type]`.
- `ScriptedDetector(rules)`: zero-shot selector, applies user rules to
  `Context`.
- `MUCB(pool=None, window_w=20, threshold_b=1.0, gamma=0.05)`: UCB1 with forced
  exploration and sliding-window restart on large recent/older mean shifts.
- `CUSUMUCB(pool=None, threshold_h=5.0, drift=0.0, gamma=0.05)`: UCB with a
  lightweight CUSUM-style restart.
- `LinUCB(alpha=1.0)`: disjoint linear UCB over `ctx.features`.
- `LinTS(alpha=1.0, v=1.0)`: disjoint linear Thompson sampling.
- `FewShotClassifierSelector(N, best_response, model)`: fit with `fit(X, y)`,
  predicts type then maps to `best_response`; `N=0` behaves randomly.
- `PLASTICPolicySelector(type_models, best_response, prior=None, value_table=None)`:
  Bayesian type posterior updated with type-model log-likelihoods.
- `PPOScheduler(pool=None, feature_dim=None, hidden=64)`: small CPU torch PPO
  policy with `train(env_factory, n_episodes, seed)`.
- `LLMSelector(...)`: chat-completion selector with strict pool parsing,
  fallback, retries, latency, usage accounting, `single`,
  `hypothesis_first`, and `react` modes.
- `LLMInitLinUCB(llm_prior, alpha=1.0, prior_strength=1.0)`: LinUCB whose first
  estimates are seeded by prior strategy scores.

All selectors must return a pool element after fallback handling.

## Encoders

```python
class Encoder:
    mode: Literal["semantic", "numeric", "opaque", "permuted"]
    def __call__(self, raw: dict, ctx_partial: Context) -> Context: ...
```

Factory:

```python
make_encoder(mode, semantic_template, feature_fn, type_names, permutation_seed,
             keep_numeric=False, feature_names=None)
```

Mode semantics:

- `semantic`: semantic text and numeric features.
- `numeric`: neutral feature listing without type-name replacement.
- `opaque`: deterministic per-type token text and one-hot token features,
  optionally appending numeric features.
- `permuted`: semantic text with type names replaced by a deterministic
  derangement; features unchanged.

`permute_type_names(text, mapping)` performs whole-name replacement.

## Probes

Environment protocol:

```python
class EpisodeEnv(Protocol):
    def reset(self, seed: int) -> Context: ...
    def step_window(self, tau: str) -> tuple[Context | None, float, bool, dict]: ...
    def clone(self) -> "EpisodeEnv": ...
```

Functions:

- `fixed_strategy_sweep(make_env, pool, seeds)`: returns per-strategy means and
  per-seed episode returns.
- `best_fixed(results)`: returns `(tau, mean)`.
- `per_window_oracle(make_env, pool, default, seeds, placebo=False, v_star=None,
  v_default=None)`: returns `Delta_def`, `placebo`, per-window deltas, and `H_D`
  when `v_star`/`v_default` are supplied.
- `oracle_h_d(delta_def, placebo, v_star, v_default)`: helper implementing
  `H_D = (Delta_def - placebo) - (V_star - V_default)`.

Probes must not call selectors.

## Analysis

- `paired_bootstrap(a, b, B=10000, seed=0)`: paired bootstrap mean difference
  CI.
- `wilson(k, n)`: 95% Wilson interval.
- `capture(v, v_star, h)`: `(v - v_star) / h`, with zero-denominator handling.
- `capture_curve(rewards, v_star, h)`: cumulative capture for `T=1..n`.
- `crossover(c_learner_curve, c_llm, B=10000, seed=0)`: first crossing and
  bootstrap CI over prefix crossings.
- `scaling_fit(x, y)`: log-log slope and bootstrap CI.
- `master_row(...)`: exact master-table row schema:
  `cell, M, W, encoding, H_D, V_star, best_non_llm{name,margin,capture},
  llm{model:{margin,capture}}, conditions{C1,C2,C3}, prediction, verdict`.

## Runner

```python
run_deployment(make_env, selector, seeds, T, encoder=None, log_path=None)
```

Runs `T` episodes using the supplied seed sequence, resets the selector for the
run and per episode, logs JSONL records with:

```json
{
  "method": "selector-name",
  "episode": 0,
  "seed": 0,
  "type": "hidden-type-or-null",
  "windows": [{"window": 0, "tau": "STRATEGY", "reward": 1.0, "fallback": false}],
  "total_reward": 1.0,
  "success": true,
  "provenance": []
}
```

The runner masks `type_label` unless `selector.reads_type_label` is true.

`calibration_split(seeds=None)` returns calibration seeds `100..119` and held-out
seeds `0..99` by default, or partitions a provided seed iterable by the same
numeric rule.
