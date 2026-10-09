"""Small PPO high-level policy baseline implemented with torch CPU."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

from v8lib.arms.base import Selector
from v8lib.context import Context


class PPOScheduler(Selector):
    """Torch MLP policy trained with PPO; source: Schulman et al., 2017."""

    name = "ppo_scheduler"
    role = "online"

    def __init__(
        self,
        pool: list[str] | None = None,
        feature_dim: int | None = None,
        hidden: int = 64,
        lr: float = 3e-3,
        online: bool = False,
        rollout_size: int = 8,
    ) -> None:
        super().__init__()
        self.pool = list(pool or [])
        self.feature_dim = feature_dim
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.online = bool(online)
        self.rollout_size = int(rollout_size)
        self.model = None
        self.value = None
        self.opt = None
        self.trained = False
        self.checkpoint_hash: str | None = None
        self._rollout = []
        self._last_transition = None
        self.rng = np.random.default_rng(0)

    def reset_run(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        try:
            import torch

            torch.manual_seed(int(seed))
        except ImportError:
            pass

    def _ensure_model(self, ctx: Context):
        import torch
        import torch.nn as nn

        if not self.pool:
            self.pool = list(ctx.pool)
        if self.feature_dim is None:
            self.feature_dim = int(np.asarray(ctx.features).size)
        if self.model is None:
            self.model = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden),
                nn.Tanh(),
                nn.Linear(self.hidden, len(self.pool)),
            )
            self.value = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden),
                nn.Tanh(),
                nn.Linear(self.hidden, 1),
            )
            self.opt = torch.optim.Adam(
                list(self.model.parameters()) + list(self.value.parameters()), lr=self.lr
            )

    def select(self, ctx: Context) -> str:
        import torch

        if not self.online and not self.trained:
            raise RuntimeError("PPOScheduler requires train() or load() before offline deployment")
        self._ensure_model(ctx)
        x = torch.as_tensor(np.asarray(ctx.features, dtype=np.float32).ravel()).unsqueeze(0)
        logits = self.model(x)
        dist = torch.distributions.Categorical(logits=logits)
        if self.online:
            act = dist.sample()
            idx = int(act.item())
            self._last_transition = {
                "obs": x.squeeze(0).detach(),
                "act": act.detach(),
                "old_logp": dist.log_prob(act).detach(),
                "value": self.value(x).squeeze().detach(),
            }
            probs = torch.softmax(logits, dim=-1).detach().numpy().ravel()
        else:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1).numpy().ravel()
            idx = int(self.rng.choice(np.arange(len(self.pool)), p=probs / probs.sum()))
        tau = self.pool[idx]
        return self._record(ctx, tau, extra={"prob": float(probs[idx])})

    def update(
        self, ctx: Context, tau: str, reward: float, done: bool, info: dict
    ) -> None:
        if not self.online:
            return None
        if self._last_transition is None:
            return None
        item = dict(self._last_transition)
        item["reward"] = float(reward)
        self._rollout.append(item)
        self._last_transition = None
        boundary = bool(done) or len(self._rollout) >= self.rollout_size or bool(info.get("ppo_update", False))
        if boundary:
            self._ppo_update(self._rollout)
            self._rollout = []
            self.trained = True
            self.checkpoint_hash = self._state_hash()

    def train(self, env_factory, n_episodes: int, seed: int = 0, checkpoint_path=None) -> list[float]:
        import torch

        self.reset_run(seed)
        returns = []
        for ep in range(int(n_episodes)):
            env = env_factory()
            ctx = env.reset(seed + ep)
            self._ensure_model(ctx)
            rollout = []
            rewards = []
            done = False
            while not done:
                x = torch.as_tensor(np.asarray(ctx.features, dtype=np.float32).ravel()).unsqueeze(0)
                logits = self.model(x)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                val = self.value(x).squeeze()
                next_ctx, reward, done, info = env.step_window(self.pool[int(act)])
                rollout.append(
                    {
                        "obs": x.squeeze(0).detach(),
                        "act": act.detach(),
                        "old_logp": dist.log_prob(act).detach(),
                        "value": val.detach(),
                        "reward": float(reward),
                    }
                )
                rewards.append(float(reward))
                ctx = next_ctx if next_ctx is not None else ctx
            returns.append(float(sum(rewards)))
            self._ppo_update(rollout)
        self.trained = True
        self.checkpoint_hash = self._state_hash()
        if checkpoint_path is not None:
            self.save(checkpoint_path)
        return returns

    def save(self, path) -> str:
        import torch

        if self.model is None or not self.trained:
            raise RuntimeError("cannot save an untrained PPO scheduler")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pool": self.pool,
            "feature_dim": self.feature_dim,
            "hidden": self.hidden,
            "lr": self.lr,
            "online": self.online,
            "rollout_size": self.rollout_size,
            "model": self.model.state_dict(),
            "value": self.value.state_dict(),
        }
        torch.save(payload, path)
        self.checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.checkpoint_hash

    @classmethod
    def load(cls, path) -> "PPOScheduler":
        import torch

        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            pool=payload["pool"],
            feature_dim=payload["feature_dim"],
            hidden=payload["hidden"],
            lr=payload["lr"],
            online=payload.get("online", False),
            rollout_size=payload.get("rollout_size", 8),
        )
        obj._ensure_model(Context(obj.pool, np.zeros(obj.feature_dim), "", 0, 0, obj.pool[0]))
        obj.model.load_state_dict(payload["model"])
        obj.value.load_state_dict(payload["value"])
        obj.trained = True
        obj.checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        return obj

    def parameter_vector(self) -> np.ndarray:
        if self.model is None:
            return np.array([], dtype=float)
        params = [p.detach().cpu().numpy().ravel() for p in self.model.parameters()]
        params += [p.detach().cpu().numpy().ravel() for p in self.value.parameters()]
        return np.concatenate(params) if params else np.array([], dtype=float)

    def _ppo_update(self, rollout) -> None:
        if not rollout:
            return None
        import torch
        import torch.nn.functional as F

        rewards = [float(item["reward"]) for item in rollout]
        R, returns = 0.0, []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.append(R)
        returns_t = torch.as_tensor(list(reversed(returns)), dtype=torch.float32)
        X = torch.stack([item["obs"] for item in rollout])
        A = torch.stack([item["act"] for item in rollout])
        old = torch.stack([item["old_logp"] for item in rollout])
        values = torch.stack([item["value"] for item in rollout])
        advantages = returns_t - values.detach()
        for _ in range(3):
            dist = torch.distributions.Categorical(logits=self.model(X))
            ratio = torch.exp(dist.log_prob(A) - old)
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(ratio * advantages, clipped).mean()
            value_loss = F.mse_loss(self.value(X).squeeze(-1), returns_t)
            loss = policy_loss + 0.5 * value_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def _state_hash(self) -> str:
        import torch

        if self.model is None:
            return ""
        payload = {
            "model": self.model.state_dict(),
            "value": self.value.state_dict(),
            "pool": self.pool,
            "feature_dim": self.feature_dim,
            "hidden": self.hidden,
        }
        buf = io.BytesIO()
        torch.save(payload, buf)
        return hashlib.sha256(buf.getvalue()).hexdigest()
