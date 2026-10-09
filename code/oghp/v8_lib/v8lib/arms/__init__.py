"""Selector arms exposed by v8lib."""

from .bandit import CUSUMUCB, MUCB
from .base import Selector
from .classifier import FewShotClassifierSelector
from .contextual import ContextUCB, LLMInitLinUCB, LinTS, LinUCB
from .llm import LLMSelector
from .plastic import PLASTICPolicySelector
from .ppo import PPOScheduler
from .simple import FixedSelector, RandomSelector, ScriptedDetector, TypeOracleSelector

__all__ = [
    "Selector",
    "RandomSelector",
    "FixedSelector",
    "TypeOracleSelector",
    "ScriptedDetector",
    "MUCB",
    "CUSUMUCB",
    "LinUCB",
    "LinTS",
    "ContextUCB",
    "FewShotClassifierSelector",
    "PLASTICPolicySelector",
    "PPOScheduler",
    "LLMSelector",
    "LLMInitLinUCB",
]
