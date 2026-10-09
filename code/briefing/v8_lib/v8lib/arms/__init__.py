"""Selector arms exposed by v8lib."""

from .bandit import CUSUMUCB, MUCB
from .base import Selector
from .classifier import FewShotClassifierSelector
from .contextual import ContextUCB, LLMInitLinUCB, LinTS, LinUCB
from .llm import FramedLLMSelector, LLMSelector
from .plastic import PLASTICPolicySelector
from .ppo import PPOScheduler
from .simple import FixedSelector, RandomSelector, ScriptedDetector, TypeOracleSelector
from .text import ScriptedTextSelector, SentenceEmbedder, TextBowSelector, TextEmbedSelector, TextLinUCB

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
    "FramedLLMSelector",
    "LLMInitLinUCB",
    "ScriptedTextSelector",
    "SentenceEmbedder",
    "TextBowSelector",
    "TextEmbedSelector",
    "TextLinUCB",
]
