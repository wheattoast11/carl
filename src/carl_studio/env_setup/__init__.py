"""carl env — progressive-disclosure environment-setup wizard (v0.12 MVP).

Scope for v0.12: 4 core questions building a training-config YAML.
Full 7-question flow from ``docs/v09_carl_env_design.md`` is targeted
for v0.13 once this MVP is in user hands.

Architecture is functor-composed: each question is a typed transition
``EnvState → (Question, Answer → EnvState)``. State is
JSON-serializable so users can resume via ``--resume``.
"""

from __future__ import annotations

from carl_studio.env_setup.questions import QUESTION_REGISTRY, Question
from carl_studio.env_setup.render import render_training_config_yaml
from carl_studio.env_setup.state import EnvState

__all__ = [
    "EnvState",
    "Question",
    "QUESTION_REGISTRY",
    "render_training_config_yaml",
]
