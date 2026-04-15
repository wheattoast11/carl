"""Built-in CARL skills (merit badges)."""
from carl_studio.skills.builtins.deployer import DeployerSkill
from carl_studio.skills.builtins.grader import GraderSkill
from carl_studio.skills.builtins.observer import ObserverSkill
from carl_studio.skills.builtins.synthesizer import SynthesizerSkill
from carl_studio.skills.builtins.trainer import TrainerSkill

BUILTIN_SKILLS: list[ObserverSkill | GraderSkill | TrainerSkill | SynthesizerSkill | DeployerSkill] = [
    ObserverSkill(),
    GraderSkill(),
    TrainerSkill(),
    SynthesizerSkill(),
    DeployerSkill(),
]

__all__ = [
    "ObserverSkill",
    "GraderSkill",
    "TrainerSkill",
    "SynthesizerSkill",
    "DeployerSkill",
    "BUILTIN_SKILLS",
]
