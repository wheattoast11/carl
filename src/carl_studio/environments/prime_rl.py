"""Prime-RL environment adapter -- for prime-rl's async training framework."""
from __future__ import annotations


class PrimeRLAdapter:
    """
    Adapter for Prime Intellect's prime-rl framework.

    prime-rl uses TOML configs and runs on large GPU clusters.
    This adapter generates the TOML config with CARL reward
    functions injected.

    Note: prime-rl is a separate framework -- this adapter creates
    the bridge, not the training loop.
    """

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path

    def generate_config(
        self,
        model_id: str,
        dataset_id: str,
        carl_weight: float = 0.3,
    ) -> str:
        """Generate a prime-rl TOML config with CARL rewards."""
        return f'''
[model]
name = "{model_id}"

[training]
method = "grpo"
carl_weight = {carl_weight}

[data]
dataset = "{dataset_id}"
'''
