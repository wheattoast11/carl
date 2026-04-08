"""Test hub integration."""
from carl_studio.hub import generate_model_card


def test_model_card_has_metadata():
    card = generate_model_card(
        model_id="test/model",
        base_model="Qwen/Qwen3-8B",
        method="grpo",
        dataset="test/data",
        phi_mean=0.72,
        cloud_quality=0.45,
    )
    assert "carl-studio" in card
    assert "Qwen/Qwen3-8B" in card
    assert "0.7200" in card
    assert "0.4500" in card
    assert "kappa" in card.lower()


def test_model_card_has_conservation_law():
    card = generate_model_card(
        model_id="test/model",
        base_model="test",
        method="sft",
        dataset="test",
    )
    assert "21.333333" in card  # KAPPA
    assert "0.187500" in card   # SIGMA
    assert "4.0" in card        # KAPPA * SIGMA
