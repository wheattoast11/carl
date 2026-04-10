"""Test hub integration."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from carl_studio.hub import generate_model_card
from carl_studio.hub.models import push_with_metadata


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
    assert "0.187500" in card  # SIGMA
    assert "4.0" in card  # KAPPA * SIGMA


@pytest.mark.anyio
async def test_push_with_metadata_uploads_folder_and_model_card(tmp_path: Path):
    artifact_dir = tmp_path / "adapter"
    artifact_dir.mkdir()
    (artifact_dir / "adapter_model.safetensors").write_text("weights")

    mock_api = Mock()
    with (
        patch("huggingface_hub.HfApi", return_value=mock_api),
        patch("huggingface_hub.get_token", return_value="hf_test"),
    ):
        url = await push_with_metadata(
            model_path=str(artifact_dir),
            repo_id="org/model",
            base_model="Tesslate/OmniCoder-9B",
            method="grpo",
            dataset="org/data",
            private=True,
        )

    mock_api.create_repo.assert_called_once_with("org/model", exist_ok=True, private=True)
    mock_api.upload_folder.assert_called_once()
    upload_folder_kwargs = mock_api.upload_folder.call_args.kwargs
    assert upload_folder_kwargs["folder_path"] == str(artifact_dir)
    assert upload_folder_kwargs["repo_id"] == "org/model"
    mock_api.upload_file.assert_called_once()
    upload_file_kwargs = mock_api.upload_file.call_args.kwargs
    assert upload_file_kwargs["path_in_repo"] == "README.md"
    assert url == "https://huggingface.co/org/model"


@pytest.mark.anyio
async def test_push_with_metadata_uploads_single_file(tmp_path: Path):
    artifact_file = tmp_path / "adapter_model.safetensors"
    artifact_file.write_text("weights")

    mock_api = Mock()
    with (
        patch("huggingface_hub.HfApi", return_value=mock_api),
        patch("huggingface_hub.get_token", return_value="hf_test"),
    ):
        await push_with_metadata(
            model_path=str(artifact_file),
            repo_id="org/model",
            base_model="Tesslate/OmniCoder-9B",
            method="grpo",
            dataset="org/data",
        )

    mock_api.upload_folder.assert_not_called()
    assert mock_api.upload_file.call_count == 2
    artifact_upload = mock_api.upload_file.call_args_list[0].kwargs
    assert artifact_upload["path_in_repo"] == artifact_file.name


@pytest.mark.anyio
async def test_push_with_metadata_missing_path_raises(tmp_path: Path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Model path not found"):
        await push_with_metadata(
            model_path=str(missing),
            repo_id="org/model",
            base_model="base/model",
            method="grpo",
            dataset="org/data",
        )
