"""Test reward functions."""
import pytest
from carl_studio.training.rewards.base import extract_text, extract_json
from carl_studio.training.rewards.task import (
    tool_call_format_reward,
    tool_selection_reward,
    conciseness_reward,
)
from carl_studio.training.rewards.vlm import (
    coordinate_format_reward,
    click_accuracy_reward,
    precision_reward,
)


class TestBase:
    def test_extract_text_string(self):
        assert extract_text("hello") == "hello"

    def test_extract_text_conversational(self):
        conv = [{"role": "assistant", "content": "world"}]
        assert extract_text(conv) == "world"

    def test_extract_json_fenced(self):
        text = '```json\n{"name": "read_file", "arguments": {"path": "/x"}}\n```'
        result = extract_json(text)
        assert result is not None
        assert result["name"] == "read_file"

    def test_extract_json_bare(self):
        text = 'Some text {"name": "write_file"} more text'
        result = extract_json(text)
        assert result is not None
        assert result["name"] == "write_file"


class TestTaskRewards:
    def test_format_reward_perfect(self):
        scores = tool_call_format_reward(
            ['{"name": "read_file", "arguments": {"path": "/x"}}']
        )
        assert scores == [1.0]

    def test_format_reward_no_json(self):
        scores = tool_call_format_reward(["just text"])
        assert scores == [0.0]

    def test_selection_reward_correct(self):
        scores = tool_selection_reward(
            ['{"name": "read_file"}'],
            expected_tools=[["read_file"]]
        )
        assert scores == [1.0]

    def test_conciseness_short(self):
        scores = conciseness_reward(["short"])
        assert scores[0] > 0.9


class TestVLMRewards:
    def test_coordinate_format_clean(self):
        scores = coordinate_format_reward(["(320, 450)"])
        assert scores == [1.0]

    def test_coordinate_format_verbose(self):
        scores = coordinate_format_reward(["The button is at (320, 450) in the screenshot"])
        assert scores == [0.6]

    def test_coordinate_format_missing(self):
        scores = coordinate_format_reward(["no coordinates here"])
        assert scores == [0.0]

    def test_click_accuracy_inside(self):
        scores = click_accuracy_reward(
            ["(100, 100)"],
            bbox=[[50, 50, 150, 150]]
        )
        assert scores == [1.0]

    def test_click_accuracy_outside(self):
        scores = click_accuracy_reward(
            ["(500, 500)"],
            bbox=[[50, 50, 150, 150]]
        )
        assert scores[0] < 0.5

    def test_precision_center(self):
        scores = precision_reward(
            ["(100, 100)"],
            bbox=[[50, 50, 150, 150]]
        )
        assert scores[0] > 0.9  # Very close to center

    def test_precision_far(self):
        scores = precision_reward(
            ["(500, 500)"],
            bbox=[[50, 50, 150, 150]]
        )
        # 1000px scale: dist=~566px → score=~0.43 (gradient signal at medium distance)
        assert 0.3 < scores[0] < 0.6
