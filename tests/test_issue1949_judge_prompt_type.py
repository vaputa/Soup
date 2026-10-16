"""Regression tests for #1949: the judge JSONL reader forwarded a non-string
``prompt`` straight to its generator, and a non-object row died at ``row.get``
with no physical-line or field diagnostic.

After the fix every malformed row is rejected with the physical line number
and the offending type before ``generate_fn`` is called, and the diagnostic
surfaces through the per-task error the same way every other judge failure
does. A missing ``prompt`` key keeps its historical empty-string behaviour
(out of scope here).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

JUDGE_BASE = "http://localhost:8000"


def _suite(prompts: str) -> EvalSuite:
    return EvalSuite(
        suite="s",
        tasks=[GateTask(type="judge", name="quality", threshold=0.7,
                        prompts=prompts, judge_model=f"{JUDGE_BASE}/m")],
    )


def _write_prompts(directory: Path, rows: list) -> str:
    prompts = directory / "prompts.jsonl"
    lines = []
    for row in rows:
        lines.append(row if isinstance(row, str) else json.dumps(row))
    prompts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(prompts)


def _run(rows: list) -> tuple[object, list[str]]:
    """Run the gate over ``rows`` and return (task_result, prompts_seen).

    GateTask.prompts must resolve under cwd (#1902 policy), so the fixture
    lives in a TemporaryDirectory anchored at the cwd like the issue repro;
    the caller holds the directory open for the whole gate run.
    """
    seen: list[str] = []
    with tempfile.TemporaryDirectory(dir=".") as directory:
        prompts = _write_prompts(Path(directory), rows)
        result = run_gate(
            _suite(prompts), generate_fn=lambda prompt: (seen.append(prompt), "x")[1],
        )
        (task,) = result.task_results
        return task, seen


class TestIssue1949JudgePromptType:
    @pytest.mark.parametrize("bad,prompt_type", [
        ({"prompt": None}, "NoneType"),
        ({"prompt": 1}, "int"),
        ({"prompt": True}, "bool"),
        ({"prompt": []}, "list"),
        ({"prompt": {}}, "dict"),
        ({"prompt": 3.5}, "float"),
    ])
    def test_non_string_prompt_rejected_with_line_and_type(
        self, bad, prompt_type: str,
    ) -> None:
        task, seen = _run([bad, {"prompt": "control"}])
        assert seen == []  # nothing reached the generator
        assert task.score is None
        assert task.passed is False
        assert "line 1" in task.error
        assert f"'prompt' must be a string, got {prompt_type}" in task.error

    @pytest.mark.parametrize("bad", [1, '"just a string"', True, [1, 2], 3.5])
    def test_non_object_row_rejected_with_line(
        self, bad,
    ) -> None:
        task, seen = _run([bad])
        assert seen == []
        assert task.score is None
        assert "line 1" in task.error
        assert "expected a JSON object with a 'prompt' field" in task.error

    def test_json_error_carries_line_number(self) -> None:
        task, _ = _run(["{not json"])
        assert task.score is None
        assert "line 1" in task.error
        assert "invalid JSONL" in task.error

    def test_line_number_counts_physical_lines(self) -> None:
        # Blank lines between rows still count toward the physical line number.
        seen: list[str] = []
        with tempfile.TemporaryDirectory(dir=".") as directory:
            prompts = Path(directory) / "prompts.jsonl"
            prompts.write_text(
                json.dumps({"prompt": "ok"}) + "\n\n" + "42" + "\n", encoding="utf-8",
            )
            result = run_gate(
                _suite(str(prompts)),
                generate_fn=lambda prompt: (seen.append(prompt), "x")[1],
            )
        (task,) = result.task_results
        assert seen == ["ok"]  # first row generated before the failure
        assert "line 3" in task.error

    def test_string_prompts_still_reach_the_generator(self) -> None:
        task, seen = _run(
            [{"prompt": "first"}, {"prompt": "second"}, {"prompt": ""}],
        )
        assert seen == ["first", "second", ""]

    def test_missing_prompt_key_keeps_empty_string_behaviour(
        self,
    ) -> None:
        # Out of scope for #1949: a missing key is not an explicit non-string.
        task, seen = _run([{"response_hint": "no prompt field"}])
        assert seen == [""]
