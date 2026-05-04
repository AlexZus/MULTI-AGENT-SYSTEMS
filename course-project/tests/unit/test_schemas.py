"""Unit tests for agents/schemas.py — no external services required."""

import pytest
from pydantic import ValidationError

from agents.schemas import SpecOutput, CodeOutput, ReviewOutput


# ---------------------------------------------------------------------------
# SpecOutput
# ---------------------------------------------------------------------------

class TestSpecOutput:
    def _valid(self, **overrides):
        data = {
            "title": "Calculator App",
            "requirements": ["Add two numbers", "Subtract two numbers", "Handle division by zero"],
            "acceptance_criteria": ["add(2, 3) returns 5", "div(1, 0) raises ValueError"],
            "estimated_complexity": "simple",
        }
        data.update(overrides)
        return data

    def test_valid_spec(self):
        spec = SpecOutput(**self._valid())
        assert spec.title == "Calculator App"
        assert len(spec.requirements) == 3
        assert spec.estimated_complexity == "simple"
        assert spec.notes == ""  # default

    def test_valid_with_notes(self):
        spec = SpecOutput(**self._valid(notes="Use Python 3.11+"))
        assert spec.notes == "Use Python 3.11+"

    def test_all_complexity_values(self):
        for complexity in ("simple", "medium", "complex"):
            spec = SpecOutput(**self._valid(estimated_complexity=complexity))
            assert spec.estimated_complexity == complexity

    def test_invalid_complexity(self):
        with pytest.raises(ValidationError):
            SpecOutput(**self._valid(estimated_complexity="easy"))

    def test_empty_requirements_raises(self):
        with pytest.raises(ValidationError):
            SpecOutput(**self._valid(requirements=[]))

    def test_empty_acceptance_criteria_raises(self):
        with pytest.raises(ValidationError):
            SpecOutput(**self._valid(acceptance_criteria=[]))

    def test_missing_title_raises(self):
        data = self._valid()
        del data["title"]
        with pytest.raises(ValidationError):
            SpecOutput(**data)

    def test_missing_estimated_complexity_raises(self):
        data = self._valid()
        del data["estimated_complexity"]
        with pytest.raises(ValidationError):
            SpecOutput(**data)

    def test_model_dump_roundtrip(self):
        spec = SpecOutput(**self._valid())
        dumped = spec.model_dump()
        restored = SpecOutput(**dumped)
        assert restored == spec


# ---------------------------------------------------------------------------
# CodeOutput
# ---------------------------------------------------------------------------

class TestCodeOutput:
    def _valid(self, **overrides):
        data = {
            "summary": "Implemented calculator with four operations",
            "files_created": ["calculator/main.py", "calculator/tests/test_main.py"],
        }
        data.update(overrides)
        return data

    def test_valid_code_output(self):
        out = CodeOutput(**self._valid())
        assert len(out.files_created) == 2
        assert out.tests_passed is False  # default
        assert out.dependencies_installed == []  # default

    def test_with_optional_fields(self):
        out = CodeOutput(**self._valid(
            dependencies_installed=["requests"],
            tests_passed=True,
            notes="All tests passing",
        ))
        assert out.tests_passed is True
        assert "requests" in out.dependencies_installed

    def test_empty_files_created_raises(self):
        with pytest.raises(ValidationError):
            CodeOutput(**self._valid(files_created=[]))

    def test_missing_summary_raises(self):
        data = self._valid()
        del data["summary"]
        with pytest.raises(ValidationError):
            CodeOutput(**data)

    def test_model_dump_roundtrip(self):
        out = CodeOutput(**self._valid(tests_passed=True))
        restored = CodeOutput(**out.model_dump())
        assert restored == out


# ---------------------------------------------------------------------------
# ReviewOutput
# ---------------------------------------------------------------------------

class TestReviewOutput:
    def _valid_approved(self, **overrides):
        data = {
            "verdict": "APPROVED",
            "score": 0.9,
            "tests_run": 5,
            "tests_passed": 5,
        }
        data.update(overrides)
        return data

    def _valid_revision(self, **overrides):
        data = {
            "verdict": "REVISION_NEEDED",
            "score": 0.4,
            "issues": ["Missing error handling", "No docstrings"],
            "tests_run": 3,
            "tests_passed": 1,
        }
        data.update(overrides)
        return data

    def test_valid_approved(self):
        review = ReviewOutput(**self._valid_approved())
        assert review.verdict == "APPROVED"
        assert review.score == 0.9
        assert review.issues == []  # default

    def test_valid_revision_needed(self):
        review = ReviewOutput(**self._valid_revision())
        assert review.verdict == "REVISION_NEEDED"
        assert len(review.issues) == 2

    def test_score_bounds(self):
        ReviewOutput(**self._valid_approved(score=0.0))
        ReviewOutput(**self._valid_approved(score=1.0))

    def test_score_above_1_raises(self):
        with pytest.raises(ValidationError):
            ReviewOutput(**self._valid_approved(score=1.1))

    def test_score_below_0_raises(self):
        with pytest.raises(ValidationError):
            ReviewOutput(**self._valid_approved(score=-0.1))

    def test_invalid_verdict_raises(self):
        with pytest.raises(ValidationError):
            ReviewOutput(**self._valid_approved(verdict="PASS"))

    def test_missing_verdict_raises(self):
        data = self._valid_approved()
        del data["verdict"]
        with pytest.raises(ValidationError):
            ReviewOutput(**data)

    def test_missing_score_raises(self):
        data = self._valid_approved()
        del data["score"]
        with pytest.raises(ValidationError):
            ReviewOutput(**data)

    def test_model_dump_roundtrip(self):
        review = ReviewOutput(**self._valid_revision())
        restored = ReviewOutput(**review.model_dump())
        assert restored == review

    def test_suggestions_default_empty(self):
        review = ReviewOutput(**self._valid_approved())
        assert review.suggestions == []

    def test_with_suggestions(self):
        review = ReviewOutput(**self._valid_approved(suggestions=["Add type hints"]))
        assert review.suggestions == ["Add type hints"]
