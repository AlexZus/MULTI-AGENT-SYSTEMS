"""Structured output schemas for agent responses."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SpecOutput(BaseModel):
    """Business Analyst output — project specification."""

    title: str = Field(description="Short title for the task")
    requirements: list[str] = Field(
        description="Functional requirements (at least 3)",
        min_length=1,
    )
    acceptance_criteria: list[str] = Field(
        description="Acceptance criteria the implementation must satisfy (at least 2)",
        min_length=1,
    )
    estimated_complexity: Literal["simple", "medium", "complex"] = Field(
        description="Estimated implementation complexity"
    )
    notes: str = Field(default="", description="Any additional technical notes or constraints")

    @field_validator("requirements")
    @classmethod
    def at_least_one_requirement(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("requirements must not be empty")
        return v

    @field_validator("acceptance_criteria")
    @classmethod
    def at_least_one_criterion(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("acceptance_criteria must not be empty")
        return v


class CodeOutput(BaseModel):
    """Developer output — code implementation summary."""

    summary: str = Field(description="Brief description of what was implemented")
    files_created: list[str] = Field(
        description="List of project-relative file paths created or modified",
        min_length=1,
    )
    dependencies_installed: list[str] = Field(
        default_factory=list,
        description="Python packages installed via pip (if any)",
    )
    tests_passed: bool = Field(
        default=False,
        description="Whether automated tests were run and passed",
    )
    notes: str = Field(default="", description="Any known issues or follow-up items")

    @field_validator("files_created")
    @classmethod
    def at_least_one_file(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("files_created must not be empty")
        return v


class ReviewOutput(BaseModel):
    """QA agent output — code review result."""

    verdict: Literal["APPROVED", "REVISION_NEEDED"] = Field(
        description="Whether the implementation passes quality review"
    )
    score: float = Field(
        description="Quality score between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of issues found (empty if APPROVED)",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Improvement suggestions (can be present even when APPROVED)",
    )
    tests_run: int = Field(
        default=0,
        description="Number of tests executed during review",
    )
    tests_passed: int = Field(
        default=0,
        description="Number of tests that passed",
    )
