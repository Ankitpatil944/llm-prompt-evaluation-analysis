"""
Prompt design utilities for an LLM evaluation study.

This module builds structured evaluator prompts for a controlled experiment on
prompt formats used in LLM response assessment workflows.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvaluationCase:
    """Container for a single response-evaluation scenario."""

    study_id: str
    task: str
    candidate_response: str
    rubric: Dict[str, str]
    expected_traits: List[str] = field(default_factory=list)
    output_format: str = "JSON with per-dimension scores, rationale, and final verdict."
    domain: str = "general"
    difficulty: str = "medium"


def _format_bullets(items: List[str], fallback: str) -> str:
    """Render a list as a bullet block."""
    if not items:
        return f"- {fallback}"
    return "\n".join(f"- {item}" for item in items)


def _format_rubric(rubric: Dict[str, str]) -> str:
    """Convert rubric dimensions into a readable block."""
    return "\n".join(f"- {dimension}: {description}" for dimension, description in rubric.items())


def generate_evaluation_prompt(case: EvaluationCase, prompt_format: str = "rubric_strict") -> str:
    """
    Build a structured evaluator prompt for one study case.

    Supported prompt formats:
    - baseline
    - rubric_strict
    - comparative
    - research_audit
    """

    rubric_block = _format_rubric(case.rubric)
    trait_block = _format_bullets(case.expected_traits, "No explicit expected traits supplied.")

    templates = {
        "baseline": "\n".join(
            [
                "You are evaluating a candidate LLM response.",
                "",
                f"Study ID: {case.study_id}",
                f"Domain: {case.domain}",
                f"Difficulty: {case.difficulty}",
                "",
                "Task:",
                case.task,
                "",
                "Candidate Response:",
                case.candidate_response,
                "",
                "Rubric:",
                rubric_block,
                "",
                "Expected Traits:",
                trait_block,
                "",
                "Return Format:",
                case.output_format,
            ]
        ),
        "rubric_strict": "\n".join(
            [
                "System Role: Senior evaluator in an LLM prompt-engineering study.",
                "",
                f"Study ID: {case.study_id}",
                f"Domain: {case.domain}",
                f"Difficulty: {case.difficulty}",
                "Objective: Judge the candidate response against a fixed research rubric.",
                "",
                "Evaluation Task:",
                case.task,
                "",
                "Candidate Response Under Review:",
                case.candidate_response,
                "",
                "Rubric Dimensions:",
                rubric_block,
                "",
                "Expected Response Traits:",
                trait_block,
                "",
                "Instructions:",
                "- Score each dimension from 1 to 5.",
                "- Use concise evidence-based reasoning.",
                "- Penalize unsupported certainty, missing constraints, and shallow coverage.",
                "- End with one recommendation to improve the candidate response.",
                "",
                "Required Output:",
                case.output_format,
            ]
        ),
        "comparative": "\n".join(
            [
                "You are comparing a candidate LLM response against an ideal answer profile.",
                "",
                f"Study ID: {case.study_id}",
                f"Domain: {case.domain}",
                f"Difficulty: {case.difficulty}",
                "",
                "Task Context:",
                case.task,
                "",
                "Ideal Qualities:",
                trait_block,
                "",
                "Candidate Response:",
                case.candidate_response,
                "",
                "Scoring Rubric:",
                rubric_block,
                "",
                "Evaluation Method:",
                "- Identify the strongest qualities first.",
                "- Identify the most important weaknesses second.",
                "- Score all dimensions from 1 to 5.",
                "- Call out tradeoffs between brevity, completeness, and caution when relevant.",
                "",
                "Deliverable:",
                case.output_format,
            ]
        ),
        "research_audit": "\n".join(
            [
                "Research Audit Prompt",
                "",
                "You are assisting with an LLM evaluation pipeline study.",
                "Produce a rigorous, structured judgment suitable for experiment logging.",
                "",
                "Metadata:",
                f"- Study ID: {case.study_id}",
                f"- Domain: {case.domain}",
                f"- Difficulty: {case.difficulty}",
                "- Prompt Format: research_audit",
                "",
                "Source Task:",
                case.task,
                "",
                "Response Being Audited:",
                case.candidate_response,
                "",
                "Audit Rubric:",
                rubric_block,
                "",
                "Desired Response Properties:",
                trait_block,
                "",
                "Audit Requirements:",
                "- Provide dimension scores from 1 to 5.",
                "- Note whether the response appears calibrated or overconfident.",
                "- Highlight missing constraints, weak evidence, or vague phrasing.",
                "- Provide a short audit summary for downstream experiment analysis.",
                "",
                "Output Contract:",
                case.output_format,
            ]
        ),
    }

    if prompt_format not in templates:
        supported = ", ".join(sorted(templates))
        raise ValueError(f"Unsupported prompt format '{prompt_format}'. Supported formats: {supported}")

    return templates[prompt_format]


def build_prompt_suite(case: EvaluationCase) -> Dict[str, str]:
    """Generate all supported prompt variants for a single case."""
    formats = ["baseline", "rubric_strict", "comparative", "research_audit"]
    return {prompt_format: generate_evaluation_prompt(case, prompt_format) for prompt_format in formats}


def build_study_cases() -> List[EvaluationCase]:
    """Create a small set of experiment cases across different domains."""
    rubric = {
        "relevance": "Does the response address the task directly and stay on scope?",
        "completeness": "Does it cover the key concepts, risks, or requested actions?",
        "clarity": "Is the response readable, organized, and easy to interpret?",
        "factual_caution": "Does it avoid unjustified certainty and acknowledge risk appropriately?",
        "instruction_adherence": "Does it follow the requested framing, constraints, and level of detail?",
    }

    return [
        EvaluationCase(
            study_id="study-001",
            domain="healthcare",
            difficulty="high",
            task=(
                "Summarize the risks of using LLM outputs in a medical support workflow "
                "and suggest safeguards for human oversight."
            ),
            candidate_response=(
                "LLMs can help clinicians save time, but they may hallucinate facts or miss context. "
                "A safe workflow should require clinician review, logging, uncertainty escalation, "
                "and clear boundaries on when automated suggestions can be used."
            ),
            rubric=rubric,
            expected_traits=[
                "Mentions hallucination or factual inaccuracy risk",
                "Notes the need for human review",
                "Includes operational safeguards",
                "Uses cautious, calibrated language",
            ],
        ),
        EvaluationCase(
            study_id="study-002",
            domain="enterprise_policy",
            difficulty="medium",
            task=(
                "Explain how an internal LLM assistant should handle confidential financial documents "
                "and summarize the main governance controls required."
            ),
            candidate_response=(
                "The assistant should not freely expose confidential financial content. "
                "Access controls, audit logs, redaction rules, role-based permissions, "
                "and human approval for sensitive disclosures are important safeguards."
            ),
            rubric=rubric,
            expected_traits=[
                "References confidential or sensitive data handling",
                "Mentions access control or permissions",
                "Includes auditability or logging",
                "States governance or approval requirements",
            ],
        ),
        EvaluationCase(
            study_id="study-003",
            domain="customer_support",
            difficulty="medium",
            task=(
                "Draft a concise policy for when an LLM customer-support agent should escalate "
                "a case to a human representative."
            ),
            candidate_response=(
                "The agent should escalate when the user is frustrated, asks for refunds, "
                "mentions legal or security concerns, or when the model is uncertain. "
                "Escalation should include a short case summary and confidence note."
            ),
            rubric=rubric,
            expected_traits=[
                "Defines clear escalation triggers",
                "Mentions uncertainty handling",
                "Includes operational handoff guidance",
                "Keeps the policy concise and actionable",
            ],
        ),
    ]


def example_case() -> EvaluationCase:
    """Return the first study case for simple examples."""
    return build_study_cases()[0]


if __name__ == "__main__":
    # Example usage: generate and inspect all evaluator prompt variants.
    case = example_case()
    prompt_suite = build_prompt_suite(case)

    for name, prompt in prompt_suite.items():
        print("=" * 88)
        print(f"PROMPT FORMAT: {name}")
        print("=" * 88)
        print(prompt)
        print()
