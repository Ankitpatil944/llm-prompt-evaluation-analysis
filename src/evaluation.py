"""
Simulated evaluation pipeline for an LLM response study.

This module runs a small prompt-format experiment over multiple evaluation
cases, scores each response with transparent heuristics, and exports results
for later analysis.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from prompt_design import EvaluationCase, build_prompt_suite, build_study_cases


@dataclass
class EvaluationResult:
    """Represents the outcome of one simulated evaluation run."""

    study_id: str
    domain: str
    prompt_format: str
    dimension_scores: Dict[str, float]
    weighted_total: float
    verdict: str
    notes: List[str]


PROMPT_STYLE_BONUS = {
    "baseline": {"clarity": 0.0, "instruction_adherence": 0.0},
    "rubric_strict": {"clarity": 0.2, "instruction_adherence": 0.4, "completeness": 0.15},
    "comparative": {"relevance": 0.2, "completeness": 0.3},
    "research_audit": {"factual_caution": 0.4, "instruction_adherence": 0.3, "relevance": 0.1},
}

DIMENSION_WEIGHTS = {
    "relevance": 0.25,
    "completeness": 0.25,
    "clarity": 0.20,
    "factual_caution": 0.15,
    "instruction_adherence": 0.15,
}


def _normalize_score(raw_score: float) -> float:
    """Clamp scores to a realistic 1-5 evaluation scale."""
    return round(max(1.0, min(5.0, raw_score)), 2)


def _keyword_overlap(task: str, response: str) -> int:
    """Count rough keyword overlap between task and candidate response."""
    task_tokens = {word.strip(".,:;").lower() for word in task.split() if len(word) > 4}
    response_tokens = {word.strip(".,:;").lower() for word in response.split()}
    return len(task_tokens.intersection(response_tokens))


def _trait_coverage(expected_traits: List[str], response: str) -> int:
    """Estimate how many desired concepts appear in the response."""
    response_lower = response.lower()
    hits = 0
    for trait in expected_traits:
        keywords = [word.lower().strip(".,:;") for word in trait.split() if len(word) > 5]
        if any(keyword in response_lower for keyword in keywords):
            hits += 1
    return hits


def _clarity_score(response: str) -> float:
    """Approximate clarity from sentence structure and response length."""
    sentences = [segment.strip() for segment in response.split(".") if segment.strip()]
    word_count = len(response.split())

    score = 3.0
    if 18 <= word_count <= 95:
        score += 1.0
    if len(sentences) >= 2:
        score += 0.45
    if any(marker in response.lower() for marker in ["should", "when", "because", "for example"]):
        score += 0.25
    return _normalize_score(score)


def _factual_caution_score(response: str) -> float:
    """Reward calibrated language and penalize unsupported certainty."""
    response_lower = response.lower()
    score = 3.0

    cautious_markers = ["may", "can", "could", "uncertain", "risk", "review", "should"]
    certainty_markers = ["always", "never", "guaranteed", "definitely", "proves"]

    score += 0.22 * sum(marker in response_lower for marker in cautious_markers)
    score -= 0.35 * sum(marker in response_lower for marker in certainty_markers)

    return _normalize_score(score)


def simulate_response_scoring(case: EvaluationCase, prompt_format: str) -> EvaluationResult:
    """
    Simulate evaluator scoring for a candidate response under a prompt style.

    The scoring is heuristic by design. It is intended to resemble a compact
    LLM-as-a-judge prototype while remaining fully deterministic.
    """

    overlap = _keyword_overlap(case.task, case.candidate_response)
    trait_hits = _trait_coverage(case.expected_traits, case.candidate_response)
    style_bonus = PROMPT_STYLE_BONUS.get(prompt_format, {})

    dimension_scores = {
        "relevance": _normalize_score(2.7 + 0.32 * overlap + style_bonus.get("relevance", 0.0)),
        "completeness": _normalize_score(2.35 + 0.52 * trait_hits + style_bonus.get("completeness", 0.0)),
        "clarity": _normalize_score(_clarity_score(case.candidate_response) + style_bonus.get("clarity", 0.0)),
        "factual_caution": _normalize_score(
            _factual_caution_score(case.candidate_response) + style_bonus.get("factual_caution", 0.0)
        ),
        "instruction_adherence": _normalize_score(
            2.65 + 0.42 * trait_hits + style_bonus.get("instruction_adherence", 0.0)
        ),
    }

    weighted_total = round(
        sum(dimension_scores[dimension] * weight for dimension, weight in DIMENSION_WEIGHTS.items()),
        2,
    )

    if weighted_total >= 4.4:
        verdict = "strong"
    elif weighted_total >= 3.6:
        verdict = "acceptable"
    else:
        verdict = "needs_revision"

    notes = _build_notes(case, dimension_scores, weighted_total, prompt_format)

    return EvaluationResult(
        study_id=case.study_id,
        domain=case.domain,
        prompt_format=prompt_format,
        dimension_scores=dimension_scores,
        weighted_total=weighted_total,
        verdict=verdict,
        notes=notes,
    )


def _build_notes(
    case: EvaluationCase, dimension_scores: Dict[str, float], weighted_total: float, prompt_format: str
) -> List[str]:
    """Generate brief analyst notes that resemble experiment observations."""
    strongest_dimension = max(dimension_scores, key=dimension_scores.get)
    weakest_dimension = min(dimension_scores, key=dimension_scores.get)

    notes = [
        f"{case.study_id} in domain '{case.domain}' produced an aggregate score of {weighted_total}.",
        f"Strongest dimension: {strongest_dimension} ({dimension_scores[strongest_dimension]}).",
        f"Weakest dimension: {weakest_dimension} ({dimension_scores[weakest_dimension]}).",
    ]

    if prompt_format == "research_audit":
        notes.append("Audit framing improved caution-sensitive scoring and experiment traceability.")
    elif prompt_format == "rubric_strict":
        notes.append("Strict rubric framing increased structure and instruction-following pressure.")
    elif prompt_format == "comparative":
        notes.append("Comparative framing favored broader coverage and tradeoff-aware evaluation.")
    else:
        notes.append("Baseline framing behaved like a low-control evaluation condition.")

    return notes


def run_prompt_format_experiment(case: EvaluationCase) -> List[EvaluationResult]:
    """Evaluate one case across all supported prompt formats."""
    prompt_suite = build_prompt_suite(case)
    return [simulate_response_scoring(case, prompt_format) for prompt_format in prompt_suite]


def run_full_study(cases: List[EvaluationCase]) -> List[EvaluationResult]:
    """Run the experiment across every study case."""
    all_results: List[EvaluationResult] = []
    for case in cases:
        all_results.extend(run_prompt_format_experiment(case))
    return all_results


def summarize_results(results: List[EvaluationResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate weighted totals by prompt format."""
    prompt_formats = sorted({result.prompt_format for result in results})
    summary: Dict[str, Dict[str, float]] = {}

    for prompt_format in prompt_formats:
        format_results = [result for result in results if result.prompt_format == prompt_format]
        summary[prompt_format] = {
            "average_weighted_total": round(mean(result.weighted_total for result in format_results), 2),
            "best_weighted_total": round(max(result.weighted_total for result in format_results), 2),
            "worst_weighted_total": round(min(result.weighted_total for result in format_results), 2),
        }

    return summary


def export_results(results: List[EvaluationResult], summary: Dict[str, Dict[str, float]]) -> Path:
    """Write the latest study output to a JSON artifact."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "latest_study_results.json"
    payload = {
        "study_name": "LLM Evaluation Pipeline Study",
        "num_cases": len({result.study_id for result in results}),
        "num_runs": len(results),
        "results": [asdict(result) for result in results],
        "summary": summary,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def print_experiment_report(results: List[EvaluationResult], summary: Dict[str, Dict[str, float]]) -> None:
    """Display a research-style console report."""
    print("\nLLM Evaluation Pipeline Study")
    print("=" * 88)

    current_case = None
    for result in results:
        if result.study_id != current_case:
            current_case = result.study_id
            print(f"\nCase: {result.study_id} | Domain: {result.domain}")
            print("-" * 88)

        print(f"Prompt Format      : {result.prompt_format}")
        print(f"Weighted Total     : {result.weighted_total}")
        print(f"Verdict            : {result.verdict}")
        print("Dimension Scores   :")
        for dimension, score in result.dimension_scores.items():
            print(f"  - {dimension:<20} {score}")
        print("Analyst Notes      :")
        for note in result.notes:
            print(f"  - {note}")
        print("-" * 88)

    print("\nCross-Case Summary")
    print("=" * 88)
    for prompt_format, metrics in summary.items():
        print(f"{prompt_format}")
        for metric_name, metric_value in metrics.items():
            print(f"  - {metric_name:<24} {metric_value}")

    best_prompt = max(summary.items(), key=lambda item: item[1]["average_weighted_total"])
    print("\nBest Overall Prompt Format")
    print("=" * 88)
    print(f"{best_prompt[0]} with average weighted total {best_prompt[1]['average_weighted_total']}")


if __name__ == "__main__":
    # Example usage: run the full study, print the report, and export JSON results.
    study_cases = build_study_cases()
    study_results = run_full_study(study_cases)
    study_summary = summarize_results(study_results)
    artifact_path = export_results(study_results, study_summary)

    print_experiment_report(study_results, study_summary)
    print(f"\nResults artifact written to: {artifact_path}")
