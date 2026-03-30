# LLM Evaluation Pipeline Study

## Objective

This repository simulates a research-oriented LLM evaluation pipeline focused on prompt engineering, evaluator prompt design, and structured scoring. The goal is to study how different evaluator prompt formats influence downstream judgment quality when assessing candidate LLM responses.

The project is intentionally framed like an early-stage experiment:

- fixed evaluation cases
- controlled prompt variants
- multi-dimensional scoring
- aggregate summaries across runs
- logged artifacts for later inspection

## Project Structure

```text
llm-prompt-evaluation-analysis/
+-- README.md
+-- src/
    +-- evaluation.py
    +-- prompt_design.py
```

## Approach

The pipeline is split into two layers.

### 1. Prompt Design Layer

The prompt design module generates evaluator prompts from structured experiment cases. Each case includes:

- a study identifier
- the original user task
- a candidate LLM response
- a rubric with scoring dimensions
- expected traits for a strong answer
- an evaluator output contract

Multiple evaluator prompt formats are supported so the same case can be tested under different prompting strategies.

### 2. Evaluation Layer

The evaluation module simulates an LLM-as-a-judge workflow with deterministic heuristics. Rather than calling a live model, it estimates evaluator behavior using:

- task keyword overlap
- expected-trait coverage
- response clarity signals
- calibrated versus overconfident phrasing
- instruction-following cues

Scores are produced per dimension, combined using weights, and summarized into experiment-style findings.

## Experiments

### Prompt Variations

The study compares four evaluator prompt styles:

- `baseline`: minimal evaluator instruction, useful as a control
- `rubric_strict`: explicit rubric enforcement for consistency
- `comparative`: prompts the evaluator to reason against an ideal answer profile
- `research_audit`: audit-style framing optimized for traceability and experiment logging

### Evaluation Dimensions

Each response is scored across the following axes:

- `relevance`
- `completeness`
- `clarity`
- `factual_caution`
- `instruction_adherence`

### Study Cases

The experiment runner evaluates multiple candidate responses across distinct settings such as:

- medical support workflows
- enterprise policy summarization
- customer support escalation behavior

This makes the project feel closer to a small benchmark study than a single hard-coded example.

### Scoring Logic

The scoring system is simulated but intentionally realistic:

- stronger task-response overlap improves relevance
- broader concept coverage improves completeness
- concise multi-sentence structure improves clarity
- calibrated language improves factual caution
- prompt-style bonuses simulate how evaluator framing changes scoring rigor

The final result includes:

- dimension-level scores
- weighted total
- verdict label
- evaluator notes
- cross-case prompt-format averages

## Observations

The current prototype suggests several useful patterns:

- `rubric_strict` and `research_audit` produce more stable structured judgments than `baseline`
- `comparative` tends to reward broader coverage and tradeoff discussion
- `research_audit` favors caution-heavy responses and improves evaluator traceability
- short candidate responses can remain clear while still underperforming on completeness

These findings are simulated, but they mirror the types of observations researchers often report in prompt evaluation studies.

## Status

Current status: `Research Prototype`

Implemented:

- structured evaluator prompt generation
- prompt-format ablation across multiple cases
- weighted scoring engine
- experiment summary reporting
- JSON artifact export for study outputs

Planned extensions:

- replace heuristic scoring with real judge-model calls
- add repeated trials and variance estimation
- compare system prompts versus user-only evaluation prompts
- log outputs to CSV and visualization notebooks

## Quick Start

Run the prompt generation demo:

```bash
python src/prompt_design.py
```

Run the evaluation study:

```bash
python src/evaluation.py
```

When the study runs, it prints:

- case-by-case prompt format results
- average scores by prompt format
- best-performing prompt format

It also exports a machine-readable JSON artifact:

```text
results/latest_study_results.json
```

## Research Framing

This is not just a script that assigns arbitrary scores. It is structured as a compact evaluation study designed to demonstrate how prompt engineering decisions can influence response assessment pipelines. The code is simple enough to read quickly, but organized enough to be extended into a fuller LLM evaluation framework.
