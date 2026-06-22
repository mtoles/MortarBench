# MortarBench: Evaluating Mortgage Loan Origination Agents

A benchmark for LLM agents on mortgage underwriting. Given a synthetic bank
statement and a ULAD loan application, the agent answers underwriting questions
(e.g. *"What is the total value of the applicant's unsecured loans?"*). The best
closed-source model reaches 77.1% exact match; our CRIT confidence-calibration
method raises this to 80.5%.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Add the API keys you plan to use to a `.env` in the repo root:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

LLM calls are cached with `joblib`; set `DISABLE_LLM_CACHE=1` to force fresh samples.

## Reproducing the main experiment

Run a baseline and a CRIT (`threshold`) evaluation per model:

```bash
python3 eval.py --seed 2 --model_id gpt-5 --model_type baseline  --results_dir main
python3 eval.py --seed 2 --model_id gpt-5 --model_type threshold --confidence_threshold 5 --results_dir main
```

Repeat for `claude-sonnet-4-6` and `gemini-3.1-pro-preview`. Results are written
to `main/<model>/<type>/<timestamp>/*.jsonl`.

## Citation

```bibtex
@article{toles2026mortarbench,
  title   = {MortarBench: Evaluating Mortgage Loan Origination Agents},
  author  = {Toles, Matthew and others},
  year    = {2026},
  journal = {arXiv preprint arXiv:2606.19416},
  url     = {https://arxiv.org/abs/2606.19416}
}
```
