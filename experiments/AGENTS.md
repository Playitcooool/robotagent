# Experiments Guide

This folder contains evaluation and analysis scripts for research results.

- `academic_rag_eval.py` and `academic_rag_compare.py` evaluate retrieval behavior.
- `experience_transferability_eval.py` and `task_attempt_analysis_eval.py` analyze task/experience performance.
- `generate_charts.py` creates experiment figures.
- `data/` and `results/` contain experiment inputs and generated outputs.

Favor reproducible scripts that read from explicit data paths and write to `results/`. Do not move experiment assumptions into production modules without checking the backend, tools, and prompts contracts.
