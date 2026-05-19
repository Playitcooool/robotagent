# Training-Free GRPO Guide

This folder contains experience collection and conversion code for training-free optimization experiments.

- `collect.py` gathers task attempts or trajectories.
- `extract_experiences.py` extracts reusable experience records.
- `convert_experiences.py` converts raw experience data into downstream formats.
- `experience_tools.py` contains helpers used by the collection/conversion flow.
- `output/` is generated experiment output.

Keep this area separate from the production FastAPI request path unless the task explicitly asks to integrate an experiment. Generated outputs should not be treated as source of truth for backend behavior.
