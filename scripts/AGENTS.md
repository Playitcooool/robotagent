# Scripts Guide

This folder contains utility scripts for generated presentations, document formula handling, and visual/demo assets.

These scripts are not part of the core web runtime. Before editing, identify the specific artifact pipeline they support, usually under `documents/`, `output/`, or presentation assets.

Use `.venv` for Python scripts and keep Node scripts scoped to their existing package assumptions. Avoid broad cleanup of generated files unless the task explicitly requests it.
