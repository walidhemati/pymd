#!/usr/bin/env bash
# Build the paper: figures + results.tex must exist (python figures/make_figures.py).
set -euo pipefail
cd "$(dirname "$0")"
tectonic paper.tex
