#!/usr/bin/env bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# ── Part A: run benchmark on experiments branch ──────────────────────────────
echo "=== Switching to experiments branch ==="
git checkout experiments

echo ""
echo "=== Running Part A benchmark (may take several minutes) ==="
rm -f experiments/results/coalition_benchmark.csv
python experiments/benchmark.py

echo ""
echo "=== Committing Part A results ==="
git add experiments/benchmark.py experiments/results/
git commit -m "benchmark: extend n_agents to 100, re-run for 30-60s coverage" || echo "(nothing to commit)"

echo ""
echo "=== Pushing experiments ==="
git push origin experiments

# ── Part B: run benchmark on performance-improvement branch ──────────────────
echo ""
echo "=== Switching to performance-improvement branch ==="
git checkout performance-improvement

echo ""
echo "=== Running Part B benchmark (before: read from experiments branch) ==="
# Copy Part A results as the "before" baseline
mkdir -p experiments/results
git show experiments:experiments/results/coalition_benchmark.csv > experiments/results/coalition_benchmark.csv

rm -f experiments/results/coalition_benchmark_optimized.csv
rm -f experiments/results/comparison.csv
rm -f experiments/results/before_vs_after_runtime.png
python experiments/benchmark_improvement.py

echo ""
echo "=== Committing Part B results ==="
git add experiments/benchmark.py experiments/benchmark_improvement.py experiments/results/
git commit -m "benchmark: extend n_agents to 100, re-run before/after comparison" || echo "(nothing to commit)"

echo ""
echo "=== Pushing performance-improvement ==="
git push origin performance-improvement

echo ""
echo "=== Done ==="
echo "Part A: https://github.com/HILLELOH/pref_voting/tree/experiments/experiments"
echo "Part B: https://github.com/HILLELOH/pref_voting/tree/performance-improvement/experiments"
