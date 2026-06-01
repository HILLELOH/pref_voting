"""
Part B: Before vs after performance comparison.
Runs the optimized coalition_formation on the same inputs used in Part A,
then plots runtime before vs after (Part A results must exist).
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_csv import Experiment
from coalition_formation import run_coalition_formation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BEFORE_CSV = os.path.join(RESULTS_DIR, "coalition_benchmark.csv")
AFTER_CSV = os.path.join(RESULTS_DIR, "coalition_benchmark_optimized.csv")
TIME_LIMIT = 60

_POLICY_POOL = [
    "We must invest heavily in renewable energy to combat climate change.",
    "Economic growth requires affordable fossil fuels for developing nations.",
    "Carbon taxes are the most efficient way to reduce emissions.",
    "Nuclear energy provides reliable low-carbon baseload power.",
    "Public transportation investment reduces urban carbon footprints.",
    "Individual behavioral change is essential for environmental progress.",
    "International cooperation is key to solving global climate issues.",
    "Technological innovation will solve climate change without restrictions.",
    "Strict industrial regulations are needed to protect the environment.",
    "Market mechanisms should drive the transition to clean energy.",
    "Local food systems reduce transportation emissions significantly.",
    "Green building standards should be mandatory for new construction.",
    "Reforestation programs can offset significant carbon emissions.",
    "Electric vehicles need government incentives to achieve mass adoption.",
    "Climate adaptation is as important as mitigation strategies.",
    "Ocean conservation protects critical carbon-absorbing ecosystems.",
    "Agricultural reform can dramatically reduce methane emissions.",
    "Energy efficiency standards for appliances reduce consumption.",
    "Circular economy principles eliminate waste and save resources.",
    "Environmental education builds long-term cultural change.",
]


def _make_agents(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool = _POLICY_POOL.copy()
    rng.shuffle(pool)
    selected = pool[:n] if n <= len(pool) else pool + [
        f"Policy position number {i} on sustainable development." for i in range(n - len(pool))
    ]
    return [{"name": f"Agent{i}", "ideal": selected[i]} for i in range(n)]


def single_run(n_agents: int, majority_quota: float, seed: int = 42) -> dict:
    agents = _make_agents(n_agents, seed)
    result = run_coalition_formation(
        agents_info=agents,
        majority_quota=majority_quota,
        seed=seed,
    )
    coalition_size = len(result["coalition"])
    yes_votes = sum(1 for v in result["votes"] if v["voted"])
    return {
        "iterations": result["iterations"],
        "coalition_size": coalition_size,
        "coalition_fraction": coalition_size / n_agents,
        "yes_votes": yes_votes,
        "yes_fraction": yes_votes / n_agents,
    }


if __name__ == "__main__":
    import pandas as pd

    os.makedirs(RESULTS_DIR, exist_ok=True)

    exp = Experiment(results_folder=RESULTS_DIR, results_filename="coalition_benchmark_optimized.csv")
    exp.run_with_time_limit(
        single_run=single_run,
        input_ranges={
            "n_agents": [3, 5, 7, 10, 12, 15],
            "majority_quota": [0.5, 0.67, 1.0],
            "seed": [42, 123, 777],
        },
        time_limit=TIME_LIMIT,
        runtime_field_name="runtime",
    )

    # Combine before/after into single CSV for comparison plot
    if os.path.exists(BEFORE_CSV) and os.path.exists(AFTER_CSV):
        import matplotlib.pyplot as plt
        df_before = pd.read_csv(BEFORE_CSV)
        df_after = pd.read_csv(AFTER_CSV)
        df_before["version"] = "before"
        df_after["version"] = "after"
        combined = pd.concat([df_before, df_after], ignore_index=True)
        combined_path = os.path.join(RESULTS_DIR, "comparison.csv")
        combined.to_csv(combined_path, index=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        for version, style in [("before", "--"), ("after", "-")]:
            subset = (
                combined[combined["version"] == version]
                .groupby("n_agents")["runtime"]
                .mean()
            )
            ax.plot(subset.index, subset.values, marker="o", linestyle=style, label=version)
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("Runtime before vs after optimization (mean over all quotas/seeds)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, "before_vs_after_runtime.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Comparison plot saved to {out}")
    else:
        print("Run experiments/benchmark.py first to generate before-optimization results.")
