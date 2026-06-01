"""
Part A: Performance benchmarking of coalition formation algorithm.
Compares majority_quota=0.5, 0.67, 1.0 across increasing number of agents.
Uses experiments-csv for experiment management and plotting.
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_csv import Experiment
from experiments_csv.plot_results import multi_plot_results
from coalition_formation import run_coalition_formation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_FILE = "coalition_benchmark.csv"
TIME_LIMIT = 60  # seconds

# Diverse policy positions for random agent generation
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
    exp = Experiment(results_folder=RESULTS_DIR, results_filename=RESULTS_FILE)

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

    results_csv = os.path.join(RESULTS_DIR, RESULTS_FILE)

    # Plot runtime vs n_agents for each majority_quota
    multi_plot_results(
        results_csv_file=results_csv,
        filter={},
        x_field="n_agents",
        y_field="runtime",
        z_field="majority_quota",
        mean=True,
        subplot_field=None,
        subplot_rows=1,
        subplot_cols=1,
        save_to_file=os.path.join(RESULTS_DIR, "runtime_vs_agents.png"),
    )

    # Plot iterations vs n_agents for each majority_quota
    multi_plot_results(
        results_csv_file=results_csv,
        filter={},
        x_field="n_agents",
        y_field="iterations",
        z_field="majority_quota",
        mean=True,
        subplot_field=None,
        subplot_rows=1,
        subplot_cols=1,
        save_to_file=os.path.join(RESULTS_DIR, "iterations_vs_agents.png"),
    )

    # Plot coalition_fraction vs n_agents for each majority_quota
    multi_plot_results(
        results_csv_file=results_csv,
        filter={},
        x_field="n_agents",
        y_field="coalition_fraction",
        z_field="majority_quota",
        mean=True,
        subplot_field=None,
        subplot_rows=1,
        subplot_cols=1,
        save_to_file=os.path.join(RESULTS_DIR, "coalition_fraction_vs_agents.png"),
    )

    print(f"\nResults saved to {results_csv}")
    print(f"Plots saved to {RESULTS_DIR}/")
