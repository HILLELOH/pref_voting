"""
Flask web application for AI-mediated coalition formation
"""

import logging
import os
import sys
from flask import Flask, render_template, request, jsonify

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "app.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")


class _FlushFileHandler(logging.FileHandler):
    """File handler that flushes after every record — survives hard process kills."""
    def emit(self, record):
        super().emit(record)
        self.flush()


_file_handler = _FlushFileHandler(LOG_PATH)
_file_handler.setFormatter(_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stream_handler])

logging.info(f"App starting. Python: {sys.executable}  cwd: {os.getcwd()}")

app = Flask(__name__)

# ============================================================================
# ROUTES
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", show_modal=True)


@app.route("/random-input", methods=["GET"])
def random_input():
    n = int(request.args.get('n', 5))

    proposals = [
        "Subsidise electric vehicles for consumers.",
        "Establish a global climate emergency fund.",
        "Implement a global carbon tax now.",
        "Switch entirely to solar and wind energy.",
        "Invest heavily in carbon capture technologies.",
        "Protect and restore ocean ecosystems.",
        "Mandate renewable energy for all new buildings.",
        "Phase out coal power plants by 2030.",
        "Electrify all transportation systems.",
        "Ban single-use plastics immediately.",
        "Create green jobs through clean energy investment.",
        "Retrofit all buildings for energy efficiency.",
        "End deforestation through international law.",
        "Plant trees to absorb CO2 emissions globally.",
        "Improve public transport to cut car usage.",
        "Reduce meat consumption to lower emissions.",
        "Develop green hydrogen as a fuel source.",
        "Require companies to disclose carbon footprints.",
        "Promote circular economy to reduce waste.",
    ]

    import random
    selected = random.sample(proposals, min(n, len(proposals)))

    agents = [
        {"name": f"Agent {i+1}", "sentence": proposal}
        for i, proposal in enumerate(selected)
    ]

    status_quo_options = [
        "Continue current energy policy with minor adjustments.",
        "Maintain existing environmental regulations as they are.",
        "No major policy changes, focus on voluntary industry action.",
        "Keep current carbon targets without new enforcement mechanisms.",
    ]
    status_quo = random.choice(status_quo_options)

    return jsonify({"agents": agents, "status_quo": status_quo})


@app.route("/run", methods=["POST"])
def run():
    try:
        names = request.form.getlist("agent_name")
        sentences = request.form.getlist("agent_sentence")
        status_quo = request.form.get("status_quo", "").strip()
        majority_quota = float(request.form.get("majority_quota", 0.5))
        sigma = float(request.form.get("sigma", 0.0))

        # Input validation
        errors = []
        if not status_quo:
            errors.append("Status quo cannot be empty.")
        agents_info = [
            {"name": n.strip(), "ideal": s.strip()}
            for n, s in zip(names, sentences)
            if n.strip() and s.strip()
        ]
        if len(agents_info) < 2:
            errors.append("At least 2 agents with non-empty name and sentence are required.")
        if not (0 < majority_quota <= 1):
            errors.append("Majority quota must be between 0 (exclusive) and 1 (inclusive).")

        if errors:
            return render_template(
                "index.html",
                errors=errors,
                show_modal=False,
                form_data=request.form,
                prev_names=names,
                prev_sentences=sentences,
            ), 400

        from coalition_formation import run_coalition_formation

        result = run_coalition_formation(
            agents_info=agents_info,
            status_quo=status_quo,
            majority_quota=majority_quota,
            sigma=sigma,
        )

        coalition_names = result["coalition"]
        total_agents = len(agents_info)
        coalition_size = len(coalition_names)
        majority_quota_pct = round(majority_quota * 100)
        coalition_pct = round(coalition_size / total_agents * 100) if total_agents else 0

        votes_by_name = {v["name"]: v for v in result.get("votes", [])}
        proof_rows = [
            {
                "name": a["name"],
                "ideal": a["ideal"],
                "d_proposal": round(votes_by_name.get(a["name"], {}).get("d_proposal", 0), 4),
                "d_sq": round(votes_by_name.get(a["name"], {}).get("d_status_quo", 0), 4),
                "voted_yes": a["name"] in coalition_names,
            }
            for a in agents_info
        ]

        return render_template(
            "result.html",
            result_sentence=result["result"],
            coalition_names=coalition_names,
            coalition_size=coalition_size,
            total_agents=total_agents,
            coalition_pct=coalition_pct,
            majority_quota_pct=majority_quota_pct,
            status_quo=status_quo,
            proof_rows=proof_rows,
            iterations=result.get("iterations", "?"),
        )

    except Exception as e:
        logging.error(f"Error in /run: {e}", exc_info=True)
        return render_template(
            "index.html",
            errors=[str(e)],
            show_modal=False,
            form_data=request.form,
            prev_names=request.form.getlist("agent_name"),
            prev_sentences=request.form.getlist("agent_sentence"),
        ), 500


@app.route("/logs", methods=["GET"])
def logs():
    try:
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
        content = "".join(lines[-500:])  # last 500 lines
    except FileNotFoundError:
        content = "(No log file found yet.)"
    return render_template("logs.html", content=content)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", errors=["Page not found."], show_modal=False), 404


@app.errorhandler(500)
def server_error(e):
    logging.error(f"Server error: {e}", exc_info=True)
    return render_template("index.html", errors=["Internal server error."], show_modal=False), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5003,
        debug=True,
        use_reloader=True,
        threaded=True,
    )
