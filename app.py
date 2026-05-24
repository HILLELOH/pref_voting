"""
Flask web application for AI-mediated coalition formation
"""

import logging
import os
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

app = Flask(__name__)

# ============================================================================
# ROUTES
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/random-input", methods=["GET"])
def random_input():
    """Generate random agent proposals."""
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
    """Run the coalition formation algorithm."""
    try:
        names = request.form.getlist("agent_name")
        sentences = request.form.getlist("agent_sentence")
        status_quo = request.form.get("status_quo", "Do nothing about climate change.")
        majority_quota = float(request.form.get("majority_quota", 0.5))

        agents_info = [
            {"name": n, "ideal": s}
            for n, s in zip(names, sentences)
            if n.strip() and s.strip()
        ]

        from coalition_formation import run_coalition_formation

        result = run_coalition_formation(
            agents_info=agents_info,
            status_quo=status_quo,
            majority_quota=majority_quota,
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
            form_data=request.form,
            prev_names=names,
            prev_sentences=sentences,
        )

    except Exception as e:
        logging.error(f"Error in /run: {e}", exc_info=True)
        return render_template("index.html", error=str(e), form_data=request.form,
                               prev_names=request.form.getlist("agent_name"),
                               prev_sentences=request.form.getlist("agent_sentence")), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    logging.error(f"Server error: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


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