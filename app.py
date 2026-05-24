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
        {"name": f"Agent {i+1}", "ideal": proposal}
        for i, proposal in enumerate(selected)
    ]
    
    return jsonify({"agents": agents})


@app.route("/run", methods=["POST"])
def run():
    """Run the coalition formation algorithm."""
    try:
        data = request.json
        agents_info = data.get("agents", [])
        status_quo = data.get("status_quo", "Do nothing about climate change.")
        majority_quota = float(data.get("majority_quota", 0.5))
        
        # Import here (after route is called, not at startup)
        from coalition_formation import run_coalition_formation
        
        result = run_coalition_formation(
            agents_info=agents_info,
            status_quo=status_quo,
            majority_quota=majority_quota,
        )
        
        return jsonify({
            "success": True,
            "result": result["result"],
            "coalition": result["coalition"],
            "iterations": result["iterations"],
            "votes": result["votes"],
        })
    
    except Exception as e:
        logging.error(f"Error in /run: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


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