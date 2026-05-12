import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# Allow importing pref_voting from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

LOG_FILE = Path(__file__).parent / "logs" / "app.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

SAMPLE_SENTENCES = [
    "Plant trees to absorb CO2 emissions globally.",
    "Switch entirely to solar and wind energy.",
    "Reduce meat consumption to lower emissions.",
    "Invest heavily in carbon capture technologies.",
    "Improve public transport to cut car usage.",
    "Ban single-use plastics immediately.",
    "Implement a global carbon tax now.",
    "Protect and restore ocean ecosystems.",
    "Develop green hydrogen as a fuel source.",
    "Retrofit all buildings for energy efficiency.",
    "Electrify all transportation systems.",
    "Subsidise electric vehicles for consumers.",
    "End deforestation through international law.",
    "Fund climate adaptation in developing nations.",
    "Phase out coal power plants by 2030.",
    "Create green jobs through clean energy investment.",
    "Establish a global climate emergency fund.",
    "Require companies to disclose carbon footprints.",
    "Promote circular economy to reduce waste.",
    "Mandate renewable energy for all new buildings.",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    from coalition_formation import (
        coalition_formation,
        embed_text,
        cosine_dissimilarity,
    )

    api_key = (
        request.form.get("api_key", "").strip()
        or os.environ.get("OPENAI_API_KEY")
        or None
    )

    names = request.form.getlist("agent_name")
    sentences = request.form.getlist("agent_sentence")
    status_quo = request.form.get("status_quo", "").strip()

    errors = []
    if not status_quo:
        errors.append("Status quo sentence is required.")

    agents = {}
    for i, (name, sent) in enumerate(zip(names, sentences)):
        name = name.strip()
        sent = sent.strip()
        if not name:
            errors.append(f"Agent {i + 1}: name is empty.")
        if not sent:
            errors.append(f"Agent {i + 1}: ideal sentence is empty.")
        if name or sent:
            agents[i] = {"name": name or f"Agent {i + 1}", "sentence": sent}

    if not agents:
        errors.append("At least one agent is required.")

    try:
        majority_quota = float(request.form.get("majority_quota", 0.5))
        if not (0 < majority_quota <= 1):
            errors.append("Majority quota must be between 0 (exclusive) and 1 (inclusive).")
    except ValueError:
        errors.append("Majority quota must be a decimal number.")
        majority_quota = 0.5

    try:
        sigma = float(request.form.get("sigma", 0.0))
        if sigma < 0:
            errors.append("Sigma must be >= 0.")
    except ValueError:
        errors.append("Sigma must be a decimal number.")
        sigma = 0.0

    seed_str = request.form.get("seed", "").strip()
    try:
        seed = int(seed_str) if seed_str else None
    except ValueError:
        seed = None

    if errors:
        return render_template(
            "index.html",
            errors=errors,
            form_data=request.form,
            prev_names=names,
            prev_sentences=sentences,
        )

    ideal = {i: a["sentence"] for i, a in agents.items()}
    agent_names = {i: a["name"] for i, a in agents.items()}

    logger.info(
        "Run: %d agents, majority_quota=%s, sigma=%s, seed=%s, status_quo=%r",
        len(ideal), majority_quota, sigma, seed, status_quo,
    )
    for i, a in agents.items():
        logger.info("  Agent %d (%s): %r", i, a["name"], a["sentence"])

    try:
        result_sentence, coalition_ids = coalition_formation(
            ideal,
            status_quo,
            majority_quota=majority_quota,
            sigma=sigma,
            seed=seed,
            api_key=api_key,
        )
    except Exception as e:
        logger.error("Algorithm error: %s", e, exc_info=True)
        return render_template(
            "index.html",
            errors=[f"Algorithm error: {e}"],
            form_data=request.form,
            prev_names=names,
            prev_sentences=sentences,
        )

    logger.info("Result sentence: %r", result_sentence)
    logger.info("Coalition: %s", [agent_names[i] for i in coalition_ids])

    proposal_emb = embed_text(result_sentence)
    sq_emb = embed_text(status_quo)

    proof_rows = []
    for i, agent in agents.items():
        ideal_emb = embed_text(agent["sentence"])
        d_proposal = cosine_dissimilarity(ideal_emb, proposal_emb)
        d_sq = cosine_dissimilarity(ideal_emb, sq_emb)
        voted_yes = i in coalition_ids
        proof_rows.append({
            "name": agent["name"],
            "ideal": agent["sentence"],
            "d_proposal": round(d_proposal, 4),
            "d_sq": round(d_sq, 4),
            "voted_yes": voted_yes,
        })
        logger.info(
            "  %s: d(ideal→proposal)=%.4f, d(ideal→status_quo)=%.4f, voted=%s",
            agent["name"], d_proposal, d_sq, voted_yes,
        )

    coalition_size = len(coalition_ids)
    total = len(agents)
    pct = round(100 * coalition_size / total, 1) if total else 0

    return render_template(
        "result.html",
        result_sentence=result_sentence,
        coalition_names=[agent_names[i] for i in coalition_ids],
        coalition_size=coalition_size,
        total_agents=total,
        coalition_pct=pct,
        majority_quota=majority_quota,
        majority_quota_pct=round(majority_quota * 100, 1),
        status_quo=status_quo,
        proof_rows=proof_rows,
    )


@app.route("/random-input")
def random_input():
    n = request.args.get("n", 5, type=int)
    n = max(1, min(n, len(SAMPLE_SENTENCES)))
    selected = random.sample(SAMPLE_SENTENCES, n)
    return jsonify({
        "agents": [
            {"name": f"Agent {i + 1}", "sentence": s}
            for i, s in enumerate(selected)
        ],
        "status_quo": "Do nothing about climate change.",
    })


@app.route("/logs")
def logs():
    try:
        content = LOG_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = "(No logs yet.)"
    return render_template("logs.html", content=content)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
