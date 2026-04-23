"""
An implementation of the algorithm in:
"AI-Generated Compromises for Coalition Formation", by Eyal Briman, Ehud Shapiro, and Nimrod Talmon (2024), https://arxiv.org/abs/2410.21440

Programmer: Hillel Ohayon.
Date: 2025-04-04.
"""

from __future__ import annotations
import logging
import math
import random
from typing import Optional

logger = logging.getLogger(__name__)

Vector = list[float]  # 512-dim embedding (Universal Sentence Encoder)
Coalition = tuple[frozenset[int], str]  # (agent_indices, compromise_sentence)

# ---------------------------------------------------------------------------
# Sentence embedding — lazy-loaded and cached
#
# The paper (Section 4.1) uses Google's Universal Sentence Encoder (USE) which
# produces 512-dim vectors via TensorFlow Hub.  We wrap a PyTorch-based
# sentence-transformer (paraphrase-multilingual-MiniLM-L12-v2, 384-dim) and
# zero-pad to 512 dimensions.  Zero-padding preserves cosine similarity exactly:
#   dot(pad(a), pad(b)) = dot(a,b)   norm(pad(v)) = norm(v)
# so cosine_dissimilarity is identical whether computed in 384-d or 512-d.
# ---------------------------------------------------------------------------
_st_model = None
_ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_DIM = 512  # target dimension (USE output size, as in the paper)


def _get_st_model():
    """Load the sentence-transformer model (cached after first call)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info("Loading sentence-transformer model '%s'…", _ST_MODEL_NAME)
        _st_model = SentenceTransformer(_ST_MODEL_NAME)
        logger.info("Sentence-transformer model ready.")
    return _st_model


# ---------------------------------------------------------------------------
# Core helper: weighted vector arithmetic
# ---------------------------------------------------------------------------

def _weighted_avg(v1: Vector, w1: float, v2: Vector, w2: float) -> Vector:
    """Return (w1*v1 + w2*v2) / (w1 + w2) — the size-weighted centroid of two points.

    This is used in Section 2.3 to compute the compromise embedding p:
        p = argmin_x [ (|C_i|/(|C_i|+|C_j|)) * d(p_i, x)
                      + (|C_j|/(|C_i|+|C_j|)) * d(p_j, x) ]
    In Euclidean (embedding) space the argmin reduces to the weighted mean.

    Args:
        v1 (Vector): The embedding vector of the first coalition's compromise sentence.
        w1 (float): The weight of the first coalition (typically its size / number of agents).
        v2 (Vector): The embedding vector of the second coalition's compromise sentence.
        w2 (float): The weight of the second coalition (typically its size / number of agents).

    Returns:
        Vector: The size-weighted average embedding vector (centroid) of the two input vectors.

    >>> _weighted_avg([1.0, 0.0], 1, [0.0, 1.0], 1)
    [0.5, 0.5]
    >>> _weighted_avg([2.0, 0.0], 3, [0.0, 2.0], 1)
    [1.5, 0.5]
    """
    total = w1 + w2
    return [(w1 * a + w2 * b) / total for a, b in zip(v1, v2)]


def _vector_sum(vectors: list[Vector], weights: list[float]) -> Vector:
    """Return the weighted sum (not normalized) of vectors.

    Used to compute the global centroid in Section 2.3:
        centroid(D) = argmin_x (1/n) * Σ |C_i| * d(x, p_i)
    In Euclidean space this is Σ(|C_i|*p_i) / n.

    Args:
        vectors (list[Vector]): A list of embedding vectors (e.g., the compromise points of various coalitions).
        weights (list[float]): A list of numerical weights corresponding to each vector (e.g., the size of each coalition).

    Returns:
        Vector: A single vector representing the weighted sum of all input vectors.

    >>> _vector_sum([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])
    [1.0, 1.0]
    """
    dim = len(vectors[0])
    result = [0.0] * dim
    for v, w in zip(vectors, weights):
        for i in range(dim):
            result[i] += w * v[i]
    return result

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(text: str) -> Vector:
    """Embed a sentence into a 512-dimensional semantic vector (Section 4.1).

    Uses Google's Universal Sentence Encoder. Semantically similar sentences
    are placed close together in the embedding space.

    Args:
        text (str): A natural-language sentence.

    Returns:
        Vector: A list of 512 floats.

    >>> v = embed_text("We must reduce carbon emissions.")
    >>> len(v)
    512
    >>> all(isinstance(x, float) for x in v)
    True

    >>> x,y = embed_text("We must reduce carbon emissions."), embed_text("We must reduce carbon emissions.")
    >>> (x==y) == True
    True
    """
    model = _get_st_model()
    logger.debug("Embedding text: %r", text[:60])
    raw: list[float] = model.encode(text, convert_to_numpy=True).tolist()
    # Zero-pad to 512 dimensions (USE output size, Section 4.1).
    # Padding preserves cosine similarity: dot/norms are unchanged by zero entries.
    padded = raw + [0.0] * (_EMBED_DIM - len(raw))
    return padded[:_EMBED_DIM]


def cosine_dissimilarity(v1: Vector, v2: Vector) -> float:
    """Compute the square-root cosine-based dissimilarity between two vectors.

    Defined as sqrt(2 - 2*cos(theta)) where theta is the angle between v1 and v2
    (Section 4.1, footnote 6).  Returns values in [0, sqrt(2)].

    Formally: d_cos(A, B) = sqrt(2 - 2 * (A·B)/(||A||*||B||))

    Args:
        v1 (Vector): First embedding vector.
        v2 (Vector): Second embedding vector.

    Returns:
        float: Distance in [0, sqrt(2)].

    >>> cosine_dissimilarity([1.0, 0.0], [1.0, 0.0])
    0.0
    >>> round(cosine_dissimilarity([1.0, 0.0], [0.0, 1.0]), 4)
    1.4142
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    cos_sim = dot / (norm1 * norm2)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return math.sqrt(max(0.0, 2.0 - 2.0 * cos_sim))


def agent_votes(
    ideal: str,
    proposal: str,
    status_quo: str,
    sigma: float = 0.0,
) -> bool:
    """Return True if the agent accepts the proposed compromise sentence.

    Deterministic (sigma=0): accept iff cosine_dissimilarity(ideal, proposal)
    <= cosine_dissimilarity(ideal, status_quo).
    Probabilistic (sigma>0): accept with half-Gaussian probability when the
    proposal is farther than the status quo (Definition 5, Section 2.1).

    The acceptance probability when the proposal is worse is:
        F = sqrt(2/pi) / sigma * exp(-d(pv,p)^2 / (2*sigma^2))
    (half-normal PDF evaluated at d(pv,p), clamped to [0,1]).

    Args:
        ideal (str): The agent's ideal sentence.
        proposal (str): The mediator's proposed sentence.
        status_quo (str): The current status-quo sentence.
        sigma (float): Flexibility >= 0. 0 means fully deterministic.

    Returns:
        bool: True if the agent votes to accept.

    >>> agent_votes("Cut carbon now.", "Cut carbon now.", "Do nothing.", sigma=0.0)
    True
    >>> agent_votes("Cut carbon now.", "Do nothing.", "Cut carbon now.", sigma=0.0)
    False
    """
    v_ideal = embed_text(ideal)
    v_proposal = embed_text(proposal)
    v_sq = embed_text(status_quo)

    d_proposal = cosine_dissimilarity(v_ideal, v_proposal)
    d_sq = cosine_dissimilarity(v_ideal, v_sq)

    logger.debug(
        "agent_votes: d(ideal,proposal)=%.4f  d(ideal,sq)=%.4f  sigma=%.2f",
        d_proposal, d_sq, sigma,
    )

    if d_proposal <= d_sq:
        logger.debug("agent_votes: proposal accepted (closer than status quo)")
        return True

    if sigma == 0.0:
        logger.debug("agent_votes: proposal rejected (deterministic, farther than sq)")
        return False

    # Half-Gaussian probability (Definition 5): prob shrinks as proposal moves further
    # from ideal; grows as sigma grows (more altruistic / flexible).
    prob = min(1.0, math.sqrt(2.0 / math.pi) / sigma * math.exp(-d_proposal ** 2 / (2.0 * sigma ** 2)))
    accepted = random.random() < prob
    logger.debug("agent_votes: probabilistic result=%s (prob=%.4f)", accepted, prob)
    return bool(accepted)


def generate_compromise_sentences(
    sentence1: str,
    sentence2: str,
    n: int = 10,
    api_key: str = None,
) -> list[str]:
    """Ask GPT-3.5-turbo to generate n sentences aggregating the two inputs.

    Uses the Mediator-1 prompt from Section 4.2: structured prompt + zero-shot
    chain-of-thought, with a max of 15 words per output sentence.

    When OPENAI_API_KEY is not set, falls back to simple template-based
    combination so that tests and offline runs still produce valid output.

    Args:
        sentence1 (str): Compromise sentence of the first coalition.
        sentence2 (str): Compromise sentence of the second coalition.
        n (int): Number of candidate sentences to generate.

    Returns:
        list[str]: n candidate compromise sentences.

    >>> sentences = generate_compromise_sentences(
    ...     "Plant trees to fight climate change.",
    ...     "Switch to renewable energy sources.",
    ...     n=3,
    ... )
    >>> len(sentences)
    3
    >>> all(isinstance(s, str) and len(s) > 0 for s in sentences)
    True
    """
    import os  # noqa: PLC0415
    actual_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not actual_key:
        logger.warning(
            "OPENAI_API_KEY not set and no api_key provided — using template-based fallback (Section 4.2 offline mode)."
        )
        return _fallback_compromise_sentences(sentence1, sentence2, n)

    return _gpt_compromise_sentences(sentence1, sentence2, n, actual_key)


def _gpt_compromise_sentences(sentence1: str, sentence2: str, n: int, api_key: str) -> list[str]:
    """Call GPT-3.5-turbo with the Mediator-1 prompt (Section 4.2, Option 1).

    Prompt: "Generate {n} possible different well-structured sentences that
    aggregate the following two sentences. Make sure each sentence has at most
    15 words. Number your answers …"
    Message: mediator role instruction (zero-shot chain-of-thought).
    """
    import openai  # noqa: PLC0415
    import logging
    logger = logging.getLogger(__name__)

    client = openai.OpenAI(api_key=api_key)

    prompt = (
        f"Generate {n} possible different well-structured sentences that aggregate "
        f"the following two sentences. Make sure each sentence has at most 15 words. "
        f"Number your answers (i.e., 1), 2), 3), 4), 5), and so on) for each sentence "
        f"you propose.\n\n"
        f'Sentence 1: "{sentence1}"\n'
        f'Sentence 2: "{sentence2}"'
    )
    message = (
        "You are a mediator trying to find agreed wording based on existing sentences. "
        "Give a straightforward answer with no introduction to help people reach an "
        "agreed wording of a coherent sentence."
    )

    logger.info(
        "Requesting %d compromise sentences from GPT-3.5-turbo for: %r | %r",
        n, sentence1[:40], sentence2[:40],
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.75,
        messages=[
            {"role": "system", "content": message},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content or ""
    logger.debug("GPT raw response:\n%s", raw)

    sentences = _parse_numbered_list(raw, n)
    logger.info("Parsed %d sentences from GPT response.", len(sentences))
    return sentences


def _fallback_compromise_sentences(sentence1: str, sentence2: str, n: int) -> list[str]:
    """Generate candidate compromise sentences without an LLM.

    Produces n distinct short sentences (≤15 words) combining key words from
    sentence1 and sentence2 via simple templates.  Used when OPENAI_API_KEY is
    absent — preserves structural correctness for offline testing.

    >>> sents = _fallback_compromise_sentences("Cut emissions now.", "Plant more trees.", 5)
    >>> len(sents)
    5
    >>> all(isinstance(s, str) and len(s) > 0 for s in sents)
    True
    >>> all(len(s.split()) <= 15 for s in sents)
    True
    """
    w1 = sentence1.rstrip(".!?")
    w2 = sentence2.rstrip(".!?")
    templates = [
        f"{w1} and {w2}.",
        f"We should {w1.lower()} while also working to {w2.lower()}.",
        f"Both {w1.lower()} and {w2.lower()} are essential.",
        f"To address our goals: {w1.lower()} and {w2.lower()}.",
        f"Combine efforts to {w1.lower()} and {w2.lower()}.",
        f"{w1} alongside {w2}.",
        f"Support {w1.lower()} and promote {w2.lower()}.",
        f"Together we must {w1.lower()} and {w2.lower()}.",
        f"Our plan: {w1.lower()} and {w2.lower()}.",
        f"Let us {w1.lower()} and also {w2.lower()}.",
    ]
    # Truncate each template to at most 15 words
    results = []
    for t in templates[:n]:
        words = t.split()
        results.append(" ".join(words[:15]))
    # Pad if n > len(templates)
    while len(results) < n:
        results.append(results[-1])
    return results[:n]


def _parse_numbered_list(text: str, n: int) -> list[str]:
    """Parse a numbered list from GPT output into a list of plain strings.

    Handles formats like "1) ...", "1. ...", "1: ...".
    If fewer than n sentences are parsed, the last sentence is repeated to fill.

    >>> _parse_numbered_list("1) Hello world\\n2) Goodbye world", 2)
    ['Hello world', 'Goodbye world']
    >>> len(_parse_numbered_list("1) Only one sentence here.", 3))
    3
    """
    import re  # noqa: PLC0415
    lines = text.strip().splitlines()
    results: list[str] = []
    for line in lines:
        line = line.strip()
        # Match "1)", "1.", "1:" at the start
        m = re.match(r"^\d+[.):\-]\s*(.+)", line)
        if m:
            sentence = m.group(1).strip().rstrip(".")
            if sentence:
                results.append(sentence)
    # Pad if GPT returned fewer than n
    if results and len(results) < n:
        results.extend([results[-1]] * (n - len(results)))
    # Truncate if GPT returned more than n
    return results[:n] if results else [""] * n


def choose_best_sentence(candidates: list[str], target: Vector) -> str:
    """Return the candidate sentence whose embedding is closest to the target.

    The target is the size-weighted average of the two coalition embeddings.
    Selects argmin cosine_dissimilarity(embed(s), target) (Section 4.1).

    Args:
        candidates (list[str]): Candidate sentences from generate_compromise_sentences.
        target (Vector): Weighted-average embedding of the two coalition points.

    Returns:
        str: The best candidate sentence.

    >>> choose_best_sentence(["hello world", "goodbye world"], [1.0, 0.0])  # doctest: +SKIP
    'hello world'
    """
    if len(candidates) == 1:
        return candidates[0]

    best_sentence = candidates[0]
    best_dist = float("inf")
    for s in candidates:
        emb = embed_text(s)
        d = cosine_dissimilarity(emb, target)
        if d < best_dist:
            best_dist = d
            best_sentence = s
    logger.debug("Best compromise sentence (dist=%.4f): %r", best_dist, best_sentence[:60])
    return best_sentence


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def coalition_formation(
    ideal_sentences: dict[int, str],
    status_quo: str,
    majority_quota: float = 0.5,
    sigma: float = 0.0,
    alpha: float = 0.0,
    coalition_discipline: bool = False,
    max_iterations: int = 10_000,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
) -> tuple[str, list[int]]:
    """Run the AI-mediated coalition formation algorithm in a textual metric space.

    Algorithm 1 (Section 1.2):
      1. Initialise: each agent forms a singleton coalition around its ideal sentence.
      2. Loop:
         a. Select two coalitions via centroid-based scoring (Section 2.3).
         b. Generate a compromise sentence using the LLM mediator.
         c. Agents vote; the constitution assigns them to old or new coalition.
         d. Remove empty coalitions; check halting condition.
      3. Halt when a coalition covers >= majority_quota fraction of all agents.

    Args:
        ideal_sentences (dict[int, str]): Agent index -> ideal sentence.
        status_quo (str): The starting status-quo sentence.
        majority_quota (float): Fraction of agents needed to halt (default 0.5).
        sigma (float): Agent flexibility (0 = deterministic).
        alpha (float): Mediator coalition-selection bias in [-1, 1].
        coalition_discipline (bool): Enforce whole-coalition voting.
        max_iterations (int): Safety cap on iterations.
        seed (Optional[int]): Random seed for reproducibility.
        api_key (Optional[str]): OpenAI API key. Falls back to OPENAI_API_KEY env var,
            then to template-based offline mode if neither is set.

    Returns:
        tuple[str, list[int]]:
            - The winning compromise sentence.
            - Sorted list of agent indices in the majority coalition.

    Single agent trivially wins:

    >>> sentence, agents = coalition_formation({0: "Protect the forests."}, "Do nothing.")
    >>> agents
    [0]
    """
    # --- Input validation ---
    if majority_quota < 0.0 or majority_quota > 1.0:
        raise ValueError(f"majority_quota must be in [0, 1], got {majority_quota}")
    if alpha < -1.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [-1, 1], got {alpha}")

    # --- Edge cases ---
    if not ideal_sentences:
        logger.info("coalition_formation: empty input, returning status quo.")
        return status_quo, []

    n_agents = len(ideal_sentences)

    if seed is not None:
        random.seed(seed)

    # --- Initialisation (Section 1.2) ---
    # Each agent i starts in its own singleton coalition.
    # State: list of dicts with keys 'agents', 'sentence', 'embedding'
    logger.info(
        "coalition_formation: initialising %d singleton coalitions. "
        "quota=%.2f  sigma=%.2f  alpha=%.2f  discipline=%s",
        n_agents, majority_quota, sigma, alpha, coalition_discipline,
    )
    coalitions: list[dict] = []
    for agent_idx, sentence in ideal_sentences.items():
        emb = embed_text(sentence)
        coalitions.append({
            "agents": {agent_idx},
            "sentence": sentence,
            "embedding": emb,
        })
        logger.debug("Init coalition: agent=%d  sentence=%r", agent_idx, sentence[:50])

    # Check trivial halting condition: singleton already satisfies quota
    for c in coalitions:
        if len(c["agents"]) / n_agents >= majority_quota:
            winner = c
            logger.info("Halting immediately: coalition size %d >= quota.", len(winner["agents"]))
            return winner["sentence"], sorted(winner["agents"])

    # --- Iterative process ---
    for iteration in range(1, max_iterations + 1):
        logger.info("--- Iteration %d  (coalitions: %d) ---", iteration, len(coalitions))

        if len(coalitions) == 1:
            # Only one coalition left — it must be the majority.
            break

        # (a) Compute global centroid (Section 2.3) —
        #     centroid = Σ(|C_i| * p_i) / n  (Euclidean argmin of weighted distances)
        sizes = [len(c["agents"]) for c in coalitions]
        centroid = _vector_sum(
            [c["embedding"] for c in coalitions],
            [float(s) for s in sizes],
        )
        total_weight = float(sum(sizes))
        centroid = [x / total_weight for x in centroid]

        # Distances from each coalition to the centroid
        dists_to_centroid = [
            cosine_dissimilarity(c["embedding"], centroid) for c in coalitions
        ]
        max_dist = max(dists_to_centroid) or 1.0  # avoid division by zero

        # (b) Score coalitions: S_i = exp(alpha * d'(p_i, centroid))
        #     where d'(p_i, centroid) = d(p_i, centroid) / max_j d(p_j, centroid)
        scores = [
            math.exp(alpha * (d / max_dist)) for d in dists_to_centroid
        ]
        total_score = sum(scores)
        probs = [s / total_score for s in scores]

        logger.debug(
            "Coalition scores: %s  probs: %s",
            [round(s, 3) for s in scores],
            [round(p, 3) for p in probs],
        )

        # (c) Select d_i by weighted-random sampling (Section 2.3)
        r = random.random()
        cumulative = 0.0
        idx_i = len(coalitions) - 1
        for k, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                idx_i = k
                break

        # (d) Select d_j = closest coalition to d_i (excluding d_i itself)
        d_i_emb = coalitions[idx_i]["embedding"]
        idx_j = min(
            (k for k in range(len(coalitions)) if k != idx_i),
            key=lambda k: cosine_dissimilarity(coalitions[k]["embedding"], d_i_emb),
        )

        c_i = coalitions[idx_i]
        c_j = coalitions[idx_j]
        logger.info(
            "Selected coalitions: i=%d (size=%d, %r…)  j=%d (size=%d, %r…)",
            idx_i, len(c_i["agents"]), c_i["sentence"][:30],
            idx_j, len(c_j["agents"]), c_j["sentence"][:30],
        )

        # (e) Compute compromise target embedding — size-weighted average of p_i and p_j
        #     p = argmin_x [ (|C_i|/(|C_i|+|C_j|)) * d(p_i,x)
        #                   + (|C_j|/(|C_i|+|C_j|)) * d(p_j,x) ]
        #     In Euclidean space: weighted mean (Section 2.3)
        size_i = float(len(c_i["agents"]))
        size_j = float(len(c_j["agents"]))
        target_emb = _weighted_avg(c_i["embedding"], size_i, c_j["embedding"], size_j)

        # (f) Generate compromise sentences via LLM mediator (Mediator-1, Section 4.2)
        candidates = generate_compromise_sentences(c_i["sentence"], c_j["sentence"], api_key=api_key)
        logger.info("Generated %d candidate sentences.", len(candidates))

        # (g) Choose best sentence — argmin d_cos(embed(s), target)
        compromise_sentence = choose_best_sentence(candidates, target_emb)
        compromise_emb = embed_text(compromise_sentence)
        logger.info("Compromise sentence chosen: %r", compromise_sentence)

        # (h) Agents vote and apply constitution (Section 2.2)
        new_coalition_agents: set[int] = set()
        remaining_i: set[int] = set()
        remaining_j: set[int] = set()

        # Votes for C_i
        votes_i = {
            agent: agent_votes(
                ideal_sentences[agent], compromise_sentence, status_quo, sigma
            )
            for agent in c_i["agents"]
        }
        # Votes for C_j
        votes_j = {
            agent: agent_votes(
                ideal_sentences[agent], compromise_sentence, status_quo, sigma
            )
            for agent in c_j["agents"]
        }

        logger.debug(
            "Votes C_i: yes=%d/%d  votes C_j: yes=%d/%d",
            sum(votes_i.values()), len(c_i["agents"]),
            sum(votes_j.values()), len(c_j["agents"]),
        )

        if coalition_discipline:
            # Coalition Discipline (Section 2.2): if majority of C_i vote yes,
            # then individuals who voted yes join dp; others stay. If threshold
            # not met, all stay.
            threshold_i = math.ceil(len(c_i["agents"]) / 2)
            threshold_j = math.ceil(len(c_j["agents"]) / 2)

            if sum(votes_i.values()) >= threshold_i:
                for agent, voted_yes in votes_i.items():
                    if voted_yes:
                        new_coalition_agents.add(agent)
                    else:
                        remaining_i.add(agent)
            else:
                remaining_i = set(c_i["agents"])

            if sum(votes_j.values()) >= threshold_j:
                for agent, voted_yes in votes_j.items():
                    if voted_yes:
                        new_coalition_agents.add(agent)
                    else:
                        remaining_j.add(agent)
            else:
                remaining_j = set(c_j["agents"])
        else:
            # No Coalition Discipline (Section 2.2, Q=0):
            # each agent independently joins if it voted yes
            for agent, voted_yes in votes_i.items():
                if voted_yes:
                    new_coalition_agents.add(agent)
                else:
                    remaining_i.add(agent)
            for agent, voted_yes in votes_j.items():
                if voted_yes:
                    new_coalition_agents.add(agent)
                else:
                    remaining_j.add(agent)

        logger.info(
            "After vote: new coalition size=%d  remaining_i=%d  remaining_j=%d",
            len(new_coalition_agents), len(remaining_i), len(remaining_j),
        )

        # (i) Update coalition structure D' (Section 1.2)
        new_coalitions: list[dict] = []
        for k, c in enumerate(coalitions):
            if k == idx_i:
                if remaining_i:
                    new_coalitions.append({
                        "agents": remaining_i,
                        "sentence": c_i["sentence"],
                        "embedding": c_i["embedding"],
                    })
            elif k == idx_j:
                if remaining_j:
                    new_coalitions.append({
                        "agents": remaining_j,
                        "sentence": c_j["sentence"],
                        "embedding": c_j["embedding"],
                    })
            else:
                new_coalitions.append(c)

        if new_coalition_agents:
            new_coalitions.append({
                "agents": new_coalition_agents,
                "sentence": compromise_sentence,
                "embedding": compromise_emb,
            })

        coalitions = new_coalitions

        # (j) Check halting condition: |C| / n >= majority_quota
        for c in coalitions:
            if len(c["agents"]) / n_agents >= majority_quota:
                logger.info(
                    "Halting at iteration %d: coalition size %d (%.1f%%) >= quota %.1f%%.",
                    iteration, len(c["agents"]),
                    100 * len(c["agents"]) / n_agents,
                    100 * majority_quota,
                )
                return c["sentence"], sorted(c["agents"])

    # Safety: max_iterations reached — return largest coalition
    logger.warning(
        "Max iterations (%d) reached without halting condition. Returning largest coalition.",
        max_iterations,
    )
    winner = max(coalitions, key=lambda c: len(c["agents"]))
    return winner["sentence"], sorted(winner["agents"])
