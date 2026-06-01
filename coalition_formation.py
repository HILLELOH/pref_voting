"""
AI-Mediated Coalition Formation Implementation
Based on: Briman, Shapiro, Talmon 2024
https://arxiv.org/pdf/2512.05983
"""

import gc
import json
import logging
import os
import re
import threading
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# SENTENCE TRANSFORMER SINGLETON (with thread-safe double-check locking)
# ============================================================================
_st_model_lock = threading.Lock()
_st_model = None
_ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def _get_st_model():
    """Get or lazily load the sentence-transformer model (thread-safe)."""
    global _st_model
    if _st_model is None:
        with _st_model_lock:
            if _st_model is None:
                logger.info(f"Loading sentence-transformer model '{_ST_MODEL_NAME}'…")
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer(_ST_MODEL_NAME, device="cpu")
    return _st_model


# ============================================================================
# LLAMA-CPP MODEL (per-request, not global - FIX FOR SECOND RUN CRASH)
# ============================================================================
_MODEL_FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
_MODEL_FILENAME_FALLBACK = "qwen2.5-0.5b-instruct-q3_k_m.gguf"
_MODEL_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "models", _MODEL_FILENAME),
    "/home/hilleloh/app/models/" + _MODEL_FILENAME,
    os.path.join(os.path.dirname(__file__), "models", _MODEL_FILENAME_FALLBACK),
    "/home/hilleloh/app/models/" + _MODEL_FILENAME_FALLBACK,
]

def _get_llm_model():
    """Load local GGUF model via llama-cpp."""
    from llama_cpp import Llama

    model_path = next((p for p in _MODEL_SEARCH_PATHS if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(
            f"LLM model not found. Searched: {_MODEL_SEARCH_PATHS}"
        )

    logger.info(f"Loading model from {model_path}…")
    try:
        return Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
            chat_format="chatml",
        )
    except Exception as e:
        logger.error(f"Llama() constructor failed: {e}", exc_info=True)
        raise


# ============================================================================
# COALITION FORMATION ALGORITHM
# ============================================================================

def _cosine_similarity(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _cosine_dissimilarity(a, b):
    return 1 - _cosine_similarity(a, b)


def _batch_cosine_dissimilarity(matrix: "np.ndarray", vec: "np.ndarray") -> "np.ndarray":
    """Dissimilarity of each row in matrix vs vec. Single vectorized op."""
    import numpy as np
    norms_m = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm_v = np.linalg.norm(vec)
    if norm_v == 0:
        return np.ones(len(matrix), dtype=np.float32)
    safe = np.where(norms_m == 0, 1.0, norms_m)
    sims = (matrix / safe) @ vec / norm_v
    return 1.0 - sims


def _max_consecutive_common_words(text: str, reference: str) -> int:
    """Longest run of consecutive words from text that appears verbatim in reference."""
    words = text.lower().split()
    ref = reference.lower()
    max_run = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            if ' '.join(words[i:j]) in ref:
                max_run = max(max_run, j - i)
            else:
                break
    return max_run


_INCOMPLETE_ENDINGS = {
    'a', 'an', 'the', 'with', 'for', 'to', 'in', 'on', 'at', 'by', 'of',
    'and', 'or', 'but', 'if', 'as', 'from', 'while', 'that', 'which',
    'into', 'during', 'before', 'after', 'both', 'its', 'their', 'our',
    'this', 'these', 'those', 'also', 'while', 'though', 'since',
}

def _trim_to_max_words(text: str, max_words: int = 15) -> str:
    """Truncate to max_words words, stripping trailing incomplete words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    words = words[:max_words]
    # Strip trailing articles/prepositions/conjunctions that leave sentence dangling
    while words and words[-1].rstrip('.,;:!?"\'').lower() in _INCOMPLETE_ENDINGS:
        words.pop()
    if not words:
        return ' '.join(text.split()[:max_words]).rstrip(',;') + '.'
    result = ' '.join(words)
    if result[-1] not in '.!?':
        result = result.rstrip(',;') + '.'
    return result


def _is_valid_compromise(candidate: str, all_ideals: list, max_common: int = 2) -> bool:
    """Return True if candidate shares no more than max_common consecutive words with any ideal."""
    for ideal in all_ideals:
        if _max_consecutive_common_words(candidate, ideal) > max_common:
            return False
    return True


@lru_cache(maxsize=4096)
def _encode_sentence(sentence: str) -> tuple:
    """Encode sentence to embeddings. Cached — same string never re-encoded."""
    st_model = _get_st_model()
    embedding = st_model.encode(sentence, convert_to_numpy=True)
    return tuple(embedding.tolist())


def _parse_json_response(text: str, n: int, sentence1: str, sentence2: str) -> list[str]:
    """Parse JSON response into a list of n compromise strings."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try to close incomplete JSON (stop token may have cut the closing brace)
    for suffix in ["", "}", "]}"]:
        try:
            data = json.loads(cleaned + suffix)
            compromises = data.get("compromises", [])
            if isinstance(compromises, list):
                result = [c for c in compromises if isinstance(c, str) and c.strip()][:n]
                if result:
                    return result
        except (json.JSONDecodeError, AttributeError):
            continue

    # Fallback: extract quoted strings that look like sentences (20+ chars, contain a space)
    _JSON_KEYS = {"compromises", "result", "proposals", "sentences", "options"}
    strings = [
        s for s in re.findall(r'"([^"]{20,})"', cleaned)
        if ' ' in s and s.lower().strip() not in _JSON_KEYS
    ]
    if strings:
        return strings[:n]

    logger.warning(f"Failed to parse compromise from: {text!r}")
    return [sentence1, sentence2][:n]


def _call_llm_local(sentence1: str, sentence2: str, n: int = 2, llm=None) -> list[str]:
    """Call local LLM for compromise generation (supports Llama and Qwen instruct)."""
    logger.info(f"Calling LLM for {n} compromise candidates...")

    _own_llm = llm is None
    if _own_llm:
        llm = _get_llm_model()
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a policy mediator. Reply only with valid JSON.",
            },
            {
                "role": "user",
                "content": (
                    f'Two agents disagree on policy:\n'
                    f'- "{sentence1}"\n'
                    f'- "{sentence2}"\n\n'
                    f'Write {n} NEW compromise sentences (8-15 words each) that blend both ideas.\n'
                    f'Do NOT copy the input sentences. Write new sentences only.\n\n'
                    f'Reply ONLY with JSON like this:\n'
                    f'{{"compromises": ["Adopt balanced measures addressing both goals simultaneously.", "Combine approaches to achieve shared environmental outcomes."]}}'
                ),
            },
        ]

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.7,
            top_p=0.95,
        )

        raw_text = response["choices"][0]["message"]["content"]
        logger.info(f"LLM raw output: {raw_text!r}")

        compromises = _parse_json_response(raw_text, n, sentence1, sentence2)

        # Reject any output that is just a copy of the inputs
        s1_lower, s2_lower = sentence1.lower().strip(), sentence2.lower().strip()
        filtered = [
            c for c in compromises
            if c.lower().strip() not in (s1_lower, s2_lower)
            and s1_lower not in c.lower()
            and s2_lower not in c.lower()
        ]
        if not filtered:
            logger.warning("Llama only produced copies of input — using fallback blending")
            filtered = ["Implement coordinated policies that address both goals for shared sustainable outcomes."]

        logger.info(f"Compromise(s): {filtered}")
        return filtered
    finally:
        if _own_llm:
            del llm


# ============================================================================
# COALITION FORMATION
# ============================================================================

class Agent:
    """Represents an agent with an ideal proposal and voting logic."""

    def __init__(self, name: str, ideal: str):
        import numpy as np
        self.name = name
        self.ideal = ideal
        # Store as numpy array for vectorized ops; lru_cache returns tuple
        self.ideal_embedding = np.asarray(_encode_sentence(ideal), dtype=np.float32)

    def vote(self, proposal: str, status_quo: str) -> tuple:
        """Vote yes if proposal is closer to ideal than status quo."""
        proposal_embedding = _encode_sentence(proposal)
        status_quo_embedding = _encode_sentence(status_quo)

        d_proposal = _cosine_dissimilarity(self.ideal_embedding, proposal_embedding)
        d_status_quo = _cosine_dissimilarity(self.ideal_embedding, status_quo_embedding)

        voted = d_proposal < d_status_quo
        logger.info(
            f"   {self.name}: d(ideal→proposal)={d_proposal:.4f}, "
            f"d(ideal→status_quo)={d_status_quo:.4f}, voted={voted}"
        )

        return voted, d_proposal, d_status_quo


def run_coalition_formation(
    agents_info: list[dict],
    majority_quota: float = 0.5,
    sigma: float = 0.0,
    seed: Optional[int] = None,
    status_quo: str = "Do nothing about climate change.",
    progress_callback=None,
) -> dict:
    """
    Run bottom-up coalition formation algorithm.

    Each agent starts as individual coalition. Each iteration: pick 2 coalitions,
    generate LLM compromise, each coalition votes internally. If both accept
    (strict majority within each group), merge. Continue until merged coalition
    reaches majority_quota of all agents.
    """
    import random
    import numpy as np
    from sklearn.decomposition import PCA as _PCA2D

    n_agents = len(agents_info)
    agents = [Agent(info["name"], info["ideal"]) for info in agents_info]

    logger.info(f"Run: {n_agents} agents, majority_quota={majority_quota}, sigma={sigma}, seed={seed}, status_quo='{status_quo}'")
    for i, info in enumerate(agents_info, 1):
        logger.info(f"   Agent {i-1} ({info['name']}): '{info['ideal']}'")

    # Each coalition: {"members": [agent_index, ...], "representative": sentence}
    coalitions = [
        {"members": [i], "representative": agents[i].ideal}
        for i in range(n_agents)
    ]

    all_ideals = [a.ideal for a in agents]

    # PCA-2D setup: fit once on agent ideals, reuse transform every snapshot
    _ideal_embs = np.array([a.ideal_embedding for a in agents])
    _pca_2d = _PCA2D(n_components=min(2, n_agents), random_state=42)
    _pca_2d.fit(_ideal_embs)
    _emb_2d_cache: dict = {a.ideal: a.ideal_embedding for a in agents}

    def _snap_2d():
        pts = []
        for c in coalitions:
            rep = c["representative"]
            if rep not in _emb_2d_cache:
                _emb_2d_cache[rep] = _encode_sentence(rep)
            xy = _pca_2d.transform([_emb_2d_cache[rep]])[0]
            pts.append({
                "x": round(float(xy[0]), 4),
                "y": round(float(xy[1]), 4) if len(xy) > 1 else 0.0,
                "size": len(c["members"]),
                "label": ", ".join(agents[idx].name for idx in c["members"]),
                "rep": c["representative"][:80],
                "first_idx": min(c["members"]),
            })
        return pts
    iteration = 0
    winning_coalition = None
    winning_sentence = None

    llm = _get_llm_model()
    try:
        while True:
            iteration += 1
            progress_msg = f"Iteration {iteration} | {len(coalitions)} coalitions remaining (need {majority_quota*100:.0f}% of {n_agents} agents)"
            logger.info(f"--- {progress_msg} ---")
            print(f"[{progress_msg}]", flush=True)
            if progress_callback:
                progress_callback({
                    "iteration": iteration,
                    "coalitions": len(coalitions),
                    "n_agents": n_agents,
                    "majority_quota": majority_quota,
                    "event": "start",
                    "proposal": None,
                    "merged_size": None,
                    "coalitions_2d": _snap_2d(),
                })

            if seed is not None:
                random.seed(seed + iteration)

            if len(coalitions) < 2:
                winning_coalition = coalitions[0]
                winning_sentence = winning_coalition["representative"]
                break

            ci, cj = random.sample(range(len(coalitions)), 2)
            coal_i = coalitions[ci]
            coal_j = coalitions[cj]

            if progress_callback:
                progress_callback({
                    "iteration": iteration,
                    "coalitions": len(coalitions),
                    "n_agents": n_agents,
                    "majority_quota": majority_quota,
                    "event": "generating",
                    "rep_i": coal_i["representative"],
                    "rep_j": coal_j["representative"],
                    "size_i": len(coal_i["members"]),
                    "size_j": len(coal_j["members"]),
                    "proposal": None,
                    "merged_size": None,
                })

            candidates = _call_llm_local(coal_i["representative"], coal_j["representative"], n=2, llm=llm)

            valid = [c for c in candidates if _is_valid_compromise(c, all_ideals)]
            if valid:
                proposal = _trim_to_max_words(valid[0])
            elif candidates:
                proposal = _trim_to_max_words(min(
                    candidates,
                    key=lambda c: max(_max_consecutive_common_words(c, ideal) for ideal in all_ideals),
                ))
                logger.warning(f"No candidate passed overlap filter, using least-overlap: {proposal!r}")
            else:
                logger.warning("Compromise generation failed, skipping merge this iteration")
                if iteration > n_agents * 3:
                    break
                continue

            if progress_callback:
                progress_callback({
                    "iteration": iteration,
                    "coalitions": len(coalitions),
                    "n_agents": n_agents,
                    "majority_quota": majority_quota,
                    "event": "voting",
                    "rep_i": coal_i["representative"],
                    "rep_j": coal_j["representative"],
                    "size_i": len(coal_i["members"]),
                    "size_j": len(coal_j["members"]),
                    "proposal": proposal,
                    "merged_size": None,
                })

            def coalition_accepts(members: list) -> tuple:
                import numpy as np
                ideal_matrix = np.stack([agents[idx].ideal_embedding for idx in members])
                prop_vec = np.asarray(_encode_sentence(proposal), dtype=np.float32)
                sq_vec = np.asarray(_encode_sentence(status_quo), dtype=np.float32)
                d_prop = _batch_cosine_dissimilarity(ideal_matrix, prop_vec)
                d_sq = _batch_cosine_dissimilarity(ideal_matrix, sq_vec)
                yes = int(np.sum(d_prop < d_sq))
                for k, idx in enumerate(members):
                    logger.info(
                        f"   {agents[idx].name}: d(ideal→proposal)={d_prop[k]:.4f}, "
                        f"d(ideal→status_quo)={d_sq[k]:.4f}, voted={d_prop[k] < d_sq[k]}"
                    )
                return yes, len(members)

            yes_i, total_i = coalition_accepts(coal_i["members"])
            yes_j, total_j = coalition_accepts(coal_j["members"])
            merged_count = total_i + total_j

            logger.info(
                f"Vote result: yes_i={yes_i}/{total_i}  yes_j={yes_j}/{total_j}  "
                f"merged={merged_count}"
            )

            i_accepts = yes_i / total_i > 0.5
            j_accepts = yes_j / total_j > 0.5

            if i_accepts and j_accepts:
                merged = {
                    "members": coal_i["members"] + coal_j["members"],
                    "representative": proposal,
                }
                for idx in sorted([ci, cj], reverse=True):
                    coalitions.pop(idx)
                coalitions.append(merged)
                logger.info(f"Merged → coalition of {merged_count} agents, rep: '{proposal}'")
                print(f"  ✓ Merged! Coalition now has {merged_count}/{n_agents} agents.", flush=True)
                if progress_callback:
                    progress_callback({
                        "iteration": iteration,
                        "coalitions": len(coalitions),
                        "n_agents": n_agents,
                        "majority_quota": majority_quota,
                        "event": "merged",
                        "proposal": proposal,
                        "merged_size": merged_count,
                        "yes_i": yes_i, "total_i": total_i,
                        "yes_j": yes_j, "total_j": total_j,
                        "coalitions_2d": _snap_2d(),
                    })

                if merged_count / n_agents >= majority_quota:
                    winning_coalition = merged
                    winning_sentence = proposal
                    logger.info(f"Majority reached: {merged_count}/{n_agents} >= {majority_quota}")
                    print(f"  ★ Majority reached: {merged_count}/{n_agents} agents agreed!", flush=True)
                    if progress_callback:
                        progress_callback({
                            "iteration": iteration,
                            "coalitions": len(coalitions),
                            "n_agents": n_agents,
                            "majority_quota": majority_quota,
                            "event": "majority",
                            "proposal": proposal,
                            "merged_size": merged_count,
                            "coalitions_2d": _snap_2d(),
                        })
                    break
            else:
                logger.info(f"Merge rejected: i_accepts={i_accepts}, j_accepts={j_accepts}")
                print(f"  ✗ Merge rejected (i={yes_i}/{total_i}, j={yes_j}/{total_j}).", flush=True)
                if progress_callback:
                    progress_callback({
                        "iteration": iteration,
                        "coalitions": len(coalitions),
                        "n_agents": n_agents,
                        "majority_quota": majority_quota,
                        "event": "rejected",
                        "proposal": proposal,
                        "merged_size": None,
                        "yes_i": yes_i, "total_i": total_i,
                        "yes_j": yes_j, "total_j": total_j,
                    })

            if iteration > n_agents * 5:
                logger.warning("Max iterations reached")
                winning_coalition = max(coalitions, key=lambda c: len(c["members"]))
                winning_sentence = winning_coalition["representative"]
                break
    finally:
        del llm

    # Final full-vote of ALL agents on winning sentence for the proof table
    if winning_coalition is None:
        winning_coalition = max(coalitions, key=lambda c: len(c["members"]))
        winning_sentence = winning_coalition["representative"]

    import numpy as np
    ideal_matrix = np.stack([a.ideal_embedding for a in agents])
    prop_vec = np.asarray(_encode_sentence(winning_sentence), dtype=np.float32)
    sq_vec = np.asarray(_encode_sentence(status_quo), dtype=np.float32)
    d_props = _batch_cosine_dissimilarity(ideal_matrix, prop_vec)
    d_sqs = _batch_cosine_dissimilarity(ideal_matrix, sq_vec)
    voting_results = [
        {
            "name": agents[i].name,
            "d_proposal": float(d_props[i]),
            "d_status_quo": float(d_sqs[i]),
            "voted": bool(d_props[i] < d_sqs[i]),
        }
        for i in range(len(agents))
    ]

    coalition_names = [agents[idx].name for idx in winning_coalition["members"]]
    logger.info(f"Result sentence: '{winning_sentence}'")
    logger.info(f"Coalition: {coalition_names}")

    return {
        "result": winning_sentence,
        "coalition": coalition_names,
        "votes": voting_results,
        "iterations": iteration,
    }