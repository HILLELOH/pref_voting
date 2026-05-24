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
_MODEL_FILENAME = "qwen2.5-0.5b-instruct-q3_k_m.gguf"
_MODEL_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "models", _MODEL_FILENAME),
    "/home/hilleloh/app/models/" + _MODEL_FILENAME,
]

def _get_qwen_model():
    """Load Qwen locally via llama-cpp (fresh instance per request)."""
    from llama_cpp import Llama

    model_path = next((p for p in _MODEL_SEARCH_PATHS if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(
            f"Qwen model not found. Searched: {_MODEL_SEARCH_PATHS}"
        )

    logger.info(f"Loading Qwen2.5-0.5B from {model_path}…")
    try:
        return Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"Llama() constructor failed: {e}", exc_info=True)
        raise


# ============================================================================
# COALITION FORMATION ALGORITHM
# ============================================================================

def _cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    import numpy as np
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_dissimilarity(a, b):
    """Compute cosine dissimilarity (1 - similarity)."""
    return 1 - _cosine_similarity(a, b)


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


def _is_valid_compromise(candidate: str, all_ideals: list, max_common: int = 2) -> bool:
    """Return True if candidate shares no more than max_common consecutive words with any ideal."""
    for ideal in all_ideals:
        if _max_consecutive_common_words(candidate, ideal) > max_common:
            return False
    return True


def _encode_sentence(sentence: str) -> list[float]:
    """Encode a sentence to embeddings."""
    st_model = _get_st_model()
    embedding = st_model.encode(sentence, convert_to_numpy=True)
    return embedding.tolist()


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


def _call_qwen_local(sentence1: str, sentence2: str, n: int = 2) -> list[str]:
    """Call local Qwen2.5-0.5B for compromise generation."""
    logger.info(f"Calling Qwen for {n} compromise candidates...")

    llm = _get_qwen_model()
    try:
        prompt = (
            f'Two agents disagree on policy:\n'
            f'- "{sentence1}"\n'
            f'- "{sentence2}"\n\n'
            f'Write {n} NEW compromise sentences (8-15 words each) that blend both ideas.\n'
            f'Do NOT copy the input sentences. Write new sentences only.\n\n'
            f'Example format:\n'
            f'{{"compromises": ["Adopt balanced measures addressing both goals simultaneously.", "Combine approaches to achieve shared environmental outcomes."]}}\n\n'
            f'Your answer:\n'
        )

        response = llm(
            prompt,
            max_tokens=400,
            temperature=0.7,
            top_p=0.95,
        )

        raw_text = response["choices"][0]["text"]
        logger.info(f"Qwen raw output: {raw_text!r}")

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
            logger.warning("Qwen only produced copies of input — using fallback blending")
            words1 = sentence1.rstrip('.').split()
            words2 = sentence2.rstrip('.').split()
            filtered = [f"{' '.join(words1[:4])} and {words2[-1].lower()} sustainably."]

        logger.info(f"Compromise(s): {filtered}")
        return filtered
    finally:
        del llm


# ============================================================================
# COALITION FORMATION
# ============================================================================

class Agent:
    """Represents an agent with an ideal proposal and voting logic."""
    
    def __init__(self, name: str, ideal: str):
        self.name = name
        self.ideal = ideal
        self.ideal_embedding = _encode_sentence(ideal)
    
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
) -> dict:
    """
    Run bottom-up coalition formation algorithm.

    Each agent starts as individual coalition. Each iteration: pick 2 coalitions,
    generate LLM compromise, each coalition votes internally. If both accept
    (strict majority within each group), merge. Continue until merged coalition
    reaches majority_quota of all agents.
    """
    import random

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
    iteration = 0
    winning_coalition = None
    winning_sentence = None

    while True:
        iteration += 1
        logger.info(f"--- Iteration {iteration}  (coalitions: {len(coalitions)}) ---")

        if seed is not None:
            random.seed(seed + iteration)

        # Pick 2 distinct coalitions to try to merge
        if len(coalitions) < 2:
            # Only one coalition left — it's the winner
            winning_coalition = coalitions[0]
            winning_sentence = winning_coalition["representative"]
            break

        ci, cj = random.sample(range(len(coalitions)), 2)
        coal_i = coalitions[ci]
        coal_j = coalitions[cj]

        rep_i = coal_i["representative"]
        rep_j = coal_j["representative"]

        # Generate compromise between the two coalition representatives
        candidates = _call_qwen_local(rep_i, rep_j, n=2)

        valid = [c for c in candidates if _is_valid_compromise(c, all_ideals)]
        if valid:
            proposal = valid[0]
        elif candidates:
            proposal = min(
                candidates,
                key=lambda c: max(_max_consecutive_common_words(c, ideal) for ideal in all_ideals),
            )
            logger.warning(f"No candidate passed overlap filter, using least-overlap: {proposal!r}")
        else:
            logger.warning("Compromise generation failed, skipping merge this iteration")
            if iteration > n_agents * 3:
                break
            continue

        # Each coalition votes internally: strict majority of its members must accept
        def coalition_accepts(members: list) -> tuple:
            yes = sum(1 for idx in members if agents[idx].vote(proposal, status_quo)[0])
            return yes, len(members)

        yes_i, total_i = coalition_accepts(coal_i["members"])
        yes_j, total_j = coalition_accepts(coal_j["members"])

        merged_count = total_i + total_j
        logger.info(
            f"Vote result: yes_i={yes_i}/{total_i}  yes_j={yes_j}/{total_j}  "
            f"merged={merged_count}"
        )

        # Both coalitions must have strict majority acceptance
        i_accepts = yes_i / total_i > 0.5
        j_accepts = yes_j / total_j > 0.5

        if i_accepts and j_accepts:
            merged = {
                "members": coal_i["members"] + coal_j["members"],
                "representative": proposal,
            }
            # Remove old coalitions (higher index first to preserve lower index)
            for idx in sorted([ci, cj], reverse=True):
                coalitions.pop(idx)
            coalitions.append(merged)

            logger.info(f"Merged → coalition of {merged_count} agents, rep: '{proposal}'")

            # Check if merged coalition satisfies majority_quota
            if merged_count / n_agents >= majority_quota:
                winning_coalition = merged
                winning_sentence = proposal
                logger.info(f"Majority reached: {merged_count}/{n_agents} >= {majority_quota}")
                break
        else:
            logger.info(
                f"Merge rejected: i_accepts={i_accepts}, j_accepts={j_accepts}"
            )

        # Safety valve
        if iteration > n_agents * 5:
            logger.warning("Max iterations reached")
            # Pick largest coalition as winner
            winning_coalition = max(coalitions, key=lambda c: len(c["members"]))
            winning_sentence = winning_coalition["representative"]
            break

    # Final full-vote of ALL agents on winning sentence for the proof table
    if winning_coalition is None:
        winning_coalition = max(coalitions, key=lambda c: len(c["members"]))
        winning_sentence = winning_coalition["representative"]

    voting_results = []
    for agent in agents:
        voted, d_prop, d_sq = agent.vote(winning_sentence, status_quo)
        voting_results.append({
            "name": agent.name,
            "d_proposal": d_prop,
            "d_status_quo": d_sq,
            "voted": voted,
        })

    coalition_names = [agents[idx].name for idx in winning_coalition["members"]]
    logger.info(f"Result sentence: '{winning_sentence}'")
    logger.info(f"Coalition: {coalition_names}")

    return {
        "result": winning_sentence,
        "coalition": coalition_names,
        "votes": voting_results,
        "iterations": iteration,
    }