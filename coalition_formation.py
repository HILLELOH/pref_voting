"""
An implementation of the algorithm in:
"AI-Generated Compromises for Coalition Formation", by Eyal Briman, Ehud Shapiro, and Nimrod Talmon (2024), https://arxiv.org/pdf/2512.05983

Programmer: Hillel Ohayon.
Date: 2025-04-04.
"""

from __future__ import annotations
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

# Prevent PyTorch from reserving too much RAM, which starves llama-cpp
os.environ["OMP_NUM_THREADS"] = "4"

logger = logging.getLogger(__name__)

_st_model = None
_ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_EMBED_DIM = 512

def _get_st_model():
    """Load the sentence-transformer model (cached after first call)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer model '%s'…", _ST_MODEL_NAME)
        # Force CPU to avoid VRAM conflicts if a GPU is present
        _st_model = SentenceTransformer(_ST_MODEL_NAME, device="cpu")
    return _st_model

@dataclass
class _Coalition:
    agents: set[int]
    sentence: str
    embedding: np.ndarray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def embed_text(text: str) -> np.ndarray:
    """Embed a sentence into a 512-dimensional semantic vector."""
    logger.debug("Computing embedding (cache miss) for: %.60r", text)
    raw = _get_st_model().encode(text, convert_to_numpy=True)
    padded = np.zeros(_EMBED_DIM)
    padded[:len(raw)] = raw
    return padded


def cosine_dissimilarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """sqrt(2 - 2*cos(theta)). Returns values in [0, 2]."""
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.sqrt(max(0.0, 2.0 - 2.0 * cos))


def agent_votes(ideal: str, proposal: str, status_quo: str, sigma: float = 0.0) -> bool:
    """Return True if the agent accepts the proposed compromise sentence."""
    d_proposal = cosine_dissimilarity(embed_text(ideal), embed_text(proposal))
    d_sq = cosine_dissimilarity(embed_text(ideal), embed_text(status_quo))

    if d_proposal <= d_sq:
        return True
    if sigma == 0.0:
        return False

    prob = min(1.0, math.sqrt(2.0 / math.pi) / sigma * math.exp(-d_proposal**2 / (2.0 * sigma**2)))
    return bool(random.random() < prob)


_llama_cpp_model = None

def _get_llama_cpp_model():
    """Load Qwen model locally strictly via llama-cpp."""
    global _llama_cpp_model
    if _llama_cpp_model is None:
        from llama_cpp import Llama
        logger.info("Loading Qwen2.5-0.5B from local file via llama-cpp…")
        model_path = os.path.join(os.path.dirname(__file__), "models", "qwen2.5-0.5b-instruct-q3_k_m.gguf")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logger.debug("Model path: %s", model_path)
        _llama_cpp_model = Llama(
            model_path=model_path,
            verbose=False,
            n_ctx=1024,
            n_threads=4, # Restrict threads to prevent memory spikes
            n_batch=256  # Smaller batch size uses less memory
        )
    return _llama_cpp_model


def _build_mediator_prompt(sentence1: str, sentence2: str, n: int) -> tuple[str, str]:
    prompt = (
        f'Input Sentence 1: "{sentence1}"\n'
        f'Input Sentence 2: "{sentence2}"\n\n'
        f"Task:\n"
        f"Generate exactly {n} distinct, creative compromise sentences that bridge the core ideas of both inputs.\n\n"
        f"Strict Constraints:\n"
        f"1. Each sentence must be at most 15 words.\n"
        f"2. High Syntactic & Lexical Diversity: Do NOT reuse the same phrasing, structure, or prominent vocabulary from the input sentences. Express the middle-ground using completely different words and a fresh sentence structure.\n"
        f"3. Do not just mix or concatenate pieces of the two sentences.\n"
        f"4. Maintain the original language of the input sentences.\n\n"
        f'Return ONLY valid JSON in this format: {{"compromises": ["sentence1", "sentence2", ...]}}'
    )
    
    system_msg = (
        "You are an expert mediator AI. Your job is to find conceptual common ground between two opposing opinions. "
        "Crucially, your generations must sound entirely fresh—use unique vocabulary, different idioms, and distinct sentence structures "
        "so that they do not look superficially similar to either input. "
        "Respond strictly with valid JSON. No markdown wrappers (like ```json), no conversational filler."
    )
    return system_msg, prompt


def generate_compromise_sentences(
    sentence1: str,
    sentence2: str,
    n: int = 2,
    api_key: str = None, # Left in args to prevent breaking app.py, but ignored
) -> list[str]:
    """Generate sentences aggregating the two inputs using ONLY local Qwen."""
    llm = _get_llama_cpp_model()
    system_msg, prompt = _build_mediator_prompt(sentence1, sentence2, n)
    
    logger.info("Calling local llama-cpp (Qwen) for %d compromise candidates...", n)
    
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    raw = response["choices"][0]["message"]["content"] or ""
    logger.info("RAW QWEN OUTPUT: %s", raw)
    
    return _parse_json_response(raw, n, sentence1, sentence2)


def _parse_json_response(text: str, n: int, sentence1: str, sentence2: str) -> list[str]:
    """Parse JSON response into a list of n compromise strings."""
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    def _extract(data) -> list[str] | None:
        if isinstance(data, list):
            return [str(s).strip() for s in data if str(s).strip()] or None
        if isinstance(data, dict):
            for key in ("compromises", "sentences", "results", "suggestions"):
                if key in data and isinstance(data[key], list):
                    return [str(s).strip() for s in data[key] if str(s).strip()] or None
        return None

    attempts = [cleaned]
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m: attempts.append(m.group(0))

    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if m: attempts.append(m.group(0))

    for pfx in ('{"compromises":[', '{"compromises": ['):
        idx = cleaned.find(pfx)
        if idx >= 0:
            attempts.append(cleaned[idx:].rstrip().rstrip(',') + ']}')

    idx = cleaned.find('[')
    if idx >= 0:
        attempts.append(cleaned[idx:].rstrip().rstrip(',') + ']')

    results = None
    for attempt in attempts:
        try:
            results = _extract(json.loads(attempt))
            if results: break
        except (json.JSONDecodeError, TypeError):
            continue

    if not results:
        logger.warning("Failed to parse JSON. Returning fallback templates.")
        return [f"{sentence1.rstrip('.')} and {sentence2.lower()}"] * n

    if len(results) < n:
        results += [results[-1]] * (n - len(results))
    return results[:n]


def choose_best_sentence(
    candidates: list[str],
    target: np.ndarray,
    original_sentences: list[str] = None,
    diversity_weight: float = 0.35,
) -> str:
    """Return the candidate closest to target while penalizing similarity to originals."""
    if len(candidates) == 1:
        return candidates[0]
    
    scored = []
    for s in candidates:
        s_emb = embed_text(s)
        dist_to_target = cosine_dissimilarity(s_emb, target)
        
        if original_sentences and len(original_sentences) > 0:
            min_dist_to_original = min(
                cosine_dissimilarity(s_emb, embed_text(orig))
                for orig in original_sentences
            )
            similarity_penalty = max(0.0, 1.0 - min_dist_to_original)
            composite_score = dist_to_target + diversity_weight * similarity_penalty
            scored.append((dist_to_target, composite_score, similarity_penalty, s))
        else:
            scored.append((dist_to_target, dist_to_target, None, s))
    
    scored.sort(key=lambda x: x[1])
    return scored[0][3]


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
    """Run the AI-mediated coalition formation algorithm."""

    file_path = Path("logs/app.log")
    if file_path.is_file():
        file_path.unlink()

    if not 0.0 <= majority_quota <= 1.0:
        raise ValueError(f"majority_quota must be in [0, 1], got {majority_quota}")
    if not ideal_sentences:
        return status_quo, []

    n_agents = len(ideal_sentences)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    coalitions = [
        _Coalition(agents={i}, sentence=s, embedding=embed_text(s))
        for i, s in ideal_sentences.items()
    ]

    def meets_quota(c: _Coalition) -> bool:
        return len(c.agents) / n_agents >= majority_quota

    def cast_votes(coalition: _Coalition, compromise: str) -> dict[int, bool]:
        return {
            a: agent_votes(ideal_sentences[a], compromise, status_quo, sigma)
            for a in coalition.agents
        }

    def split(votes: dict[int, bool], all_agents: set[int], discipline: bool) -> tuple[set[int], set[int]]:
        if discipline and sum(votes.values()) < math.ceil(len(all_agents) / 2):
            return set(), set(all_agents)
        yes = {a for a, v in votes.items() if v}
        no  = {a for a, v in votes.items() if not v}
        return yes, no

    logger.info("Init: %d agents, quota=%.2f, sigma=%.2f", n_agents, majority_quota, sigma)

    for c in coalitions:
        if meets_quota(c):
            return c.sentence, sorted(c.agents)

    for iteration in range(1, max_iterations + 1):
        logger.info("--- Iteration %d  (coalitions: %d) ---", iteration, len(coalitions))

        if len(coalitions) == 1:
            break

        sizes = np.array([len(c.agents) for c in coalitions], dtype=float)
        embeddings = np.stack([c.embedding for c in coalitions])
        centroid = (sizes @ embeddings) / sizes.sum()

        dists = np.array([cosine_dissimilarity(c.embedding, centroid) for c in coalitions])
        probs = np.exp(alpha * (dists / (dists.max() or 1.0)))
        probs /= probs.sum()
        idx_i = int(np.random.choice(len(coalitions), p=probs))

        idx_j = min(
            (k for k in range(len(coalitions)) if k != idx_i),
            key=lambda k: cosine_dissimilarity(coalitions[k].embedding, coalitions[idx_i].embedding),
        )

        c_i, c_j = coalitions[idx_i], coalitions[idx_j]
        size_i, size_j = float(len(c_i.agents)), float(len(c_j.agents))

        target_emb = (size_i * c_i.embedding + size_j * c_j.embedding) / (size_i + size_j)

        # STRICTLY uses local qwen model now
        candidates = generate_compromise_sentences(c_i.sentence, c_j.sentence)
        
        compromise_sentence = choose_best_sentence(
            candidates, 
            target_emb, 
            original_sentences=[c_i.sentence, c_j.sentence]
        )
        compromise_emb = embed_text(compromise_sentence)

        votes_i = cast_votes(c_i, compromise_sentence)
        votes_j = cast_votes(c_j, compromise_sentence)
        
        new_i, rem_i = split(votes_i, c_i.agents, coalition_discipline)
        new_j, rem_j = split(votes_j, c_j.agents, coalition_discipline)
        new_agents = new_i | new_j
        
        coalitions = [c for k, c in enumerate(coalitions) if k not in (idx_i, idx_j)]
        if rem_i: coalitions.append(_Coalition(rem_i, c_i.sentence, c_i.embedding))
        if rem_j: coalitions.append(_Coalition(rem_j, c_j.sentence, c_j.embedding))
        if new_agents: coalitions.append(_Coalition(new_agents, compromise_sentence, compromise_emb))

        for c in coalitions:
            if meets_quota(c):
                return c.sentence, sorted(c.agents)

    winner = max(coalitions, key=lambda c: len(c.agents))
    return winner.sentence, sorted(winner.agents)