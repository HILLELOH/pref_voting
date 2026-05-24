"""
AI-Mediated Coalition Formation Implementation
Based on: Briman, Shapiro, Talmon 2024
https://arxiv.org/pdf/2512.05983
"""

import json
import logging
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
def _get_qwen_model():
    """Load Qwen locally via llama-cpp (fresh instance per request)."""
    from llama_cpp import Llama
    
    model_path = "/home/hilleloh/app/models/qwen2.5-0.5b-instruct-q3_k_m.gguf"
    logger.info("Loading Qwen2.5-0.5B from local file via llama-cpp…")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )
    return llm


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


def _encode_sentence(sentence: str) -> list[float]:
    """Encode a sentence to embeddings."""
    st_model = _get_st_model()
    embedding = st_model.encode(sentence, convert_to_numpy=True)
    return embedding.tolist()


def _parse_json_response(text: str, n: int, sentence1: str, sentence2: str) -> list[str]:
    """Parse JSON response into a list of n compromise strings."""
    cleaned = text.strip()
    # Remove markdown code fence markers
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    
    try:
        data = json.loads(cleaned)
        compromises = data.get("compromises", [])
        
        if isinstance(compromises, list):
            # Return requested number of compromises (or fewer if not available)
            return compromises[:n]
        else:
            logger.warning(f"Expected list in 'compromises', got {type(compromises)}")
            return [sentence1, sentence2][:n]
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}\nRaw text:\n{text}")
        return [sentence1, sentence2][:n]


def _call_qwen_local(sentence1: str, sentence2: str, n: int = 2) -> list[str]:
    """Call local Qwen2.5-0.5B for compromise generation."""
    logger.info("Calling local llama-cpp (Qwen) for 2 compromise candidates...")
    
    # Load model (FRESH INSTANCE per request - THIS IS KEY!)
    llm = _get_qwen_model()
    
    try:
        prompt = f"""Generate {n} compromise proposals between these two positions:
Position 1: {sentence1}
Position 2: {sentence2}

Return ONLY valid JSON (no markdown, no preamble):
{{"compromises": ["proposal1", "proposal2"]}}"""

        response = llm(
            prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.95,
            stop=["}\n", "}"],
        )
        
        raw_text = response["choices"][0]["text"]
        logger.info(f"RAW QWEN OUTPUT: {raw_text}")
        
        compromises = _parse_json_response(raw_text, n, sentence1, sentence2)
        return compromises
    finally:
        # CRITICAL: Clean up model to free memory and CUDA resources
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
    Run the coalition formation algorithm.
    
    Args:
        agents_info: List of dicts with 'name' and 'ideal' keys
        majority_quota: Fraction of agents needed for coalition
        sigma: Noise parameter
        seed: Random seed
        status_quo: Current status quo proposal
    
    Returns:
        dict with 'result', 'coalition', 'votes' keys
    """
    n_agents = len(agents_info)
    agents = [Agent(info["name"], info["ideal"]) for info in agents_info]
    
    logger.info(f"Run: {n_agents} agents, majority_quota={majority_quota}, sigma={sigma}, seed={seed}, status_quo='{status_quo}'")
    for i, info in enumerate(agents_info, 1):
        logger.info(f"   Agent {i-1} ({info['name']}): '{info['ideal']}'")
    
    logger.info(f"Init: {n_agents} agents, quota={majority_quota:.2f}, sigma={sigma:.2f}")
    
    current_proposal = status_quo
    iteration = 0
    
    while True:
        iteration += 1
        
        # Count agents in "coalition" (would vote for current proposal)
        votes = []
        voting_results = []
        for agent in agents:
            voted, d_prop, d_sq = agent.vote(current_proposal, status_quo)
            votes.append(voted)
            voting_results.append({"name": agent.name, "d_proposal": d_prop, "d_status_quo": d_sq, "voted": voted})
        
        coalition_size = sum(votes)
        logger.info(f"--- Iteration {iteration}  (coalitions: {coalition_size}) ---")
        
        # Check if we have a winning coalition
        if coalition_size / n_agents >= majority_quota:
            coalition_names = [agents[i].name for i in range(n_agents) if votes[i]]
            logger.info(f"Result sentence: '{current_proposal}'")
            logger.info(f"Coalition: {coalition_names}")
            for vr in voting_results:
                logger.info(f"   {vr['name']}: d(ideal→proposal)={vr['d_proposal']:.4f}, d(ideal→status_quo)={vr['d_status_quo']:.4f}, voted={vr['voted']}")
            
            return {
                "result": current_proposal,
                "coalition": coalition_names,
                "votes": voting_results,
                "iterations": iteration,
            }
        
        # Generate new compromise proposals
        if coalition_size > 0:
            # Pick two agents from coalition to drive compromise
            import random
            if seed is not None:
                random.seed(seed + iteration)
            
            coalition_indices = [i for i in range(n_agents) if votes[i]]
            idx1, idx2 = random.sample(coalition_indices, min(2, len(coalition_indices)))
            sentence1 = agents[idx1].ideal
            sentence2 = agents[idx2].ideal
        else:
            # Pick any two agents
            import random
            if seed is not None:
                random.seed(seed + iteration)
            idx1, idx2 = random.sample(range(n_agents), 2)
            sentence1 = agents[idx1].ideal
            sentence2 = agents[idx2].ideal
        
        # Get compromise proposals
        candidates = _call_qwen_local(sentence1, sentence2, n=2)
        
        if not candidates:
            logger.warning("Compromise generation failed, using fallback")
            current_proposal = sentence1
        else:
            current_proposal = candidates[0]
        
        # Safeguard: don't loop forever
        if iteration > 10:
            logger.warning("Max iterations reached, returning current proposal")
            coalition_names = [agents[i].name for i in range(n_agents) if votes[i]]
            return {
                "result": current_proposal,
                "coalition": coalition_names,
                "votes": voting_results,
                "iterations": iteration,
            }