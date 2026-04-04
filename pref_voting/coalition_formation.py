"""
An implementation of the algorithm in:
"AI-Generated Compromises for Coalition Formation", by Eyal Briman, Ehud Shapiro, and Nimrod Talmon (2024), https://arxiv.org/abs/2410.21440

Programmer: Hillel Ohayon.
Date: 2025-04-04.
"""

from __future__ import annotations
import math
from typing import Optional


# Type aliases
Vector = list[float] # 512-dim embedding (Universal Sentence Encoder)
Coalition = tuple[frozenset[int], str] # (agent_indices, compromise_sentence)


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
    """
    return [0.0]*512  # TODO: load USE model and return encoder(text).numpy().tolist()[0]


def cosine_dissimilarity(v1: Vector, v2: Vector) -> float:
    """Compute the square-root cosine-based dissimilarity between two vectors.

    Defined as sqrt(2 - 2*cos(theta)) where theta is the angle between v1 and v2
    (Section 4.1, footnote 6).  Returns values in [0, sqrt(2)].

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
    return -1  # TODO: dot(v1,v2) / (norm(v1)*norm(v2)) -> sqrt(2 - 2*cos)


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
    return False  # TODO: embed all three, compare distances, apply half-Gaussian if sigma>0


def generate_compromise_sentences(
    sentence1: str,
    sentence2: str,
    n: int = 10,
) -> list[str]:
    """Ask GPT-3.5-turbo to generate n sentences aggregating the two inputs.

    Uses the Mediator-1 prompt from Section 4.2: structured prompt + zero-shot
    chain-of-thought, with a max of 15 words per output sentence.

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
    return [""] * n  # TODO: call OpenAI API, parse numbered response into list


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
    return ""  # TODO: embed each candidate, return argmin cosine_dissimilarity to target


def coalition_formation(
    ideal_sentences: dict[int, str],
    status_quo: str,
    majority_quota: float = 0.5,
    sigma: float = 0.0,
    alpha: float = 0.0,
    coalition_discipline: bool = False,
    max_iterations: int = 10_000,
    seed: Optional[int] = None,
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

    Returns:
        tuple[str, list[int]]:
            - The winning compromise sentence.
            - Sorted list of agent indices in the majority coalition.

    Single agent trivially wins:

    >>> sentence, agents = coalition_formation({0: "Protect the forests."}, "Do nothing.")
    >>> agents
    [0]
    """
    return "", []  # TODO: full iterative loop using all helpers above