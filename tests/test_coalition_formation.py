"""
Tests for coalition_formation.py (AI textual version).

All tests currently FAIL because the implementations are empty stubs.
They will pass once the algorithm is fully implemented.
"""

import math
import pytest
from unittest.mock import patch

from pref_voting.coalition_formation import (
    embed_text,
    cosine_dissimilarity,
    agent_votes,
    generate_compromise_sentences,
    choose_best_sentence,
    coalition_formation,
)


# ---------------------------------------------------------------------------
# cosine_dissimilarity
# ---------------------------------------------------------------------------

class TestCosineDissimilarity:
    def test_identical_vectors_is_zero(self):
        assert cosine_dissimilarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors(self):
        assert cosine_dissimilarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.sqrt(2), rel=1e-3)

    def test_opposite_vectors(self):
        assert cosine_dissimilarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0, rel=1e-3)

    def test_symmetry(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert cosine_dissimilarity(a, b) == pytest.approx(cosine_dissimilarity(b, a))

    def test_result_in_valid_range(self):
        import random; random.seed(0)
        a = [random.gauss(0, 1) for _ in range(512)]
        b = [random.gauss(0, 1) for _ in range(512)]
        d = cosine_dissimilarity(a, b)
        assert 0.0 <= d <= math.sqrt(2) + 1e-9


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

class TestEmbedText:
    def test_returns_512_dimensions(self):
        v = embed_text("Reduce carbon emissions globally.")
        assert len(v) == 512

    def test_returns_floats(self):
        v = embed_text("Climate change is urgent.")
        assert all(isinstance(x, float) for x in v)

    def test_similar_sentences_are_closer(self):
        v1 = embed_text("Plant trees to fight climate change.")
        v2 = embed_text("Grow forests to combat global warming.")
        v3 = embed_text("Increase military spending now.")
        d_similar = cosine_dissimilarity(v1, v2)
        d_different = cosine_dissimilarity(v1, v3)
        assert d_similar < d_different

    def test_same_sentence_distance_zero(self):
        v = embed_text("We must act on climate change.")
        assert cosine_dissimilarity(v, v) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# agent_votes
# ---------------------------------------------------------------------------

class TestAgentVotes:
    def test_deterministic_accepts_identical_to_ideal(self):
        s = "Cut carbon emissions now."
        assert agent_votes(s, s, "Do nothing.", sigma=0.0) is True

    def test_deterministic_rejects_farther_proposal(self):
        # ideal close to proposal_bad → should reject
        assert agent_votes(
            "Plant trees everywhere.",
            "Increase military spending.",
            "Plant trees everywhere.",
            sigma=0.0,
        ) is False

    def test_deterministic_accepts_closer_proposal(self):
        assert agent_votes(
            "Use solar energy.",
            "Switch to renewable energy.",
            "Keep burning coal forever.",
            sigma=0.0,
        ) is True

    def test_probabilistic_sometimes_accepts_worse(self):
        """With large sigma, a worse proposal can still be accepted."""
        import random; random.seed(42)
        results = [
            agent_votes("Cut emissions.", "Do nothing.", "Cut emissions.", sigma=10.0)
            for _ in range(100)
        ]
        assert any(results), "High sigma should allow occasional acceptance of worse proposal"

    def test_returns_bool(self):
        result = agent_votes("a", "b", "c", sigma=0.0)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# generate_compromise_sentences
# ---------------------------------------------------------------------------

class TestGenerateCompromiseSentences:
    def test_returns_correct_count(self):
        sentences = generate_compromise_sentences(
            "Plant trees to fight climate change.",
            "Switch to renewable energy.",
            n=5,
        )
        assert len(sentences) == 5

    def test_returns_non_empty_strings(self):
        sentences = generate_compromise_sentences(
            "Plant trees to fight climate change.",
            "Switch to renewable energy.",
            n=3,
        )
        assert all(isinstance(s, str) and len(s) > 0 for s in sentences)

    def test_default_count_is_10(self):
        sentences = generate_compromise_sentences(
            "Reduce emissions.",
            "Plant trees.",
        )
        assert len(sentences) == 10

    def test_sentences_max_15_words(self):
        """Each sentence should be at most 15 words (Section 4.2 constraint)."""
        sentences = generate_compromise_sentences(
            "Protect rainforests by reducing deforestation rates globally.",
            "Invest in solar and wind energy technology.",
            n=5,
        )
        for s in sentences:
            assert len(s.split()) <= 15, f"Too long: '{s}'"


# ---------------------------------------------------------------------------
# choose_best_sentence
# ---------------------------------------------------------------------------

class TestChooseBestSentence:
    def test_returns_string(self):
        target = [1.0] + [0.0] * 511
        result = choose_best_sentence(["hello", "world"], target)
        assert isinstance(result, str)

    def test_returns_one_of_the_candidates(self):
        candidates = ["Cut emissions now.", "Plant more trees.", "Use green energy."]
        target = embed_text("Reduce carbon through renewable energy and reforestation.")
        result = choose_best_sentence(candidates, target)
        assert result in candidates

    def test_single_candidate_always_returned(self):
        target = [0.0] * 512
        result = choose_best_sentence(["only option"], target)
        assert result == "only option"


# ---------------------------------------------------------------------------
# coalition_formation (main algorithm)
# ---------------------------------------------------------------------------

class TestCoalitionFormation:
    def test_single_agent_returns_itself(self):
        sentence, agents = coalition_formation(
            {0: "Protect the forests."}, "Do nothing."
        )
        assert agents == [0]
        assert isinstance(sentence, str)

    def test_returns_majority(self):
        ideal = {
            0: "Plant trees to absorb CO2 emissions.",
            1: "Switch to solar and wind energy.",
            2: "Reduce meat consumption to lower emissions.",
            3: "Invest in carbon capture technologies.",
            4: "Improve public transport to cut car use.",
        }
        sentence, agents = coalition_formation(
            ideal, "Do nothing about climate change.", seed=0
        )
        assert len(agents) >= math.ceil(len(ideal) / 2)

    def test_returned_agents_are_valid_indices(self):
        ideal = {i: f"Policy proposal number {i}." for i in range(6)}
        sentence, agents = coalition_formation(ideal, "Status quo.", seed=1)
        assert all(a in ideal for a in agents)

    def test_compromise_is_string(self):
        ideal = {0: "Tax carbon heavily.", 1: "Subsidise green energy."}
        sentence, agents = coalition_formation(
            ideal, "Do nothing.", majority_quota=0.51, seed=0
        )
        assert isinstance(sentence, str) and len(sentence) > 0

    def test_large_random_input_converges(self):
        """With 20 agents on a global-warming topic, algorithm should converge."""
        import random; random.seed(77)
        topics = [
            "Plant trees globally.",
            "Ban fossil fuels immediately.",
            "Invest in nuclear energy.",
            "Tax carbon emissions heavily.",
            "Improve public transport networks.",
            "Subsidise electric vehicles now.",
            "Reduce meat consumption worldwide.",
            "Install rooftop solar panels.",
            "Protect existing rainforests legally.",
            "Develop carbon capture technologies.",
        ] * 2  # 20 agents
        ideal = {i: topics[i] for i in range(20)}
        sentence, agents = coalition_formation(
            ideal, "Do nothing about climate change.", sigma=1.0, seed=77
        )
        assert len(agents) >= 10

    def test_deterministic_reproducible(self):
        ideal = {0: "Cut emissions.", 1: "Plant trees.", 2: "Use renewables."}
        r1 = coalition_formation(ideal, "Do nothing.", seed=42)
        r2 = coalition_formation(ideal, "Do nothing.", seed=42)
        assert r1[1] == r2[1]