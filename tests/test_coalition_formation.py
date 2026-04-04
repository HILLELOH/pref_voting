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
        # Explanation: The distance between a vector and itself should be exactly 0.
        assert cosine_dissimilarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors(self):
        # Explanation: Perpendicular (orthogonal) vectors should have a distance of sqrt(2).
        assert cosine_dissimilarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.sqrt(2), rel=1e-3)

    def test_opposite_vectors(self):
        # Explanation: Vectors pointing in completely opposite directions should have the maximum distance of 2.
        assert cosine_dissimilarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0, rel=1e-3)

    def test_symmetry(self):
        # Explanation: Distance from A to B must be exactly the same as distance from B to A.
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert cosine_dissimilarity(a, b) == pytest.approx(cosine_dissimilarity(b, a))

    def test_result_in_valid_range(self):
        # Explanation: The output of the distance formula must always be a positive number.
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
        # Explanation: The Universal Sentence Encoder must return exactly 512 dimensions.
        v = embed_text("Reduce carbon emissions globally.")
        assert len(v) == 512

    def test_returns_floats(self):
        # Explanation: All 512 items in the embedding vector must be floating-point numbers.
        v = embed_text("Climate change is urgent.")
        assert all(isinstance(x, float) for x in v)

    def test_similar_sentences_are_closer(self):
        # Explanation: Sentences with similar meanings should have a smaller distance than completely unrelated sentences.
        v1 = embed_text("Plant trees to fight climate change.")
        v2 = embed_text("Grow forests to combat global warming.")
        v3 = embed_text("Increase military spending now.")
        d_similar = cosine_dissimilarity(v1, v2)
        d_different = cosine_dissimilarity(v1, v3)
        assert d_similar < d_different

    def test_same_sentence_distance_zero(self):
        # Explanation: Embedding the exact same text twice should yield identical vectors (distance 0).
        v = embed_text("We must act on climate change.")
        assert cosine_dissimilarity(v, v) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# agent_votes
# ---------------------------------------------------------------------------

class TestAgentVotes:
    def test_deterministic_accepts_identical_to_ideal(self):
        # Explanation: An agent should always vote True if the proposal is their exact ideal sentence.
        s = "Cut carbon emissions now."
        assert agent_votes(s, s, "Do nothing.", sigma=0.0) is True

    def test_deterministic_rejects_farther_proposal(self):
        # Explanation: A deterministic agent (sigma=0) must reject a proposal that is mathematically worse than the status quo.
        assert agent_votes(
            "Plant trees everywhere.",
            "Increase military spending.",
            "Plant trees everywhere.",
            sigma=0.0,
        ) is False

    def test_deterministic_accepts_closer_proposal(self):
        # Explanation: A deterministic agent (sigma=0) must accept a proposal that is closer to their ideal than the status quo.
        assert agent_votes(
            "Use solar energy.",
            "Switch to renewable energy.",
            "Keep burning coal forever.",
            sigma=0.0,
        ) is True

    def test_probabilistic_sometimes_accepts_worse(self):
        # Explanation: With high altruism/flexibility (sigma), the agent should occasionally vote True even if the proposal is worse.
        import random; random.seed(42)
        results = [
            agent_votes("Cut emissions.", "Do nothing.", "Cut emissions.", sigma=10.0)
            for _ in range(100)
        ]
        assert any(results), "High sigma should allow occasional acceptance of worse proposal"

    def test_returns_bool(self):
        # Explanation: The voting function must return a standard boolean (True/False).
        result = agent_votes("a", "b", "c", sigma=0.0)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# generate_compromise_sentences
# ---------------------------------------------------------------------------

class TestGenerateCompromiseSentences:
    def test_returns_correct_count(self):
        # Explanation: The LLM function should return exactly 'n' candidate sentences.
        sentences = generate_compromise_sentences(
            "Plant trees to fight climate change.",
            "Switch to renewable energy.",
            n=5,
        )
        assert len(sentences) == 5

    def test_returns_non_empty_strings(self):
        # Explanation: The LLM should return actual text strings, not empty data or None.
        sentences = generate_compromise_sentences(
            "Plant trees to fight climate change.",
            "Switch to renewable energy.",
            n=3,
        )
        assert all(isinstance(s, str) and len(s) > 0 for s in sentences)

    def test_default_count_is_10(self):
        # Explanation: If 'n' is not specified, the function must default to generating 10 sentences.
        sentences = generate_compromise_sentences(
            "Reduce emissions.",
            "Plant trees.",
        )
        assert len(sentences) == 10

    def test_sentences_max_15_words(self):
        # Explanation: As per the paper's constraints, the LLM should generate short sentences of <= 15 words.
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
        # Explanation: The selection function should return a single string (the chosen sentence).
        target = [1.0] + [0.0] * 511
        result = choose_best_sentence(["hello", "world"], target)
        assert isinstance(result, str)

    def test_returns_one_of_the_candidates(self):
        # Explanation: The returned sentence must physically exist in the list of candidates provided by the LLM.
        candidates = ["Cut emissions now.", "Plant more trees.", "Use green energy."]
        target = embed_text("Reduce carbon through renewable energy and reforestation.")
        result = choose_best_sentence(candidates, target)
        assert result in candidates

    def test_single_candidate_always_returned(self):
        # Explanation: Edge case - if the LLM only gave 1 candidate, the function must pick it by default.
        target = [0.0] * 512
        result = choose_best_sentence(["only option"], target)
        assert result == "only option"


# ---------------------------------------------------------------------------
# coalition_formation (main algorithm)
# ---------------------------------------------------------------------------

class TestCoalitionFormation:
    def test_single_agent_returns_itself(self):
        # Explanation: Edge case - a system with only 1 agent immediately halts, as that 1 agent represents 100% of the votes.
        sentence, agents = coalition_formation(
            {0: "Protect the forests."}, "Do nothing."
        )
        assert agents == [0]
        assert isinstance(sentence, str)

    def test_returns_majority(self):
        # Explanation: The algorithm must loop until the final coalition reaches the default majority threshold (50%).
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
        # Explanation: The final returned list of agents must contain valid IDs from the original input dictionary.
        ideal = {i: f"Policy proposal number {i}." for i in range(6)}
        sentence, agents = coalition_formation(ideal, "Status quo.", seed=1)
        assert all(a in ideal for a in agents)

    def test_compromise_is_string(self):
        # Explanation: The final output of the algorithm must be the actual compromise string text.
        ideal = {0: "Tax carbon heavily.", 1: "Subsidise green energy."}
        sentence, agents = coalition_formation(
            ideal, "Do nothing.", majority_quota=0.51, seed=0
        )
        assert isinstance(sentence, str) and len(sentence) > 0

    def test_large_random_input_converges(self):
        # Explanation: Large input test - checks that 20 agents voting on related topics successfully coalesce into a single majority group.
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
        # Explanation: Ensures that if we provide the exact same random seed, the algorithm makes the exact same choices and yields identical results.
        ideal = {0: "Cut emissions.", 1: "Plant trees.", 2: "Use renewables."}
        r1 = coalition_formation(ideal, "Do nothing.", seed=42)
        r2 = coalition_formation(ideal, "Do nothing.", seed=42)
        assert r1[1] == r2[1]

    def test_empty_input(self):
        # Explanation: Edge case - 0 agents passed. The system should safely return the status quo and an empty list, not crash.
        sentence, agents = coalition_formation({}, "Status quo remains.")
        assert sentence == "Status quo remains."
        assert len(agents) == 0

    def test_invalid_majority_quota_raises_error(self):
        # Explanation: Error handling test - it is mathematically impossible to require a quota greater than 1.0 (100%), so it should raise an error.
        ideal = {0: "A", 1: "B"}
        with pytest.raises(ValueError):
            coalition_formation(ideal, "Status quo", majority_quota=1.5)

    def test_invalid_alpha_raises_error(self):
        # Explanation: Error handling test - the alpha parameter dictates mediator behavior and is strictly restricted to -1, 0, or 1.
        ideal = {0: "A", 1: "B"}
        with pytest.raises(ValueError):
            coalition_formation(ideal, "Status quo", alpha=5)

    def test_very_large_random_input_subset_property(self):
        # Explanation: Large property test - creates 100 random agents and verifies structural properties (subset validity, quota minimums) rather than specific text outputs.
        import random
        random.seed(100)
        topics = ["Solar", "Wind", "Nuclear", "Geothermal", "Hydro", "Biomass"]
        ideal = {i: random.choice(topics) for i in range(100)}

        sentence, agents = coalition_formation(
            ideal, "Status quo", majority_quota=0.6, seed=100
        )
        
        assert set(agents).issubset(set(ideal.keys()))
        assert len(agents) >= 60
        assert isinstance(sentence, str)