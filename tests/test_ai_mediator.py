import pytest
import random
from pref_voting.ai_mediator import find_ai_compromise

def test_empty_input():
    # Edge Case: Empty agents list
    assert find_ai_compromise([], [0.0, 0.0]) == [0.0, 0.0]

def test_single_agent():
    # Edge Case: Only one agent (the compromise should just be the agent's exact position)
    assert find_ai_compromise([[1.0, 1.0]], [0.0, 0.0]) == [1.0, 1.0]

def test_large_input():
    # Large Input: 1000 agents all wanting the exact same point
    agents = [[5.0, 5.0] for _ in range(1000)]
    assert find_ai_compromise(agents, [0.0, 0.0]) == [5.0, 5.0]

def test_large_random_input_property():
    # Large Random Input: Testing a property instead of an exact value
    # Property check: Does the algorithm return a point with the correct number of dimensions?
    dimensions = 5
    num_agents = 500
    
    # Generate random 5D coordinates for agents and the centroid
    agents = [[random.uniform(-10, 10) for _ in range(dimensions)] for _ in range(num_agents)]
    centroid = [random.uniform(-10, 10) for _ in range(dimensions)]
    
    result = find_ai_compromise(agents, centroid)
    
    # The result MUST have the same number of dimensions as the input space
    assert len(result) == dimensions
    # The result shouldn't just be the hardcoded [0.0, 0.0] from our empty implementation
    assert result != [0.0, 0.0]