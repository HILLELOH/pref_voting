
import pytest
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies


@pytest.fixture(scope="session", autouse=True)
def preload_sentence_model():
    """Load the sentence-transformer model once for the entire test session.

    Without this, the first test that calls embed_text pays the ~15s model load.
    With session scope + autouse, the model is warm before any test runs.
    """
    from pref_voting.coalition_formation import _get_st_model
    _get_st_model()

@pytest.fixture
def condorcet_cycle():
    return Profile([
        [0, 1, 2], 
        [1, 2, 0], 
        [2, 0, 1]])

@pytest.fixture
def linear_profile_0():
    return Profile([
        [0, 1, 2], 
        [2, 1, 0]], 
        rcounts=[2, 1])

@pytest.fixture
def profile_with_ties_linear_0():
    return ProfileWithTies([
        {0:1, 1:2, 2:3}, 
        {0:3, 1:2, 2:1}],
        rcounts=[2, 1])

@pytest.fixture
def profile_with_ties():
    return ProfileWithTies([
        {0:1, 1:1, 2:2}, 
        {0:2, 1:2, 2:1}],
        rcounts=[2, 1])

@pytest.fixture
def profile_single_voter():
    return Profile([[0, 1, 2, 3]])