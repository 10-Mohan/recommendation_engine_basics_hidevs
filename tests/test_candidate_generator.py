"""
Unit tests for the candidate_generator module.
"""

from candidate_generator import CandidateGenerator


def test_generate_candidates_known_user():
    """Test candidate generation for a known user returns unrated high-rated items from similar users."""
    gen = CandidateGenerator()
    user_ratings = {
        "user_a": {"item1": 5, "item2": 0, "item3": 0},
        "user_b": {"item1": 5, "item2": 4, "item3": 2},
    }
    item_tags = {
        "item1": {"action"},
        "item2": {"comedy"},
        "item3": {"drama"},
    }
    
    candidates = gen.generate_candidates("user_a", user_ratings, item_tags, top_n_neighbors=1)
    assert candidates == {"item2"}


def test_generate_candidates_unknown_user():
    """Test candidate generation for an unknown user returns an empty set."""
    gen = CandidateGenerator()
    user_ratings = {
        "user_a": {"item1": 5, "item2": 3},
        "user_b": {"item1": 4, "item2": 5},
    }
    item_tags = {
        "item1": {"action"},
        "item2": {"comedy"},
    }
    
    candidates = gen.generate_candidates("non_existent_user", user_ratings, item_tags)
    assert candidates == set()


def test_generate_candidates_user_rated_everything():
    """Test candidate generation for a user who has rated all items returns an empty set."""
    gen = CandidateGenerator()
    user_ratings = {
        "user_a": {"item1": 5, "item2": 4, "item3": 3},
        "user_b": {"item1": 4, "item2": 5, "item3": 4},
    }
    item_tags = {
        "item1": {"action"},
        "item2": {"comedy"},
        "item3": {"drama"},
    }
    
    candidates = gen.generate_candidates("user_a", user_ratings, item_tags)
    assert candidates == set()
