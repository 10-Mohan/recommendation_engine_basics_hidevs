"""
Unit tests for the scorer module.
"""

from scorer import Scorer


def test_score_candidates_sorted_results():
    """Test score_candidates returns results sorted by score in descending order."""
    scorer = Scorer()
    user_ratings = {
        "target": {"item1": 5},
        "neighbor": {"item1": 5, "item2": 5, "item3": 2},
    }
    item_tags = {
        "item1": {"action"},
        "item2": {"comedy"},
        "item3": {"drama"},
    }
    candidates = {"item2", "item3"}
    
    results = scorer.score_candidates("target", candidates, user_ratings, item_tags)
    
    assert len(results) == 2
    # Ensure scores are sorted in descending order
    assert results[0][1] >= results[1][1]
    assert results[0][0] == "item2"
    assert results[1][0] == "item3"


def test_get_top_n_respects_n():
    """Test get_top_n returns at most n items from scored items."""
    scorer = Scorer()
    scored_items = [
        ("item1", 4.8),
        ("item2", 4.2),
        ("item3", 3.5),
        ("item4", 2.1),
    ]
    
    top_2 = scorer.get_top_n(scored_items, n=2)
    assert top_2 == [("item1", 4.8), ("item2", 4.2)]
    assert len(top_2) == 2


def test_get_top_n_empty_input():
    """Test get_top_n handles empty scored items input gracefully."""
    scorer = Scorer()
    
    assert scorer.get_top_n([], n=3) == []
    assert scorer.get_top_n([], n=0) == []
