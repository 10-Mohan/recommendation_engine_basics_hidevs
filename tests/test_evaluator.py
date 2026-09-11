"""
Unit tests for the evaluator module.
"""

from evaluator import Evaluator


def test_precision_at_k_perfect_match():
    """Test precision_at_k returns 1.0 when all recommended items are relevant."""
    evaluator = Evaluator()
    recommended_items = ["item1", "item2", "item3"]
    relevant_items = {"item1", "item2", "item3"}
    
    score = evaluator.precision_at_k(recommended_items, relevant_items, k=3)
    assert score == 1.0


def test_precision_at_k_no_match():
    """Test precision_at_k returns 0.0 when none of the recommended items are relevant."""
    evaluator = Evaluator()
    recommended_items = ["item1", "item2"]
    relevant_items = {"item3", "item4"}
    
    score = evaluator.precision_at_k(recommended_items, relevant_items, k=2)
    assert score == 0.0


def test_precision_at_k_empty_inputs():
    """Test precision_at_k returns 0.0 for various empty input edge cases."""
    evaluator = Evaluator()
    
    # Empty recommended items
    assert evaluator.precision_at_k([], {"item1", "item2"}) == 0.0
    
    # Empty relevant items
    assert evaluator.precision_at_k(["item1", "item2"], set()) == 0.0
    
    # Both empty
    assert evaluator.precision_at_k([], set()) == 0.0
    
    # k = 0
    assert evaluator.precision_at_k(["item1"], {"item1"}, k=0) == 0.0
