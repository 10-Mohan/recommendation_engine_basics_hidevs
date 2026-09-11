"""
Unit tests for the similarity module.
"""

import pytest
from similarity import SimilarityCalculator


def test_cosine_similarity_identical_vectors():
    """Test cosine similarity between identical vectors returns 1.0."""
    calc = SimilarityCalculator()
    v1 = {"item1": 5, "item2": 3, "item3": 4}
    v2 = {"item1": 5, "item2": 3, "item3": 4}
    
    result = calc.cosine_similarity(v1, v2)
    assert result == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    """Test cosine similarity between orthogonal (completely disjoint) vectors returns 0.0."""
    calc = SimilarityCalculator()
    v1 = {"item1": 5, "item2": 3}
    v2 = {"item3": 4, "item4": 2}
    
    result = calc.cosine_similarity(v1, v2)
    assert result == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    """Test cosine similarity with zero vectors returns 0.0 without division by zero."""
    calc = SimilarityCalculator()
    v_nonzero = {"item1": 5, "item2": 3}
    v_zero = {"item1": 0, "item2": 0}
    v_empty = {}
    
    assert calc.cosine_similarity(v_nonzero, v_zero) == 0.0
    assert calc.cosine_similarity(v_zero, v_nonzero) == 0.0
    assert calc.cosine_similarity(v_zero, v_zero) == 0.0
    assert calc.cosine_similarity(v_nonzero, v_empty) == 0.0


def test_jaccard_similarity_identical_sets():
    """Test Jaccard similarity between identical non-empty sets returns 1.0."""
    calc = SimilarityCalculator()
    s1 = {"action", "sci-fi", "thriller"}
    s2 = {"action", "sci-fi", "thriller"}
    
    result = calc.jaccard_similarity(s1, s2)
    assert result == pytest.approx(1.0)


def test_jaccard_similarity_disjoint_sets():
    """Test Jaccard similarity between disjoint sets returns 0.0."""
    calc = SimilarityCalculator()
    s1 = {"action", "sci-fi"}
    s2 = {"romance", "drama"}
    
    result = calc.jaccard_similarity(s1, s2)
    assert result == pytest.approx(0.0)


def test_jaccard_similarity_both_empty():
    """Test Jaccard similarity when both sets are empty returns 1.0."""
    calc = SimilarityCalculator()
    s1 = set()
    s2 = set()
    
    result = calc.jaccard_similarity(s1, s2)
    assert result == 1.0
