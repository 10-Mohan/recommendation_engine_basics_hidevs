"""
Similarity calculation module for the recommendation engine.

Provides methods for computing similarity between users based on their ratings
and similarity between items based on their tags.
"""

import math


class SimilarityCalculator:
    """
    A class for computing various similarity measures used in recommendations.
    """

    @staticmethod
    def cosine_similarity(vec1: dict, vec2: dict) -> float:
        """
        Compute cosine similarity between two rating vectors.

        Cosine similarity measures the cosine of the angle between two vectors.
        It is commonly used to compare user rating patterns, where a higher
        similarity indicates that two users have rated items in similar ways.

        Formula: cos(θ) = (A · B) / (||A|| * ||B||)
        
        Where:
        - A · B is the dot product (sum of element-wise products)
        - ||A|| is the magnitude (Euclidean norm) of vector A
        - ||B|| is the magnitude of vector B

        Args:
            vec1 (dict): First vector as a dictionary (e.g., {item_id: rating})
            vec2 (dict): Second vector as a dictionary (e.g., {item_id: rating})

        Returns:
            float: Cosine similarity in range [0, 1], or 0 if either vector is zero

        Examples:
            >>> calc = SimilarityCalculator()
            >>> v1 = {"item1": 5, "item2": 3}
            >>> v2 = {"item1": 5, "item2": 3}
            >>> calc.cosine_similarity(v1, v2)
            1.0
            
            >>> v3 = {"item1": 0, "item2": 0}
            >>> calc.cosine_similarity(v1, v3)
            0.0
        """
        # Find all unique keys across both vectors
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        # Compute dot product (A · B)
        dot_product = 0.0
        for key in all_keys:
            # Treat missing keys as 0 (standard practice for sparse vectors)
            val1 = vec1.get(key, 0)
            val2 = vec2.get(key, 0)
            dot_product += val1 * val2
        
        # Compute magnitude of vec1 (||A||)
        mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        
        # Compute magnitude of vec2 (||B||)
        mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        # Handle zero-vector edge case to avoid division by zero
        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0
        
        # Return the cosine similarity
        return dot_product / (mag1 * mag2)

    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """
        Compute Jaccard similarity between two sets.

        Jaccard similarity (also known as Jaccard index) measures the overlap
        between two sets as a fraction of their union. It is commonly used to
        compare item characteristics (tags, genres, etc.).

        Formula: J(A, B) = |A ∩ B| / |A ∪ B|

        Where:
        - A ∩ B is the intersection (common elements)
        - A ∪ B is the union (all unique elements)

        Args:
            set1 (set): First set (e.g., tags of item A)
            set2 (set): Second set (e.g., tags of item B)

        Returns:
            float: Jaccard similarity in range [0, 1]
                   Returns 1.0 if both sets are empty (they are identical/equivalent)

        Note:
            When both sets are empty, we return 1.0 because two empty sets are
            considered identical. This makes sense in the context of recommendations:
            if two items have no tags, they are equally (lack of) information.
        """
        # Compute intersection (common elements)
        intersection = set1 & set2
        
        # Compute union (all unique elements)
        union = set1 | set2
        
        # Handle the case where both sets are empty
        # Two empty sets are considered identical, so similarity = 1.0
        if len(union) == 0:
            return 1.0
        
        # Return the Jaccard similarity
        return len(intersection) / len(union)


if __name__ == "__main__":
    """
    Demonstration of similarity calculations using sample data.
    """
    from sample_data import user_ratings, item_tags
    
    calc = SimilarityCalculator()
    
    print("=" * 60)
    print("COSINE SIMILARITY - User Rating Patterns")
    print("=" * 60)
    
    # Compare Alice and Bob (both like item1, limited overlap)
    alice_ratings = user_ratings["alice"]
    bob_ratings = user_ratings["bob"]
    similarity_ab = calc.cosine_similarity(alice_ratings, bob_ratings)
    print(f"Alice vs Bob: {similarity_ab:.4f}")
    
    # Compare Alice and Carol (different preferences)
    carol_ratings = user_ratings["carol"]
    similarity_ac = calc.cosine_similarity(alice_ratings, carol_ratings)
    print(f"Alice vs Carol: {similarity_ac:.4f}")
    
    # Compare Bob and Carol
    similarity_bc = calc.cosine_similarity(bob_ratings, carol_ratings)
    print(f"Bob vs Carol: {similarity_bc:.4f}")
    
    print("\n" + "=" * 60)
    print("JACCARD SIMILARITY - Item Tag Overlap")
    print("=" * 60)
    
    # Compare item1 and item4 (both have action tag)
    item1_tags = item_tags["item1"]
    item4_tags = item_tags["item4"]
    similarity_14 = calc.jaccard_similarity(item1_tags, item4_tags)
    print(f"item1 vs item4: {similarity_14:.4f}")
    
    # Compare item1 and item2 (no overlapping tags)
    item2_tags = item_tags["item2"]
    similarity_12 = calc.jaccard_similarity(item1_tags, item2_tags)
    print(f"item1 vs item2: {similarity_12:.4f}")
    
    # Compare item3 and item4 (item3 has comedy, item4 has comedy + action)
    item3_tags = item_tags["item3"]
    similarity_34 = calc.jaccard_similarity(item3_tags, item4_tags)
    print(f"item3 vs item4: {similarity_34:.4f}")
    
    print("\n" + "=" * 60)
