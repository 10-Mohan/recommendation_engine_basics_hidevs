"""
Scorer module for the recommendation engine.

Responsible for scoring and ranking candidate items
based on similarity-weighted averaging of neighbor ratings.
"""

from similarity import SimilarityCalculator


class Scorer:
    """
    Scores and ranks candidate items for recommendations.
    
    Uses similarity-weighted averaging: for each candidate item, compute a score
    based on how similar users rated that item. The score is a weighted average
    where the weights are the similarity scores between the target user and
    their similar neighbors.
    """

    def __init__(self):
        """Initialize the scorer with a similarity calculator."""
        self.similarity_calc = SimilarityCalculator()

    def score_candidates(
        self,
        target_user: str,
        candidates: set,
        user_ratings: dict,
        item_tags: dict,
        top_n_neighbors: int = 2
    ) -> list:
        """
        Score candidate items using similarity-weighted averaging.

        For each candidate item, compute a score based on how similar users
        rated that item. The score is a weighted average where weights are
        the similarity scores between the target user and their neighbors.

        Scoring Formula:
            score(item) = sum(similarity(target_user, neighbor) * neighbor_rating[item])
                         / sum(similarity(target_user, neighbor))

        Calculation Details:
        - Find the top N most similar users to the target user
        - For each candidate item, only consider neighbors who rated it (rating > 0)
        - Multiply each neighbor's rating by their similarity score to the target user
        - Sum these weighted ratings and divide by the sum of similarities
        - If no neighbor rated the item, the score is 0.0

        Args:
            target_user (str): The user ID to score recommendations for
            candidates (set): Set of item IDs to score
            user_ratings (dict): Dictionary mapping user_id -> {item_id -> rating}
            item_tags (dict): Dictionary mapping item_id -> set of tags (included for API,
                            not used in this weighted-average scoring method)
            top_n_neighbors (int): Number of similar users to consider. Default is 2.

        Returns:
            list: List of (item_id, score) tuples sorted by score in descending order
                  (highest scores first). Returns empty list if candidates is empty or
                  target_user not found.

        Edge Cases:
            - Empty candidates set: returns empty list
            - target_user not in user_ratings: returns empty list
            - No neighbors rated an item: score is 0.0 for that item

        Example:
            >>> scorer = Scorer()
            >>> candidates = {'item3', 'item4'}
            >>> scored = scorer.score_candidates("alice", candidates, user_ratings, item_tags)
            >>> print(scored)  # [('item3', 4.5), ('item4', 2.1)]
        """
        # Validate inputs
        if not candidates or not user_ratings:
            return []
        
        if target_user not in user_ratings:
            return []
        
        # Get target user's rating vector
        target_ratings = user_ratings[target_user]
        
        # Find similar users (same logic as CandidateGenerator)
        similarities = {}
        for other_user, other_ratings in user_ratings.items():
            if other_user == target_user:
                continue
            
            # Calculate cosine similarity between rating vectors
            similarity_score = self.similarity_calc.cosine_similarity(
                target_ratings, other_ratings
            )
            similarities[other_user] = similarity_score
        
        # If no other users exist, return empty list
        if not similarities:
            return []
        
        # Sort by similarity (descending) and take top N
        sorted_neighbors = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_neighbors = sorted_neighbors[:top_n_neighbors]
        top_neighbor_ids = [neighbor[0] for neighbor in top_neighbors]
        
        # Score each candidate item using weighted average
        scored_items = []
        
        for item_id in candidates:
            # Accumulate weighted sum and similarity sum for this item
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            # Iterate through similar neighbors
            for neighbor_id in top_neighbor_ids:
                neighbor_ratings = user_ratings[neighbor_id]
                neighbor_rating = neighbor_ratings.get(item_id, 0)
                
                # Only include neighbors who rated this item (rating > 0)
                if neighbor_rating > 0:
                    neighbor_similarity = similarities[neighbor_id]
                    # Weight the rating by the similarity score
                    weighted_sum += neighbor_similarity * neighbor_rating
                    similarity_sum += neighbor_similarity
            
            # Compute the final score
            # If no neighbor rated this item, score is 0.0
            if similarity_sum > 0:
                score = weighted_sum / similarity_sum
            else:
                score = 0.0
            
            scored_items.append((item_id, score))
        
        # Sort by score in descending order (highest scores first)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        return scored_items

    def get_top_n(
        self,
        scored_items: list,
        n: int = 3
    ) -> list:
        """
        Extract the top N items from a scored list.

        Args:
            scored_items (list): List of (item_id, score) tuples, assumed to be
                               pre-sorted by score in descending order
            n (int): Number of top items to return. Default is 3.

        Returns:
            list: Top n items from the list. If the list has fewer than n items,
                  returns all items. If the list is empty, returns an empty list.

        Edge Cases:
            - If len(scored_items) < n: returns all items (no error)
            - If scored_items is empty: returns empty list
            - If n <= 0: returns empty list (Python slicing behavior)

        Example:
            >>> scorer = Scorer()
            >>> scored = [('item1', 5.0), ('item2', 3.0), ('item3', 1.0)]
            >>> top_2 = scorer.get_top_n(scored, n=2)
            >>> print(top_2)  # [('item1', 5.0), ('item2', 3.0)]
        """
        return scored_items[:n]


if __name__ == "__main__":
    """
    Demonstration of candidate scoring and ranking.
    """
    from sample_data import user_ratings, item_tags
    from candidate_generator import CandidateGenerator
    
    scorer = Scorer()
    gen = CandidateGenerator()
    
    print("=" * 70)
    print("SCORER - Candidate Ranking Demo")
    print("=" * 70)
    print()
    
    # Generate candidates for Alice
    target_user = "alice"
    print(f"Target user: {target_user}")
    print(f"Target user ratings: {user_ratings[target_user]}")
    print()
    
    candidates = gen.generate_candidates(
        target_user,
        user_ratings,
        item_tags,
        top_n_neighbors=2
    )
    print(f"Generated candidates: {candidates}")
    print()
    
    # Show the similar neighbors and their ratings for transparency
    print("Similar neighbors and their ratings for candidate items:")
    print("-" * 70)
    similarities = {}
    for other_user, other_ratings in user_ratings.items():
        if other_user != target_user:
            sim = scorer.similarity_calc.cosine_similarity(
                user_ratings[target_user], other_ratings
            )
            similarities[other_user] = sim
    
    sorted_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for neighbor_id, sim_score in sorted_neighbors[:2]:
        print(f"  {neighbor_id} (similarity={sim_score:.4f}):")
        neighbor_ratings = user_ratings[neighbor_id]
        for item_id in candidates:
            rating = neighbor_ratings.get(item_id, 0)
            if rating > 0:
                print(f"    - {item_id}: {rating}")
    print()
    
    # Score the candidates
    scored = scorer.score_candidates(
        target_user,
        candidates,
        user_ratings,
        item_tags,
        top_n_neighbors=2
    )
    
    print("Scored candidates (sorted by score, highest first):")
    print("-" * 70)
    for item_id, score in scored:
        print(f"  {item_id}: {score:.4f}")
    print()
    
    # Get top 2 recommendations
    top_2 = scorer.get_top_n(scored, n=2)
    print(f"Top 2 recommendations for {target_user}:")
    print("-" * 70)
    for item_id, score in top_2:
        print(f"  {item_id}: {score:.4f}")
    print()
