"""
Candidate generator module for the recommendation engine.

Responsible for generating a set of candidate items to recommend
based on various strategies (collaborative filtering, content-based, etc.).
"""

from similarity import SimilarityCalculator


class CandidateGenerator:
    """
    Generates candidate recommendations using collaborative filtering.
    
    The approach:
    1. Find users similar to the target user based on their rating patterns
    2. Look at items highly rated by those similar users
    3. Return items the target user hasn't yet rated
    """

    def __init__(self):
        """Initialize the candidate generator with a similarity calculator."""
        self.similarity_calc = SimilarityCalculator()

    def generate_candidates(
        self,
        target_user: str,
        user_ratings: dict,
        item_tags: dict,
        top_n_neighbors: int = 2
    ) -> set:
        """
        Generate candidate items for a user using collaborative filtering.

        This method uses a two-stage approach:
        
        Stage 1 - Find Similar Users:
            Compare the target user's rating vector against all other users
            using cosine similarity. Select the top N most similar users.
        
        Stage 2 - Extract Candidates:
            Collect items that similar users rated highly (rating >= 4) but
            that the target user has not yet rated (rating == 0 or missing).

        Args:
            target_user (str): The ID of the user to generate recommendations for
            user_ratings (dict): Dictionary mapping user_id -> {item_id -> rating}
            item_tags (dict): Dictionary mapping item_id -> set of tags (unused in this
                            basic collaborative filtering, included for API completeness)
            top_n_neighbors (int): Number of similar users to consider. Default is 2.

        Returns:
            set: A set of item IDs that are candidates for recommendation
                 Returns empty set if target_user not found, has no unrated items,
                 or if user_ratings is empty.

        Edge Cases:
            - If target_user not in user_ratings: returns empty set
            - If fewer than top_n_neighbors other users exist: uses all available users
            - If target_user has rated all items: returns empty set
            - If user_ratings or item_tags is empty: returns empty set

        Example:
            >>> gen = CandidateGenerator()
            >>> candidates = gen.generate_candidates("alice", user_ratings, item_tags, top_n_neighbors=2)
            >>> print(candidates)  # e.g., {'item3', 'item4'}
        """
        # Validate inputs: return empty set if data is empty or target user not found
        if not user_ratings or not item_tags:
            return set()
        
        if target_user not in user_ratings:
            return set()
        
        # Get target user's rating vector
        target_ratings = user_ratings[target_user]
        
        # ===== STAGE 1: Find Similar Users =====
        # Calculate similarity between target user and all other users
        similarities = {}
        for other_user, other_ratings in user_ratings.items():
            # Skip the target user themselves
            if other_user == target_user:
                continue
            
            # Compute cosine similarity between rating vectors
            similarity_score = self.similarity_calc.cosine_similarity(
                target_ratings, other_ratings
            )
            similarities[other_user] = similarity_score
        
        # If no other users exist, return empty set
        if not similarities:
            return set()
        
        # Sort by similarity (descending) and take top N
        # We sort in descending order so the most similar users come first
        sorted_neighbors = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_neighbors = sorted_neighbors[:top_n_neighbors]
        top_neighbor_ids = [neighbor[0] for neighbor in top_neighbors]
        
        # ===== STAGE 2: Generate Candidates =====
        # Collect items highly rated by similar users that target user hasn't rated
        candidates = set()
        
        for neighbor_id in top_neighbor_ids:
            neighbor_ratings = user_ratings[neighbor_id]
            
            # Look at each item the neighbor has rated
            for item_id, rating in neighbor_ratings.items():
                # Only consider items the neighbor rated highly (>= 4)
                if rating >= 4:
                    # Only add to candidates if target user hasn't rated it
                    # (rating == 0 or item not in target user's ratings)
                    target_rating = target_ratings.get(item_id, 0)
                    if target_rating == 0:
                        candidates.add(item_id)
        
        return candidates


if __name__ == "__main__":
    """
    Demonstration of candidate generation for collaborative filtering.
    """
    from sample_data import user_ratings, item_tags
    
    gen = CandidateGenerator()
    
    print("=" * 70)
    print("CANDIDATE GENERATOR - Collaborative Filtering Demo")
    print("=" * 70)
    print()
    
    # Generate candidates for Alice
    print("Generating candidates for: alice")
    print("-" * 70)
    print(f"Alice's current ratings: {user_ratings['alice']}")
    print()
    
    candidates_alice = gen.generate_candidates(
        "alice",
        user_ratings,
        item_tags,
        top_n_neighbors=2
    )
    
    print(f"Top 2 similar users to Alice and their high-rated items:")
    
    # Show the similarity scores and high-rated items for transparency
    similarities = {}
    for other_user, other_ratings in user_ratings.items():
        if other_user != "alice":
            sim = gen.similarity_calc.cosine_similarity(
                user_ratings["alice"], other_ratings
            )
            similarities[other_user] = sim
    
    sorted_neighbors = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for neighbor_id, sim_score in sorted_neighbors[:2]:
        high_rated = {
            item: rating
            for item, rating in user_ratings[neighbor_id].items()
            if rating >= 4
        }
        print(f"  {neighbor_id}: similarity={sim_score:.4f}, high-rated={high_rated}")
    
    print()
    print(f"Candidate items for Alice: {candidates_alice}")
    print()
    
    # Generate candidates for Bob
    print("=" * 70)
    print("Generating candidates for: bob")
    print("-" * 70)
    print(f"Bob's current ratings: {user_ratings['bob']}")
    print()
    
    candidates_bob = gen.generate_candidates(
        "bob",
        user_ratings,
        item_tags,
        top_n_neighbors=2
    )
    
    print(f"Candidate items for Bob: {candidates_bob}")
    print()
    
    # Generate candidates for Carol
    print("=" * 70)
    print("Generating candidates for: carol")
    print("-" * 70)
    print(f"Carol's current ratings: {user_ratings['carol']}")
    print()
    
    candidates_carol = gen.generate_candidates(
        "carol",
        user_ratings,
        item_tags,
        top_n_neighbors=2
    )
    
    print(f"Candidate items for Carol: {candidates_carol}")
    print()
    
    # Test edge case: non-existent user
    print("=" * 70)
    print("Testing edge case: non-existent user")
    print("-" * 70)
    candidates_unknown = gen.generate_candidates(
        "unknown_user",
        user_ratings,
        item_tags,
        top_n_neighbors=2
    )
    print(f"Candidates for 'unknown_user': {candidates_unknown} (should be empty)")
    print()
