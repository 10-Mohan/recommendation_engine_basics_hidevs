"""
Evaluator module for the recommendation engine.

Responsible for evaluating the quality of recommendations
using various metrics (precision, recall, NDCG, etc.).
"""


class Evaluator:
    """
    Evaluates the quality of recommendations using standard metrics.
    """

    @staticmethod
    def precision_at_k(
        recommended_items: list,
        relevant_items: set,
        k: int = None
    ) -> float:
        """
        Compute Precision@K, a standard recommender-systems evaluation metric.

        Precision@K measures the fraction of the top-k recommendations that are
        relevant (in the ground-truth set). It answers the question: "Of the top
        k items recommended, how many are actually relevant to the user?"

        Formula:
            precision@k = (number of top-k recommendations that are relevant) / k

        Key Characteristics:
        - Range: [0.0, 1.0] where 1.0 means all top-k items are relevant
        - Lower values indicate more irrelevant items in the top-k
        - Useful for evaluating recommendation quality at different cutoff points

        Concrete Example:
            recommended = ['item1', 'item2', 'item3', 'item4']
            relevant = {'item1', 'item3', 'item5'}  # ground truth
            precision_at_k(recommended, relevant, k=2) = 1/2 = 0.5
              (only 'item1' is relevant in the top-2)
            precision_at_k(recommended, relevant, k=4) = 2/4 = 0.5
              ('item1' and 'item3' are relevant in the top-4)

        Args:
            recommended_items (list): List of recommended item IDs, OR
                                     list of (item_id, score) tuples.
                                     The method auto-detects and extracts item names.
            relevant_items (set): Set of item IDs considered relevant (ground truth)
            k (int): The cutoff point for top-k evaluation. If None, use the full
                    length of recommended_items. Default is None.

        Returns:
            float: Precision@K score in range [0.0, 1.0]
                   Returns 0.0 if k=0, recommended_items is empty,
                   or no recommendations are relevant.

        Edge Cases:
            - If k is None: use len(recommended_items) as k
            - If k = 0: return 0.0 (avoid division by zero)
            - If recommended_items is empty: return 0.0
            - If relevant_items is empty: return 0.0 (nothing is relevant)
            - If k > len(recommended_items): use available items but divide by
              the requested k. This is standard precision@k behavior: if you ask
              for top-5 but only get 3 items, you still divide by 5, penalizing
              the inability to provide k recommendations.
            - Ties in recommendations: stable order is preserved from input

        Example:
            >>> evaluator = Evaluator()
            >>> recs = [('item1', 0.9), ('item2', 0.7), ('item3', 0.5)]
            >>> relevant = {'item1', 'item3'}
            >>> evaluator.precision_at_k(recs, relevant, k=3)
            0.6667  # 2 out of 3 are relevant
        """
        # Handle None k: use full length of recommended_items
        if k is None:
            k = len(recommended_items)
        
        # Edge case: k is 0 or no recommendations
        if k == 0 or not recommended_items:
            return 0.0
        
        # Edge case: no relevant items exist
        if not relevant_items:
            return 0.0
        
        # Extract item names from recommended_items
        # Handle both: list of item names AND list of (item_name, score) tuples
        top_k_items = []
        for item in recommended_items[:k]:
            # Check if item is a tuple (item_name, score)
            if isinstance(item, tuple):
                top_k_items.append(item[0])
            else:
                # Assume it's just an item name
                top_k_items.append(item)
        
        # Count how many of the top-k items are in the relevant set
        num_relevant_found = sum(1 for item in top_k_items if item in relevant_items)
        
        # Compute precision@k
        # Note: We divide by k (the requested cutoff), not len(top_k_items).
        # This penalizes cases where we have fewer than k recommendations available.
        precision = num_relevant_found / k
        
        return precision

    @staticmethod
    def evaluate(
        recommended_items: list,
        relevant_items: set,
        k: int = None
    ) -> dict:
        """
        Evaluate recommendations and return multiple metrics in one call.

        Args:
            recommended_items (list): List of recommended item IDs (or tuples)
            relevant_items (set): Set of relevant item IDs (ground truth)
            k (int): Cutoff point for top-k evaluation. If None, uses full list.

        Returns:
            dict: Dictionary containing:
                - "precision_at_k" (float): Precision@K score [0.0, 1.0]
                - "k" (int): The actual k value used
                - "num_relevant_found" (int): Count of relevant items in top-k
                - "num_recommended" (int): Total items recommended
                - "num_relevant_total" (int): Total relevant items in ground truth

        Example:
            >>> evaluator = Evaluator()
            >>> recs = ['item1', 'item2', 'item3']
            >>> relevant = {'item1', 'item3'}
            >>> result = evaluator.evaluate(recs, relevant, k=3)
            >>> print(result)
            {'precision_at_k': 0.6667, 'k': 3, 'num_relevant_found': 2,
             'num_recommended': 3, 'num_relevant_total': 2}
        """
        # Determine k value (default to full length if None)
        actual_k = k if k is not None else len(recommended_items)
        
        # Extract item names from recommended_items (same logic as precision_at_k)
        top_k_items = []
        for item in recommended_items[:actual_k]:
            if isinstance(item, tuple):
                top_k_items.append(item[0])
            else:
                top_k_items.append(item)
        
        # Count relevant items in top-k
        num_relevant_found = sum(1 for item in top_k_items if item in relevant_items)
        
        # Compute precision
        precision = Evaluator.precision_at_k(
            recommended_items, relevant_items, k=k
        )
        
        return {
            "precision_at_k": precision,
            "k": actual_k,
            "num_relevant_found": num_relevant_found,
            "num_recommended": len(recommended_items),
            "num_relevant_total": len(relevant_items),
        }


if __name__ == "__main__":
    """
    Demonstration of recommendation evaluation.
    """
    from sample_data import user_ratings, item_tags
    from candidate_generator import CandidateGenerator
    from scorer import Scorer
    
    gen = CandidateGenerator()
    scorer = Scorer()
    evaluator = Evaluator()
    
    print("=" * 70)
    print("EVALUATOR - Recommendation Quality Demo")
    print("=" * 70)
    print()
    
    # Run the full recommendation pipeline for "alice"
    target_user = "alice"
    print(f"Target user: {target_user}")
    print(f"User ratings: {user_ratings[target_user]}")
    print()
    
    # Step 1: Generate candidates
    candidates = gen.generate_candidates(
        target_user,
        user_ratings,
        item_tags,
        top_n_neighbors=2
    )
    print(f"Step 1 - Generated candidates: {candidates}")
    print()
    
    # Step 2: Score candidates
    scored = scorer.score_candidates(
        target_user,
        candidates,
        user_ratings,
        item_tags,
        top_n_neighbors=2
    )
    print("Step 2 - Scored candidates:")
    for item_id, score in scored:
        print(f"  {item_id}: {score:.4f}")
    print()
    
    # Step 3: Get top 3 recommendations
    top_3_recs = scorer.get_top_n(scored, n=3)
    print(f"Step 3 - Top 3 recommendations: {top_3_recs}")
    print()
    
    # Step 4: Define ground truth and evaluate
    # For Alice, we define that item3 is relevant to her preferences
    relevant_for_alice = {"item3"}
    print(f"Ground truth (relevant items for {target_user}): {relevant_for_alice}")
    print()
    
    # Evaluate the recommendations
    evaluation = evaluator.evaluate(top_3_recs, relevant_for_alice, k=3)
    
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    for key, value in evaluation.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Show interpretation
    print("Interpretation:")
    print("-" * 70)
    if evaluation["num_relevant_found"] == 0:
        print("  No relevant items found in the top recommendations. ❌")
    elif evaluation["num_relevant_found"] == evaluation["num_relevant_total"]:
        print("  All relevant items were found in the recommendations! ✓")
    else:
        print(f"  Found {evaluation['num_relevant_found']} out of "
              f"{evaluation['num_relevant_total']} relevant items.")
    print()
