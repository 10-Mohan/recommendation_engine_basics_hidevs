"""
Main entry point for the recommendation engine.

Orchestrates the recommendation pipeline: candidate generation,
scoring, and evaluation for all users.
"""

from similarity import SimilarityCalculator
from candidate_generator import CandidateGenerator
from scorer import Scorer
from evaluator import Evaluator
from sample_data import user_ratings, item_tags


def recommend_for_user(
    target_user: str,
    user_ratings: dict,
    item_tags: dict,
    top_n_neighbors: int = 2,
    top_k: int = 3
) -> list:
    """
    Generate and score recommendations for a target user.

    Runs the full recommendation pipeline end-to-end:
    1. Find similar users and generate candidate items (CandidateGenerator)
    2. Score candidates based on how similar users rated them (Scorer)
    3. Return the top K picks

    Args:
        target_user (str): The user ID to generate recommendations for
        user_ratings (dict): Dictionary mapping user_id -> {item_id -> rating}
        item_tags (dict): Dictionary mapping item_id -> set of tags
        top_n_neighbors (int): Number of similar users to consider. Default is 2.
        top_k (int): Number of top recommendations to return. Default is 3.

    Returns:
        list: List of (item_id, score) tuples in descending order by score.
              Returns empty list if no candidates or recommendations found.

    Pipeline:
        - Stage 1: Generate candidates from similar users' highly-rated items
        - Stage 2: Score each candidate based on similarity-weighted average
        - Stage 3: Return top-k ranked items
    """
    # Stage 1: Generate candidates using collaborative filtering
    gen = CandidateGenerator()
    candidates = gen.generate_candidates(
        target_user,
        user_ratings,
        item_tags,
        top_n_neighbors=top_n_neighbors
    )
    
    # If no candidates found, return empty list
    if not candidates:
        return []
    
    # Stage 2: Score candidates based on similar users' ratings
    scorer = Scorer()
    scored = scorer.score_candidates(
        target_user,
        candidates,
        user_ratings,
        item_tags,
        top_n_neighbors=top_n_neighbors
    )
    
    # Stage 3: Return top-k recommendations
    recommendations = scorer.get_top_n(scored, n=top_k)
    
    return recommendations


def main():
    """
    Main execution: run recommendations for all users and evaluate.
    """
    print("=" * 80)
    print("SIMPLE RECOMMENDATION ENGINE - FULL PIPELINE")
    print("=" * 80)
    print()
    
    # ===== Part 1: Generate recommendations for all users =====
    print("PART 1: GENERATING RECOMMENDATIONS FOR ALL USERS")
    print("-" * 80)
    print()
    
    all_recommendations = {}  # Store for later evaluation
    
    for user_id in user_ratings.keys():
        # Generate recommendations for this user
        recommendations = recommend_for_user(
            user_id,
            user_ratings,
            item_tags,
            top_n_neighbors=2,
            top_k=3
        )
        
        all_recommendations[user_id] = recommendations
        
        # Print a readable report for this user
        print(f"=== Recommendations for {user_id} ===")
        print(f"Current ratings: {user_ratings[user_id]}")
        print()
        
        if recommendations:
            print("Top recommendations:")
            for rank, (item_id, score) in enumerate(recommendations, 1):
                print(f"  {rank}. {item_id} (score: {score:.2f})")
        else:
            print("No new recommendations — this user has rated everything,")
            print("or no similar users found.")
        
        print()
    
    # ===== Part 2: Evaluate recommendations for one user =====
    print()
    print("=" * 80)
    print("PART 2: EVALUATION - Quality Metrics for Sample User")
    print("-" * 80)
    print()
    
    # Pick "alice" for detailed evaluation
    eval_user = "alice"
    eval_recs = all_recommendations[eval_user]
    
    # Define ground truth: items we consider relevant for alice
    # (In practice, this would come from actual user behavior data)
    relevant_items = {"item3"}  # item3 is highly relevant to alice
    
    print(f"Evaluating recommendations for: {eval_user}")
    print(f"Recommendations: {eval_recs}")
    print(f"Ground truth (relevant items): {relevant_items}")
    print()
    
    # Run evaluation
    evaluator = Evaluator()
    eval_result = evaluator.evaluate(eval_recs, relevant_items, k=3)
    
    print("Evaluation Results:")
    print("-" * 80)
    for key, value in eval_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Interpret the results
    print("Interpretation:")
    print("-" * 80)
    precision = eval_result["precision_at_k"]
    found = eval_result["num_relevant_found"]
    total_relevant = eval_result["num_relevant_total"]
    
    if found == 0:
        print(f"  ❌ No relevant items found in recommendations.")
    elif found == total_relevant:
        print(f"  ✓ All relevant items were found in the recommendations!")
    else:
        print(f"  ◐ Found {found}/{total_relevant} relevant items.")
    
    if precision > 0.7:
        print(f"  → High precision ({precision:.2%}): most recommendations are relevant.")
    elif precision > 0.4:
        print(f"  → Medium precision ({precision:.2%}): some recommendations are relevant.")
    else:
        print(f"  → Low precision ({precision:.2%}): few recommendations are relevant.")
    
    print()
    print("=" * 80)
    print("END OF PIPELINE")
    print("=" * 80)


if __name__ == "__main__":
    main()
