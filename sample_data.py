"""
Sample data for the recommendation engine.
Contains user ratings and item tag information.
"""

# Dictionary mapping user IDs to their ratings for items.
# Each user has a dict of item_id -> rating (0-5 scale)
user_ratings = {
    "alice": {"item1": 5, "item2": 3, "item3": 0, "item4": 1},
    "bob":   {"item1": 4, "item2": 0, "item3": 0, "item4": 1},
    "carol": {"item1": 1, "item2": 1, "item3": 5, "item4": 4},
}

# Dictionary mapping item IDs to sets of tags/genres.
item_tags = {
    "item1": {"action", "sci-fi"},
    "item2": {"romance", "drama"},
    "item3": {"comedy"},
    "item4": {"action", "comedy"},
}
