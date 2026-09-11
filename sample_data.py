"""
Sample data for the recommendation engine.
Loads user ratings and item tag information from data/sample_data.json.
"""

import json
from pathlib import Path

# Path to the sample data JSON file
_DATA_FILE = Path(__file__).parent / "data" / "sample_data.json"

with open(_DATA_FILE, "r", encoding="utf-8") as _f:
    _data = json.load(_f)

# Dictionary mapping user IDs to their ratings for items.
# Each user has a dict of item_id -> rating (0-5 scale)
user_ratings = _data["user_ratings"]

# Dictionary mapping item IDs to sets of tags/genres.
item_tags = {item_id: set(tags) for item_id, tags in _data["item_tags"].items()}
