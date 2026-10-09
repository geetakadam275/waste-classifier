"""Map a predicted class to recyclable status and disposal guidance."""
import json

from config import DISPOSAL_GUIDE_PATH


def load_guide(path=DISPOSAL_GUIDE_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_guidance(label: str, guide: dict | None = None) -> dict:
    """Return {'recyclable': bool, 'bin': str, 'tip': str} for a class label."""
    guide = guide or load_guide()
    default = {
        "recyclable": False,
        "bin": "General waste",
        "tip": "Unknown item. Check your local waste guidelines.",
    }
    return guide.get(label, default)
