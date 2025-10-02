import re
from typing import Dict

def inaturalist_mc_doc_to_visual(doc):
    """Return image path."""
    # Convert to RGB to ensure compatibility
    if hasattr(doc["image"], 'convert'):
        return [doc["image"].convert("RGB")]
    return [doc["image"]]

def inaturalist_mc_doc_to_text(doc):
    """Return multiple choice question."""
    return doc["conversations"][0]["value"]

def inaturalist_mc_doc_to_target(doc):
    """Return correct letter answer."""
    return doc["conversations"][1]["value"]

def inaturalist_mc_process_results(doc, results):
    """Process model output for multiple choice accuracy."""
    pred = results[0].strip().upper()
    gold = doc["conversations"][1]["value"].strip().upper()
    
    # Extract just the letter if model outputs more
    # e.g., "(A)" -> "A", "A." -> "A", "Answer: A" -> "A"
    letter_match = re.search(r'[A-Z]', pred)
    if letter_match:
        pred = letter_match.group(0)
    
    # Simple exact match for letters
    is_correct = (pred == gold)
    
    return {"inat_mc_accuracy": float(is_correct)}
