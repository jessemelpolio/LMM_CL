import re
from typing import Dict

def eurosat_doc_to_visual(doc):
    """Return image path."""
    return doc["image"]

def eurosat_doc_to_text(doc):
    """Return land use classification prompt."""
    # Extract the text after removing <image> tag
    text = doc["conversations"][0]["value"]
    text = text.replace("<image>\n", "")
    return text

def eurosat_doc_to_target(doc):
    """Return ground truth land use class."""
    return doc["conversations"][1]["value"]

def eurosat_process_results(doc, results):
    """Process model output and compute accuracy for land use classification."""
    pred = results[0].strip()
    gold = doc["conversations"][1]["value"].strip()
    
    # Normalize answers
    pred_normalized = normalize_class_name(pred)
    gold_normalized = normalize_class_name(gold)
    
    # Check various matching strategies
    is_correct = check_class_match(pred_normalized, gold_normalized)
    
    return {"eurosat_accuracy": float(is_correct)}

def normalize_class_name(name):
    """Normalize class name for comparison."""
    # Convert to lowercase
    name = name.lower()
    
    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    # Handle camelCase by adding spaces
    # e.g., "AnnualCrop" -> "annual crop"
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name).lower()
    
    return name

def check_class_match(pred, gold):
    """Check if predicted class matches ground truth with various strategies."""
    # Exact match
    if pred == gold:
        return True
    
    # EuroSAT class names and their variations
    eurosat_variations = {
        "annualcrop": ["annual crop", "annual crops", "cropland", "farmland", "agriculture", "agricultural", "farm"],
        "forest": ["forests", "woodland", "woods", "trees", "forested", "tree cover"],
        "herbaceousvegetation": ["herbaceous vegetation", "grassland", "grass", "meadow", "vegetation", "herbs"],
        "highway": ["highways", "road", "roads", "motorway", "freeway", "street"],
        "industrial": ["industry", "industrial area", "factory", "industrial zone", "warehouse", "industrial building"],
        "pasture": ["pastures", "grazing land", "grassland", "pastoral", "grazing"],
        "permanentcrop": ["permanent crop", "permanent crops", "orchard", "vineyard", "plantation"],
        "residential": ["residential area", "houses", "housing", "suburb", "urban", "neighborhood"],
        "river": ["rivers", "water", "stream", "waterway", "watercourse"],
        "sealake": ["sea lake", "sea", "lake", "ocean", "water body", "water", "coastal"]
    }
    
    # Normalize gold for lookup
    gold_key = gold.replace(" ", "")
    
    # Check if prediction matches any variation
    if gold_key in eurosat_variations:
        variations = eurosat_variations[gold_key]
        if pred in variations:
            return True
        
        # Check partial matches
        for variation in variations:
            if variation in pred or pred in variation:
                return True
    
    # Reverse check - see if pred is a key and gold is a variation
    for key, variations in eurosat_variations.items():
        if pred.replace(" ", "") == key and gold in variations:
            return True
    
    # Check word overlap
    pred_words = set(pred.split())
    gold_words = set(gold.split())
    
    # If significant overlap
    if len(pred_words) > 0 and len(gold_words) > 0:
        overlap = len(pred_words & gold_words)
        min_len = min(len(pred_words), len(gold_words))
        if overlap / min_len >= 0.5:  # At least 50% word overlap
            return True
    
    # Special cases for compound words
    compound_mappings = {
        "herbaceous vegetation": ["herbaceousvegetation"],
        "annual crop": ["annualcrop"],
        "permanent crop": ["permanentcrop"],
        "sea lake": ["sealake"]
    }
    
    for compound, singles in compound_mappings.items():
        if (pred == compound and gold in singles) or (gold == compound and pred in singles):
            return True
    
    return False