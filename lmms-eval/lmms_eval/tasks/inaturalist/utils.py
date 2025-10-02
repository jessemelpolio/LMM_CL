import re
from typing import Dict

def inaturalist_doc_to_visual(doc):
    """Return image path."""
    # Convert to RGB to ensure compatibility
    if hasattr(doc["image"], 'convert'):
        return [doc["image"].convert("RGB")]
    return [doc["image"]]

def inaturalist_doc_to_text(doc):
    """Return species identification prompt."""
    # Extract the text after removing <image> tag
    text = doc["conversations"][0]["value"]
    text = text.replace("<image>\n", "")
    return text

def inaturalist_doc_to_target(doc):
    """Return ground truth species name."""
    return doc["conversations"][1]["value"]

def inaturalist_process_results(doc, results):
    """Process model output and compute accuracy for species identification."""
    pred = results[0].strip()
    gold = doc["conversations"][1]["value"].strip()
    
    # Normalize answers
    pred_normalized = normalize_species_name(pred)
    gold_normalized = normalize_species_name(gold)
    
    # Check various matching strategies
    is_correct = check_species_match(pred_normalized, gold_normalized)
    
    return {"inat_accuracy": float(is_correct)}

def normalize_species_name(name):
    """Normalize species name for comparison."""
    # Convert to lowercase
    name = name.lower()
    
    # Remove common prefixes/suffixes
    name = re.sub(r'^(the|a|an)\s+', '', name)
    
    # Remove punctuation except parentheses and hyphens
    name = re.sub(r'[^\w\s\(\)\-]', '', name)
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name

def check_species_match(pred, gold):
    """Check if predicted species matches ground truth with various strategies."""
    # Exact match
    if pred == gold:
        return True
    
    # Check if scientific name and common name format
    # Format: "Common Name (Scientific name)"
    if '(' in gold and ')' in gold:
        # Extract common and scientific names
        common_match = re.match(r'^([^(]+)\s*\(', gold)
        scientific_match = re.search(r'\(([^)]+)\)', gold)
        
        if common_match:
            common_name = common_match.group(1).strip()
            if pred == common_name.lower():
                return True
                
        if scientific_match:
            scientific_name = scientific_match.group(1).strip()
            if pred == scientific_name.lower():
                return True
            
            # Check genus only (first word of scientific name)
            genus = scientific_name.split()[0] if scientific_name.split() else ""
            if genus and pred == genus.lower():
                return True
    
    # Reverse check - if prediction has parentheses
    if '(' in pred and ')' in pred:
        pred_common_match = re.match(r'^([^(]+)\s*\(', pred)
        pred_scientific_match = re.search(r'\(([^)]+)\)', pred)
        
        if pred_common_match:
            pred_common = pred_common_match.group(1).strip()
            if gold == pred_common.lower():
                return True
                
        if pred_scientific_match:
            pred_scientific = pred_scientific_match.group(1).strip()
            if gold == pred_scientific.lower():
                return True
    
    # Check if one is contained in the other (for partial matches)
    if len(pred) > 3 and len(gold) > 3:  # Avoid short substring matches
        if pred in gold or gold in pred:
            return True
    
    # Check word overlap for multi-word names
    pred_words = set(pred.split())
    gold_words = set(gold.split())
    
    # If all words match (order independent)
    if pred_words == gold_words:
        return True
    
    # If significant overlap (at least 2/3 of words match)
    if len(pred_words) >= 2 and len(gold_words) >= 2:
        overlap = len(pred_words & gold_words)
        min_len = min(len(pred_words), len(gold_words))
        if overlap / min_len >= 0.67:
            return True
    
    # Handle common variations
    variations = get_species_variations(gold)
    if pred in variations:
        return True
    
    return False

def get_species_variations(species_name):
    """Get common variations of species names."""
    variations = [species_name]
    
    # Common abbreviations and variations
    replacements = {
        "species": ["sp.", "spp."],
        "subspecies": ["subsp.", "ssp."],
        "variety": ["var.", "v."],
        "form": ["f.", "forma"],
    }
    
    for full, abbrevs in replacements.items():
        if full in species_name:
            for abbrev in abbrevs:
                variations.append(species_name.replace(full, abbrev))
    
    # Remove common words that might be optional
    optional_words = ["the", "common", "american", "european", "asian", "african"]
    for word in optional_words:
        if word in species_name:
            variations.append(species_name.replace(word + " ", ""))
    
    return variations