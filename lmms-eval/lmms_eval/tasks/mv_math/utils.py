import re
from typing import Dict

def mv_math_doc_to_visual(doc):
    """Return image path."""
    return doc["image"]

def mv_math_doc_to_text(doc):
    """Return math problem prompt."""
    # Extract the text after removing <image> tag
    text = doc["conversations"][0]["value"]
    text = text.replace("<image>\n", "")
    return text

def mv_math_doc_to_target(doc):
    """Return ground truth answer."""
    return doc["conversations"][1]["value"]

def mv_math_process_results(doc, results):
    """Process model output and compute accuracy."""
    pred = results[0].strip()
    gold = doc["conversations"][1]["value"].strip()
    
    # Normalize answers
    pred_normalized = extract_math_answer(pred)
    gold_normalized = extract_math_answer(gold)
    
    # Check exact match or numeric equivalence
    is_correct = check_math_equivalence(pred_normalized, gold_normalized)
    
    return {"mv_math_acc": float(is_correct)}

def extract_math_answer(text):
    """Extract mathematical answer from text."""
    # Remove units and extra text
    text = text.lower()
    
    # Common units to remove
    units = [
        'cm', 'mm', 'm', 'km', 'inches', 'feet', 'yards', 'miles',
        'g', 'kg', 'pounds', 'lbs', 'oz',
        'ml', 'l', 'liters', 'gallons',
        'degrees', '°', 'celsius', 'fahrenheit',
        'seconds', 's', 'minutes', 'min', 'hours', 'h',
        'dollars', '$', '¥', '€', '£',
        '元', '米', '千米', '克', '千克', '升', '毫升', '度', '秒', '分钟', '小时'
    ]
    
    # Remove units
    for unit in units:
        text = re.sub(rf'\b{unit}\b', '', text)
    
    # Try to extract number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        # Return the last number found (usually the answer)
        return numbers[-1]
    
    # For algebraic expressions, try to extract variable = value
    var_match = re.search(r'([a-zA-Z])\s*=\s*(-?\d+\.?\d*)', text)
    if var_match:
        return var_match.group(2)
    
    # For fractions
    fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
    if fraction_match:
        try:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            if denominator != 0:
                return str(numerator / denominator)
        except:
            pass
    
    # Clean up and return
    text = re.sub(r'[^\w\d\+\-\*/\^\s\.]', '', text).strip()
    return text

def check_math_equivalence(pred, gold, tolerance=0.001):
    """Check if two mathematical answers are equivalent."""
    # First try exact string match
    if pred == gold:
        return True
    
    # Try numeric comparison
    try:
        pred_num = float(pred)
        gold_num = float(gold)
        return abs(pred_num - gold_num) < tolerance
    except:
        pass
    
    # Try to handle percentage vs decimal
    try:
        # Check if one is percentage and other is decimal
        if '%' in pred or '%' in gold:
            pred_clean = pred.replace('%', '')
            gold_clean = gold.replace('%', '')
            pred_num = float(pred_clean)
            gold_num = float(gold_clean)
            
            # If one has % and other doesn't, convert
            if '%' in pred and '%' not in gold:
                pred_num = pred_num / 100
            elif '%' in gold and '%' not in pred:
                gold_num = gold_num / 100
                
            return abs(pred_num - gold_num) < tolerance
    except:
        pass
    
    # Fall back to normalized string comparison
    pred_normalized = re.sub(r'\s+', '', pred)
    gold_normalized = re.sub(r'\s+', '', gold)
    return pred_normalized == gold_normalized