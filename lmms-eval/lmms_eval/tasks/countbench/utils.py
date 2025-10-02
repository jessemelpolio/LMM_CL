import re

def countbench_doc_to_visual(doc):
    """Extract image from CountBench document."""
    return [doc["image"].convert("RGB")]

def countbench_doc_to_text(doc, pre_prompt="", post_prompt=""):
    """Convert CountBench document to input text."""
    question = doc["question"]
    # remove \n<image>
    question = question.replace("\n<image>", "")
    return f"{pre_prompt}{question}{post_prompt}"

def countbench_doc_to_target(doc):
    """Extract target number from CountBench document."""
    return str(doc["number"])

def parse_count(text):
    """Parse number from text, handling various formats."""
    # Convert to lowercase and remove periods for consistency
    text = text.lower().replace('.', '')
    
    # Dictionary for converting word numbers to digits
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }
    
    # First try to find any digits in the text
    number_match = re.search(r'\d+', text)
    if number_match:
        return int(number_match.group())
    
    # If no digits found, try to match word numbers
    for word, num in word_to_num.items():
        if word in text:
            return int(num)
    
    return None

def countbench_process_results(doc, results):
    """Process model outputs for CountBench evaluation."""
    # Extract predicted number from model output
    # Results is a list with a single string for generate_until tasks
    pred_text = results[0].strip()
    
    # Parse prediction using the enhanced parser
    pred_number = parse_count(pred_text)
    if pred_number is None:
        return {"countbench_accuracy": 0}
    
    # Get ground truth
    true_number = doc["number"]
    
    # Return 1 if prediction matches ground truth, 0 otherwise
    return {"countbench_accuracy": int(pred_number == true_number)}
