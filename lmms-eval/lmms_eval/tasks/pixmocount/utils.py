import re
from PIL import Image
import io

def pixmocount_doc_to_visual(doc):
    """Extract image from PixmoCount document."""
    image = doc["image"]
    if isinstance(image, dict) and "bytes" in image:
        # Convert bytes to PIL Image if needed
        image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    return [image]

def pixmocount_doc_to_text(doc, pre_prompt="", post_prompt=""):
    """Convert PixmoCount document to input text."""
    question = doc["question"]
    # remove \n<image>
    question = question.replace("\n<image>", "")
    return f"{pre_prompt}{question}{post_prompt}"

def pixmocount_doc_to_target(doc):
    """Extract target answer from PixmoCount document."""
    # Extract number from the answer string
    number_match = re.search(r'\d+', doc['answer'])
    if number_match:
        return number_match.group()
    return doc['answer']  # fallback to full answer if no number found

def parse_count(text):
    """Parse number from text, handling both digits and word numbers."""
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
    
    # Replace word numbers with digits
    for word, num in word_to_num.items():
        text = text.replace(word, num)
    
    # Try to find number in different formats
    patterns = [
        r'\d+',  # Raw digits
        r'there (?:is|are) (\d+)',  # "there are X"
        r'(\d+)\s+\w+',  # "X" followed by any word(s)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # If the pattern has a group, use it, otherwise use the full match
            number = match.group(1) if len(match.groups()) > 0 else match.group(0)
            return int(number)
    
    return None

def pixmocount_process_results(doc, results):
    """Process model outputs for PixmoCount evaluation."""
    # Extract predicted number from model output
    pred_text = results[0].strip()
    
    # Parse prediction using the enhanced parser
    pred_number = parse_count(pred_text)
    if pred_number is None:
        return {"pixmocount_accuracy": 0}
    
    # Get ground truth number
    true_answer = doc['answer']
    true_number = parse_count(true_answer)
    if true_number is None:
        return {"pixmocount_accuracy": 0}
    
    # Return 1 if prediction matches ground truth, 0 otherwise
    return {"pixmocount_accuracy": int(pred_number == true_number)} 