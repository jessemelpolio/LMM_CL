import re
from PIL import Image
import io

def openimgclock_doc_to_visual(doc):
    """Extract image from OpenImages clock document."""
    image = doc["image"]
    if isinstance(image, dict) and "bytes" in image:
        # Convert bytes to PIL Image if needed
        image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    return [image]

def openimgclock_doc_to_text(doc, pre_prompt="", post_prompt=""):
    """Convert clock document to input text."""
    # remove \n<image>
    question = doc["question"].replace("\n<image>", "")
    return f"{pre_prompt}{question}{post_prompt}"

def openimgclock_doc_to_target(doc):
    """Extract target time from clock document."""
    return doc["answer"]

def parse_time(text):
    """Parse hour and minute from text."""
    # Convert to lowercase and remove periods for consistency
    text = text.lower().replace('.', '')
    
    # Try to find time in format HH:MM
    time_match = re.search(r'(\d{1,2}):(\d{1,2})', text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        return hour, minute
    
    # Try to find time in natural language format "the time is X:Y"
    time_match = re.search(r'time (?:is|displayed|shown).*?(\d{1,2}):(\d{1,2})', text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        return hour, minute
    
    # Try to find separate hour and minute numbers
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        hour = int(numbers[0])
        minute = int(numbers[1])
        return hour, minute
    
    return None, None

def openimgclock_process_results(doc, results):
    """Process model outputs for clock reading evaluation."""
    # Parse predicted time
    pred_text = results[0].strip()
    pred_hour, pred_minute = parse_time(pred_text)
    
    # Parse ground truth time
    true_text = doc["answer"]
    true_hour, true_minute = parse_time(true_text)
    
    if pred_hour is None or true_hour is None:
        return {
            "hour_accuracy": 0,
            "minute_accuracy": 0,
            "both_accuracy": 0
        }
    
    # Calculate accuracies following the ItsAboutTime paper metrics
    hour_correct = (pred_hour == true_hour)
    minute_correct = abs(pred_minute - true_minute) <= 1 or abs(pred_minute - true_minute) == 59
    both_correct = hour_correct and minute_correct
    
    return {
        "hour_accuracy": int(hour_correct),
        "minute_accuracy": int(minute_correct),
        "both_accuracy": int(both_correct)
    } 