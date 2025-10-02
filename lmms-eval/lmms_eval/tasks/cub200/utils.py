import re
import numpy as np
from PIL import Image
import io

def cub200_doc_to_visual(doc):
    """
    Extract the image from the document.
    
    Args:
        doc: A document from the dataset with an image field.
        
    Returns:
        list: A list containing the PIL.Image extracted from the document.
    """
    # Handle different image formats
    image = doc["image"]
    
    # Handle case where image is already a PIL Image object
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    # Handle case where image is a dictionary with bytes
    elif isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    # Handle other possible formats
    else:
        raise ValueError(f"Unexpected image format: {type(image)}")
        
    return [image]  # Return as a list of images

def cub200_doc_to_text(doc, pre_prompt="", post_prompt=""):
    """
    Extract the question text from the document and add pre/post prompts.
    
    Args:
        doc: A document from the dataset with a question field.
        pre_prompt: Text to add before the question.
        post_prompt: Text to add after the question.
        
    Returns:
        str: The formatted question text with prompts.
    """
    # Get the question text
    question = doc["question"]
    
    # Remove <image> tag if present (for consistency with other datasets)
    question = question.replace("\n<image>", "").replace("<image>", "")
    
    # Return the formatted question with prompts
    return f"{pre_prompt}{question}{post_prompt}"

def cub200_doc_to_target(doc):
    """
    Extract the answer (target) from the document.
    
    Args:
        doc: A document from the dataset with an answer field.
        
    Returns:
        str: The answer (target).
    """
    # Return the answer from the document
    return doc["answer"]

def cub200_process_results(doc, results):
    """
    Process the model predictions to calculate accuracy metric.
    
    Args:
        doc: The original document/sample from the dataset.
        results: The model's prediction for this document.
        
    Returns:
        dict: A dictionary containing the processed results for accuracy calculation.
    """
    # Get the ground truth answer (target)
    gold = doc["answer"].lower()
    
    # Handle case where results is a list (take the first item)
    if isinstance(results, list):
        if len(results) > 0:
            results = results[0]
        else:
            results = ""
    
    # Clean the prediction (remove whitespace, extract just the letter)
    # Multiple formats are possible: "A", "A.", "The answer is A", etc.
    cleaned_pred = str(results).strip()
    
    # Try to extract just the letter using a regex
    letter_match = re.search(r'\b([A-Za-z])[.)]?\b', cleaned_pred)
    if letter_match:
        cleaned_pred = letter_match.group(1).lower()
    else:
        # If no match, just use the first character if it's a letter
        if cleaned_pred and cleaned_pred[0].isalpha():
            cleaned_pred = cleaned_pred[0].lower()
        else:
            cleaned_pred = cleaned_pred.lower()
    
    # Return a result with the accuracy (1 for correct, 0 for incorrect)
    is_correct = int(gold == cleaned_pred)
    
    return {
        "acc": is_correct,
        "gold": gold, 
        "pred": cleaned_pred
    } 