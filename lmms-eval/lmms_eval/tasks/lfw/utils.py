"""
Utility functions for LFW face recognition evaluation.
"""

def lfw_doc_to_visual(doc):
    """Extract image from document."""
    # Convert to RGB to ensure compatibility
    if hasattr(doc["image"], 'convert'):
        return [doc["image"].convert("RGB")]
    return [doc["image"]]

def lfw_doc_to_text(doc, pre_prompt="", post_prompt=""):
    """Extract question text from document."""
    # Check if question field exists, otherwise extract from conversations
    if "question" in doc:
        question = doc["question"]
    elif "conversations" in doc:
        # Handle conversations as JSON string if needed
        import json
        if isinstance(doc["conversations"], str):
            conversations = json.loads(doc["conversations"])
        else:
            conversations = doc["conversations"]
            
        if len(conversations) > 0:
            # Extract text after removing <image> tag
            text = conversations[0]["value"]
            question = text.replace("<image>\n", "").strip()
        else:
            question = "What is the name of the person in this image?"
    else:
        question = "What is the name of the person in this image?"
    
    # Return with pre and post prompts
    return f"{pre_prompt}{question}{post_prompt}"

def lfw_doc_to_target(doc):
    """Extract target answer (person's name) from document."""
    # Try multiple fields for flexibility
    if "answer" in doc:
        return doc["answer"]
    elif "person_name" in doc:
        return doc["person_name"]
    elif "conversations" in doc:
        # Handle conversations as JSON string if needed
        import json
        if isinstance(doc["conversations"], str):
            conversations = json.loads(doc["conversations"])
        else:
            conversations = doc["conversations"]
            
        if len(conversations) > 1:
            return conversations[1]["value"]
        else:
            return ""
    else:
        return ""

def lfw_process_results(doc, results):
    """Process model outputs for evaluation."""
    prediction = results[0] if results else ""
    target = lfw_doc_to_target(doc)
    
    # Clean up prediction
    prediction = prediction.strip()
    
    # Remove common suffixes/prefixes that models might add
    # Remove trailing punctuation
    while prediction and prediction[-1] in '.,!?;:':
        prediction = prediction[:-1].strip()
    
    # Remove common prefixes like "The person is", "This is", etc.
    prefixes_to_remove = [
        "the person is ",
        "the person in the image is ",
        "the person shown is ",
        "this is ",
        "it is ",
        "that is ",
        "this person is ",
        "the name is ",
    ]
    prediction_lower = prediction.lower()
    for prefix in prefixes_to_remove:
        if prediction_lower.startswith(prefix):
            prediction = prediction[len(prefix):].strip()
            break
    
    # Handle quotation marks
    if prediction.startswith('"') and prediction.endswith('"'):
        prediction = prediction[1:-1].strip()
    elif prediction.startswith("'") and prediction.endswith("'"):
        prediction = prediction[1:-1].strip()
    
    # Exact match
    exact_match = int(prediction == target)
    
    # Case-insensitive match
    exact_match_case_insensitive = int(
        prediction.lower() == target.lower()
    )
    
    return {
        "exact_match": exact_match,
        "exact_match_case_insensitive": exact_match_case_insensitive,
        "prediction": prediction,
        "target": target
    }