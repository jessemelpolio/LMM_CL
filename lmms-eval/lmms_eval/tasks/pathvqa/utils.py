import re
import math
from collections import defaultdict
from typing import List, Dict, Union, Any

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

def pathvqa_doc_to_visual(doc):
    """Returns the visual content from the document."""
    return [doc['image'].convert('RGB')]

def pathvqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    """Formats the question with appropriate prompts."""
    question = doc['question']
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{question}{post_prompt}"

def pathvqa_doc_to_target(doc):
    """Returns the target answer from the document."""
    return doc['answer']

def is_yes_no_question(question):
    """
    Determines if a question is a binary (yes/no) question based solely on its structure.
    
    This function examines the structure of the question to identify if it's likely to 
    be answered with 'yes' or 'no'. It looks for common prefixes that indicate a binary
    question regardless of the actual answer.
    
    Args:
        question: The question text to analyze
        
    Returns:
        bool: True if the question appears to be a yes/no question, False otherwise
    """
    question = question.lower().strip()
    
    # Common prefixes for yes/no questions
    binary_prefixes = [
        'is ', 'are ', 'was ', 'were ', 
        'does ', 'do ', 'did ',
        'can ', 'could ', 'would ', 'should ',
        'has ', 'have ', 'had ',
        'will ', 'shall ',
        'am i ', 'are there ', 'is there ', 'are these ', 'is this '
    ]
    
    # Check if the question starts with any of the binary question prefixes
    return any(question.startswith(prefix) for prefix in binary_prefixes)

# Helper functions for official PathVQA evaluation
def split_sentence(sentence, n):
    """
    Split sentence into n-grams and count their occurrences.
    This is the exact implementation from the official evaluation script.
    """
    words = defaultdict(int)
    tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words

def brevity_penalty(candidate, references):
    """
    Calculate brevity penalty for BLEU score.
    This is the exact implementation from the official evaluation script.
    """
    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))
    
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)

def modified_precision(candidate, references, n):
    """
    Calculate modified precision for BLEU score.
    This is the exact implementation from the official evaluation script.
    """
    max_frequency = defaultdict(int)
    min_frequency = defaultdict(int)
    
    candidate_words = split_sentence(candidate, n)
    
    for reference in references:
        reference_words = split_sentence(reference, n)
        for word in candidate_words:
            max_frequency[word] = max(max_frequency[word], reference_words[word])
    for word in candidate_words:
            min_frequency[word] = min(max_frequency[word], candidate_words[word])
    
    if sum(candidate_words.values()) == 0:
        return 0
    
    P = sum(min_frequency.values()) / sum(candidate_words.values())
    return P

def calculate_bleu(weights, pn, n, bp):
    """
    Calculate BLEU score using the official formula.
    This is the exact implementation from the official evaluation script.
    """
    sum_wlogp = 0
    for i in range(n):
        if pn[i] != 0:
            sum_wlogp += float(weights[i]) * math.log(pn[i])
    bleu_result = bp * math.exp(sum_wlogp)
    return bleu_result

def bleu(candidate, references, n, weights):
    """
    Calculate BLEU-n score.
    This is the exact implementation from the official evaluation script.
    """
    pn = []
    bp = brevity_penalty(candidate, references)
    for i in range(n):
        pn.append(modified_precision(candidate, references, i + 1))
    
    if len(weights) > len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(weights[i])
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return bleu_result
    elif len(weights) < len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(0)
        for i in range(len(weights)):
            tmp_weights[i] = weights[i]
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return bleu_result
    else:
        bleu_result = calculate_bleu(weights, pn, n, bp)
        return bleu_result

def calculate_exactmatch(candidate, reference):
    """
    Calculate exact match score.
    This is the exact implementation from the official evaluation script.
    """
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]
        
    if total == 0:
        return 0
    else:
        return count / total

def calculate_f1score(candidate, reference):
    """
    Calculate F1 score.
    This is the exact implementation from the official evaluation script.
    """
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)
    
    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]
    
    if len(candidate_words) == 0:
        return 0
    elif len(reference_words) == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)

def pathvqa_process_results(doc, results):
    """
    Process model outputs using metrics defined in the official PathVQA evaluation script.
    """
    assert len(results) == 1, f"The result should be a list of length 1, but got {len(results)}."
    prediction = results[0].strip()
    ground_truth = doc['answer']
    question = doc['question']
    
    # Check if it's a binary question ONLY based on question structure
    is_binary = is_yes_no_question(question)
    
    # Calculate metrics using the official implementation
    try:
        exact_match_score = calculate_exactmatch(prediction, ground_truth)
        f1_score = calculate_f1score(prediction, ground_truth)
    except Exception as e:
        # Handle potential errors in metric calculation, especially for empty strings
        print(f"Error calculating metrics: {e}")
        exact_match_score = 0.0
        f1_score = 0.0
    
    # Calculate BLEU scores with uniform weights
    references = [ground_truth]  # PathVQA has only one reference per question
    
    try:
        bleu1_score = bleu(prediction, references, 1, [1.0])
        bleu2_score = bleu(prediction, references, 2, [0.5, 0.5])
        bleu3_score = bleu(prediction, references, 3, [1/3, 1/3, 1/3])
    except Exception as e:
        # Handle potential errors in BLEU calculation
        print(f"Error calculating BLEU scores: {e}")
        bleu1_score = 0.0
        bleu2_score = 0.0
        bleu3_score = 0.0
    
    # Binary accuracy (for yes/no questions based on question structure)
    # For binary questions, we compute binary accuracy by comparing lowercase answers
    binary_accuracy = float('nan')  # default for non-binary questions
    
    if is_binary:
        pred_norm = prediction.lower().strip()
        gt_norm = ground_truth.lower().strip()
        binary_accuracy = 1.0 if pred_norm == gt_norm else 0.0
    
    return {
        "pathvqa_exact_match": exact_match_score,
        "pathvqa_f1": f1_score,
        "pathvqa_bleu1": bleu1_score,
        "pathvqa_bleu2": bleu2_score,
        "pathvqa_bleu3": bleu3_score,
        "pathvqa_binary_accuracy": binary_accuracy,
        "submission": {
            "question_id": doc.get("question_id", "unknown"),
            "answer": prediction,
        }
    }

def pathvqa_aggregate_binary_accuracy(items):
    """
    Custom aggregation function for binary_accuracy metric.
    
    This function filters out NaN values (which represent non-binary questions) 
    before computing the mean. The result is the average accuracy across 
    only binary (yes/no) questions.
    
    Args:
        items: List of binary accuracy scores, where NaN means the question wasn't binary
        
    Returns:
        float: The mean accuracy for binary questions, or NaN if no binary questions exist
    """
    import math
    
    # Filter out NaN values (non-binary questions)
    valid_scores = [score for score in items if not (isinstance(score, float) and math.isnan(score))]
    
    # If no binary questions were evaluated, return NaN
    if len(valid_scores) == 0:
        return float('nan')
    
    # Compute the mean of valid scores
    return sum(valid_scores) / len(valid_scores)

def pathvqa_aggregate_submissions(items):
    """
    Aggregate submission items.
    
    Args:
        items: List of submission dictionaries
    
    Returns:
        list: The list of submissions
    """
    return items