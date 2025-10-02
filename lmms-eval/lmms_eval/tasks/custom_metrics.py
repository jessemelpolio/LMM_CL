from lmms_eval.api.registry import register_metric
import math

@register_metric(
    metric="hour_accuracy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def hour_accuracy(items):
    """Compute hour accuracy for clock reading."""
    return items

@register_metric(
    metric="minute_accuracy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def minute_accuracy(items):
    """Compute minute accuracy for clock reading."""
    return items

@register_metric(
    metric="both_accuracy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def both_accuracy(items):
    """Compute combined accuracy for clock reading."""
    return items

@register_metric(
    metric="countbench_accuracy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def countbench_accuracy(items):
    """Compute accuracy for counting task."""
    return items

@register_metric(
    metric="pixmocount_accuracy",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def pixmocount_accuracy(items):
    """Compute accuracy for pixmocount task."""
    return items

@register_metric(
    metric="pathvqa_vqa_score",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def pathvqa_vqa_score(items):
    """Compute standard VQA accuracy score for PathVQA."""
    return items

@register_metric(
    metric="pathvqa_exact_match",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def pathvqa_exact_match(items):
    """Compute token-based exact match for PathVQA task (from official implementation)."""
    return [item["pathvqa_exact_match"] for item in items]

@register_metric(
    metric="pathvqa_f1",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def pathvqa_f1(items):
    """Compute F1 score for PathVQA task (from official implementation)."""
    return [item["pathvqa_f1"] for item in items]

@register_metric(
    metric="pathvqa_bleu1",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def pathvqa_bleu1(items):
    """Compute BLEU-1 score for PathVQA task (from official implementation)."""
    return [item["pathvqa_bleu1"] for item in items]

@register_metric(
    metric="pathvqa_bleu2",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def pathvqa_bleu2(items):
    """Compute BLEU-2 score for PathVQA task (from official implementation)."""
    return [item["pathvqa_bleu2"] for item in items]

@register_metric(
    metric="pathvqa_bleu3",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def pathvqa_bleu3(items):
    """Compute BLEU-3 score for PathVQA task (from official implementation)."""
    return [item["pathvqa_bleu3"] for item in items]

@register_metric(
    metric="pathvqa_binary_accuracy",
    higher_is_better=True,
    output_type="generate_until"
)
def pathvqa_binary_accuracy(items):
    """
    Compute binary question accuracy for PathVQA task.
    
    This returns the accuracy for binary (yes/no) questions only.
    Non-binary questions are represented as NaN values.
    The actual aggregation is handled by a custom function defined in utils.py.
    """
    return [item["pathvqa_binary_accuracy"] for item in items]

@register_metric(
    metric="pathvqa_open_ended_score",
    higher_is_better=True,
    output_type="generate_until",
    aggregation="mean"
)
def pathvqa_open_ended_score(items):
    """Compute accuracy for open-ended questions in PathVQA."""
    return items 