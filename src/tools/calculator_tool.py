"""Deterministic variance calculator tool."""


def calculate_variance(current_val: float, prior_val: float) -> dict:
    """Compute absolute change, percentage variance, and trend between two values."""
    for name, value in (("current_val", current_val), ("prior_val", prior_val)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if prior_val == 0:
        raise ValueError("prior_val must not be zero (percentage variance undefined)")

    absolute_change = current_val - prior_val
    percentage_variance = round((absolute_change / prior_val) * 100, 2)
    if absolute_change > 0:
        trend = "increase"
    elif absolute_change < 0:
        trend = "decrease"
    else:
        trend = "no_change"

    return {
        "absolute_change": absolute_change,
        "percentage_variance": percentage_variance,
        "trend": trend,
    }
