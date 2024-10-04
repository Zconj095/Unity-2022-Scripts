def expanded_mmr(difficulty, context, processing_time, extra_energy):
    """
    Calculates the Manual Memory Recall (MMR) using the expanded equation.

    Args:
        difficulty: The difficulty of the recall task (float).
        context: The context in which the information was stored (float).
        processing_time: The time it takes to retrieve the information (float).
        extra_energy: The additional energy required for manual recall (float).

    Returns:
        The Manual Memory Recall (MMR) score.
    """

    # Calculate the numerator of the expanded equation.
    numerator = context * extra_energy * processing_time + context * processing_time * processing_time + extra_energy * processing_time

    # Calculate the denominator of the expanded equation.
    denominator = context

    # Calculate the expanded Manual Memory Recall score.
    expanded_mmr = numerator / denominator

    return expanded_mmr

# Example usage
difficulty = 0.7  # Higher value indicates greater difficulty
context = 0.5  # Higher value indicates easier recall due to context
processing_time = 2.0  # Time in seconds
extra_energy = 1.5  # Additional energy expenditure

expanded_mmr_score = expanded_mmr(difficulty, context, processing_time, extra_energy)

print(f"Expanded Manual Memory Recall score: {expanded_mmr_score:.2f}")
