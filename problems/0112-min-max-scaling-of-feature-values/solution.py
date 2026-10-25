def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    x_min = min(x)
    x_max = max(x)

    return [(x_i - x_min) / (x_max - x_min) for x_i in x]