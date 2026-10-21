"""
Statistics helpers for the anchor pilot.
"""

import math


def clopper_pearson_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """
    Compute the Clopper-Pearson exact binomial confidence interval's lower bound.

    Uses binary search over p in [0, 1] to find the value where the binomial
    survival function (probability of k or more successes) equals the complement
    of the confidence level, halved (i.e., alpha/2).

    Args:
        successes: Number of successful outcomes (k). Must satisfy 0 <= successes <= n.
        n: Total number of trials. Must be > 0.
        confidence: Confidence level (default 0.95 for 95% CI). Must be in (0, 1).

    Returns:
        The lower bound of the confidence interval on the true success probability.

    Raises:
        ValueError: If n <= 0, successes < 0, or successes > n.
    """
    # Input validation
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes}")
    if successes > n:
        raise ValueError(f"successes ({successes}) cannot exceed n ({n})")

    # Special case: no successes means the lower bound is 0
    if successes == 0:
        return 0.0

    # Special case: perfect successes (successes == n)
    # Clopper-Pearson's k==n special case solves p**n == alpha/2 directly
    if successes == n:
        alpha = 1.0 - confidence
        return (alpha / 2.0) ** (1.0 / n)

    # General case: binary search for p such that the survival function equals alpha/2
    alpha = 1.0 - confidence
    target = alpha / 2.0

    # Binomial survival function: P(X >= successes) when X ~ Binomial(n, p)
    def survival_function(p: float) -> float:
        """Compute the binomial survival function at p."""
        result = 0.0
        for i in range(successes, n + 1):
            result += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
        return result

    # Binary search for the lower bound
    # The survival function is monotonically increasing in p
    left, right = 0.0, 1.0
    tolerance = 1e-10

    while right - left > tolerance:
        mid = (left + right) / 2.0
        sf = survival_function(mid)

        if sf < target:
            # Need a larger p to increase the survival function
            left = mid
        else:
            # Need a smaller p to decrease the survival function
            right = mid

    return (left + right) / 2.0
