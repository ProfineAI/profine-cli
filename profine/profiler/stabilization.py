"""Warmup detection via stabilization analysis.

Determines when step times have stabilized (compile/cudagraph overhead
is gone) so we can split warmup from steady-state measurements.
"""

from __future__ import annotations

import statistics

_WINDOW_SIZE = 5
_CV_THRESHOLD = 0.10          # coefficient of variation < 10%
_SLOPE_THRESHOLD = 0.05       # normalized slope < 5%
_TAIL_CLOSENESS_THRESHOLD = 0.12
_REQUIRED_CONSECUTIVE = 3     # consecutive stable windows before accepting


def detect_stabilization_point(
    step_times: list[float],
    min_warmup: int = 0,
) -> int:
    """Find the first step index where times are stable.

    Returns the index into step_times where steady-state begins.
    If no stabilization is detected, returns min_warmup.
    """
    n = len(step_times)
    if n < _WINDOW_SIZE + _REQUIRED_CONSECUTIVE:
        return min(min_warmup, max(0, n - 1))

    # Compute tail median (last 20% of steps) as reference
    tail_start = max(n - n // 5, n // 2)
    tail = step_times[tail_start:]
    if not tail:
        return min_warmup
    tail_median = statistics.median(tail)
    if tail_median <= 0:
        return min_warmup

    # Adjust closeness threshold based on tail variability
    tail_cv = _cv(tail) if len(tail) >= 2 else 0.0
    closeness = _TAIL_CLOSENESS_THRESHOLD + min(tail_cv, 0.05)

    consecutive = 0
    for i in range(max(min_warmup, 0), n - _WINDOW_SIZE + 1):
        window = step_times[i : i + _WINDOW_SIZE]
        if _is_stable_window(window, tail_median, closeness):
            consecutive += 1
            if consecutive >= _REQUIRED_CONSECUTIVE:
                return i - _REQUIRED_CONSECUTIVE + 1
        else:
            consecutive = 0

    return min_warmup


def _is_stable_window(
    window: list[float],
    tail_median: float,
    closeness: float,
) -> bool:
    """Check if a window of step times is stable."""
    if len(window) < 2:
        return False

    # Low coefficient of variation
    cv = _cv(window)
    if cv > _CV_THRESHOLD:
        return False

    # Low slope (not trending up or down)
    slope = _normalized_slope(window)
    if abs(slope) > _SLOPE_THRESHOLD:
        return False

    # Close to tail median
    window_median = statistics.median(window)
    if tail_median > 0:
        deviation = abs(window_median - tail_median) / tail_median
        if deviation > closeness:
            return False

    return True


def _cv(values: list[float]) -> float:
    """Coefficient of variation."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean <= 0:
        return 0.0
    return statistics.stdev(values) / mean


def _normalized_slope(values: list[float]) -> float:
    """Slope normalized by mean, using simple linear regression."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_val = statistics.mean(values)
    if mean_val <= 0:
        return 0.0

    mean_x = (n - 1) / 2
    numerator = sum((i - mean_x) * (v - mean_val) for i, v in enumerate(values))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0

    slope = numerator / denominator
    return slope / mean_val
