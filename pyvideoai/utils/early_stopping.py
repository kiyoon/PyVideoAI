import math

def has_gotten_lower(values: list, allow_same: bool = True, EPS: float = 1e-6) -> bool:
    """
    Check if any of the elements is lower than the first element.
    Used for early stopping.

    ```
    if not has_gotten_lower(val_loss[-20:]):
        logger.info("Validation loss hasn't decreased for 20 epochs. Stopping training..")
        raise Exception("Early stopping triggered.")
    ```
    """

    values = [value for value in values if value is not None and not math.isnan(value)]

    if len(values) <= 1:
        raise ValueError("Can't determine if values got lower with 0 or 1 value.")

    if allow_same:
        for value in values[1:]:
            if values[0] > value - EPS:
                return True
    else:
        for value in values[1:]:
            if values[0] > value + EPS:
                return True

    return False


def has_gotten_higher(values: list, allow_same: bool = True, EPS: float = 1e-6) -> bool:
    """
    Check if any of the elements is higher than the first element.
    Used for early stopping.
    If the list contains None, ignore them.

    ```
    if not has_gotten_higher(val_acc[-20:]):
        logger.info("Validation accuracy hasn't increased for 20 epochs. Stopping training..")
        raise Exception("Early stopping triggered.")
    ```
    """

    values = [value for value in values if value is not None and not math.isnan(value)]

    if len(values) <= 1:
        raise ValueError("Can't determine if values got higher with 0 or 1 value.")

    if allow_same:
        for value in values[1:]:
            if values[0] < value + EPS:
                return True
    else:
        for value in values[1:]:
            if values[0] < value - EPS:
                return True

    return False


def has_gotten_better(values: list, is_better_func: callable, allow_same: bool = True) -> bool:
    """
    Check if any of the elements is better than the first element.
    Used for early stopping.
    If the list contains None, ignore them.

    ```
    if not has_gotten_better(val_acc[-20:], is_better_func=lambda a,b: a>b):
        logger.info("Validation accuracy hasn't increased for 20 epochs. Stopping training..")
        raise Exception("Early stopping triggered.")
    ```
    """

    values = [value for value in values if value is not None and not math.isnan(value)]

    if len(values) <= 1:
        raise ValueError("Can't determine if values got better with 0 or 1 value.")

    if allow_same:
        for value in values[1:]:
            if not is_better_func(values[0], value):
                return True
    else:
        for value in values[1:]:
            if is_better_func(value, values[0]):
                return True

    return False


def min_value_within_lastN(values: list, last_N: int = 20, allow_same: bool = True) -> bool:
    return best_value_within_lastN(values, last_N, is_better_func=lambda a,b: a<b, allow_same=allow_same)
def max_value_within_lastN(values: list, last_N: int = 20, allow_same: bool = True) -> bool:
    return best_value_within_lastN(values, last_N, is_better_func=lambda a,b: a>b, allow_same=allow_same)

def best_value_within_lastN(values: list, last_N: int = 20, is_better_func: callable = lambda a,b: a>b, allow_same: bool = True) -> bool:
    """
    Check if one of last N values is the best value of the array.
    Used for early stopping.
    If the list contains None, consider it as always worse than best.

    ```
    if not best_value_within_lastN(val_acc, 20, is_better_func=lambda a,b: a>b):
        logger.info("Validation accuracy hasn't increased for 20 epochs. Stopping training..")
        raise Exception("Early stopping triggered.")
    ```
    """

    values_filtered = [(idx, value) for idx, value in enumerate(values) if value is not None and not math.isnan(value)]

    if len(values_filtered) == 0:
        raise ValueError("Can't determine if values got better with no value.")

    if len(values) < last_N:
        return True

    best_idx = values_filtered[0][0]
    best_value = values_filtered[0][1]

    if allow_same:
        for idx, value in values_filtered:
            if not is_better_func(best_value, value):
                best_idx = idx
                best_value = value
    else:
        for idx, value in values_filtered:
            if is_better_func(value, best_value):
                best_idx = idx
                best_value = value

    return best_idx >= len(values) - last_N
