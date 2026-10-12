
def calculate_average(prices):
    """Calculate the average of a list of car prices."""
    if not prices:
        raise ValueError("Prices list cannot be empty")

    if any(price < 0 for price in prices):
        raise ValueError("Car prices cannot be negative")

    return sum(prices) / len(prices)
