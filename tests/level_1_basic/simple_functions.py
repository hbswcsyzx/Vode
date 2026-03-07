"""
Level 1: Basic Python Functions
Tests core function call tracing without PyTorch dependencies.
"""


def add(a, b):
    """Simple addition function."""
    return a + b


def multiply(x, y):
    """Simple multiplication function."""
    return x * y


def compute(a, b, c):
    """Function that calls other functions."""
    temp1 = add(a, b)
    temp2 = multiply(temp1, c)
    return temp2


def nested_compute(x):
    """Function with nested calls."""
    result = compute(x, x + 1, x + 2)
    return result


def main():
    """Main entry point for testing."""
    print("Level 1: Basic Python Functions")
    
    # Test simple function calls
    result1 = add(5, 3)
    print(f"add(5, 3) = {result1}")
    
    result2 = multiply(4, 7)
    print(f"multiply(4, 7) = {result2}")
    
    # Test function composition
    result3 = compute(2, 3, 4)
    print(f"compute(2, 3, 4) = {result3}")
    
    # Test nested calls
    result4 = nested_compute(10)
    print(f"nested_compute(10) = {result4}")
    
    return result4


if __name__ == "__main__":
    main()
