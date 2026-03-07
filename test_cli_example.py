def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


def compute(x, y):
    sum_val = add(x, y)
    prod_val = multiply(x, y)
    return sum_val, prod_val


if __name__ == "__main__":
    result = compute(3, 4)
    print(f"Result: {result}")
