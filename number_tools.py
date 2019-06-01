import numbers


def is_number(num):
    if isinstance(num, numbers.Number) and type(num) is not bool:  # in Python, boolean is a subclass of int
        return True
    return False
