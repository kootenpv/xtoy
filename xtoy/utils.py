def is_float(x):
    try:
        if int(x) != float(x):
            return True
    except ValueError:
        pass
    return False
