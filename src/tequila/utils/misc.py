from numpy import isclose, float64, complex64


def to_float(number) -> float:
    """
    Cast numeric type to reals
    """

    if hasattr(number, "imag"):
        if isclose(number.imag, 0.0, atol=1.e-6):
            return float64(number.real)
        else:
            raise TypeError("imaginary part detected {number}".format(number=number))
    elif hasattr(number, "evalf"):
        tmp = complex64(number.evalf())
        if hasattr(tmp, "imag") and isclose(tmp.imag, 0.0, atol=1.e-6):
            return float64(tmp.real)
        else:
            raise TypeError(
                "casting number {number} of type {type} fo float failed".format(number=number, type=type(number)))
    else:
        try:
            return float64(number)
        except TypeError:
            raise TypeError(
                "casting number {number} of type {type} fo float failed".format(number=number, type=type(number)))
