import functools
from logzero import logger, loglevel

import numpy as np


def check_value(str_to_check: str, values_to_accept: list):
    """
A convenient function to check if a string should be accepted or not.

    Parameters
    ----------
    str_to_check : str
        The string to check
    values_to_accept : list
        The list of acceptable strings.

    Returns
    -------
    Throws ValueError if string is not correct.
    """
    if str_to_check not in values_to_accept:
        raise ValueError("This value is not handled: %s. Try: %s." % (str_to_check, values_to_accept))


def check_dim(array: np.ndarray, dimension_to_accept: list or int):
    """
A convenient function to check if the dimension of an array should be accepted or not.

    Parameters
    ----------
    array : np.ndarray
        The array to check
    dimension_to_accept : int | list
        The dimension(s) to accept

    Returns
    -------
    None

    Throws ValueError if not correct.
    """
    if type(dimension_to_accept) is int:
        dimension_to_accept = [dimension_to_accept]
    if array.ndim not in dimension_to_accept:
        raise ValueError("You passed an array with dimension %d (shape: %s), when accepted dimensions are: %s." %
                         (array.ndim, array.shape, dimension_to_accept))


def custom_loglevel(func):
    @functools.wraps(func)
    def loglevel_decorator(verbose=None, *args, **kwargs):
        """
    Decorator which enables handling of custom loglevels.

        Args:
            verbose: passed to logger.loglevel. 10: debug. 20: info. 30: warn. 40: error.
        """
        if verbose is None:
            wrapper_returns = func(*args, **kwargs)
        else:
            previous_loglevel = logger.level
            loglevel(verbose)
            wrapper_returns = func(*args, **kwargs)
            loglevel(previous_loglevel)
        return wrapper_returns
    return loglevel_decorator
