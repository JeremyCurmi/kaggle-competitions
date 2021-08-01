import re
import time
from typing import List


def current_timestamp(time_format: str = '%m-%d-%Y_%H-%M-%S') -> str:
    """

    :param time_format: timestamp format
    :return: timestamp as a string
    """
    t = time.localtime()
    return time.strftime(time_format, t)


def snake_case_transformation(list_of_strings: List[str]) -> List[str]:
    """

    :param list_of_strings: list of strings
    :return: list of strings, each of which is in snake case
    """
    sc_list = []
    for string in list_of_strings:
        split_string = [s.lower() for s in re.sub(r"([A-Z])", r" \1", string).split()]
        sc_list.append('_'.join(split_string))
    return sc_list