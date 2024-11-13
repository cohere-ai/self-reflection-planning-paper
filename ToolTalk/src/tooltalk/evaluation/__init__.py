"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""


from typing import Union


def safely_divide(a: Union[int, float], b: int) -> float:
    return a / b if b != 0 else (float("nan") if a == 0 else float("inf"))
