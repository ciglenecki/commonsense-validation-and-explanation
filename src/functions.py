import inspect
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


class SocketConcatenator:
    """Merges multiple sockets (files) into one enabling writing to multiple sockets/files at
    once."""

    def __init__(self, *files):
        self.files = files
        self.encoding = "utf-8"

    def write(self, obj):
        for f in self.files:
            f.write(obj)
        self.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def stdout_to_file(file: Path):
    """Pipes standard input to standard input and to a new file."""
    print("Standard output piped to file:")
    f = open(Path(file), "w", encoding="utf-8")
    sys.stdout = SocketConcatenator(sys.stdout, f)
    sys.stderr = SocketConcatenator(sys.stderr, f)


def reset_sockets():
    """Reset stdout and stderr sockets."""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def to_yaml(data):
    return yaml.dump(data, allow_unicode=True, default_flow_style=False)


def get_timestamp(format="%m-%d-%H-%M-%S"):
    return datetime.today().strftime(format)


def one_hot_encode(index: int, size: int):
    zeros = np.zeros(shape=size)
    zeros[index] = 1
    return zeros


def add_prefix_to_keys(
    dict: dict, prefix: str, filter_fn: callable = lambda x: False
) -> dict:
    """
    Example:
        dict = {"a": 1, "b": 2}
        prefix = "text_"
        returns {"text_a": 1, "text_b": 2}

    Example:
        dict = {"abra": 1, "abrakadabra": 2, "nothing": 3}
        prefix = "text_"
        filter = lambda x: x.startswith("abra")
        returns {"text_abra": 1, "text_abrakadabra": 2, "nothing": 3}
    """
    return {(k if filter_fn(k) else prefix + k): v for k, v in dict.items()}


def flatten(list):
    """
    Example:
        list = [[1, 2], [3, 4]]
        returns [1, 2, 3, 4]
    """
    return [item for sublist in list for item in sublist]


def save_yaml(data: object, path: Path):
    print("Saving yaml file:", str(path))
    with open(path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def save_json(data: object, path: Path):
    print("Saving json file:", str(path))
    with open(path, "w") as outfile:
        json.dump(data, outfile)


def all_args(cls):
    """
    A decorator function that checks if all arguments are provided by the user when instantiating an object.
    Args:
        cls: The class to be wrapped.
    Returns:
        The wrapped class with argument checking.
    Raises:
        TypeError: If any of the required arguments are missing.
    Notes:
        - Required arguments are those that do not have a default value
    """

    def wrapper(*args, **kwargs):
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        user_args = {**dict(zip(params.keys(), args)), **kwargs}
        missing_args = set(params.keys()) - set(user_args.keys())
        if missing_args:
            missing_args_list = ", ".join(missing_args)
            raise TypeError(f"Missing required argument(s): {missing_args_list}")

        return cls(*args, **kwargs)

    return wrapper


def isfloat(x: str):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(x: str):
    try:
        a = float(x)
        b = int(x)
    except (TypeError, ValueError):
        return False
    else:
        return a == b


def parse_kwargs(kwargs_strs: list[str], list_sep=",", key_value_sep="="):
    """
    Example:
        kwargs_str = stretch_factors=0.8,1.2 freq_mask_param=30
        returns {"stretch_factors": [0.8, 1.2], "freq_mask_param": 30}

    Args:
        kwargs_str: _description_
        list_sep: _description_..
        arg_sep: _description_..
    """
    if isinstance(kwargs_strs, str):
        kwargs_strs = [kwargs_strs]

    def parse_value(value: str):
        if isint(value):
            return int(value)
        if isfloat(value):
            return float(value)
        return value

    kwargs = {}
    for key_value in kwargs_strs:
        _kv = key_value.split(key_value_sep)
        assert (
            len(_kv) == 2
        ), f"Exactly one `{key_value_sep}` should appear in {key_value}"
        key, value = _kv
        value = [parse_value(v) for v in value.split(list_sep)]
        value = value if len(value) > 1 else value[0]
        kwargs[key] = value
    return kwargs


nato_alphabet = [
    "Alpha",
    "Bravo",
    "Charlie",
    "Delta",
    "Echo",
    "Foxtrot",
    "Golf",
    "Hotel",
    "India",
    "Juliett",
    "Kilo",
    "Lima",
    "Mike",
    "November",
    "Oscar",
    "Papa",
    "Quebec",
    "Romeo",
    "Sierra",
    "Tango",
    "Uniform",
    "Victor",
    "Whiskey",
    "X-ray",
    "Yankee",
    "Zulu",
    "Sinisa",
    "Jan",
    "Alan",
]

adjectives = [
    "agile",
    "ample",
    "avid",
    "awed",
    "best",
    "bonny",
    "brave",
    "brisk",
    "calm",
    "clean",
    "clear",
    "comfy",
    "cool",
    "cozy",
    "crisp",
    "cute",
    "deft",
    "eager",
    "eased",
    "easy",
    "elite",
    "fair",
    "famed",
    "fancy",
    "fast",
    "fiery",
    "fine",
    "finer",
    "fond",
    "free",
    "freed",
    "fresh",
    "fun",
    "funny",
    "glad",
    "gold",
    "good",
    "grand",
    "great",
    "hale",
    "handy",
    "happy",
    "hardy",
    "holy",
    "hot",
    "ideal",
    "jolly",
    "keen",
    "lean",
    "like",
    "liked",
    "loved",
    "loyal",
    "lucid",
    "lucky",
    "lush",
    "magic",
    "merry",
    "neat",
    "nice",
    "nicer",
    "noble",
    "plush",
    "prize",
    "proud",
    "pure",
    "quiet",
    "rapid",
    "rapt",
    "ready",
    "regal",
    "rich",
    "right",
    "roomy",
    "rosy",
    "safe",
    "sane",
    "sexy",
    "sharp",
    "shiny",
    "sleek",
    "slick",
    "smart",
    "soft",
    "solid",
    "suave",
    "super",
    "swank",
    "sweet",
    "swift",
    "tidy",
    "top",
    "tough",
    "vivid",
    "warm",
    "well",
    "wise",
    "witty",
    "worth",
    "young",
]


def random_codeword():
    """Return e.g.:

    YoungAlpha, WiseZulu
    """
    return f"{random.choice(adjectives).capitalize()}{random.choice(nato_alphabet)}"
