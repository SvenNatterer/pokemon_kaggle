import json
from dataclasses import MISSING


def to_dataclass(dic: dict, cls: type):
    """
    Convert a dictionary to a dataclass instance recursively.

    This function matches keys in the dictionary to the fields of the given dataclass `cls`,
    and recursively constructs nested dataclass instances if needed.

    Args:
        dic (dict): The source dictionary.
        cls (type): The target dataclass type.

    Returns:
        Any: An instance of the target dataclass, or None if input is None.
    """
    if dic is None:
        return None

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    d = {}

    for key, value in dic.items():
        if key in field_types:
            if isinstance(value, dict):
                c = field_types[key]
                if hasattr(c, "__args__"):
                    c = c.__args__[0]
                d[key] = to_dataclass(value, c)
            elif isinstance(value, list):
                c = field_types[key].__args__[0]
                if hasattr(c, "__args__"):
                    c = c.__args__[0]
                    if hasattr(c, "__args__"):
                        c = c.__args__[0]
                if not hasattr(c, "__dataclass_fields__"):
                    d[key] = value
                else:
                    d[key] = [to_dataclass(v, c) if isinstance(v, dict) else v for v in value]
            else:
                d[key] = value

    for f_name, field in cls.__dataclass_fields__.items():
        if f_name not in d:
            if field.default is not MISSING or field.default_factory is not MISSING:
                continue
            origin = getattr(field.type, "__origin__", None)
            if origin is list or field.type is list:
                d[f_name] = []
            elif field.type in (int, float):
                d[f_name] = 0
            elif field.type is bool:
                d[f_name] = False
            elif field.type is str:
                d[f_name] = ""
            else:
                d[f_name] = None

    return cls(**d)


def json_to_dataclass(bs: bytes, cls: type):
    """
    Convert a JSON byte string to a dataclass instance.

    This function decodes the JSON bytes and uses `to_dataclass()` to
    recursively convert the resulting dictionary to a dataclass instance.

    Args:
        bs (bytes): JSON data in bytes.
        cls (type): The target dataclass type.

    Returns:
        Any: An instance of the target dataclass.
    """
    js = bs.decode()
    dic = json.loads(js)
    return to_dataclass(dic, cls)
