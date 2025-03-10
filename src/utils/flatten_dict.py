from typing import *

def flatten_dict(
        d: dict,
        parent_key: Optional[str] = None,
        sep: str = '.'
) -> dict:
    if parent_key is None:
        parent_key = ''
    items = []
    for key, val in d.items():
        new_key = f'{parent_key}{sep}{key}'
        if isinstance(val, dict):
            items.extend(flatten_dict(val, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, val))
    return dict(items)

def unflatten_dict(
        d: dict,
        sep: str = '.'
):
    unflattened_dict = {}
    for key, val in d.items():
        parts = key.split(sep)
        current_level = unflattened_dict
        for part in parts[:-1]:
            if not(part in current_level):
                current_level[part] = {}
            current_level = current_level[part]
        current_level[parts[:-1]] = val
    return unflattened_dict