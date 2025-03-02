from dataclasses import dataclass, fields, asdict, is_dataclass
from typing import Any, Optional, Union, get_type_hints, get_origin, get_args
from itertools import product


@dataclass
class TuningMetadata:
    choices: Optional[list[Any]] = None
    value_range: Optional[Union[tuple[int, int, int], tuple[float, float]]] = None
    length_boundary: Optional[tuple[int, int]] = None


def _is_union_list_type(field_type: Any) -> Optional[Any]:
    """
    Checks if a field type is `Union[X, list[X]]`.

    Args:
        field_type: The annotated type of a field.

    Returns:
        The base type X if the field is Union[X, list[X]], otherwise None.
    """
    if get_origin(field_type) is Union:
        args = get_args(field_type)
        if len(args) == 2 and list in map(get_origin, args):
            base, lst = args if get_origin(args[1]) is list else args[::-1]
            if base == get_args(lst)[0]:
                return base
    return None


def generate_all_configurations(config_instance: Any):
    """
    Recursively generate all candidate configurations for a dataclass instance by
    exhaustively exploring tunable fields according to their metadata.

    Field handling rules:
      - For `Union[X, list[X]]` with a provided length_boundary:
          Generate candidate lists of allowed lengths using `choices` (for `str`) or
          `value_range` (for `int`). Without a length_boundary, treat the field as scalar `X`.
      - For list fields:
          Generate candidate lists using either `choices` (e.g. for `list[str]`) or
          `value_range` (for `list[int/float]`). If a length_boundary is provided,
          generate lists for all allowed lengths; otherwise, use the current length.
      - For scalar numeric fields (`int/float`):
          Generate candidate values using `value_range`.
      - For scalar string fields:
          Generate candidate values using `choices`.
      - For fields without tuning metadata, retain the current value.

    Yields:
        New instances of the dataclass for every combination of candidate values.
    """
    if not is_dataclass(config_instance) or not hasattr(
        config_instance, "tuning_metadata"
    ):
        yield config_instance
        return

    # Dictionary to store possible values for each field
    candidate_dict = {}
    type_hints = get_type_hints(config_instance.__class__)

    tuning_metadata: dict[str, TuningMetadata] = config_instance.tuning_metadata
    for f in fields(config_instance):
        # Skip the tuning_metadata field itself.
        if f.name == "tuning_metadata":
            continue

        value = getattr(config_instance, f.name)

        meta = tuning_metadata.get(f.name, {})
        meta = asdict(meta) if is_dataclass(meta) else meta

        ftype = type_hints.get(f.name)
        candidates = []

        # Recursively generate candidates if the field is a nested dataclass.
        if is_dataclass(value):
            candidates = list(generate_all_configurations(value))
        # If no metadata, retain the current value.
        elif not meta:
            candidates = [value]
        else:
            # Case 1: Field is Union[X, list[X]]
            union_elem = _is_union_list_type(ftype)
            if union_elem is not None:
                # Generate candidate values for type X.
                if union_elem is str:
                    if not meta["choices"]:
                        # No "choices" -> retain the current value.
                        candidate_dict[f.name] = [value]
                        continue
                    possible_vals = meta["choices"]
                elif union_elem is int:
                    if not meta["value_range"]:
                        # No "value_range" -> retain the current value.
                        candidate_dict[f.name] = [value]
                        continue

                    boundaries = meta["value_range"]
                    if len(boundaries) == 3 and all(
                        isinstance(x, int) for x in boundaries
                    ):
                        min_val, max_val, step = boundaries
                        possible_vals = list(range(min_val, max_val + 1, step))
                    else:
                        # "value_range" is specified for float -> retain the current value.
                        candidate_dict[f.name] = [value]
                        continue
                else:
                    candidate_dict[f.name] = [value]
                    continue

                # Generate lists for every allowed length.
                length_boundary = meta["length_boundary"]
                if length_boundary:
                    min_len, max_len = length_boundary

                    for l in range(min_len, max_len + 1):
                        for combo in product(possible_vals, repeat=l):
                            candidates.append(list(combo))
                # Without length_boundary, treat as a scalar.
                else:
                    candidates = possible_vals

            # Case 2: Field is a list (but not a Union)
            elif get_origin(ftype) is list:
                underlying_type = get_args(ftype)[0]

                if underlying_type is str:
                    if not meta["choices"]:
                        # No "choices" -> retain the current value.
                        candidate_dict[f.name] = [value]
                        continue
                    possible_vals = meta["choices"]
                elif underlying_type is int:
                    if not meta["value_range"]:
                        # No "value_range" -> retain the current value.
                        candidate_dict[f.name] = [value]
                        continue

                    boundaries = meta["value_range"]
                    if len(boundaries) == 3 and all(
                        isinstance(x, int) for x in boundaries
                    ):
                        min_val, max_val, step = boundaries
                        possible_vals = list(range(min_val, max_val + 1, step))
                    else:
                        # "value_range" is specified for float -> retain the current value.
                        candidate_dict[f.name] = [value]
                        continue
                else:
                    candidate_dict[f.name] = [value]
                    continue

                # Determine length boundaries; default to current length if not provided.
                length_boundary = meta["length_boundary"]
                if length_boundary:
                    min_len, max_len = length_boundary
                else:
                    min_len, max_len = (len(value), len(value))

                # Generate lists for every allowed length.
                for l in range(min_len, max_len + 1):
                    for combo in product(possible_vals, repeat=l):
                        candidates.append(list(combo))

            # Case 3: Scalar field (int, float, str, etc.)
            else:
                if ftype is str:
                    if not meta["choices"]:
                        # No "choices" -> retain the current value.
                        candidate_dict[f.name] = [value]
                        continue
                    candidates = meta["choices"]
                elif ftype is int:
                    if not meta["value_range"]:
                        # No "value_range" -> retain the current value.
                        candidate_dict[f.name] = [value]
                        continue

                    boundaries = meta["value_range"]
                    if len(boundaries) == 3 and all(
                        isinstance(x, int) for x in boundaries
                    ):
                        min_val, max_val, step = boundaries
                        candidates = list(range(min_val, max_val + 1, step))
                    else:
                        # "value_range" is specified for float -> retain the current value.
                        candidate_dict[f.name] = [value]
                        continue
                else:
                    candidate_dict[f.name] = [value]
                    continue

        candidate_dict[f.name] = candidates

    # Generate the Cartesian product of candidate values for all fields.
    keys = list(candidate_dict.keys())
    for comb in product(*(candidate_dict[k] for k in keys)):
        # Build a candidate instance from the product.
        yield type(config_instance)(**dict(zip(keys, comb)))


def generate_random_configuration():
    pass
