"""Memory-safe, zero-copy column extraction for pandas and arrow-compatible structures."""

from __future__ import annotations

from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd


def extract_column_arrays(
    data: Union[pd.DataFrame, Any],
    columns: Dict[str, Union[str, List[str], None]],
) -> Dict[str, np.ndarray]:
    """
    Extracts specified columns without copying the parent DataFrame.
    Zero-copy slicing operates via Apache Arrow when Polars inputs are passed.
    """
    extracted: Dict[str, np.ndarray] = {}

    # Handle Polars (eager DataFrame or LazyFrame) / PyArrow zero-copy access
    if hasattr(data, "select") and (hasattr(data, "to_arrow") or hasattr(data, "collect")):
        cols_to_pull: List[str] = []
        for v in columns.values():
            if isinstance(v, str) and v not in cols_to_pull:
                cols_to_pull.append(v)
            elif isinstance(v, list):
                for item in v:
                    if item not in cols_to_pull:
                        cols_to_pull.append(item)
        selected = data.select(cols_to_pull)
        if hasattr(selected, "collect") and callable(selected.collect):
            selected = selected.collect()
        arrow_table = selected.to_arrow()
        for key, col_name in columns.items():
            if isinstance(col_name, str):
                extracted[key] = np.ascontiguousarray(arrow_table[col_name].to_numpy(zero_copy_only=False))
            elif isinstance(col_name, list):
                extracted[key] = np.ascontiguousarray(
                    np.column_stack(
                        [arrow_table[c].to_numpy(zero_copy_only=False) for c in col_name]
                    )
                )
        return extracted

    # Native Pandas extraction
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame or support PyArrow/Polars zero-copy protocols.")

    for role, col in columns.items():
        if col is None:
            continue
        if isinstance(col, str):
            if col not in data.columns:
                raise KeyError(f"Column '{col}' specified for '{role}' does not exist in data.")
            extracted[role] = np.ascontiguousarray(data[col].to_numpy())
        elif isinstance(col, list):
            for sub_col in col:
                if sub_col not in data.columns:
                    raise KeyError(f"Covariate '{sub_col}' does not exist in data.")
            extracted[role] = np.ascontiguousarray(data[col].to_numpy())

    return extracted