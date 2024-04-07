from typing import Optional

import numpy as np
import pandas as pd


def shuffle_meta_labels(
    dataset: pd.DataFrame, target_col: str, seed: Optional[int] = 0
) -> pd.DataFrame:
    """shuffles labels or values within a single selected column

    Parameters
    ----------
    dataset : pd.DataFrame
        dataframe containing the dataset

    target_col : str
        Column to select in order to conduct the shuffling

    seed : int
        setting random seed

    Returns
    -------
    pd.DataFrame
        shuffled dataset

    Raises
    ------
    TypeError
        raised if incorrect types are provided
    """
    # setting seed
    np.random.seed(seed)

    # type checking
    if not isinstance(target_col, str):
        raise TypeError("'target_col' must be a string type")
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("'dataset' must be a pandas dataframe")

    # selecting column, shuffle values within column, add to dataframe
    dataset[target_col] = np.random.permutation(dataset[target_col].values)
    return dataset


def shuffle_features(feature_vals: np.array, seed: Optional[int] = 0) -> np.array:
    """Shuffles all values within feature space

    Parameters
    ----------
    feature_vals : np.array
        Values to be shuffled.

    seed : Optional[int]
        setting random seed

    Returns
    -------
    np.array
        Returns shuffled values within the feature space

    Raises
    ------
    TypeError
        Raised if a numpy array is not provided
    """
    # setting seed
    np.random.seed(seed)

    # shuffle given array
    if not isinstance(feature_vals, np.ndarray):
        raise TypeError("'feature_vals' must be a numpy array")
    if feature_vals.ndim != 2:
        raise TypeError("'feature_vals' must be a 2x2 matrix")

    # shuffling feature space
    n_cols = feature_vals.shape[1]
    for col_idx in range(0, n_cols):
        # selecting column, shuffle, and update:
        feature_vals[:, col_idx] = np.random.permutation(feature_vals[:, col_idx])

    return feature_vals
