import numpy as np


def group_tsplit(
    X: np.ndarray, groups: np.ndarray, order: int = 1, lag: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_pres, X_post, split_groups = [], [], []

    for group, mask in iter_groups(groups):
        assert is_contiguous(mask), "groups are not temporally contiguous"
        Xi = X[mask]
        Xi_pres, Xi_post = tsplit(Xi, order=order, lag=lag)
        X_pres.append(Xi_pres)
        X_post.append(Xi_post)
        split_groups.append(np.full(len(Xi_pres), group))

    X_pres = np.concatenate(X_pres, axis=0)
    X_post = np.concatenate(X_post, axis=0)
    split_groups = np.concatenate(split_groups, axis=0)
    return X_pres, X_post, split_groups


def iter_groups(groups: np.ndarray):
    uniq_groups, uniq_index = np.unique(groups, return_index=True)
    group_order = np.argsort(uniq_index)
    for group in uniq_groups[group_order]:
        mask = groups == group
        yield group, mask


def is_contiguous(mask: np.ndarray) -> bool:
    """Check if a 1d boolean mask is contiguous."""
    (indices,) = mask.nonzero()
    if len(indices) < 2:
        return True
    max_diff = np.max(np.diff(indices))
    return max_diff == 1


def tsplit(
    X: np.ndarray,
    order: int = 1,
    lag: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    assert len(X) > 2 * (lag + order), f"timeseries too short for {order=} {lag=}"

    # X_pres: (n_timepoints, order, n_features)
    length = len(X) - lag - order + 1
    X_pres = [X[start : start + length] for start in range(order)]
    X_pres = np.stack(X_pres, axis=1)

    # X_post: (n_timepoints, n_features)
    start = order - 1 + lag
    X_post = X[start : start + length]
    return X_pres, X_post
