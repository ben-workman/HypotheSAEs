"""Methods for selecting relevant neurons based on target variables."""

import time
import numpy as np
from typing import List, Optional, Callable, Tuple
from sklearn.linear_model import Lasso, LogisticRegression, lasso_path, logistic_regression_path
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


def select_neurons_lasso(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    classification: bool = False,
    alpha: Optional[float] = None,
    max_iter: int = 1000,
    verbose: bool = False,
    group_ids: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using an L1-regularized linear model (LASSO)
    Returns (indices, coefficients) tuple
    """
    if group_ids is not None:
        unique_ids = np.unique(group_ids)
        aggregated_activations = []
        aggregated_target = []
        for uid in unique_ids:
            indices = np.where(group_ids == uid)[0]
            aggregated_activations.append(activations[indices].mean(axis=0))
            aggregated_target.append(target[indices].mean())
        activations = np.vstack(aggregated_activations)
        target = np.array(aggregated_target)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)

    if alpha is not None:
        if verbose:
            print(f"Using provided alpha: {alpha:.2e}")
        start_time = time.time()
        if classification:
            model = LogisticRegression(
                penalty='l1', 
                solver='liblinear',
                C=1/alpha,
                max_iter=max_iter
            )
        else:
            model = Lasso(alpha=alpha, max_iter=max_iter)
        model.fit(X_scaled, target)
        coef = model.coef_.flatten()
        if verbose:
            print(f"Fitting took {time.time() - start_time:.2f}s")
    else:
        alpha_low, alpha_high = 1e-6, 1e4
        if verbose:
            print(f"{'LASSO iteration':>8} {'L1 Alpha':>10} {'# Features':>10} {'Time (s)':>10}")
            print("-" * 40)
        total_start_time = time.time()
        for iteration in range(20):
            iter_start_time = time.time()
            alpha = np.sqrt(alpha_low * alpha_high)
            if classification:
                model = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=1/alpha,
                    max_iter=max_iter
                )
            else:
                model = Lasso(alpha=alpha, max_iter=max_iter)
            model.fit(X_scaled, target)
            coef = model.coef_.flatten()
            n_nonzero = np.sum(coef != 0)
            iter_time = time.time() - iter_start_time
            if verbose:
                print(f"{iteration:8d} {alpha:10.2e} {n_nonzero:10d} {iter_time:10.2f}")
            if n_nonzero == n_select:
                break
            elif n_nonzero < n_select:
                alpha_high = alpha
            else:
                alpha_low = alpha
        total_time = time.time() - total_start_time
        if verbose and n_nonzero == n_select:
            print(f"\nFound alpha={alpha:.2e} yielding exactly {n_select} features")
            print(f"Total search time: {total_time:.2f}s")
        if n_nonzero != n_select:
            print(f"Warning: Search ended with {n_nonzero} features (target: {n_select})")

    sorted_indices = np.argsort(-np.abs(coef))[:n_select]
    selected_coefs = coef[sorted_indices]  
    return sorted_indices.tolist(), selected_coefs.tolist()


def select_neurons_correlation(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    group_ids: Optional[np.ndarray] = None
) -> Tuple[List[int], List[float]]:
    """
    Select neurons with the highest correlation with the target
    """
    if group_ids is not None:
        unique_ids = np.unique(group_ids)
        aggregated_activations = []
        aggregated_target = []
        for uid in unique_ids:
            indices = np.where(group_ids == uid)[0]
            aggregated_activations.append(activations[indices].mean(axis=0))
            aggregated_target.append(target[indices].mean())
        activations = np.vstack(aggregated_activations)
        target = np.array(aggregated_target)

    correlations = np.array([
        pearsonr(activations[:, i], target)[0]
        for i in range(activations.shape[1])
    ])

    sorted_indices = np.argsort(-np.abs(correlations))[:n_select]
    selected_correlations = correlations[sorted_indices]

    return sorted_indices.tolist(), selected_correlations.tolist()


def select_neurons_separation_score(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    n_top_activating: int = 100,
    n_zero_activating: Optional[int] = None
) -> Tuple[List[int], List[float]]:
    """
    Select neurons based on separation between top activations and zero activations
    """
    scores = []
    for i in range(activations.shape[1]):
        neuron_acts = activations[:, i]
        sorted_indices = np.argsort(-neuron_acts)
        top_mean = np.mean(target[sorted_indices[:n_top_activating]])
        zero_indices = neuron_acts == 0
        if n_zero_activating is not None:
            zero_idx = np.where(zero_indices)[0]
            rand_zero_idx = np.random.choice(zero_idx, size=n_zero_activating, replace=False)
            zero_mean = np.mean(target[rand_zero_idx])
        else:
            zero_mean = np.mean(target[zero_indices])
        scores.append(top_mean - zero_mean)
    scores = np.array(scores)
    sorted_indices = np.argsort(-np.abs(scores))[:n_select]
    selected_scores = scores[sorted_indices]
    return sorted_indices.tolist(), selected_scores.tolist()


def select_neurons_custom(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    metric_fn: Callable[[np.ndarray, np.ndarray], float]
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using a custom metric function
    """
    scores = np.array([
        metric_fn(activations[:, i], target)
        for i in range(activations.shape[1])
    ])
    sorted_indices = np.argsort(scores)[-n_select:]
    selected_scores = scores[sorted_indices]
    return sorted_indices.tolist(), selected_scores.tolist()


def select_neurons_presence_correlation(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    group_ids: Optional[np.ndarray] = None
) -> Tuple[List[int], List[float]]:
    """
    Select neurons by presence correlation: binary presence vs zero activation
    aggregate by group and compute correlation with target
    """
    if group_ids is not None:
        unique_ids = np.unique(group_ids)
        agg_presence = []
        agg_target = []
        for uid in unique_ids:
            idx = np.where(group_ids == uid)[0]
            agg_presence.append((activations[idx] > 0).mean(axis=0))
            agg_target.append(target[idx].mean())
        presence = np.vstack(agg_presence)
        target_group = np.array(agg_target)
    else:
        presence = (activations > 0).astype(float)
        target_group = target

    correlations = np.array([
        pearsonr(presence[:, i], target_group)[0]
        for i in range(presence.shape[1])
    ])
    sorted_indices = np.argsort(-np.abs(correlations))[:n_select]
    return sorted_indices.tolist(), correlations[sorted_indices].tolist()

def select_neurons_stability(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    classification: bool = False,
    group_ids: Optional[np.ndarray] = None,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.5,
    alphas: Optional[np.ndarray] = None,
    Cs: Optional[np.ndarray] = None,
    pi_threshold: Optional[float] = None,
    max_iter: int = 1000,
    random_state: Optional[int] = 0,
    scale: bool = True,
    verbose: bool = False,
) -> Tuple[List[int], List[float]]:
    X = activations
    y = target
    if group_ids is not None:
        u = np.unique(group_ids)
        X = np.vstack([activations[group_ids == g].mean(axis=0) for g in u])
        y = np.array([target[group_ids == g].mean() for g in u])
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    rng = np.random.RandomState(random_state)
    n, p = X.shape
    m = max(1, int(np.floor(sample_fraction * n)))
    counts = np.zeros(p, dtype=np.int32)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=m, replace=False)
        Xb = X[idx]
        yb = y[idx]
        if classification:
            if Cs is None:
                Cs = np.logspace(-2, 2, 20)
            _, coefs, _ = logistic_regression_path(Xb, yb, Cs=Cs, penalty="l1", solver="liblinear", max_iter=max_iter)
            if coefs.ndim == 3:
                sel = (np.abs(coefs).sum(axis=0) > 0).any(axis=1)
            else:
                sel = (np.abs(coefs) > 0).any(axis=1)
        else:
            _, _, coefs = lasso_path(Xb, yb, alphas=alphas, max_iter=max_iter)
            sel = (np.abs(coefs) > 0).any(axis=1)
        counts += sel.astype(np.int32)
        if verbose and (b + 1) % max(1, n_bootstrap // 10) == 0:
            pass
    stability = counts.astype(float) / float(n_bootstrap)
    if pi_threshold is not None:
        idxs = np.where(stability >= pi_threshold)[0]
        scores = stability[idxs]
        return idxs.tolist(), scores.tolist()
    order = np.argsort(-stability)[:n_select]
    return order.tolist(), stability[order].tolist()

def select_neurons(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    method: str = "lasso",
    classification: bool = False,
    **kwargs
) -> Tuple[List[int], List[float]]:
    if classification and len(np.unique(target)) > 2:
        raise ValueError("classification=True, but the target variable has more than 2 classes")
    if method == "lasso":
        return select_neurons_lasso(
            activations=activations,
            target=target,
            n_select=n_select,
            classification=classification,
            **kwargs
        )
    elif method == "correlation":
        return select_neurons_correlation(
            activations=activations,
            target=target,
            n_select=n_select,
            **kwargs
        )
    elif method == "separation_score":
        return select_neurons_separation_score(
            activations=activations,
            target=target,
            n_select=n_select,
            **kwargs
        )
    elif method == "presence_correlation":
        return select_neurons_presence_correlation(
            activations=activations,
            target=target,
            n_select=n_select,
            group_ids=kwargs.get("group_ids", None)
        )
    elif method == "custom":
        if "metric_fn" not in kwargs:
            raise ValueError("Must provide metric_fn for custom method")
        return select_neurons_custom(
            activations=activations,
            target=target,
            n_select=n_select,
            **kwargs
        )
    elif method == "stability":
        return select_neurons_stability(
            activations=activations,
            target=target,
            n_select=n_select,
            classification=classification,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")
