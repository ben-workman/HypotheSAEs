"""Methods for selecting relevant neurons based on target variables."""

import time
import numpy as np
from typing import List, Optional, Callable, Tuple
from sklearn.linear_model import Lasso, LogisticRegression, lasso_path, LinearRegression 
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

def stability_select_lasso(
    X, y, *,
    B=200,
    frac=0.5,
    q=20,
    pi_thr=0.7,
    random_state=123,
    n_alphas=100,
    standardize=True,
    groups=None,
    group_subsample=False,
    cpps=False,
    jitter_range=(1.0, 1.0),
    return_refit=False,
    return_diagnostics=False
):
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    y_orig = y.copy()

    X_raw = X.copy()
    if standardize:
        X = StandardScaler().fit_transform(X)
    y = y - y.mean()

    n, p = X.shape
    alphas_fixed, coefs_full, _ = lasso_path(X, y, n_alphas=n_alphas, copy_X=False)

    if groups is not None and group_subsample:
        groups = np.asarray(groups)
        uniq_groups = np.unique(groups)
        def sample_indices_k_rows(k):
            g_order = rng.permutation(uniq_groups)
            chosen_rows = []
            for g in g_order:
                chosen_rows.append(np.flatnonzero(groups == g))
                if sum(len(ix) for ix in chosen_rows) >= k:
                    break
            idx_pool = np.concatenate(chosen_rows) if chosen_rows else rng.choice(n, size=k, replace=False)
            if idx_pool.size > k:
                idx_pool = rng.choice(idx_pool, size=k, replace=False)
            return idx_pool
    else:
        def sample_indices_k_rows(k):
            return rng.choice(n, size=k, replace=False)

    def pick_take_n():
        return max(1, (n // 2) if cpps else int(np.ceil(frac * n)))

    def run_once(idx):
        Xb = X[idx]
        yb = y[idx]
        if jitter_range != (1.0, 1.0):
            w = rng.uniform(jitter_range[0], jitter_range[1], size=p)
            Xb_tilt = Xb / w[np.newaxis, :]
        else:
            Xb_tilt = Xb
        alphas, coefs, _ = lasso_path(Xb_tilt, yb, alphas=alphas_fixed, copy_X=False)
        support_path = (coefs != 0)
        sizes = support_path.sum(axis=0)
        diffs = np.abs(sizes - q)
        j_star = int(np.argmin(diffs))
        if np.any(sizes <= q):
            j_star = int(np.argmin(np.where(sizes <= q, diffs, np.inf)))
        support = support_path[:, j_star].astype(float)
        return support, float(sizes[j_star]), int(j_star)

    sel_counts = np.zeros(p, dtype=float)
    support_sizes = []
    chosen_j_hist = []
    total_fits = 0

    for _ in range(B):
        take_n = pick_take_n()
        idx = sample_indices_k_rows(take_n)
        s, sz, j = run_once(idx)
        sel_counts += s
        support_sizes.append(sz)
        chosen_j_hist.append(j)
        total_fits += 1
        if cpps:
            comp = np.setdiff1d(np.arange(n), idx, assume_unique=False)
            if comp.size == 0:
                comp = sample_indices_k_rows(take_n)
            s2, sz2, j2 = run_once(comp)
            sel_counts += s2
            support_sizes.append(sz2)
            chosen_j_hist.append(j2)
            total_fits += 1

    pi = sel_counts / max(1, total_fits)
    selected = np.where(pi >= pi_thr)[0]
    if selected.size == 0:
        selected = np.array([int(np.argmax(pi))])
    order = np.argsort(-pi[selected])
    selected = selected[order]
    selected_pi = pi[selected]

    outs = [selected.tolist(), selected_pi.tolist(), pi]

    if return_refit:
        coef_full = np.zeros(p, dtype=float)
        if selected.size > 0:
            ref = LinearRegression()
            ref.fit(X_raw[:, selected], y_orig)
            coef_full[selected] = ref.coef_
            intercept = ref.intercept_
        else:
            intercept = float(np.mean(y_orig))
        outs.extend([coef_full, intercept])

    if return_diagnostics:
        avg_support = float(np.mean(support_sizes)) if support_sizes else 0.0
        mb_bound = None
        if pi_thr > 0.5:
            mb_bound = (q**2) / ((2.0 * pi_thr - 1.0) * p)
        diagnostics = {
            "avg_support_per_fit": avg_support,
            "alphas": alphas_fixed.tolist(),
            "chosen_lambda_index_hist": chosen_j_hist,
            "total_fits": int(total_fits),
            "mb_expected_false_positives_bound": mb_bound
        }
        outs.append(diagnostics)

    return tuple(outs)

def select_neurons_stability(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int = 0,
    *,
    group_ids: Optional[np.ndarray] = None,
    n_bootstrap: int = 200,
    sample_fraction: float = 0.5,
    q: Optional[int] = None,
    pi_threshold: float = 0.7,
    random_state: Optional[int] = None,
    n_alphas: int = 100,
    standardize: bool = True,
    group_subsample: bool = True,
    cpps: bool = False,
    jitter_range: tuple = (1.0, 1.0),
    return_refit: bool = False,
    return_diagnostics: bool = False,
    return_full_pi: bool = False,
):
    """
    Select neurons using stability selection with LASSO.
    
    Args:
        activations: (n_samples, n_features) array
        target: (n_samples,) target variable
        n_select: Not used (kept for API consistency)
        group_ids: Optional group IDs for group-level subsampling
        n_bootstrap: Number of bootstrap iterations
        sample_fraction: Fraction of samples per bootstrap
        q: Target number of features per fit
        pi_threshold: Selection probability threshold
        random_state: Random seed
        n_alphas: Number of regularization alphas to try
        standardize: Whether to standardize features
        group_subsample: Whether to subsample by group
        cpps: Whether to use complementary pairs stability selection
        jitter_range: Jitter range for randomized LASSO
        return_refit: Whether to return refitted coefficients
        return_diagnostics: Whether to return diagnostics dict
        return_full_pi: Whether to return full pi array for all features
    
    Returns:
        By default: (selected_indices, selected_pi)
        If return_full_pi=True: adds pi_full array
        If return_refit=True: adds (coef_full, intercept)
        If return_diagnostics=True: adds diagnostics dict
    """
    p = activations.shape[1]
    if q is None:
        q = max(5, min(50, p // 10))
    results = stability_select_lasso(
        activations, target,
        B=n_bootstrap,
        frac=sample_fraction,
        q=q,
        pi_thr=pi_threshold,
        random_state=random_state if random_state is not None else 0,
        n_alphas=n_alphas,
        standardize=standardize,
        groups=group_ids,
        group_subsample=group_subsample,
        cpps=cpps,
        jitter_range=jitter_range,
        return_refit=return_refit,
        return_diagnostics=return_diagnostics
    )
    # stability_select_lasso always returns at least (selected, selected_pi, pi_full)
    selected, selected_pi, pi_full = results[:3]
    
    # Build output tuple based on what was requested
    outs = [selected, selected_pi]
    if return_full_pi:
        outs.append(pi_full)
    
    idx = 3
    if return_refit:
        outs.extend([results[idx], results[idx + 1]])
        idx += 2
    if return_diagnostics:
        outs.append(results[idx])
    
    return tuple(outs) if len(outs) > 2 else (outs[0], outs[1])

def select_neurons(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    method: str = "lasso",
    classification: bool = False,
    **kwargs
) -> Tuple[List[int], List[float]]:
    if classification and len(np.unique(target)) > 2 and method != "stability":
        raise ValueError("classification=True, but the target variable has more than 2 classes")
    if method == "lasso":
        return select_neurons_lasso(activations=activations, target=target, n_select=n_select, classification=classification, **kwargs)
    elif method == "correlation":
        return select_neurons_correlation(activations=activations, target=target, n_select=n_select, **kwargs)
    elif method == "separation_score":
        return select_neurons_separation_score(activations=activations, target=target, n_select=n_select, **kwargs)
    elif method == "presence_correlation":
        return select_neurons_presence_correlation(activations=activations, target=target, n_select=n_select, group_ids=kwargs.get("group_ids", None))
    elif method == "custom":
        if "metric_fn" not in kwargs:
            raise ValueError("Must provide metric_fn for custom method")
        return select_neurons_custom(activations=activations, target=target, n_select=n_select, **kwargs)
    elif method == "stability":
        return select_neurons_stability(activations=activations, target=target, n_select=n_select, **kwargs)
    else:
        raise ValueError(f"Unknown selection method: {method}")
