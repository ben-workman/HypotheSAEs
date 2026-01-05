"""Meta-feature stability: ensemble SAEs, cross-run alignment, clustering, and pooled activations."""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from .sae import SparseAutoencoder, dictionary_from_model
from .quickstart import train_sae, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_sae_ensemble(
    embeddings: np.ndarray,
    M: Union[int, List[int]],
    K: int,
    seeds: List[int],
    *,
    checkpoint_dir: Optional[str] = None,
    archetypal_opts: Optional[dict] = None,
    freeze_prototypes: bool = True,
    **train_kwargs
) -> Tuple[List[SparseAutoencoder], List[np.ndarray]]:
    """
    Train multiple SAEs with different random seeds.
    
    Args:
        embeddings: Training embeddings (n_samples, input_dim)
        M: Number of dictionary atoms (or list for matryoshka)
        K: Number of active neurons per input
        seeds: List of random seeds for each run
        checkpoint_dir: Directory to save/load checkpoints
        archetypal_opts: Options for archetypal (RA) decoder; None for free decoder
        freeze_prototypes: If True and archetypal_opts provided, compute prototypes
                          once and reuse across all runs
        **train_kwargs: Additional arguments passed to train_sae
    
    Returns:
        Tuple of (list of trained SAEs, list of decoder dictionaries)
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    input_dim = embeddings.shape[1]
    
    # Freeze prototypes across runs if using archetypal decoder
    fixed_opts = archetypal_opts
    if freeze_prototypes and archetypal_opts is not None:
        from .quickstart import _freeze_prototypes_for_benchmark
        fixed_opts = _freeze_prototypes_for_benchmark(
            embeddings=embeddings,
            archetypal_opts=archetypal_opts,
            checkpoint_dir=checkpoint_dir,
            input_dim=input_dim
        )
    
    models = []
    dictionaries = []
    
    for seed in tqdm(seeds, desc="Training SAE ensemble"):
        set_seed(seed)
        model = train_sae(
            embeddings=embeddings,
            M=M,
            K=K,
            checkpoint_dir=checkpoint_dir,
            archetypal_opts=fixed_opts,
            **train_kwargs
        )
        models.append(model)
        dictionaries.append(dictionary_from_model(model))
    
    return models, dictionaries


def _get_all_activations(
    sae_list: List[SparseAutoencoder],
    X: np.ndarray
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Get activations from all SAEs concatenated.
    
    Returns:
        activations: (n_samples, total_features) array
        feature_info: List of (run_idx, local_feat_idx) for each column
    """
    if isinstance(X, np.ndarray):
        X_tensor = torch.from_numpy(X).float().to(device)
    else:
        X_tensor = X.to(device)
    
    all_acts = []
    feature_info = []
    
    for run_idx, sae in enumerate(sae_list):
        acts = sae.get_activations(X_tensor)
        all_acts.append(acts)
        n_features = acts.shape[1]
        feature_info.extend([(run_idx, j) for j in range(n_features)])
    
    return np.concatenate(all_acts, axis=1), feature_info


def compute_cross_run_similarity(
    sae_list: List[SparseAutoencoder],
    dictionaries: List[np.ndarray],
    X_align: np.ndarray,
    alpha: float = 0.5,
    show_progress: bool = True
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute pairwise similarity between all features across all runs.
    
    Similarity = alpha * |cos(decoder_i, decoder_j)| + (1-alpha) * |corr(act_i, act_j)|
    
    Args:
        sae_list: List of trained SAEs
        dictionaries: List of decoder dictionaries (one per SAE)
        X_align: Alignment dataset to compute activation correlations
        alpha: Weight for decoder cosine vs activation correlation
        show_progress: Whether to show progress bar
    
    Returns:
        similarity: (total_features, total_features) similarity matrix
        feature_info: List of (run_idx, local_feat_idx) for each row/column
    """
    # Get all activations
    all_acts, feature_info = _get_all_activations(sae_list, X_align)
    n_total = all_acts.shape[1]
    
    # Stack all decoder vectors (normalized)
    all_decoders = []
    for D in dictionaries:
        D_norm = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        all_decoders.append(D_norm)
    all_decoders = np.vstack(all_decoders)  # (total_features, input_dim)
    
    # Compute decoder cosine similarity
    decoder_sim = np.abs(all_decoders @ all_decoders.T)
    
    # Compute activation correlation
    # Standardize activations for correlation
    act_mean = all_acts.mean(axis=0, keepdims=True)
    act_std = all_acts.std(axis=0, keepdims=True) + 1e-12
    act_z = (all_acts - act_mean) / act_std
    
    # Correlation = dot product of z-scored vectors / n
    act_corr = np.abs(act_z.T @ act_z / all_acts.shape[0])
    
    # Blend similarities
    similarity = alpha * decoder_sim + (1 - alpha) * act_corr
    
    # Zero out self-similarity for features from the same run
    # (we don't want to cluster features from the same run together)
    for i in range(n_total):
        run_i = feature_info[i][0]
        for j in range(n_total):
            run_j = feature_info[j][0]
            if run_i == run_j and i != j:
                # Keep similarity but it's valid - features from same run
                # can still be matched to different clusters
                pass
    
    return similarity, feature_info


def cluster_meta_features(
    similarity: np.ndarray,
    feature_info: List[Tuple[int, int]],
    n_runs: int,
    threshold: float = 0.7,
    method: str = "average",
    min_support: float = 0.0
) -> Tuple[List[List[Tuple[int, int]]], np.ndarray, np.ndarray]:
    """
    Cluster features into meta-features using hierarchical clustering.
    
    Args:
        similarity: (n_features, n_features) similarity matrix
        feature_info: List of (run_idx, local_feat_idx) for each feature
        n_runs: Total number of SAE runs
        threshold: Similarity threshold for clustering (higher = tighter clusters)
        method: Linkage method ('average', 'complete', 'single')
        min_support: Minimum fraction of runs a cluster must appear in
    
    Returns:
        clusters: List of clusters, each cluster is list of (run_idx, local_feat_idx)
        support: Fraction of runs represented in each cluster
        coherence: Mean intra-cluster similarity for each cluster
    """
    n_features = similarity.shape[0]
    
    # Convert similarity to distance
    # Clip to [0, 1] to avoid numerical issues
    sim_clipped = np.clip(similarity, 0, 1)
    distance = 1 - sim_clipped
    
    # Make symmetric and zero diagonal
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    
    # Convert to condensed form for scipy
    condensed = squareform(distance, checks=False)
    
    # Hierarchical clustering
    Z = linkage(condensed, method=method)
    
    # Cut tree at threshold (convert similarity threshold to distance)
    dist_threshold = 1 - threshold
    labels = fcluster(Z, t=dist_threshold, criterion='distance')
    
    # Group features by cluster
    cluster_dict: Dict[int, List[Tuple[int, int]]] = {}
    for feat_idx, cluster_id in enumerate(labels):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(feature_info[feat_idx])
    
    # Compute support and coherence for each cluster
    clusters = []
    support_list = []
    coherence_list = []
    
    for cluster_id, members in cluster_dict.items():
        # Support: fraction of runs with at least one member
        runs_in_cluster = set(run_idx for run_idx, _ in members)
        support = len(runs_in_cluster) / n_runs
        
        # Skip clusters with insufficient support
        if support < min_support:
            continue
        
        # Coherence: mean pairwise similarity within cluster
        if len(members) > 1:
            member_indices = [
                i for i, fi in enumerate(feature_info) if fi in members
            ]
            pairwise_sims = []
            for i, idx_i in enumerate(member_indices):
                for idx_j in member_indices[i+1:]:
                    pairwise_sims.append(similarity[idx_i, idx_j])
            coherence = np.mean(pairwise_sims) if pairwise_sims else 1.0
        else:
            coherence = 1.0
        
        clusters.append(members)
        support_list.append(support)
        coherence_list.append(coherence)
    
    # Sort by support (descending), then coherence
    order = np.lexsort((coherence_list, [-s for s in support_list]))
    clusters = [clusters[i] for i in order]
    support_arr = np.array([support_list[i] for i in order])
    coherence_arr = np.array([coherence_list[i] for i in order])
    
    return clusters, support_arr, coherence_arr


@dataclass
class MetaFeatureSet:
    """
    Container for meta-features with methods to compute pooled activations.
    
    Attributes:
        clusters: List of clusters, each cluster is list of (run_idx, local_feat_idx)
        support: Fraction of runs represented in each cluster
        coherence: Mean intra-cluster similarity for each cluster
        sae_list: List of SAEs used to build the meta-features
        _act_stats: Cached activation statistics for standardization
    """
    clusters: List[List[Tuple[int, int]]]
    support: np.ndarray
    coherence: np.ndarray
    sae_list: List[SparseAutoencoder]
    _act_stats: Optional[Dict[Tuple[int, int], Tuple[float, float]]] = field(
        default=None, repr=False
    )
    
    @property
    def n_meta_features(self) -> int:
        return len(self.clusters)
    
    @property
    def n_runs(self) -> int:
        return len(self.sae_list)
    
    def compute_activation_stats(self, X_align: np.ndarray) -> None:
        """
        Compute per-feature mean and std on alignment set for standardization.
        """
        all_acts, feature_info = _get_all_activations(self.sae_list, X_align)
        
        self._act_stats = {}
        for feat_idx, (run_idx, local_idx) in enumerate(feature_info):
            acts = all_acts[:, feat_idx]
            self._act_stats[(run_idx, local_idx)] = (
                float(np.mean(acts)),
                float(np.std(acts) + 1e-12)
            )
    
    def get_activations(
        self,
        X: np.ndarray,
        pooling: str = "softmax",
        tau: float = 1.0,
        standardize: bool = True,
        X_align: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute pooled meta-feature activations.
        
        Args:
            X: Input embeddings (n_samples, input_dim)
            pooling: Pooling method - 'max', 'softmax', or 'mean'
            tau: Temperature for softmax pooling
            standardize: Whether to z-score activations before pooling
            X_align: If standardize=True and stats not cached, compute stats on this
        
        Returns:
            meta_activations: (n_samples, n_meta_features) array
        """
        # Compute activation stats if needed
        if standardize and self._act_stats is None:
            if X_align is None:
                X_align = X
            self.compute_activation_stats(X_align)
        
        # Get all activations
        all_acts, feature_info = _get_all_activations(self.sae_list, X)
        n_samples = all_acts.shape[0]
        
        # Build index for fast lookup
        feat_to_col = {fi: i for i, fi in enumerate(feature_info)}
        
        # Compute pooled activations for each meta-feature
        meta_acts = np.zeros((n_samples, self.n_meta_features), dtype=np.float32)
        
        for mf_idx, cluster in enumerate(self.clusters):
            # Gather activations for all features in this cluster
            cluster_acts = []
            for run_idx, local_idx in cluster:
                col = feat_to_col[(run_idx, local_idx)]
                acts = all_acts[:, col].copy()
                
                # Standardize if requested
                if standardize and self._act_stats is not None:
                    mean, std = self._act_stats[(run_idx, local_idx)]
                    acts = (acts - mean) / std
                
                cluster_acts.append(acts)
            
            cluster_acts = np.stack(cluster_acts, axis=1)  # (n_samples, cluster_size)
            
            # Pool across cluster members
            if pooling == "max":
                meta_acts[:, mf_idx] = np.max(cluster_acts, axis=1)
            elif pooling == "softmax":
                # Softmax pooling: tau * log(sum(exp(a/tau)))
                scaled = cluster_acts / tau
                max_scaled = np.max(scaled, axis=1, keepdims=True)
                exp_scaled = np.exp(scaled - max_scaled)  # numerical stability
                meta_acts[:, mf_idx] = tau * (max_scaled.squeeze() + np.log(np.sum(exp_scaled, axis=1)))
            elif pooling == "mean":
                meta_acts[:, mf_idx] = np.mean(cluster_acts, axis=1)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
        
        return meta_acts
    
    def get_cluster_info(self) -> List[Dict[str, Any]]:
        """Get summary info for each meta-feature cluster."""
        info = []
        for mf_idx, cluster in enumerate(self.clusters):
            runs_in_cluster = set(run_idx for run_idx, _ in cluster)
            info.append({
                "meta_feature_idx": mf_idx,
                "n_members": len(cluster),
                "n_runs": len(runs_in_cluster),
                "support": float(self.support[mf_idx]),
                "coherence": float(self.coherence[mf_idx]),
                "members": cluster
            })
        return info
    
    def get_representative_features(
        self,
        n_per_cluster: int = 1
    ) -> List[List[Tuple[int, int]]]:
        """
        Get representative feature(s) from each cluster for interpretation.
        
        Returns list of lists, where each inner list has up to n_per_cluster
        (run_idx, local_feat_idx) tuples representing diverse runs.
        """
        representatives = []
        for cluster in self.clusters:
            # Pick features from different runs
            by_run: Dict[int, List[int]] = {}
            for run_idx, local_idx in cluster:
                if run_idx not in by_run:
                    by_run[run_idx] = []
                by_run[run_idx].append(local_idx)
            
            # Take one from each run until we have enough
            reps = []
            run_order = sorted(by_run.keys())
            for run_idx in run_order:
                if len(reps) >= n_per_cluster:
                    break
                reps.append((run_idx, by_run[run_idx][0]))
            
            representatives.append(reps)
        
        return representatives


def build_meta_features(
    embeddings: np.ndarray,
    M: Union[int, List[int]],
    K: int,
    seeds: List[int],
    *,
    X_align: Optional[np.ndarray] = None,
    checkpoint_dir: Optional[str] = None,
    archetypal_opts: Optional[dict] = None,
    alpha: float = 0.5,
    cluster_threshold: float = 0.7,
    cluster_method: str = "average",
    min_support: float = 0.3,
    show_progress: bool = True,
    **train_kwargs
) -> MetaFeatureSet:
    """
    End-to-end meta-feature construction: train ensemble, align, cluster.
    
    Args:
        embeddings: Training embeddings (n_samples, input_dim)
        M: Number of dictionary atoms (or list for matryoshka)
        K: Number of active neurons per input
        seeds: List of random seeds for each SAE run
        X_align: Alignment set for computing feature similarity; if None, uses embeddings
        checkpoint_dir: Directory to save/load checkpoints
        archetypal_opts: Options for archetypal decoder; None for free decoder
        alpha: Weight for decoder cosine vs activation correlation in similarity
        cluster_threshold: Similarity threshold for clustering
        cluster_method: Linkage method for hierarchical clustering
        min_support: Minimum fraction of runs a cluster must appear in
        show_progress: Whether to show progress bars
        **train_kwargs: Additional arguments passed to train_sae
    
    Returns:
        MetaFeatureSet ready for computing pooled activations
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    
    if X_align is None:
        X_align = embeddings
    else:
        X_align = np.array(X_align, dtype=np.float32)
    
    # Train ensemble
    sae_list, dictionaries = train_sae_ensemble(
        embeddings=embeddings,
        M=M,
        K=K,
        seeds=seeds,
        checkpoint_dir=checkpoint_dir,
        archetypal_opts=archetypal_opts,
        **train_kwargs
    )
    
    # Compute cross-run similarity
    if show_progress:
        print("Computing cross-run feature similarity...")
    similarity, feature_info = compute_cross_run_similarity(
        sae_list=sae_list,
        dictionaries=dictionaries,
        X_align=X_align,
        alpha=alpha,
        show_progress=show_progress
    )
    
    # Cluster into meta-features
    if show_progress:
        print("Clustering into meta-features...")
    clusters, support, coherence = cluster_meta_features(
        similarity=similarity,
        feature_info=feature_info,
        n_runs=len(seeds),
        threshold=cluster_threshold,
        method=cluster_method,
        min_support=min_support
    )
    
    if show_progress:
        print(f"Found {len(clusters)} meta-features with min_support >= {min_support}")
    
    # Build MetaFeatureSet
    mf_set = MetaFeatureSet(
        clusters=clusters,
        support=support,
        coherence=coherence,
        sae_list=sae_list
    )
    
    # Pre-compute activation stats on alignment set
    mf_set.compute_activation_stats(X_align)
    
    return mf_set

def synthesize_meta_interpretation(
    member_interpretations: List[str],
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    timeout: float = 60.0
) -> str:
    """
    Synthesize a meta-feature description from individual member interpretations.
    
    Args:
        member_interpretations: List of interpretation strings from cluster members
        model: LLM model to use
        max_retries: Number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        Synthesized meta-feature description
    """
    from .llm_api import get_completion
    
    if not member_interpretations:
        return "No interpretations available"
    
    if len(member_interpretations) == 1:
        return member_interpretations[0]
    
    # Filter out empty interpretations
    valid_interps = [i for i in member_interpretations if i and i.strip()]
    if not valid_interps:
        return "No valid interpretations available"
    
    prompt = (
        "You are analyzing a cluster of related features from multiple sparse autoencoder runs. "
        "Each feature in the cluster captures a similar concept, but may describe it slightly differently.\n\n"
        "Here are the individual feature interpretations:\n"
    )
    
    for i, interp in enumerate(valid_interps, 1):
        prompt += f"{i}. {interp}\n"
    
    prompt += (
        "\nPlease synthesize these into a single, concise description that captures the common theme. "
        "The description should be specific and actionable (something one could look for in text). "
        "Respond with only the synthesized description, no explanation."
    )
    
    for attempt in range(max_retries):
        try:
            response = get_completion(
                prompt=prompt,
                model=model,
                max_tokens=100,
                temperature=0.3,
                timeout=timeout
            )
            return response.strip().strip('"').strip("- ")
        except Exception as e:
            if attempt == max_retries - 1:
                # Return the first interpretation as fallback
                return valid_interps[0]
            import time
            time.sleep(2 ** attempt)
    
    return valid_interps[0]


def interpret_meta_features(
    texts: List[str],
    embeddings: np.ndarray,
    meta_feature_set: "MetaFeatureSet",
    meta_feature_indices: List[int],
    *,
    n_members_per_cluster: int = 3,
    n_candidates_per_member: int = 1,
    n_examples_for_interpretation: int = 20,
    max_words_per_example: int = 256,
    interpreter_model: str = "gpt-4o",
    synthesizer_model: str = "gpt-4o-mini",
    annotator_model: str = "gpt-4o-mini",
    cache_name: Optional[str] = None,
    n_workers: int = 6,
    show_progress: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Interpret meta-features by first interpreting member features, then synthesizing.
    
    Args:
        texts: Input text examples
        embeddings: Input embeddings
        meta_feature_set: The MetaFeatureSet to interpret
        meta_feature_indices: Which meta-features to interpret
        n_members_per_cluster: Number of cluster members to interpret per meta-feature
        n_candidates_per_member: Number of interpretation candidates per member
        interpreter_model: LLM for generating member interpretations
        synthesizer_model: LLM for synthesizing meta-interpretation
        annotator_model: LLM for scoring interpretations
        cache_name: Cache name for interpretations
        n_workers: Number of parallel workers
        show_progress: Whether to show progress
    
    Returns:
        Dict mapping meta_feature_idx to {
            'member_interpretations': [(run_idx, local_idx, [interps]), ...],
            'synthesized_interpretation': str,
            'support': float,
            'coherence': float
        }
    """
    from .interpret_neurons import (
        NeuronInterpreter, InterpretConfig, LLMConfig, SamplingConfig, sample_top_zero
    )
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Get all activations from all SAEs
    all_acts, feature_info = _get_all_activations(meta_feature_set.sae_list, embeddings)
    feat_to_col = {fi: i for i, fi in enumerate(feature_info)}
    
    # Collect all features to interpret
    features_to_interpret = []  # (mf_idx, run_idx, local_idx)
    
    for mf_idx in meta_feature_indices:
        cluster = meta_feature_set.clusters[mf_idx]
        
        # Select diverse members
        by_run: Dict[int, List[int]] = {}
        for run_idx, local_idx in cluster:
            if run_idx not in by_run:
                by_run[run_idx] = []
            by_run[run_idx].append(local_idx)
        
        count = 0
        for run_idx in sorted(by_run.keys()):
            if count >= n_members_per_cluster:
                break
            features_to_interpret.append((mf_idx, run_idx, by_run[run_idx][0]))
            count += 1
    
    if not features_to_interpret:
        return {}
    
    # Build activation matrix for interpretation
    # Map global feature indices to local interpretation indices
    interp_cols = []
    interp_mapping = []  # (mf_idx, run_idx, local_idx, interp_col)
    
    for i, (mf_idx, run_idx, local_idx) in enumerate(features_to_interpret):
        col = feat_to_col[(run_idx, local_idx)]
        interp_cols.append(col)
        interp_mapping.append((mf_idx, run_idx, local_idx, i))
    
    interp_activations = all_acts[:, interp_cols]
    interp_indices = list(range(len(interp_cols)))
    
    # Interpret all features
    interpreter = NeuronInterpreter(
        interpreter_model=interpreter_model,
        annotator_model=annotator_model,
        n_workers_interpretation=n_workers,
        cache_name=cache_name
    )
    
    config = InterpretConfig(
        sampling=SamplingConfig(
            function=sample_top_zero,
            n_examples=n_examples_for_interpretation,
            max_words_per_example=max_words_per_example
        ),
        llm=LLMConfig(
            temperature=0.7,
            max_interpretation_tokens=50,
            timeout=120
        ),
        n_candidates=n_candidates_per_member
    )
    
    if show_progress:
        print(f"Interpreting {len(interp_indices)} cluster members...")
    
    raw_interpretations = interpreter.interpret_neurons(
        texts=texts,
        activations=interp_activations,
        neuron_indices=interp_indices,
        config=config
    )
    
    # Organize by meta-feature
    mf_member_interps: Dict[int, List[Tuple[int, int, List[str]]]] = {}
    for mf_idx, run_idx, local_idx, interp_col in interp_mapping:
        if mf_idx not in mf_member_interps:
            mf_member_interps[mf_idx] = []
        interps = raw_interpretations.get(interp_col, [])
        mf_member_interps[mf_idx].append((run_idx, local_idx, interps))
    
    # Synthesize meta-interpretations
    if show_progress:
        print(f"Synthesizing {len(meta_feature_indices)} meta-feature descriptions...")
    
    results = {}
    cluster_info = meta_feature_set.get_cluster_info()
    
    for mf_idx in meta_feature_indices:
        member_interps = mf_member_interps.get(mf_idx, [])
        
        # Flatten all interpretations for synthesis
        all_interp_texts = []
        for run_idx, local_idx, interps in member_interps:
            all_interp_texts.extend(interps)
        
        # Synthesize
        synthesized = synthesize_meta_interpretation(
            member_interpretations=all_interp_texts,
            model=synthesizer_model
        )
        
        info = cluster_info[mf_idx]
        results[mf_idx] = {
            'member_interpretations': member_interps,
            'synthesized_interpretation': synthesized,
            'support': info['support'],
            'coherence': info['coherence'],
            'n_members': info['n_members'],
            'n_runs': info['n_runs']
        }
    
    return results

