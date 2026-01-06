"""HypotheSAEs: SAEs for hypothesis generation."""

# Version information
__version__ = "0.0.3"

# Import key functions and classes to expose at the package level
from .quickstart import (
    set_seed,
    train_sae,
    interpret_sae,
    generate_hypotheses,
    generate_hypotheses_meta,
    evaluate_hypotheses
)

from .meta_features import (
    MetaFeatureSet,
    build_meta_features,
    train_sae_ensemble,
    compute_cross_run_similarity,
    cluster_meta_features,
    interpret_meta_features,
    synthesize_meta_interpretation
)

from .sae import (
    SparseAutoencoder,
    load_model,
    get_multiple_sae_activations
)

from .embedding import (
    get_openai_embeddings,
    get_local_embeddings
)

from .interpret_neurons import (
    NeuronInterpreter,
    InterpretConfig,
    ScoringConfig,
    LLMConfig,
    SamplingConfig,
    sample_top_zero,
    sample_percentile_bins
)

from .select_neurons import select_neurons, select_neurons_stability, select_neurons_elasticnet_cv

from .evaluation import score_hypotheses

from .annotate import annotate_texts_with_concepts

from .utils import get_text_for_printing

# Define what gets imported with "from hypothesaes import *"
__all__ = [
    # Main workflow functions
    "set_seed",
    "train_sae",
    "interpret_sae", 
    "generate_hypotheses",
    "generate_hypotheses_meta",
    "evaluate_hypotheses",
    
    # Core classes
    "SparseAutoencoder",
    "load_model",
    "get_multiple_sae_activations",
    
    # Meta-features
    "MetaFeatureSet",
    "build_meta_features",
    "train_sae_ensemble",
    "compute_cross_run_similarity",
    "cluster_meta_features",
    "interpret_meta_features",
    "synthesize_meta_interpretation",
    
    # Embedding functions
    "get_openai_embeddings",
    "get_local_embeddings",
    
    # Interpretation classes
    "NeuronInterpreter",
    "InterpretConfig",
    "ScoringConfig",
    "LLMConfig",
    "SamplingConfig",
    "sample_top_zero",
    "sample_percentile_bins",
    
    # Selection and evaluation
    "select_neurons",
    "select_neurons_stability",
    "select_neurons_elasticnet_cv",
    "score_hypotheses",
    "annotate_texts_with_concepts",
    
    # Utilities
    "get_text_for_printing"
]