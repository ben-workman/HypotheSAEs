"""High-level functions for hypothesis generation using SAEs."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple, Dict
import torch, os, random
from pathlib import Path

from .sae import SparseAutoencoder, load_model, get_sae_checkpoint_name
from .select_neurons import select_neurons
from .interpret_neurons import NeuronInterpreter, InterpretConfig, ScoringConfig, LLMConfig, SamplingConfig
from .utils import get_text_for_printing
from .annotate import annotate_texts_with_concepts
from .evaluation import score_hypotheses
from .llm_api import get_completion

BASE_DIR = Path(__file__).parent.parent

def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_sae(
    embeddings: Union[List, np.ndarray],
    M: int,
    K: int,
    *,
    matryoshka_prefix_lengths: Optional[List[int]] = None,
    batch_topk: bool = False,
    checkpoint_dir: Optional[str] = None,
    overwrite_checkpoint: bool = False,
    val_embeddings: Optional[Union[List, np.ndarray]] = None,
    aux_k: Optional[int] = None,
    multi_k: Optional[int] = None,
    dead_neuron_threshold_steps: int = 256,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    n_epochs: int = 100,
    aux_coef: float = 1/32,
    multi_coef: float = 0.0,
    patience: int = 3,
    clip_grad: float = 1.0,
    show_progress: bool = True,
    seed: Optional[int] = 123,
) -> SparseAutoencoder:
    """Train a Sparse Autoencoder or load an existing one."""
    if seed is not None:
        set_seed(seed)

    embeddings = np.array(embeddings)
    input_dim = embeddings.shape[1]
    
    X = torch.tensor(embeddings, dtype=torch.float)
    X_val = torch.tensor(val_embeddings, dtype=torch.float) if val_embeddings is not None else None
    
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = get_sae_checkpoint_name(M, K, matryoshka_prefix_lengths)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.exists(checkpoint_path) and not overwrite_checkpoint:
            return load_model(checkpoint_path)
    
    sae = SparseAutoencoder(
        input_dim=input_dim,
        m_total_neurons=M,
        k_active_neurons=K,
        aux_k=aux_k,
        multi_k=multi_k,
        dead_neuron_threshold_steps=dead_neuron_threshold_steps,
        prefix_lengths=matryoshka_prefix_lengths,
        use_batch_topk=batch_topk,
    )
    
    sae.fit(
        X_train=X,
        X_val=X_val,
        save_dir=checkpoint_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        aux_coef=aux_coef,
        multi_coef=multi_coef,
        patience=patience,
        clip_grad=clip_grad,
        show_progress=show_progress,
        seed=seed,
    )

    return sae

def interpret_sae(
    texts: List[str],
    embeddings: Union[List, np.ndarray],
    sae: SparseAutoencoder,
    *,
    neuron_indices: Optional[List[int]] = None,
    n_random_neurons: Optional[int] = None,
    n_top_neurons: Optional[int] = None,
    interpreter_model: str = "gpt-4.1",
    n_examples_for_interpretation: int = 20,
    max_words_per_example: int = 256,
    interpret_temperature: float = 0.7,
    max_interpretation_tokens: int = 50,
    n_candidates: int = 1,
    print_examples_n: int = 3,
    print_examples_max_chars: int = 1024,
    task_specific_instructions: Optional[str] = None,
    random_seed: Optional[int] = 0,
) -> Dict:
    """Interpret neurons in a Sparse Autoencoder."""
    selection_params = [neuron_indices, n_random_neurons, n_top_neurons]
    if sum(p is not None for p in selection_params) != 1:
        raise ValueError("Exactly one of neuron_indices, n_random_neurons, or n_top_neurons must be provided")
    
    if not isinstance(embeddings, torch.Tensor):
        X = torch.tensor(embeddings, dtype=torch.float)
    else:
        X = embeddings
    
    # Get activations from SAE
    activations = sae.get_activations(X)
    print(f"Activations shape: {activations.shape}")
    # Compute prevalence for each neuron (percentage of examples where activation != 0)
    activation_counts = (activations != 0).sum(axis=0)
    activation_percent = activation_counts / activations.shape[0] * 100
    
    # Select neurons to interpret
    total_neurons = activations.shape[1]
    if neuron_indices is None:
        if n_random_neurons is not None:
            rng = np.random.default_rng(random_seed)
            neuron_indices = rng.choice(total_neurons, size=n_random_neurons, replace=False)
        else:  # n_top_neurons is not None
            if n_top_neurons > total_neurons:
                raise ValueError(f"n_top_neurons ({n_top_neurons}) cannot exceed total neurons ({total_neurons})")
            neuron_indices = np.argsort(activation_counts)[-n_top_neurons:][::-1]
    
    # Set up interpreter
    interpreter = NeuronInterpreter(
        interpreter_model=interpreter_model,
    )

    interpret_config = InterpretConfig(
        sampling=SamplingConfig(
            n_examples=n_examples_for_interpretation,
            max_words_per_example=max_words_per_example,
            random_seed=random_seed,
        ),
        llm=LLMConfig(
            temperature=interpret_temperature,
            max_interpretation_tokens=max_interpretation_tokens,
        ),
        n_candidates=n_candidates,
        task_specific_instructions=task_specific_instructions,
    )

    # Get interpretations
    interpretations = interpreter.interpret_neurons(
        texts=texts,
        activations=activations,
        neuron_indices=neuron_indices,
        config=interpret_config,
    )

    # Find top activating examples for each neuron if requested
    results_list = []
    for idx in neuron_indices:
        neuron_activations = activations[:, idx]
        result_dict = {
            "neuron_idx": int(idx),
            "interpretation": interpretations[idx][0] if n_candidates == 1 else interpretations[idx]
        }
        
        if print_examples_n > 0:
            top_indices = np.argsort(neuron_activations)[-print_examples_n:][::-1]
            top_examples = [texts[i] for i in top_indices]
            print(f"\nNeuron {idx} ({activation_percent[idx]:.1f}% active): {interpretations[idx][0]}")
            print(f"\nTop activating examples:")
            for i, example in enumerate(top_examples, 1):
                print(f"{i}. {get_text_for_printing(example, max_chars=print_examples_max_chars)}")
                result_dict[f"top_example_{i}"] = example
            print("-"*100)
                
        results_list.append(result_dict)

    return pd.DataFrame(results_list)

def generate_hypotheses(
    texts: List[str],
    labels: Union[List[int], List[float], np.ndarray],
    embeddings: Union[List, np.ndarray],
    sae: SparseAutoencoder,
    *,
    cache_name: Optional[str] = None,
    classification: Optional[bool] = None,
    selection_method: str = "separation_score",
    n_selected_neurons: int = 20,
    interpreter_model: str = "gpt-4.1",
    annotator_model: str = "gpt-4.1-mini",
    n_examples_for_interpretation: int = 20,
    max_words_per_example: int = 256,
    interpret_temperature: float = 0.7,
    max_interpretation_tokens: int = 50,
    n_candidate_interpretations: int = 1,
    n_scoring_examples: int = 100,
    scoring_metric: str = "f1",
    n_workers_interpretation: int = 10,
    n_workers_annotation: int = 30,
    task_specific_instructions: Optional[str] = None,
    # NEW:
    group_ids: Optional[np.ndarray] = None,
    filter: bool = False,
    random_state: Optional[int] = 123,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """Generate interpretable hypotheses from text data using SAEs."""
    if random_state is not None:
        set_seed(random_state)

    labels = np.array(labels)
    if not isinstance(embeddings, torch.Tensor):
        X = torch.tensor(embeddings, dtype=torch.float)
    else:
        X = embeddings
    
    if classification is None:  # Heuristic check for classification
        classification = np.all(np.isin(np.random.choice(labels, size=min(1000, labels.shape[0]), replace=True), [0, 1]))
    
    print(f"Embeddings shape: {X.shape}")

    # Get activations from SAE
    activations = sae.get_activations(X)
    print(f"Activations shape: {activations.shape}")

    print(f"\nStep 1: Selecting top {n_selected_neurons} predictive neurons")
    if n_selected_neurons > activations.shape[1]:
        raise ValueError(f"n_selected_neurons ({n_selected_neurons}) can be at most the total number of neurons ({activations.shape[1]})")
    
    extra_args: Dict = {}
    if group_ids is not None:
        extra_args["group_ids"] = group_ids

    selected_neurons, scores = select_neurons(
        activations=activations,
        target=labels,
        n_select=n_selected_neurons,
        method=selection_method,
        classification=classification,
        **extra_args,
    )

    print(f"\nStep 2: Interpreting selected neurons")
    interpreter = NeuronInterpreter(
        cache_name=cache_name,
        interpreter_model=interpreter_model,
        annotator_model=annotator_model,
        n_workers_interpretation=n_workers_interpretation,
        n_workers_annotation=n_workers_annotation,
    )

    interpret_config = InterpretConfig(
        sampling=SamplingConfig(
            n_examples=n_examples_for_interpretation,
            max_words_per_example=max_words_per_example,
            random_seed=random_state,
        ),
        llm=LLMConfig(
            temperature=interpret_temperature,
            max_interpretation_tokens=max_interpretation_tokens,
        ),
        n_candidates=n_candidate_interpretations,
        task_specific_instructions=task_specific_instructions,
    )

    interpretations = interpreter.interpret_neurons(
        texts=texts,
        activations=activations,
        neuron_indices=selected_neurons,
        config=interpret_config,
    )

    def _mentions_relevant_feature(text: Optional[str]) -> bool:
        """LLM-based filter; defaults to 'yes' on failure to be permissive."""
        if text is None:
            return False
        prompt = (
            "Answer 'yes' if the line below describes a teacher/student behavior, speech pattern, "
            "or teaching style; answer 'no' if it mainly describes a specific math concept or a physical classroom object.\n\n"
            f"Text: {text}\n\nAnswer:"
        )
        try:
            resp = get_completion(
                model=annotator_model,
                prompt=prompt,
                max_tokens=2,
                timeout=15.0,
            )
            return str(resp).strip().lower().startswith("y")
        except Exception:
            return True  # permissive default

    # Prepare results dataframe
    results = []
    if n_scoring_examples == 0:
        # Skip scoring entirely; optionally filter out whole neurons
        for idx, score in zip(selected_neurons, scores):
            if filter and not any(_mentions_relevant_feature(t) for t in interpretations[idx]):
                continue
            results.append({
                'neuron_idx': idx,
                f'target_{selection_method}': score,
                'interpretation': interpretations[idx][0]
            })
    else:
        print(f"\nStep 3: Scoring Interpretations")
        scoring_config = ScoringConfig(n_examples=n_scoring_examples)

        interps_for_scoring = (
            {i: [t for t in lst if _mentions_relevant_feature(t)] for i, lst in interpretations.items()}
            if filter else interpretations
        )
        # Remove neurons that have no surviving candidate after filtering
        interps_for_scoring = {i: lst for i, lst in interps_for_scoring.items() if len(lst) > 0}

        metrics = interpreter.score_interpretations(
            texts=texts,
            activations=activations,
            interpretations=interps_for_scoring,
            config=scoring_config
        )
        
        for idx, score in zip(selected_neurons, scores):
            if filter and idx not in interps_for_scoring:
                continue
            cand_list = interps_for_scoring[idx] if filter else interpretations[idx]
            # Find best interpretation and its score
            best_interp = max(
                cand_list,
                key=lambda interp: metrics[idx][interp][scoring_metric]
            )
            best_score = metrics[idx][best_interp][scoring_metric]
            
            results.append({
                'neuron_idx': idx,
                f'target_{selection_method}': score,
                'interpretation': best_interp,
                f'{scoring_metric}_fidelity_score': best_score
            })

    df = pd.DataFrame(results)
    return df

def evaluate_hypotheses(
    hypotheses_df: pd.DataFrame,
    texts: List[str],
    labels: Union[List[int], List[float], np.ndarray],
    *,
    cache_name: Optional[str] = None,
    annotator_model: str = "gpt-4.1-mini",
    max_words_per_example: int = 256,
    classification: Optional[bool] = None,
    n_workers_annotation: int = 30,
    corrected_pval_threshold: float = 0.1,
) -> pd.DataFrame:
    """Evaluate hypotheses on a heldout dataset.
    
    Args:
        hypotheses_df: DataFrame from generate_hypotheses()
        texts: Heldout text examples
        labels: Heldout labels
        annotator_model: Model to use for annotation
        max_words_per_example: Maximum words per example for annotation
        classification: Whether this is a classification task. If None, inferred from labels
        cache_name: Optional string prefix for storing annotation cache
        
    Returns:
        DataFrame with original columns plus evaluation metrics
    """
    labels = np.array(labels)
    
    # Infer classification if not specified
    if classification is None:
        classification = np.all(np.isin(np.random.choice(labels, size=1000, replace=True), [0, 1]))

    # Extract hypotheses from dataframe
    hypotheses = hypotheses_df['interpretation'].tolist()
    
    # Step 1: Get annotations for each hypothesis on the texts
    print(f"Step 1: Annotating texts with {len(hypotheses)} hypotheses")
    hypothesis_annotations = annotate_texts_with_concepts(
        texts=texts,
        concepts=hypotheses,
        max_words_per_example=max_words_per_example,
        model=annotator_model,
        cache_name=cache_name,
        n_workers=n_workers_annotation,
    )
    
    # Step 2: Evaluate annotations against the true labels
    print("Step 2: Computing predictiveness of hypothesis annotations")
    metrics, evaluation_df = score_hypotheses(
        hypothesis_annotations=hypothesis_annotations,
        y_true=np.array(labels),
        classification=classification,
        corrected_pval_threshold=corrected_pval_threshold,
    )
    
    return metrics, evaluation_df
